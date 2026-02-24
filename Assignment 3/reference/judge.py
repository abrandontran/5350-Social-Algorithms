#!/usr/bin/env python3
"""Judge Scattergories answer files with an OpenAI model.

Usage:
    python3 judge.py answers_a.csv
    python3 judge.py answers_a.csv answers_b.csv --out scores.csv
    python3 judge.py answers_a.csv answers_b.csv answers_c.csv --out scores.csv --details judged_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
REQUIRED_COLUMNS = ["question_id", "letter", "category", "round_idx", "answer"]


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text)
    return text


@dataclass
class AnswerRow:
    source_file: str
    player_id: str
    question_id: str
    letter: str
    category: str
    round_idx: str
    answer_raw: str
    answer_norm: str


class JudgeCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {}
        self.hits = 0
        self.calls = 0

    def _key(self, letter: str, category: str, answer_norm: str) -> str:
        return json.dumps([letter.lower(), category.lower(), answer_norm], ensure_ascii=True)

    def get(self, letter: str, category: str, answer_norm: str) -> bool | None:
        key = self._key(letter, category, answer_norm)
        if key in self.data:
            self.hits += 1
            return bool(self.data[key])
        return None

    def put(self, letter: str, category: str, answer_norm: str, value: bool) -> None:
        key = self._key(letter, category, answer_norm)
        self.data[key] = bool(value)

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")


class OpenAIJudge:
    def __init__(self, model: str, cache: JudgeCache, temperature: float | None, max_completion_tokens: int):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.model = model
        self.cache = cache
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    def is_valid(self, letter: str, category: str, answer_norm: str) -> bool:
        if not answer_norm:
            return False
        cached = self.cache.get(letter, category, answer_norm)
        if cached is not None:
            return cached

        self.cache.calls += 1
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are judging Scattergories answers. "
                        "Be strict but fair. Return only yes or no."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Letter: {letter}\n"
                        f"Category: {category}\n"
                        f"Answer: {answer_norm}\n"
                        "Question: Is this answer valid for this letter and category? "
                        "Return only yes or no."
                    ),
                },
            ],
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        req = urllib.request.Request(
            OPENAI_CHAT_URL,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            .lower()
        )
        value = content.startswith("y")
        self.cache.put(letter, category, answer_norm, value)
        return value


def load_answers(paths: list[str]) -> list[AnswerRow]:
    rows: list[AnswerRow] = []
    for path in paths:
        p = Path(path)
        player_id = p.stem
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path} missing required columns: {missing}")
            for row in reader:
                raw = row.get("answer", "")
                rows.append(
                    AnswerRow(
                        source_file=str(p),
                        player_id=player_id,
                        question_id=str(row["question_id"]),
                        letter=str(row["letter"]),
                        category=str(row["category"]),
                        round_idx=str(row["round_idx"]),
                        answer_raw=str(raw),
                        answer_norm=normalize_answer(str(raw)),
                    )
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def judge_rows(rows: list[AnswerRow], judge: OpenAIJudge, sleep_s: float) -> list[dict]:
    judged: list[dict] = []
    by_round: dict[tuple[str, str], list[int]] = defaultdict(list)

    for idx, row in enumerate(rows):
        valid = judge.is_valid(row.letter, row.category, row.answer_norm)
        judged.append(
            {
                "source_file": row.source_file,
                "player_id": row.player_id,
                "question_id": row.question_id,
                "letter": row.letter,
                "category": row.category,
                "round_idx": row.round_idx,
                "answer_raw": row.answer_raw,
                "answer_norm": row.answer_norm,
                "valid": int(valid),
            }
        )
        by_round[(row.question_id, row.round_idx)].append(idx)
        if sleep_s > 0:
            time.sleep(sleep_s)

    for key, idxs in by_round.items():
        valid_answers = [
            judged[i]["answer_norm"]
            for i in idxs
            if judged[i]["valid"] == 1 and judged[i]["answer_norm"] != ""
        ]
        counts = Counter(valid_answers)
        for i in idxs:
            ans = judged[i]["answer_norm"]
            valid = judged[i]["valid"] == 1
            collision = valid and ans != "" and counts.get(ans, 0) > 1
            score = int(valid and not collision)
            judged[i]["collision"] = int(collision)
            judged[i]["score"] = score
            judged[i]["round_key"] = f"{key[0]}::{key[1]}"
    return judged


def summarize_scores(judged_rows: list[dict], cache: JudgeCache) -> list[dict]:
    by_player: dict[str, list[dict]] = defaultdict(list)
    for row in judged_rows:
        by_player[row["player_id"]].append(row)

    summaries: list[dict] = []
    for player_id, rows in sorted(by_player.items()):
        total = len(rows)
        valid = sum(r["valid"] for r in rows)
        points = sum(r["score"] for r in rows)
        collisions = sum(r["collision"] for r in rows)
        distinct_valid = len({r["answer_norm"] for r in rows if r["valid"] == 1 and r["answer_norm"]})
        source_file = rows[0]["source_file"] if rows else ""
        summaries.append(
            {
                "player_id": player_id,
                "source_file": source_file,
                "total_answers": total,
                "valid_answers": valid,
                "valid_rate": (valid / total) if total else 0.0,
                "points": points,
                "avg_points_per_answer": (points / total) if total else 0.0,
                "collisions": collisions,
                "collision_rate": (collisions / total) if total else 0.0,
                "distinct_valid_answers": distinct_valid,
                "judge_api_calls": cache.calls,
                "judge_cache_hits": cache.hits,
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge Scattergories answer files.")
    parser.add_argument("answer_files", nargs="+", help="One or more answer CSV files.")
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI judge model.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature for judge calls. Omit for model default.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=8,
        help="Max completion tokens for judge response.",
    )
    parser.add_argument("--out", default="scores.csv", help="Summary score CSV output path.")
    parser.add_argument(
        "--details",
        default="judged_rows.csv",
        help="Row-level judged output CSV path.",
    )
    parser.add_argument(
        "--cache",
        default=".judge_cache.json",
        help="Path to JSON cache for (letter,category,answer)->validity.",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between judge calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_answers(args.answer_files)
    cache = JudgeCache(Path(args.cache))
    judge = OpenAIJudge(
        model=args.model,
        cache=cache,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )

    judged_rows = judge_rows(rows, judge=judge, sleep_s=args.sleep)
    scores = summarize_scores(judged_rows, cache=cache)
    cache.save()

    write_csv(Path(args.details), judged_rows)
    write_csv(Path(args.out), scores)

    print(f"Wrote {args.details}")
    print(f"Wrote {args.out}")
    print(f"Judge API calls: {cache.calls}")
    print(f"Judge cache hits: {cache.hits}")


if __name__ == "__main__":
    main()
