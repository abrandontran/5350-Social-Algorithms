#!/usr/bin/env python3
"""Assignment 3 starter scaffold (player side only).

What this gives you:
- Baseline Ollama player calls
- Calibration sampling loops
- Answer-file generation in the required CSV format

What is intentionally left for you:
- Better prompts and prompt experiments
- Better normalization/answer canonicalization
- Plotting and deeper analysis for the report

Judging/scoring is intentionally separate and lives in `starter/judge.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

OLLAMA_URL = "http://localhost:11434/api/generate"


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text)
    return text


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text)


def build_day_calibration_prompt() -> str:
    return (
        "Return exactly one lowercase weekday name from this set: "
        "monday tuesday wednesday thursday friday saturday sunday. "
        "Output only the weekday."
    )


def build_fruit_b_prompt() -> str:
    return (
        "Return exactly one fruit that starts with the letter b. "
        "Output only the fruit, no punctuation, no explanation."
    )


def build_player_prompt(letter: str, category: str) -> str:
    return (
        "You are playing Scattergories. "
        f"Letter: {letter}. "
        f"Category: {category}. "
        "Return exactly one answer that starts with the required letter. "
        "Output only lowercase letters and spaces."
    )


@dataclass
class Question:
    question_id: str
    letter: str
    category: str


def load_questions(path: str | Path) -> list[Question]:
    out: list[Question] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                Question(
                    question_id=row["question_id"],
                    letter=row["letter"],
                    category=row["category"],
                )
            )
    return out


def parse_temps(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def ollama_generate(
    model: str,
    prompt: str,
    temperature: float,
    top_k: int | None = None,
    seed: int | None = None,
    max_tokens: int = 12,
) -> str:
    options: dict[str, object] = {
        "temperature": temperature,
        "num_predict": max_tokens,
    }
    if top_k is not None:
        options["top_k"] = top_k
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data.get("response", "")).strip()


def entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    return h


def kl_to_uniform(counts: Counter[str], support: Iterable[str]) -> float:
    support = list(support)
    total = sum(counts.values())
    if total == 0 or not support:
        return 0.0
    u = 1.0 / len(support)
    kl = 0.0
    for item in support:
        p = counts.get(item, 0) / total
        if p > 0:
            kl += p * math.log(p / u)
    return kl


def tv_to_uniform(counts: Counter[str], support: Iterable[str]) -> float:
    support = list(support)
    total = sum(counts.values())
    if total == 0 or not support:
        return 0.0
    u = 1.0 / len(support)
    return 0.5 * sum(abs((counts.get(item, 0) / total) - u) for item in support)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_calibration_prompt(task: str, prompt_file: str | None) -> str:
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()
    if task == "day":
        return build_day_calibration_prompt()
    if task == "fruitb":
        return build_fruit_b_prompt()
    raise ValueError(f"unknown task: {task}")


def run_calibration(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prompt = get_calibration_prompt(args.task, args.prompt_file)
    if args.task == "day":
        support = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
    else:
        support = None

    temps = parse_temps(args.temperatures)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for temp in temps:
        counts: Counter[str] = Counter()
        seed = args.seed_start
        for i in range(args.samples):
            raw = ollama_generate(
                model=args.model,
                prompt=prompt,
                temperature=temp,
                top_k=args.top_k,
                seed=seed,
                max_tokens=args.max_tokens,
            )
            if seed is not None:
                seed += 1

            ans = normalize_answer(raw)
            if args.task == "day":
                ans = ans.split()[0] if ans else ""
            counts[ans] += 1
            all_rows.append(
                {
                    "task": args.task,
                    "model": args.model,
                    "temperature": temp,
                    "sample_idx": i,
                    "prompt_id": args.prompt_id,
                    "raw": raw,
                    "answer_norm": ans,
                }
            )

        summary: dict[str, object] = {
            "task": args.task,
            "model": args.model,
            "temperature": temp,
            "samples": args.samples,
            "prompt_id": args.prompt_id,
            "unique_answers": len(counts),
            "entropy_nats": entropy_from_counts(counts),
            "top_answers": counts.most_common(20),
        }
        if support is not None:
            summary["kl_to_uniform"] = kl_to_uniform(counts, support)
            summary["tv_to_uniform"] = tv_to_uniform(counts, support)
        summaries.append(summary)

    samples_path = outdir / f"calibration_{args.task}_samples.csv"
    write_csv(samples_path, all_rows)

    summary_path = outdir / f"calibration_{args.task}_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print(f"Wrote {samples_path}")
    print(f"Wrote {summary_path}")
    print("TODO(student): make histogram plots and run prompt variants.")


def load_player_template(prompt_file: str | None) -> str | None:
    if not prompt_file:
        return None
    return Path(prompt_file).read_text(encoding="utf-8").strip()


def render_player_prompt(letter: str, category: str, template: str | None) -> str:
    if template is None:
        return build_player_prompt(letter=letter, category=category)
    try:
        return template.format(letter=letter, category=category)
    except KeyError as exc:
        raise ValueError(
            "Prompt template is missing required placeholders. "
            "Use {letter} and {category}."
        ) from exc


def run_generate_answers(args: argparse.Namespace) -> None:
    questions = load_questions(args.questions_csv)
    template = load_player_template(args.prompt_file)

    player_id = args.player_id or safe_name(args.model)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else outdir / f"answers_{player_id}.csv"

    rows: list[dict[str, object]] = []
    seed = args.seed_start
    for q in questions:
        prompt = render_player_prompt(q.letter, q.category, template)
        for r in range(args.rounds):
            raw = ollama_generate(
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed,
                max_tokens=args.max_tokens,
            )
            if seed is not None:
                seed += 1
            rows.append(
                {
                    "question_id": q.question_id,
                    "letter": q.letter,
                    "category": q.category,
                    "round_idx": r,
                    "answer": raw,
                    "model": args.model,
                    "player_id": player_id,
                    "temperature": args.temperature,
                    "top_k": args.top_k if args.top_k is not None else "",
                    "prompt_id": args.prompt_id,
                }
            )

    write_csv(out_path, rows)
    print(f"Wrote {out_path}")
    print(f"Rows: {len(rows)}")
    print("Next step: pass one or more answer CSV files to starter/judge.py.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assignment 3 starter CLI (player side)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibrate", help="Run baseline calibration sampling")
    p_cal.add_argument("--model", required=True, help="Ollama model tag")
    p_cal.add_argument("--task", choices=["day", "fruitb"], required=True)
    p_cal.add_argument("--samples", type=int, default=500)
    p_cal.add_argument("--temperatures", default="0.0,0.5,1.0,1.5,2.0,3.0,4.0,5.0,7.5,10.0")
    p_cal.add_argument("--top-k", type=int, default=None)
    p_cal.add_argument("--max-tokens", type=int, default=8)
    p_cal.add_argument("--seed-start", type=int, default=None)
    p_cal.add_argument("--prompt-file", default=None, help="Optional text file with full calibration prompt.")
    p_cal.add_argument("--prompt-id", default="baseline", help="Tag recorded in outputs.")
    p_cal.add_argument("--outdir", default="outputs")
    p_cal.set_defaults(func=run_calibration)

    p_gen = sub.add_parser("generate-answers", help="Generate one player answer CSV file")
    p_gen.add_argument("--model", required=True, help="Ollama model tag")
    p_gen.add_argument("--questions-csv", default="scattergories_questions.csv")
    p_gen.add_argument("--rounds", type=int, default=2)
    p_gen.add_argument("--temperature", type=float, default=0.9)
    p_gen.add_argument("--top-k", type=int, default=40)
    p_gen.add_argument("--max-tokens", type=int, default=16)
    p_gen.add_argument("--seed-start", type=int, default=None)
    p_gen.add_argument(
        "--prompt-file",
        default=None,
        help="Optional text template file for player prompt. Must include {letter} and {category}.",
    )
    p_gen.add_argument("--prompt-id", default="baseline", help="Tag recorded in outputs.")
    p_gen.add_argument("--player-id", default=None, help="Optional player id. Defaults to sanitized model tag.")
    p_gen.add_argument("--out", default=None, help="Optional output CSV path.")
    p_gen.add_argument("--outdir", default="outputs")
    p_gen.set_defaults(func=run_generate_answers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
