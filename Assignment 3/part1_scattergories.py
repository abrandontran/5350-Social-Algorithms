import asyncio
import pandas as pd
from collections import Counter
from tqdm.asyncio import tqdm
import requests

from reference.assignment3_starter import(
    normalize_answer,
)

MODEL = 'qwen2.5:7b'
SAMPLES = 500
TEMPERATURES = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
MAX_CONCURRENT = 15

PROMPT = (
    "We are playing a game of Scattergories. "
    "The category is 'Fruits' and the letter is 'B'. "
    "What is a fruit that starts with the letter 'B'? "
    "Output only the lowercase word: no puncuation, no explanation. "
    "For example, 'banana'"
)

async def fetch_fruit(sem, temp, session):
    async with sem:
        payload = {
            "model": MODEL,
            "prompt": PROMPT,
            "stream": False,
            "options": {
                "temperature": temp,
                "top_k": 40,
                "num_predict": 10
            }
        }

        # Using loop.run_in_executor to keep requests from blocking the event loop
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, lambda: requests.post(
                "http://localhost:11434/api/generate", json=payload, timeout=10
            ))
            raw_ans = response.json().get('response', "")
            ans = normalize_answer(raw_ans)
            ans = ans.split()[-1] if ans else "invalid"
            return {"temperature": temp, "answer": ans, "raw": raw_ans}
        except Exception as e:
            return {"temperature": temp, "answer": "error", "raw": str(e)}

async def main():
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []
    
    print(f"Starting Scattergories Fruit Race ({len(TEMPERATURES) * SAMPLES} total requests)")
    
    for temp in TEMPERATURES:
        for _ in range(SAMPLES):
            tasks.append(fetch_fruit(sem, temp, None))
            
    results = await tqdm.gather(*tasks)
    
    df = pd.DataFrame(results)
    df.to_csv('working/part1_scattergories.csv', index=False)
    
    print("\nDone! Summary of results:")
    for temp in TEMPERATURES:
        subset = df[df['temperature'] == temp]
        unique_count = subset['answer'].nunique()
        top_fruit = subset['answer'].mode()[0] if not subset.empty else "N/A"
        print(f"Temp {temp}: {unique_count} unique fruits. Most common: {top_fruit}")

if __name__ == "__main__":
    asyncio.run(main())