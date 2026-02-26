import pandas as pd
import os
import asyncio
from collections import Counter
from tqdm.asyncio import tqdm
import nest_asyncio

nest_asyncio.apply()

from reference.assignment3_starter import (
    ollama_generate,
    normalize_answer,
    entropy_from_counts,
    kl_to_uniform
)

MODEL = 'qwen2.5:7b'
SAMPLES = 50
TEMPERATURES = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
VALID_DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
MAX_CONCURRENT = 10

prompt = (
    "please pick 6 random days of the week, including weekends, with replacement. return the fifth day you choose. respond with just the day."
)

# Added 'sem' as an argument here
async def run_single_sample(sem, temp, iteration):
    async with sem:
        raw_ans = await asyncio.to_thread(
            ollama_generate,
            model=MODEL,
            prompt=prompt,
            temperature=temp,
            top_k=40,
            max_tokens=8
        )
        
        ans = normalize_answer(raw_ans)
        ans = ans.split()[0] if ans else "invalid"
        return {"temperature": temp, "raw": raw_ans, "answer": ans, "iteration": iteration}

async def main():
    # 1. Create the semaphore INSIDE the active loop
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    
    tasks = []
    print(f"Initializing {len(TEMPERATURES) * SAMPLES} samples across {MAX_CONCURRENT} concurrent workers...")

    for temp in TEMPERATURES:
        for i in range(SAMPLES):
            # 2. Pass the semaphore to the task
            tasks.append(run_single_sample(sem, temp, i + 1))

    results = await tqdm.gather(*tasks, desc="Generating Samples")

    # --- Processing Results ---
    df = pd.DataFrame(results)
    
    df.to_csv('working/part1_calibration.csv', index=False)
    
    for temp in TEMPERATURES:
        temp_counts = Counter(df[df['temperature'] == temp]['answer'])
        print(f"Unique answers at temp {temp}: {len(temp_counts)}")

if __name__ == "__main__":
    asyncio.run(main())