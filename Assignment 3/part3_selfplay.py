"""
This script completes answer generation logic for an Ollama model to follow for 
Assignment 3, Part 3 (self-play) for S&DS 5350, Social Algorithms.

Notes:
    - Arguments for answer generation are specified as global parameters
    - Three scenarios for answer generation (same combinations as Part 2):
        - Temperature of 0.9, original prompt 
        - Temperature of 2.0, original prompt
        - Temperature of 2.0, obscure prompt 
    - assignment3_starter.py script is part of the command line argument
        - Line 124/125 in assignment3_starter.py: changed timeout=60 to 
          timeout=300
    - Run Script in Terminal: 
        & "C:\Program Files\Python39\python.exe" part3_selfplay.py 

Author:
    Cailey Bobadilla

AI Acknowledgement:
    I used Gemini 3 Pro and Claude Sonnet 4.6 in this script. My main use for
    Gemini 3 Pro was for adjusting my code to work with asyncio for parallel 
    processing purposes on the Yale GPU. My main use for Claude Sonnet 4.6 was 
    for debugging, specifically increasing the timeout time for tasks called to
    Ollama in assignment3_starter.py. 
"""

import subprocess
import os
import asyncio

# Define path to starter code script and scattergories questions
STARTER_SCRIPT = "assignment3_starter.py"
QUESTIONS_CSV = "scattergories_questions.csv"

# Define server's Python path
PYTHON_PATH = r"C:\Program Files\Python39\python.exe"

# Set global parameters for answer generation
ROUNDS = "5"
# Number of tasks that can run simultaneously (3 scenarios per model)
MAX_WORKERS = 3  

# Specify the 3 local models to test
# NOTE: A dictionary mapping is used to map the full model names to shorter ones
MODELS = {
    "qwen2.5:7b": "qwen", 
    "llama3.2:3b": "llama", 
    "gemma2:2b": "gemma"
}


# Create prompt template for both the original and obscure prompts
# NOTE: render_player_prompt() function in assignment3_starter.py explicitly
# looks for {letter} and {category}
original_prompt = (
    "You are playing a game of Scattergories. "
    "Name exactly one {category} that starts with the letter {letter}. "
    "Output only the answer, with no punctuation, explanation, or conversation."
)
obscure_prompt = (
    "You are playing Scattergories. Name a highly obscure, rare, but valid "
    "{category} that starts with the letter {letter}. Your goal is to choose a "
    "correct answer that no other player will think of. Output only the "
    "answer, with no punctuation or explanation."
)


# Create a txt file for the each prompt template
with open("original_prompt.txt", "w") as f:
    f.write(original_prompt)
with open("obscure_prompt.txt", "w") as f:
    f.write(obscure_prompt) 


# Define three specific scenarios for selfplay
# NOTE: Total of 6 because there are two players per scenario
SCENARIOS = [
    {"label": "default_09", "temp": "0.9", "prompt_file": "original_prompt.txt",
     "players": ["Player_1", "Player_2"]},
    {"label": "opt_20",     "temp": "2.0", "prompt_file": "original_prompt.txt",
     "players": ["Player_1", "Player_2"]},
    {"label": "obscure_20", "temp": "2.0", "prompt_file": "obscure_prompt.txt",
     "players": ["Player_1_Obscure", "Player_2_Obscure"]},
]


async def run_generate(cmd: list):
    """
    Runs a single generation command as a background thread.

    Args:
        cmd (list): Contains command line arguments for the specific temperature
                    and player paid.
    """
    await asyncio.to_thread(subprocess.run, cmd)


async def main():
    # Make sure the output directory exists
    os.makedirs("part3_selfplay_outputs", exist_ok=True)
    
    # Semaphore limits how many tasks can run at once
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    tasks = []

    print(f"Starting generation for {len(SCENARIOS)} scenarios.")
    print(f"Total files to generate: {sum(len(s['players']) for s in SCENARIOS)}")

    # Iterate through all pairs of scenarios, models, and players
    for model, short_name in MODELS.items():    
        for scenario in SCENARIOS:
            for player in scenario['players']:
                # Create a file name for the output
                out_file = (
                    f"part3_selfplay_outputs/self_{scenario['label']}_"
                    f"{short_name}_{player}.csv"
                )
                
                # Create command line arguments for the current temperature and 
                # player
                cmd = [
                    PYTHON_PATH, STARTER_SCRIPT, "generate-answers",
                    "--model", model,
                    "--temperature", scenario['temp'],
                    "--rounds", ROUNDS,
                    "--player-id", f"{short_name}_{player}_{scenario['label']}",
                    "--prompt-file", scenario['prompt_file'],   
                    "--questions-csv", QUESTIONS_CSV,
                    "--out", out_file
                ]

                # Wrapper to respect the semaphore limit
                async def sem_task(c=cmd):
                    async with semaphore:
                        await run_generate(c)

                tasks.append(sem_task())

    # Run everything concurrently
    await asyncio.gather(*tasks)

    print("Finished generating Part 3 self-play files.")


if __name__ == "__main__":
    asyncio.run(main())