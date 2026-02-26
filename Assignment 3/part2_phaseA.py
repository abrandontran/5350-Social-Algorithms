import subprocess
import os
import asyncio

# Define path to starter code script and scattergories questions
STARTER_SCRIPT = "assignment3_starter.py"
QUESTIONS_CSV = "scattergories_questions.csv"

# Define server's Python path
PYTHON_PATH = r"C:\Program Files\Python39\python.exe"

# Set global parameters for player 1 and player 2 answer generation (same model)
MODEL_NAME = "qwen2.5:7b"
ROUNDS = "5"
PROMPT_FILE = "prompt_template.txt"
MAX_WORKERS = 15  # Limits concurrent requests to prevent VRAM overflow

# Specify temperatures and players
TEMPS = ["0.9", "2.0"]
PLAYERS = ["Player_1", "Player_2"]


# Create prompt template
# NOTE: render_player_prompt() function in assignment3_starter.py explicitly
# looks for {letter} and {category}
prompt_text = (
    "You are playing a game of Scattergories. "
    "Name exactly one {category} that starts with the letter {letter}. "
    "Output only the answer, with no punctuation, explanation, or conversation."
)

# Create a txt file for the prompt template
with open(PROMPT_FILE, "w") as f:
    f.write(prompt_text)


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
    os.makedirs("outputs", exist_ok=True)
    
    # Semaphore limits how many 'tasks' can run at once
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    tasks = []

    # Iterate through all pairs of temperature and player to build all commands
    for temp in TEMPS:
        for player in PLAYERS:
            # Create a label for the file
            label = "default" if temp == "0.9" else "opt"
            out_file = f"outputs/{label}_{temp.replace('.','')}_{player}.csv"
            
            # Create command line arguments for the current temperature and 
            # player
            cmd = [
                PYTHON_PATH, STARTER_SCRIPT, "generate-answers",
                "--model", MODEL_NAME,
                "--temperature", temp,
                "--rounds", ROUNDS,
                "--player-id", f"{player}_{label}",
                "--prompt-file", PROMPT_FILE,   
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


if __name__ == "__main__":
    asyncio.run(main())