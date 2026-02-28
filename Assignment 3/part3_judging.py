"""
This script automates judging between self-play and cross-play answer outputs 
for Assignment 3, Part 3 (judging) for S&DS 5350, Social Algorithms. 

Notes:
    - Run Script in Terminal:
        python3 part3_judging.py

Author: 
    Cailey Bobadilla

AI Acknowledgement:
    I used Claude Sonnet 4.6 for this script. My main use was for implementing
    automation using command line arguments within the script to prevent doing
    18 manual comparisons.
"""

import subprocess
import os
from itertools import combinations

# Set paths, file names, and directories needed as global parameters
PYTHON_PATH = "python3"
JUDGE_SCRIPT = "reference/judge.py"
SELFPLAY_DIR = "part3_selfplay_outputs"
CROSSPLAY_DIR = "part3_crossplay_outputs"
OUTPUT_DIR = "part3_judged_outputs"

# Set the names of the models and scenarios in the self-play/cross-play scripts 
# as global parameters
MODELS = ["qwen", "llama", "gemma"]
SCENARIOS = ["default_09", "opt_20", "obscure_20"]

# Make sure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List to hold all comparisons
comparisons = []


# Iterate through all models and scenarios for self-play (has 2 players each)
for model in MODELS:
    for scenario in SCENARIOS:
        # Set the names of the files to find in the self-play directory for 
        # comparison
        # NOTE: The obscure prompt has its own conditions because the naming is
        # different
        if scenario == "obscure_20":
            p1 = f"{SELFPLAY_DIR}/self_{scenario}_{model}_Player_1_Obscure.csv"
            p2 = f"{SELFPLAY_DIR}/self_{scenario}_{model}_Player_2_Obscure.csv"
        else:
            p1 = f"{SELFPLAY_DIR}/self_{scenario}_{model}_Player_1.csv"
            p2 = f"{SELFPLAY_DIR}/self_{scenario}_{model}_Player_2.csv"

        # Create a label for the comparison
        label = f"self_{model}_{scenario}"

        # Add a dictionary for the files and their corresponding label
        comparisons.append({"files": [p1, p2], "label": label})


# Iterate through all combinations of model comparisons and scenarios for 
# cross-play
for model_a, model_b in combinations(MODELS, 2):
    for scenario in SCENARIOS:
        # Set the names of the files to find in the cross-play directory for 
        # comparison
        fa = f"{CROSSPLAY_DIR}/{scenario}_{model_a}.csv"
        fb = f"{CROSSPLAY_DIR}/{scenario}_{model_b}.csv"

        # Create a label for the comparison
        label = f"cross_{model_a}_vs_{model_b}_{scenario}"

        # Add a dictionary for the files and their corresponding label
        comparisons.append({"files": [fa, fb], "label": label})


print(f"Start generation for {len(comparisons)} comparisons.")


# Iterate through each comparison
for i, comp in enumerate(comparisons, 1):
    # Get the label and the files for the current comparison
    label = comp["label"]
    files = comp["files"]

    # Create paths for the output scores, details, and cache
    out_scores = os.path.join(OUTPUT_DIR, f"scores_{label}.csv")
    out_judged = os.path.join(OUTPUT_DIR, f"judged_{label}.csv")
    out_cache = os.path.join(OUTPUT_DIR, f"clean_cache_{label}.json")

    print(f"[{i}/{len(comparisons)}] Judging: {label}")

    # Create command line arguments for the current comparison for judging
    cmd = [
        PYTHON_PATH, JUDGE_SCRIPT, *files, 
        "--model", "gpt-4o-mini",
        "--out", out_scores,
        "--details", out_judged,
        "--cache", out_cache
    ]

    # Get the result by running the command line arguments
    result = subprocess.run(cmd)


print("Finished generating Part 3 judging files.")