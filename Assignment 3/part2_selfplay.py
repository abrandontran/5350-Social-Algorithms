import part2_phaseA
import asyncio

# Define a new, more optimized prompt
obscure_prompt = (
    "You are playing Scattergories. Name a highly obscure, rare, but valid "
    "{category} that starts with the letter {letter}. Your goal is to choose a "
    "correct answer that no other player will think of. Output only the "
    "answer, with no punctuation or explanation."
)

# Update the variables inside the part2_phaseA library
# NOTE: We are only using temperxature=2.0 because the average score per player 
# was higher than the default temperature=0.9
part2_phaseA.TEMPS = ["2.0"]
part2_phaseA.PLAYERS = ["Player_1_Obscure", "Player_2_Obscure"]

# Overwrite the text file
# NOTE: If we don't do this, part2_phaseA will use the old prompt
with open(part2_phaseA.PROMPT_FILE, "w") as f:
    f.write(obscure_prompt)


# Run the main function from part2_phaseA
if __name__ == "__main__":
    asyncio.run(part2_phaseA.main())