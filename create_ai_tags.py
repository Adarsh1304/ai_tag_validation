import pandas as pd
import random

# Load the manual ground truth file
df = pd.read_csv("events.csv")

# Function to simulate AI mistakes
def simulate_ai_tags(manual):
    tags = [t.strip() for t in manual.split(",")]

    new_tags = []

    # AI keeps most tags, but misses some (False Negatives)
    for t in tags:
        if random.random() > 0.20:   # 80% chance to keep tag
            new_tags.append(t)

    # AI sometimes adds wrong tags (False Positives)
    if random.random() < 0.30:  # 30% chance to add noise
        extra = random.choice(["attack", "play", "movement", "pressure"])
        new_tags.append(extra)

    return ",".join(new_tags)

# Apply this to all rows
df["tags_ai"] = df["tags_manual"].apply(simulate_ai_tags)

# Save new file
df.to_csv("ai_tags.csv", index=False)

print("ai_tags.csv created successfully!")
print(df.head().to_string(index=False))





