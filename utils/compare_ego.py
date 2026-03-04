import json
import statistics

# Load the data
try:
    with open("assets/snakezero_games.json", "r") as f:
        games = json.load(f)
except json.JSONDecodeError:
    print("Error reading json")
    exit()

# Filter for the Egocentric Compass run (last 500 episodes)
conv_games = games[-500:] 

scores = [g.get('score', 0) for g in conv_games]
avg_score = statistics.mean(scores)
max_score = max(scores)
avg_steps = statistics.mean([len(g.get('steps', [])) for g in conv_games])

print(f"Compass (6x6 Training): Games={len(conv_games)}, Avg={avg_score:.2f}, Max={max_score}, Steps={avg_steps:.2f}")
