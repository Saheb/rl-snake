import json
import re
import os
import statistics

HTML_FILE = "snake_learning_journey.html"
SNAKEZERO_FILE = "snakezero_games.json"

def calculate_metrics(games, label):
    scores = [g['score'] for g in games]
    if not scores:
        return None
    
    # Check if 'steps' is a list (snakezero) or step count (html format might vary)
    # In HTML STAGES, games have 'frames' list vs snakezero has 'steps' list
    # Actually snakezero_games.json has 'steps' list.
    # HTML STAGES games have 'frames' list.
    
    steps = []
    for g in games:
        if 'steps' in g and isinstance(g['steps'], list):
             steps.append(len(g['steps']))
        elif 'frames' in g:
             steps.append(len(g['frames']))
        elif 'steps' in g and isinstance(g['steps'], int):
             steps.append(g['steps']) # Fallback if steps is just a count
        else:
             steps.append(0)

    return {
        "Label": label,
        "Count": len(games),
        "Avg Score": statistics.mean(scores),
        "Max Score": max(scores),
        "Avg Steps": statistics.mean(steps) if steps else 0
    }

def main():
    metrics = []

    # 1. Parse HTML for existing STAGES
    try:
        with open(HTML_FILE, "r") as f:
            content = f.read()
        
        match = re.search(r"const STAGES = (\[.*?\]);", content, re.DOTALL)
        if match:
            # We need to be careful with JS objects that might not be invalid JSON
            # The previous file view showed keys in quotes, so json.loads might work.
            json_str = match.group(1)
            # Remove any trailing commas in objects/lists which JSON doesn't like but JS does
            # This is a simple heuristic
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            
            stages = json.loads(json_str)
            
            for stage in stages:
                m = calculate_metrics(stage['games'], stage['label'])
                if m:
                    m["Board"] = f"{stage['board']}x{stage['board']}"
                    metrics.append(m)
        else:
            print("Could not find STAGES in HTML file.")
            
    except Exception as e:
        print(f"Error parsing HTML: {e}")

    # 2. Parse SnakeZero Data
    try:
        with open(SNAKEZERO_FILE, "r") as f:
            snakezero_games = json.load(f)
            
        # SnakeZero file format is:
        # { "episode": 0, "score": 0, "steps": [...], ... }
        
        m = calculate_metrics(snakezero_games, "SnakeZero (MCTS)")
        m["Board"] = "4x4"
        metrics.append(m)
        
    except Exception as e:
        print(f"Error reading SnakeZero data: {e}")

    # 3. Print Comparison
    print(f"{'Model':<25} | {'Board':<8} | {'Games':<6} | {'Avg Score':<10} | {'Max Score':<10} | {'Avg Steps':<10}")
    print("-" * 85)
    for m in metrics:
        print(f"{m['Label']:<25} | {m['Board']:<8} | {m['Count']:<6} | {m['Avg Score']:<10.2f} | {m['Max Score']:<10} | {m['Avg Steps']:<10.1f}")

if __name__ == "__main__":
    main()
