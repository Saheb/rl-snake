import json
import re
import os

HTML_FILE = "snake_learning_journey.html"
JSON_FILE = "snakezero_games.json"

def merge_results():
    if not os.path.exists(JSON_FILE):
        print(f"Error: {JSON_FILE} not found. Wait for training to finish.")
        return

    with open(JSON_FILE, "r") as f:
        raw_games = json.load(f)

    print(f"Loaded {len(raw_games)} recorded games.")

    # Format for visualization
    formatted_stage = {
        "label": "SnakeZero (4x4 MCTS)",
        "board": 4,
        "games": []
    }

    # Convert recorded games to proper format
    # Input: { "episode": 0, "initial_snake": [], "initial_food": [], "steps": [...], "score": 0 }
    # Output: { "frames": [{ "snake": [], "food": [], "score": 0, "step": 0 }, ...] }
    
    for game in raw_games:
        frames = []
        
        # Initial frame
        frames.append({
            "snake": game["initial_snake"],
            "food": game["initial_food"],
            "score": 0,
            "step": 0
        })
        
        score = 0
        step_count = 0
        
        for step in game["steps"]:
            step_count += 1
            # "step" dict has: action, snake, food
            
            # Did score increase?
            new_len = len(step["snake"])
            prev_len = len(frames[-1]["snake"])
            if new_len > prev_len:
                score += 1
                
            frame = {
                "snake": step["snake"],
                "food": step["food"],
                "score": score,
                "step": step_count
            }
            
            frames.append(frame)
            
        # Mark last frame as game over
        if frames:
            frames[-1]["game_over"] = True
            
        formatted_stage["games"].append({
            "frames": frames,
            "score": score,
            "episode": game["episode"]
        })

    # Read HTML
    with open(HTML_FILE, "r") as f:
        html_content = f.read()

    # 1. Add CSS class if missing
    if ".phase-snakezero" not in html_content:
        css_insert = """
        .phase-snakezero {
            --c: #ff0055;
        }
        """
        html_content = html_content.replace("</style>", css_insert + "</style>")
        print("Added .phase-snakezero CSS.")

    # 2. Inject STAGES data
    # Find existing STAGES json
    match = re.search(r"const STAGES = (\[.*?\]);", html_content, re.DOTALL)
    if match:
        current_stages_json = match.group(1)
        try:
            # We can't easily json.loads() the JS because it might have keys without quotes (though this file seems strict)
            # Actually the file uses valid JSON structure for STAGES.
            # Let's try to parse it, append, and dump strings.
            # If parsing fails (due to flexible JS syntax), we might need a simpler regex injection.
            
            # Robust approach: Find the closing bracket '];' and insert before it.
            # But we need to make sure we add a comma if previous element exists.
            
            injection_point = match.end(1) - 1 # Position of ']'
            
            new_stage_json = json.dumps(formatted_stage)
            
            # Check if there's already a comma before the closing bracket (ignoring whitespace)
            # Actually, easiest way: python string manipulation
            
            # Construct new JS array content
            # parse the full list?
            # It's huge. Let's just insert string.
            
            insert_str = ", " + new_stage_json
            
            new_html = html_content[:injection_point] + insert_str + html_content[injection_point:]
            
            with open(HTML_FILE, "w") as f:
                f.write(new_html)
            print("Successfully injected SnakeZero games into STAGES.")
            
        except Exception as e:
            print(f"Error parsing/injecting JSON: {e}")
    else:
        print("Error: Could not find 'const STAGES = ...' in HTML.")

if __name__ == "__main__":
    merge_results()
