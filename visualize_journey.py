"""
================================================================================
LEARNING JOURNEY VISUALIZATION
================================================================================

Shows the agent's progression from random to skilled:
- Records games at multiple checkpoints during training
- Groups replays by training stage
- Shows mistakes early on → mastery later

Stages captured:
  Stage 0: Random (untrained)
  Stage 1: After 5x5 IL pretraining (before any RL)
  Stage 2: After 5x5 PPO (mastered small board)
  Stage 3: Early 8x8 (500 games — still learning)
  Stage 4: Mid 8x8 (2500 games — improving)
  Stage 5: Late 8x8 (5000 games — competent)
  Stage 6: Early 10x10 (1000 games — adapting)
  Stage 7: Mid 10x10 (5000 games — getting better)
  Stage 8: Late 10x10 (10000 games — skilled)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import json
import os

from train_ppo_curriculum import (
    SnakeGame, GameState, ActorCritic, PPOAgent,
    ExpertCollector, PolicyNetwork,
    pretrain_critic, pretrain_actor, get_absolute_action
)
from train_curriculum_10x10 import collect_demos_from_ppo
import pickle

try:
    from train_tabular_q import QLearningAgent, DoubleQLearningAgent, default_q_values
except ImportError:
    QLearningAgent = None
    DoubleQLearningAgent = None
    default_q_values = None

try:
    from snake_game import SnakeGame as TabularSnakeGame
except ImportError:
    TabularSnakeGame = None

def record_tabular_games(agent, board_size, num_games=3, max_steps=500):
    """Play games with tabular agent."""
    games = []
    print(f"  Recording {num_games} tabular games...")
    
    # Use the correct environment if available
    GameClass = TabularSnakeGame if TabularSnakeGame else SnakeGame
    
    for _ in range(num_games):
        game = GameClass(board_size=board_size)
        frames = []
        done = False
        steps = 0
        last_action = 0
        
        while not done and steps < max_steps:
            frames.append({
                'snake': list(game.snake_position),
                'food': game.food_position,
                'score': game.score,
                'step': steps
            })
            
            # Tabular agent uses get_state_key which needs last_action
            state = agent.get_state_key(game, last_action)
            action = agent.choose_action(state, game)
            
            _, reward, done, info = game.step(action)
            last_action = action
            steps += 1
            
        frames.append({
            'snake': list(game.snake_position),
            'food': game.food_position,
            'score': info['score'],
            'step': steps,
            'game_over': True
        })
        games.append({'frames': frames, 'score': info['score']})
    return games


def record_games(agent, board_size, num_games=5, max_steps=5000):
    """Play games and record every frame."""
    games = []
    for _ in range(num_games):
        game = SnakeGame(board_size=board_size)
        frames = []
        done = False
        steps = 0
        while not done and steps < max_steps:
            frame = {
                'snake': list(game.snake_position),
                'food': game.food_position,
                'score': game.score,
                'step': steps
            }
            frames.append(frame)
            state = agent.get_state(game)
            action_relative, action_idx, _, _ = agent.select_action(state)
            action = get_absolute_action(action_relative, game)
            _, reward, done, info = game.step(action)
            steps += 1
        frames.append({
            'snake': list(game.snake_position),
            'food': game.food_position,
            'score': info['score'],
            'step': steps,
            'game_over': True
        })
        games.append({'frames': frames, 'score': info['score']})
    return games


def train_with_checkpoints(agent, board_size, total_games, max_steps, checkpoints):
    """Train PPO and record games at specified checkpoint intervals."""
    game = SnakeGame(board_size=board_size)
    scores = []
    record = 0
    all_checkpoint_data = []
    next_cp_idx = 0
    
    # Reset agent counters
    agent.n_games = 0
    checkpoints = sorted(checkpoints)
    
    # Check validation - record at game 0 if needed
    if next_cp_idx < len(checkpoints) and checkpoints[next_cp_idx] == 0:
        cp_games = record_games(agent, board_size, num_games=3, max_steps=max_steps)
        all_checkpoint_data.append({
            'checkpoint': 0,
            'games': cp_games,
            'mean_score': 0.0
        })
        print(f"  📸 Checkpoint @ game 0: recorded 3 games")
        next_cp_idx += 1

    total_steps = 0
    game_idx = 0
    
    while game_idx < total_games:
        game.reset()
        done = False
        steps_in_game = 0
        
        while not done and steps_in_game < max_steps:
            state = agent.get_state(game)
            action_relative, action_idx, log_prob, value = agent.select_action(state)
            action = get_absolute_action(action_relative, game)
            _, reward, done, info = game.step(action)
            steps_in_game += 1
            total_steps += 1
            
            # Store memory
            agent.store(state, action_idx, log_prob, reward, value, done)
            
            # Update PPO agent every 128 steps (matching train_ppo_curriculum.py logic)
            if total_steps % 128 == 0:
                next_state = agent.get_state(game)
                agent.update(next_state)
        
        # Game finished
        game_idx += 1
        score = info['score']
        scores.append(score)
        if score > record: record = score
            
        if game_idx % 100 == 0:
            mean = sum(scores[-100:]) / min(len(scores), 100)
            print(f"  Game {game_idx}, Score: {score}, Record: {record}, Mean: {mean:.2f}")

        # Checkpoint recording
        if next_cp_idx < len(checkpoints) and game_idx == checkpoints[next_cp_idx]:
            mean = sum(scores[-100:]) / min(len(scores), 100)
            cp_games = record_games(agent, board_size, num_games=3, max_steps=max_steps)
            all_checkpoint_data.append({
                'checkpoint': game_idx,
                'games': cp_games,
                'mean_score': round(mean, 2)
            })
            print(f"  📸 Checkpoint @ game {game_idx}: recorded 3 games")
            next_cp_idx += 1

    return all_checkpoint_data, scores


def generate_journey_html(stages, board_size=10):
    """Generate HTML that shows the learning journey across stages."""
    stages_json = json.dumps(stages)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snake AI - Learning Journey</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Inter', sans-serif;
            background: #08080d;
            color: #e0e0e0;
            min-height: 100vh;
            overflow-x: hidden;
        }}

        body::before {{
            content: '';
            position: fixed;
            inset: 0;
            background:
                radial-gradient(ellipse at 15% 50%, rgba(120,40,200,0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 85% 20%, rgba(0,200,120,0.04) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 85%, rgba(30,100,220,0.04) 0%, transparent 50%);
            z-index: -1;
        }}

        .header {{
            text-align: center;
            padding: 36px 20px 10px;
        }}

        .header h1 {{
            font-size: 2rem;
            font-weight: 900;
            background: linear-gradient(135deg, #ff6b6b, #ffa600, #00ff88, #00bbff, #8855ff);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradShift 6s ease infinite;
        }}

        @keyframes gradShift {{
            0%,100% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
        }}

        .header .sub {{ font-size: 0.85rem; color: #555; margin-top: 6px; }}

        /* Timeline */
        .timeline {{
            max-width: 900px;
            margin: 30px auto;
            padding: 0 24px;
            position: relative;
        }}

        .timeline::before {{
            content: '';
            position: absolute;
            left: 50%;
            top: 0; bottom: 0;
            width: 2px;
            background: linear-gradient(to bottom, rgba(0,255,136,0.3), rgba(0,187,255,0.3), rgba(136,85,255,0.3));
            transform: translateX(-50%);
        }}

        @media (max-width: 700px) {{
            .timeline::before {{ left: 20px; }}
        }}

        .stage {{
            position: relative;
            margin-bottom: 40px;
        }}

        .stage-badge {{
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            width: 40px; height: 40px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.8rem; font-weight: 900;
            z-index: 2;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
        }}

        @media (max-width: 700px) {{
            .stage-badge {{ left: 20px; }}
        }}

        .stage-card {{
            width: 42%;
            padding: 16px 18px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.06);
            backdrop-filter: blur(12px);
            position: relative;
        }}

        .stage:nth-child(odd) .stage-card {{ margin-left: 56%; }}
        .stage:nth-child(even) .stage-card {{ margin-right: 56%; }}

        @media (max-width: 700px) {{
            .stage-card {{ width: calc(100% - 50px) !important; margin-left: 50px !important; margin-right: 0 !important; }}
        }}

        .stage-title {{
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 4px;
        }}

        .stage-meta {{
            font-size: 0.72rem;
            color: #666;
            margin-bottom: 10px;
            font-family: 'JetBrains Mono', monospace;
        }}

        .stage-insight {{
            font-size: 0.78rem;
            color: #888;
            line-height: 1.5;
            margin-bottom: 12px;
            font-style: italic;
        }}

        .mini-canvas-row {{
            display: flex;
            gap: 8px;
            overflow-x: auto;
            padding-bottom: 4px;
        }}

        .mini-game {{
            flex-shrink: 0;
            text-align: center;
        }}

        .mini-game canvas {{
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.06);
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .mini-game canvas:hover {{
            transform: scale(1.03);
            box-shadow: 0 4px 20px rgba(0,200,100,0.15);
        }}

        .mini-game canvas.active {{
            box-shadow: 0 0 0 2px #00ff88;
        }}

        .mini-score {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            margin-top: 4px;
            color: #888;
        }}

        /* Main viewer */
        .viewer {{
            max-width: 600px;
            margin: 10px auto 30px;
            padding: 0 20px;
            text-align: center;
        }}

        .viewer-label {{
            font-size: 0.8rem;
            color: #666;
            margin-bottom: 8px;
        }}

        .viewer-stats {{
            display: flex; gap: 16px; justify-content: center;
            margin-bottom: 10px;
        }}

        .viewer-stat {{
            text-align: center;
        }}

        .viewer-stat .vlabel {{
            font-size: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #555;
        }}

        .viewer-stat .vval {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.3rem;
            font-weight: 700;
        }}

        #mainCanvas {{
            border-radius: 14px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5), 0 0 60px rgba(0,200,100,0.06);
        }}

        .viewer-controls {{
            display: flex; gap: 10px; justify-content: center; margin-top: 14px; flex-wrap: wrap;
        }}

        button {{
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 8px 18px;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            background: rgba(255,255,255,0.04);
            color: #bbb;
            cursor: pointer;
            transition: all 0.2s;
        }}

        button:hover {{
            background: rgba(255,255,255,0.08);
            color: #fff;
            transform: translateY(-1px);
        }}

        button.active {{
            background: rgba(0,200,100,0.12);
            border-color: rgba(0,200,100,0.35);
            color: #00ff88;
        }}

        .speed-ctrl {{
            display: flex; align-items: center; gap: 6px; font-size: 0.75rem; color: #555;
        }}

        input[type="range"] {{
            -webkit-appearance: none;
            width: 90px; height: 3px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
        }}

        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 12px; height: 12px;
            border-radius: 50%;
            background: #00ff88;
            cursor: pointer;
        }}

        /* Colors by phase */
        .phase-random {{ --c: #ff4444; }}
        .phase-tabular {{ --c: #d500f9; }}
        .phase-double-q {{ --c: #00e5ff; }}
        .phase-pretrain {{ --c: #ff8800; }}
        .phase-5x5 {{ --c: #ffcc00; }}
        .phase-8x8-early {{ --c: #88cc00; }}
        .phase-8x8-mid {{ --c: #00cc44; }}
        .phase-8x8-late {{ --c: #00ccaa; }}
        .phase-10x10-early {{ --c: #0088ff; }}
        .phase-10x10-mid {{ --c: #4466ff; }}
        .phase-10x10-late {{ --c: #8855ff; }}

        .stage-badge {{ background: var(--c); color: #000; }}
        .stage-card {{ background: color-mix(in srgb, var(--c) 6%, #0e0e14); }}
        .stage-title {{ color: var(--c); }}

        .footer {{
            text-align: center;
            padding: 20px;
            font-size: 0.7rem;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Learning Journey</h1>
        <p class="sub">Watch the AI evolve from random moves to skilled gameplay</p>
    </div>

    <!-- Main Viewer -->
    <div class="viewer">
        <div class="viewer-label" id="viewerLabel">Select a game below to watch</div>
        <div class="viewer-stats">
            <div class="viewer-stat"><div class="vlabel">Score</div><div class="vval" id="vScore" style="color:#00ff88">—</div></div>
            <div class="viewer-stat"><div class="vlabel">Step</div><div class="vval" id="vStep" style="color:#8855ff">—</div></div>
            <div class="viewer-stat"><div class="vlabel">Stage</div><div class="vval" id="vStage" style="color:#00bbff">—</div></div>
        </div>
        <canvas id="mainCanvas"></canvas>
        <div class="viewer-controls">
            <button id="playBtn" class="active" onclick="togglePlay()">⏸ Pause</button>
            <button onclick="restartGame()">⟲ Restart</button>
            <div class="speed-ctrl">
                Speed <input type="range" id="speedSlider" min="1" max="100" value="25" oninput="speed=101-this.value">
            </div>
        </div>
    </div>

    <!-- Timeline -->
    <div class="timeline" id="timeline"></div>

    <div class="footer">PPO + Imitation Learning + Curriculum Learning (5×5 → 8×8 → 10×10)</div>

    <script>
        const STAGES = {stages_json};
        const BOARD = {board_size};
        const CELL = 44;
        const GAP = 2;
        const PAD = 16;
        const CANVAS_SZ = PAD*2 + BOARD*CELL;
        const MINI_CELL = 16;
        const MINI_PAD = 4;
        const MINI_SZ = MINI_PAD*2 + BOARD*MINI_CELL;

        const mainCanvas = document.getElementById('mainCanvas');
        const mainCtx = mainCanvas.getContext('2d');
        mainCanvas.width = CANVAS_SZ;
        mainCanvas.height = CANVAS_SZ;

        let currentFrames = null;
        let currentFrameIdx = 0;
        let playing = true;
        let speed = 76;
        let timer = null;
        let activeStageIdx = -1;
        let activeGameIdx = -1;

        // Phase classes and metadata
        const PHASE_DATA = [
            {{ cls:'phase-random',      icon:'?',  title:'Random Policy',                insight:'No training yet. The agent moves randomly, dying almost instantly by running into walls or itself.' }},
            {{ cls:'phase-tabular',     icon:'0',  title:'Tabular Q-Learning (5x5)',      insight:'The "Zero" point. Classic Q-Learning works on small boards but fails to scale. It memorizes every state.' }},
            {{ cls:'phase-double-q',    icon:'0+', title:'Double Q-Learning (5x5)',       insight:'Improved tabular method. Uses two Q-tables to reduce overestimation bias, leading to more stable learning.' }},
            {{ cls:'phase-pretrain',     icon:'1',  title:'After Pretraining (IL)',        insight:'Behavioral cloning from REINFORCE demos. The agent learned basic movement patterns but has no strategic understanding.' }},
            {{ cls:'phase-5x5',         icon:'2',  title:'After 5×5 PPO Training',        insight:'Mastered the small board. Learned survival, food-seeking, and basic path planning.' }},
            {{ cls:'phase-8x8-early',   icon:'3',  title:'Early 8×8 (500 games)',         insight:'Just transferred to a bigger board. Struggles with the extra space, makes mistakes near edges.' }},
            {{ cls:'phase-8x8-mid',     icon:'4',  title:'Mid 8×8 (2500 games)',          insight:'Improving! Starting to find food reliably, but still makes occasional fatal mistakes.' }},
            {{ cls:'phase-8x8-late',    icon:'5',  title:'Late 8×8 (5000 games)',         insight:'Competent on 8×8. Can plan multi-step paths and avoid self-collision most of the time.' }},
            {{ cls:'phase-10x10-early', icon:'6',  title:'Early 10×10 (1000 games)',      insight:'Transferred to the largest board. Adapting to even more open space.' }},
            {{ cls:'phase-10x10-mid',   icon:'7',  title:'Mid 10×10 (5000 games)',        insight:'Showing real skill. Navigates the large board efficiently, scores consistently.' }},
            {{ cls:'phase-10x10-late',  icon:'8',  title:'Late 10×10 (10000 games)',      insight:'Peak performance. Plans long-term paths, avoids traps, and maximizes food collection.' }}
        ];

        // Build timeline
        const timelineEl = document.getElementById('timeline');
        STAGES.forEach((stage, si) => {{
            const pd = PHASE_DATA[si] || PHASE_DATA[PHASE_DATA.length-1];
            const div = document.createElement('div');
            div.className = 'stage ' + pd.cls;

            const badge = document.createElement('div');
            badge.className = 'stage-badge';
            badge.textContent = pd.icon;
            div.appendChild(badge);

            const card = document.createElement('div');
            card.className = 'stage-card';

            const title = document.createElement('div');
            title.className = 'stage-title';
            title.textContent = pd.title;
            card.appendChild(title);

            const meta = document.createElement('div');
            meta.className = 'stage-meta';
            const avgScore = stage.games.reduce((s,g) => s + g.score, 0) / stage.games.length;
            meta.textContent = 'Avg Score: ' + avgScore.toFixed(1) + ' | ' + stage.board + '×' + stage.board;
            card.appendChild(meta);

            const insight = document.createElement('div');
            insight.className = 'stage-insight';
            insight.textContent = pd.insight;
            card.appendChild(insight);

            const row = document.createElement('div');
            row.className = 'mini-canvas-row';

            stage.games.forEach((game, gi) => {{
                const wrap = document.createElement('div');
                wrap.className = 'mini-game';

                const mc = document.createElement('canvas');
                mc.width = MINI_SZ;
                mc.height = MINI_SZ;
                mc.onclick = () => selectGame(si, gi);
                mc.id = 'mini-' + si + '-' + gi;
                wrap.appendChild(mc);

                const lbl = document.createElement('div');
                lbl.className = 'mini-score';
                lbl.textContent = 'Score: ' + game.score;
                wrap.appendChild(lbl);

                row.appendChild(wrap);

                // Draw last frame on mini canvas
                drawMini(mc, game.frames[game.frames.length - 1], stage.board);
            }});

            card.appendChild(row);
            div.appendChild(card);
            timelineEl.appendChild(div);
        }});

        function drawMini(canvas, frame, boardSize) {{
            const ctx = canvas.getContext('2d');
            const cellSz = MINI_CELL;
            const pad = MINI_PAD;
            const sz = pad*2 + boardSize * cellSz;

            ctx.fillStyle = '#0e0e14';
            ctx.fillRect(0, 0, sz, sz);

            // Grid
            for (let r=0; r<boardSize; r++) {{
                for (let c=0; c<boardSize; c++) {{
                    ctx.fillStyle = (r+c)%2===0 ? '#151520' : '#181828';
                    ctx.fillRect(pad + c*cellSz + 1, pad + r*cellSz + 1, cellSz-2, cellSz-2);
                }}
            }}

            // Food
            if (frame.food) {{
                ctx.fillStyle = '#ff3344';
                ctx.fillRect(pad + frame.food[1]*cellSz + 3, pad + frame.food[0]*cellSz + 3, cellSz-6, cellSz-6);
            }}

            // Snake
            const snake = frame.snake;
            for (let i=0; i<snake.length; i++) {{
                const [r,c] = snake[i];
                const isHead = i === snake.length - 1;
                const t = i / Math.max(1, snake.length-1);
                const g = isHead ? 255 : Math.round(80 + 120*t);
                ctx.fillStyle = isHead ? '#00ff88' : `rgb(0,${{g}},${{Math.round(g*0.5)}})`;
                ctx.fillRect(pad + c*cellSz + 2, pad + r*cellSz + 2, cellSz-4, cellSz-4);
            }}
        }}

        function selectGame(stageIdx, gameIdx) {{
            // Remove previous active
            document.querySelectorAll('.mini-game canvas.active').forEach(c => c.classList.remove('active'));

            activeStageIdx = stageIdx;
            activeGameIdx = gameIdx;
            const mc = document.getElementById('mini-' + stageIdx + '-' + gameIdx);
            if (mc) mc.classList.add('active');

            const stage = STAGES[stageIdx];
            const game = stage.games[gameIdx];
            currentFrames = game.frames;
            currentFrameIdx = 0;

            const pd = PHASE_DATA[stageIdx] || PHASE_DATA[0];
            document.getElementById('viewerLabel').textContent = pd.title + ' — Game ' + (gameIdx+1);
            document.getElementById('vStage').textContent = pd.icon;

            drawMainFrame();
            if (playing) scheduleNext();
        }}

        function togglePlay() {{
            playing = !playing;
            const btn = document.getElementById('playBtn');
            btn.textContent = playing ? '⏸ Pause' : '▶ Play';
            btn.classList.toggle('active', playing);
            if (playing && currentFrames) scheduleNext();
        }}

        function restartGame() {{
            if (!currentFrames) return;
            currentFrameIdx = 0;
            drawMainFrame();
        }}

        function scheduleNext() {{
            clearTimeout(timer);
            if (!playing || !currentFrames) return;
            timer = setTimeout(() => {{
                currentFrameIdx++;
                if (currentFrameIdx >= currentFrames.length) {{
                    drawMainFrame();
                    // Auto-advance to next game after a pause
                    setTimeout(() => {{
                        advanceToNext();
                    }}, 1200);
                    return;
                }}
                drawMainFrame();
                scheduleNext();
            }}, speed);
        }}

        function advanceToNext() {{
            if (activeStageIdx < 0) return;
            let si = activeStageIdx, gi = activeGameIdx + 1;
            const stage = STAGES[si];
            if (gi >= stage.games.length) {{
                si++;
                gi = 0;
            }}
            if (si >= STAGES.length) {{
                si = 0; gi = 0;
            }}
            selectGame(si, gi);
        }}

        // ==================== MAIN CANVAS DRAWING ====================
        function drawMainFrame() {{
            if (!currentFrames) return;
            if (currentFrameIdx >= currentFrames.length) currentFrameIdx = currentFrames.length - 1;
            const frame = currentFrames[currentFrameIdx];
            const boardSize = STAGES[activeStageIdx].board;

            document.getElementById('vScore').textContent = frame.score;
            document.getElementById('vStep').textContent = frame.step;

            const ctx = mainCtx;
            ctx.clearRect(0, 0, CANVAS_SZ, CANVAS_SZ);
            ctx.fillStyle = '#0e0e14';
            ctx.fillRect(0, 0, CANVAS_SZ, CANVAS_SZ);

            // Adjust cell sizing for different boards
            const effectiveCell = (CANVAS_SZ - PAD*2) / boardSize;

            // Grid
            for (let r=0; r<boardSize; r++) {{
                for (let c=0; c<boardSize; c++) {{
                    const x = PAD + c*effectiveCell + GAP/2;
                    const y = PAD + r*effectiveCell + GAP/2;
                    const s = effectiveCell - GAP;
                    ctx.fillStyle = (r+c)%2===0 ? '#161622' : '#1a1a28';
                    ctx.beginPath();
                    roundRect(ctx, x, y, s, s, 4);
                    ctx.fill();
                }}
            }}

            // Food glow
            if (frame.food) {{
                const fx = PAD + frame.food[1]*effectiveCell + effectiveCell/2;
                const fy = PAD + frame.food[0]*effectiveCell + effectiveCell/2;
                const grad = ctx.createRadialGradient(fx, fy, 2, fx, fy, effectiveCell*1.5);
                grad.addColorStop(0, 'rgba(255,60,60,0.25)');
                grad.addColorStop(1, 'rgba(255,60,60,0)');
                ctx.fillStyle = grad;
                ctx.fillRect(fx-effectiveCell*1.5, fy-effectiveCell*1.5, effectiveCell*3, effectiveCell*3);
            }}

            // Food
            if (frame.food) {{
                const fx = PAD + frame.food[1]*effectiveCell + GAP/2;
                const fy = PAD + frame.food[0]*effectiveCell + GAP/2;
                const s = effectiveCell - GAP;
                ctx.fillStyle = '#ff3344';
                ctx.beginPath();
                roundRect(ctx, fx+3, fy+3, s-6, s-6, 8);
                ctx.fill();
                ctx.fillStyle = 'rgba(255,255,255,0.3)';
                ctx.beginPath();
                ctx.arc(fx+s*0.38, fy+s*0.32, 3, 0, Math.PI*2);
                ctx.fill();
            }}

            // Snake
            const snake = frame.snake;
            for (let i=0; i<snake.length; i++) {{
                const [r,c] = snake[i];
                const x = PAD + c*effectiveCell + GAP/2;
                const y = PAD + r*effectiveCell + GAP/2;
                const s = effectiveCell - GAP;
                const isHead = i === snake.length - 1;

                if (isHead) {{
                    // Glow
                    const hx = x + s/2, hy = y + s/2;
                    const hg = ctx.createRadialGradient(hx, hy, 2, hx, hy, effectiveCell);
                    hg.addColorStop(0, 'rgba(0,255,136,0.2)');
                    hg.addColorStop(1, 'rgba(0,255,136,0)');
                    ctx.fillStyle = hg;
                    ctx.fillRect(hx-effectiveCell, hy-effectiveCell, effectiveCell*2, effectiveCell*2);

                    ctx.fillStyle = '#00ff88';
                    ctx.beginPath();
                    roundRect(ctx, x+2, y+2, s-4, s-4, 8);
                    ctx.fill();

                    // Eyes
                    ctx.fillStyle = '#0a0a0f';
                    let dir = [0,1];
                    if (snake.length > 1) {{
                        const neck = snake[snake.length-2];
                        dir = [r-neck[0], c-neck[1]];
                    }}
                    const eOff = Math.max(4, s*0.18);
                    const eR = Math.max(2, s*0.08);
                    if (dir[0]===0) {{
                        ctx.beginPath(); ctx.arc(x+s/2+dir[1]*3, y+s/2-eOff, eR, 0, Math.PI*2); ctx.fill();
                        ctx.beginPath(); ctx.arc(x+s/2+dir[1]*3, y+s/2+eOff, eR, 0, Math.PI*2); ctx.fill();
                    }} else {{
                        ctx.beginPath(); ctx.arc(x+s/2-eOff, y+s/2+dir[0]*3, eR, 0, Math.PI*2); ctx.fill();
                        ctx.beginPath(); ctx.arc(x+s/2+eOff, y+s/2+dir[0]*3, eR, 0, Math.PI*2); ctx.fill();
                    }}
                }} else {{
                    const t = i / Math.max(1, snake.length-1);
                    const g = Math.round(60 + 140*t);
                    ctx.fillStyle = `rgb(0,${{g}},${{Math.round(g*0.5)}})`;
                    ctx.beginPath();
                    roundRect(ctx, x+3, y+3, s-6, s-6, 6);
                    ctx.fill();
                }}
            }}

            // Game over
            if (frame.game_over) {{
                ctx.fillStyle = 'rgba(10,10,15,0.65)';
                ctx.fillRect(0, 0, CANVAS_SZ, CANVAS_SZ);
                ctx.fillStyle = '#ff4444';
                ctx.font = '700 24px Inter';
                ctx.textAlign = 'center';
                ctx.fillText('💀 Game Over', CANVAS_SZ/2, CANVAS_SZ/2 - 14);
                ctx.fillStyle = '#aaa';
                ctx.font = '600 18px JetBrains Mono';
                ctx.fillText('Score: ' + frame.score, CANVAS_SZ/2, CANVAS_SZ/2 + 18);
            }}
        }}

        function roundRect(ctx, x, y, w, h, r) {{
            ctx.moveTo(x+r, y);
            ctx.lineTo(x+w-r, y); ctx.quadraticCurveTo(x+w, y, x+w, y+r);
            ctx.lineTo(x+w, y+h-r); ctx.quadraticCurveTo(x+w, y+h, x+w-r, y+h);
            ctx.lineTo(x+r, y+h); ctx.quadraticCurveTo(x, y+h, x, y+h-r);
            ctx.lineTo(x, y+r); ctx.quadraticCurveTo(x, y, x+r, y);
            ctx.closePath();
        }}

        // Auto-start with first game
        if (STAGES.length > 0 && STAGES[0].games.length > 0) {{
            selectGame(0, 0);
        }}
    </script>
</body>
</html>"""
    return html


def main():
    print("\n" + "=" * 70)
    print("🧠 LEARNING JOURNEY — Recording at Every Stage")
    print("=" * 70)

    all_stages = []

    # ========================================
    # STAGE 0: Random (untrained) on 10x10
    # ========================================
    print("\n📸 Stage 0: Recording RANDOM agent...")
    network = ActorCritic(14, 256, 3)
    agent = PPOAgent(network=network, lr=0.0003)
    games = record_games(agent, board_size=10, num_games=3, max_steps=500)
    all_stages.append({
        'label': 'Random',
        'board': 10,
        'games': games
    })
    avg = sum(g['score'] for g in games) / len(games)
    print(f"  Avg score: {avg:.1f}")

    # ========================================
    # STAGE 0.5: Tabular Q-Learning (5x5)
    # ========================================
    if QLearningAgent:
        print("\n📸 Stage 0.5: Recording Tabular Q-Learning agent...")
        try:
            with open('tabular_q_5x5.pkl', 'rb') as f:
                tabular_agent = pickle.load(f)
            games = record_tabular_games(tabular_agent, board_size=5, num_games=3, max_steps=500)
            all_stages.append({
                'label': 'Tabular Q (5x5)',
                'board': 5,
                'games': games
            })
            avg = sum(g['score'] for g in games) / len(games)
            print(f"  Avg score: {avg:.1f}")
        except FileNotFoundError:
            print("  ⚠️ tabular_q_5x5.pkl not found, skipping Tabular stage.")
    else:
        print("  ⚠️ QLearningAgent not imported, skipping Tabular stage.")

    # ========================================
    # STAGE 0.6: Double Q-Learning (5x5)
    # ========================================
    if DoubleQLearningAgent:
        print("\n📸 Stage 0.6: Recording Double Q-Learning agent...")
        try:
            with open('tabular_double_q_5x5.pkl', 'rb') as f:
                double_q_agent = pickle.load(f)
            games = record_tabular_games(double_q_agent, board_size=5, num_games=3, max_steps=500)
            all_stages.append({
                'label': 'Double Q (5x5)',
                'board': 5,
                'games': games
            })
            avg = sum(g['score'] for g in games) / len(games)
            print(f"  Avg score: {avg:.1f}")
        except FileNotFoundError:
            print("  ⚠️ tabular_double_q_5x5.pkl not found, skipping Double Q stage.")
    else:
        print("  ⚠️ DoubleQLearningAgent not imported, skipping Double Q stage.")

    # ========================================
    # STAGE 1: After IL Pretraining (5x5 demos)
    # ========================================
    print("\n📚 Training REINFORCE expert on 5x5...")
    collector = ExpertCollector(board_size=5)
    expert_data_5x5, _ = collector.collect_with_trained_reinforce(num_episodes=50, min_score=3)

    pretrain_critic(network, expert_data_5x5, epochs=50)
    pretrain_actor(network, expert_data_5x5, epochs=50)

    print("\n📸 Stage 1: Recording after IL pretraining (playing on 5x5)...")
    games = record_games(agent, board_size=5, num_games=3, max_steps=200)
    all_stages.append({
        'label': 'After Pretraining',
        'board': 5,
        'games': games
    })
    avg = sum(g['score'] for g in games) / len(games)
    print(f"  Avg score: {avg:.1f}")

    # ========================================
    # STAGE 2: After 5x5 PPO
    # ========================================
    print("\n📚 Training PPO on 5x5 (1000 games)...")
    checkpoints = [1000]
    cp_data, _ = train_with_checkpoints(agent, board_size=5, total_games=1000, max_steps=500, checkpoints=checkpoints)

    print(f"\n📸 Stage 2: Recording after 5x5 mastery...")
    games = record_games(agent, board_size=5, num_games=3, max_steps=500)
    all_stages.append({
        'label': 'After 5x5 PPO',
        'board': 5,
        'games': games
    })
    avg = sum(g['score'] for g in games) / len(games)
    print(f"  Avg score: {avg:.1f}")

    # ========================================
    # STAGE 3-5: 8x8 Training with checkpoints
    # ========================================
    print("\n📚 Training PPO on 8x8 (2000 games) with checkpoints...")
    checkpoints_8x8 = [500, 2000]
    cp_data_8x8, _ = train_with_checkpoints(agent, board_size=8, total_games=2000, max_steps=2000, checkpoints=checkpoints_8x8)

    for cp in cp_data_8x8:
        all_stages.append({
            'label': f"8x8 @ {cp['checkpoint']} games",
            'board': 8,
            'games': cp['games']
        })
        avg = sum(g['score'] for g in cp['games']) / len(cp['games'])
        print(f"  📸 8x8 checkpoint {cp['checkpoint']}: Avg score {avg:.1f}")

    # ========================================
    # Collect 8x8 demos for 10x10
    # ========================================
    print("\n📚 Collecting 8x8 demos for 10x10 transfer...")
    expert_data_8x8 = collect_demos_from_ppo(agent, board_size=8, num_episodes=100, min_score=10)
    pretrain_critic(agent.network, expert_data_8x8, epochs=30)
    pretrain_actor(agent.network, expert_data_8x8, epochs=30)
    agent.optimizer = optim.Adam(agent.network.parameters(), lr=0.0003)

    # ========================================
    # STAGE 6-8: 10x10 Training with checkpoints
    # ========================================
    print("\n📚 Training PPO on 10x10 (1000 games) with checkpoints...")
    checkpoints_10x10 = [500, 1000]
    cp_data_10x10, _ = train_with_checkpoints(agent, board_size=10, total_games=1000, max_steps=5000, checkpoints=checkpoints_10x10)

    for cp in cp_data_10x10:
        all_stages.append({
            'label': f"10x10 @ {cp['checkpoint']} games",
            'board': 10,
            'games': cp['games']
        })
        avg = sum(g['score'] for g in cp['games']) / len(cp['games'])
        print(f"  📸 10x10 checkpoint {cp['checkpoint']}: Avg score {avg:.1f}")

    # ========================================
    # Generate HTML
    # ========================================
    print("\n🎨 Generating Learning Journey HTML...")
    html = generate_journey_html(all_stages, board_size=10)

    output_path = os.path.join(os.path.dirname(__file__), 'snake_learning_journey.html')
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\n✅ Saved to: {output_path}")
    print("   Open in browser to watch the learning progression!")
    print("\n📊 Summary:")
    for i, stage in enumerate(all_stages):
        avg = sum(g['score'] for g in stage['games']) / len(stage['games'])
        print(f"  Stage {i}: {stage['label']} ({stage['board']}x{stage['board']}) — Avg: {avg:.1f}")


if __name__ == '__main__':
    main()
