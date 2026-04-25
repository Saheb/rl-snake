import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import json
    import random

    return json, mo, np, plt, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Curiosity Killed the Snake: Re-evaluating Intrinsic Motivation
    *An interactive exploration of "Curiosity-driven Exploration by Self-supervised Prediction" (Pathak et al., 2017).*

    ### The Problem: Sparse Reward Trap
    In Deep Reinforcement Learning, an agent learns by maximizing a reward signal. But what happens when the environment is vast, the board is empty, and the reward is incredibly hard to find by pure chance? And what if there are no external rewards at all?

    When an $\epsilon$-greedy agent (like a standard DQN) faces a sparse environment, its exploration is entirely random. It suffers from "Catastrophic Amnesia" of its early states, spinning in circles rather than systematically mapping the environment.

    **Play with the agent below to see how it performs when the reward is 13 steps away.**
    """)
    return


@app.cell(hide_code=True)
def _(json, mo, random):
    grid_size = 14
    start = (grid_size // 2, grid_size // 2)
    reward = (0, grid_size - 1)
    n_steps = 120

    def clamp(v): return max(0, min(grid_size - 1, v))

    # --- Random walk ---
    rng1 = random.Random(42)
    hist_random = [start]
    for _ in range(n_steps):
        y, x = hist_random[-1]
        moves = [(0,1),(0,-1),(1,0),(-1,0)]
        rng1.shuffle(moves)
        for dy, dx in moves:
            ny, nx = clamp(y+dy), clamp(x+dx)
            if (ny, nx) != (y, x):
                break
        hist_random.append((ny, nx))

    # --- Curiosity-Driven Walk (seeks unvisited/surprising tiles) ---
    rng2 = random.Random(42)
    hist_curious = [start]
    visit_counts = {start: 1}
    for _ in range(n_steps):
        y, x = hist_curious[-1]
        moves = [(0,1),(0,-1),(1,0),(-1,0)]
        valid_neighbors = []
        for dy, dx in moves:
            ny, nx = clamp(y+dy), clamp(x+dx)
            if (ny, nx) != (y, x):
                valid_neighbors.append((ny, nx))
        rng2.shuffle(valid_neighbors)
        valid_neighbors.sort(key=lambda pos: visit_counts.get(pos, 0))
        next_pos = valid_neighbors[0]
        hist_curious.append(next_pos)
        visit_counts[next_pos] = visit_counts.get(next_pos, 0) + 1

    html = f"""<!DOCTYPE html>
    <html>
    <head>
    <style>
      body {{ margin:0; font-family:sans-serif; background:white; padding:10px; }}
      .row {{ display:flex; gap:24px; justify-content:center; }}
      .panel {{ display:flex; flex-direction:column; align-items:center; }}
      .label {{ font-weight:bold; font-size:13px; margin-bottom:5px; }}
      .sublabel {{ font-size:11px; color:#666; margin-bottom:4px; }}
      canvas {{ border:1px solid #ccc; }}
      .controls {{ margin-top:10px; display:flex; gap:14px; align-items:center; justify-content:center; }}
      button {{ padding:5px 22px; font-size:14px; cursor:pointer; border-radius:4px;
            border:1px solid #aaa; background:#f5f5f5; }}
      button:hover {{ background:#e0e0e0; }}
      .legend {{ font-size:11px; color:#777; margin-top:4px; text-align:center; }}
    </style>
    </head>
    <body>
    <div class="row">
      <div class="panel">
    <div class="label">Random Walk</div>
    <div class="sublabel">No reward signal — pure chance</div>
    <canvas id="c1" width="400" height="400"></canvas>
      </div>
      <div class="panel">
    <div class="label">Count-Based Curiosity Agent</div>
    <div class="sublabel">ICM proxy — seeks lowest-visited tiles</div>
    <canvas id="c2" width="400" height="400"></canvas>
      </div>
    </div>
    <div class="controls">
      <button id="btn">▶ Play</button>
      <span id="info" style="font-size:13px;color:#444">Step 0 / {n_steps}</span>
    </div>
    <div class="legend">● Agent &nbsp;&nbsp; ■ Start &nbsp;&nbsp; ★ Reward</div>

    <script>
    const G = {grid_size}, SZ = 400, CELL = SZ / G;
    const pathA = {json.dumps(hist_random)};
    const pathB = {json.dumps(hist_curious)};
    const startPos = [{start[1]}, {start[0]}];
    const rewardPos = [{reward[1]}, {reward[0]}];
    const c1 = document.getElementById('c1').getContext('2d');
    const c2 = document.getElementById('c2').getContext('2d');
    const btn = document.getElementById('btn');
    const info = document.getElementById('info');
    let frame = 0, playing = false, timer = null;

    function cc(col, row) {{ return [(col+0.5)*CELL, (row+0.5)*CELL]; }}

    function drawStar(ctx, col, row, r) {{
      const [cx,cy] = cc(col, row);
      ctx.beginPath();
      for (let i=0;i<5;i++) {{
    const a=(i*4*Math.PI/5)-Math.PI/2;
    const px=cx+r*Math.cos(a), py=cy+r*Math.sin(a);
    i===0?ctx.moveTo(px,py):ctx.lineTo(px,py);
      }}
      ctx.closePath(); ctx.fillStyle='gold'; ctx.fill();
      ctx.strokeStyle='#888'; ctx.lineWidth=1; ctx.stroke();
    }}

    function drawCanvas(ctx, path, agentColor) {{
      ctx.clearRect(0,0,SZ,SZ);
      ctx.strokeStyle='#ccc'; ctx.lineWidth=0.5;
      for (let i=0;i<=G;i++) {{
    ctx.beginPath(); ctx.moveTo(i*CELL,0); ctx.lineTo(i*CELL,SZ); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,i*CELL); ctx.lineTo(SZ,i*CELL); ctx.stroke();
      }}
      if (frame>0) {{
    ctx.beginPath(); ctx.strokeStyle='rgba(80,80,80,0.2)'; ctx.lineWidth=1.5;
    for (let i=0;i<=frame;i++) {{
      const [cx,cy]=cc(path[i][1],path[i][0]);
      i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
    }}
    ctx.stroke();
      }}
      const [sx,sy]=cc(...startPos);
      ctx.fillStyle='royalblue'; ctx.fillRect(sx-7,sy-7,14,14);
      ctx.strokeStyle='#000'; ctx.lineWidth=1; ctx.strokeRect(sx-7,sy-7,14,14);
      drawStar(ctx, rewardPos[0], rewardPos[1], 10);
      const [ax,ay]=cc(path[frame][1],path[frame][0]);
      ctx.beginPath(); ctx.arc(ax,ay,7,0,Math.PI*2);
      ctx.fillStyle=agentColor; ctx.fill();
      ctx.strokeStyle='#000'; ctx.lineWidth=1.5; ctx.stroke();
    }}

    function render() {{
      drawCanvas(c1, pathA, '#ef4444');
      drawCanvas(c2, pathB, '#8b5cf6');
      info.textContent = 'Step ' + frame + ' / {n_steps}';
    }}

    function tick() {{
      render();
      if (playing) {{
    frame = (frame+1) % (pathA.length);
    timer = setTimeout(tick, 130);
      }}
    }}

    btn.onclick = () => {{
      playing = !playing;
      btn.textContent = playing ? '⏸ Pause' : '▶ Play';
      if (playing) tick();
    }};

    render();
    </script>
    </body>
    </html>"""

    mo.vstack([
        mo.iframe(html, height="500px"),
        mo.callout(
            mo.md(
                "**Visualization note:** The right-hand agent uses a **count-based heuristic** "
                "(always move to the lowest visit-count neighbour), not a live neural network. "
                "In a discrete grid, visit counts are a mathematically equivalent proxy for "
                "ICM's forward model prediction error — both measure how *novel* a state is. "
                "The outward-seeking behaviour is identical; only the mechanism differs."
            ),
            kind="info"
        )
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    # 1. The Narrative Text (Explaining the Forward Model)
    intro_text = mo.md(
        """
        ### The Concept: Mathematically Defining "Curiosity"

        If the environment isn't giving the agent any rewards, the agent has to generate its own. Pathak et al. achieved this by giving the agent a **Forward Model**—a neural network that acts as an internal physics engine. 

        Before the agent takes a step, the Forward Model guesses what the next state of the world will look like. The agent then takes the step and compares its guess to reality. The difference between the guess and reality is the **Prediction Error** (Mean Squared Error). 

        $$Intrinsic\\ Reward = \\frac{\\eta}{2} (Predicted\\ State - Actual\\ State)^2$$

        Where **$\\eta$ (Eta)** is the **Curiosity Weight**. It scales how much intrinsic reward the agent gets from being surprised. The key insight of this equation is that **Prediction Error = Surprise = Reward**.

        * If the agent visits a tile it has seen 100 times, its Forward Model perfectly predicts the physics of that tile. The error is zero. The agent is bored.
        * If the agent visits a completely new tile, the Forward Model's guess is completely wrong. The error is massive. The agent experiences a spike of surprise.

        **Try it yourself:** Click the tiles in the "Boredom Simulator" below. Watch how "surprise" spikes when you explore new areas, and how the reward drops to zero if you linger in the same spot.
        """
    )

    # 2. The Interactive "Boredom Simulator" (Native HTML5 Canvas)
    html2 = """<!DOCTYPE html>
    <html>
    <head>
    <style>
        body { margin:0; font-family:sans-serif; background:white; padding:10px; display:flex; flex-direction:column; align-items:center; }
        .container { display:flex; gap:30px; align-items:flex-start; margin-top: 10px;}
        .panel { display:flex; flex-direction:column; align-items:center; }
        .label { font-weight:bold; font-size:14px; margin-bottom:5px; color:#333;}
        canvas { border:1px solid #ccc; background:#f9f9f9; border-radius: 4px; cursor: pointer; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
        #chartCanvas { cursor: default; }
        .score-board { margin-top: 15px; font-size: 16px; font-weight: bold; color: #10b981; }
        .instructions { font-size: 12px; color: #666; margin-top: 5px; font-style: italic; }
    </style>
    </head>
    <body>

    <div class="container">
        <div class="panel">
        <div class="label">The Environment (Click to Explore)</div>
        <canvas id="gridCanvas" width="250" height="250"></canvas>
        <div class="instructions">Click tiles to generate intrinsic reward.</div>
        </div>

        <div class="panel">
        <div class="label">Forward Model: Prediction Error (Reward)</div>
        <canvas id="chartCanvas" width="350" height="250"></canvas>
        <div class="score-board">Total Intrinsic Reward: <span id="score">0.00</span></div>
        </div>
    </div>
    <div style="margin-top:15px;">
        <button id="resetBtn" style="padding:6px 16px; cursor:pointer; background:#f0f0f0; border:1px solid #aaa; border-radius:4px; font-weight:bold;">🔄 Reset Environment</button>
    </div>

    <script>
    // --- Grid Logic ---
    const gridCtx = document.getElementById('gridCanvas').getContext('2d');
    const chartCtx = document.getElementById('chartCanvas').getContext('2d');
    const scoreEl = document.getElementById('score');
    const resetBtn = document.getElementById('resetBtn');

    const ROWS = 5;
    const COLS = 5;
    const CELL_SIZE = 50;

    // State
    let visits = Array(ROWS).fill().map(() => Array(COLS).fill(0));
    let rewardHistory = [];
    let totalScore = 0;

    // Reward curve: 1st visit = 1.0, 2nd = 0.4, 3rd = 0.1, 4th+ = 0.0
    function getReward(visitCount) {
        if (visitCount === 1) return 1.0;
        if (visitCount === 2) return 0.4;
        if (visitCount === 3) return 0.1;
        return 0.0;
    }

    // Color mapping based on boredom (visits)
    function getCellColor(visitCount) {
        if (visitCount === 0) return '#f9f9f9'; // Unvisited
        if (visitCount === 1) return '#fde047'; // Bright Yellow (High Surprise)
        if (visitCount === 2) return '#fcd34d'; // Dull Yellow
        if (visitCount === 3) return '#d1d5db'; // Light Gray (Getting Bored)
        return '#9ca3af'; // Dark Gray (Completely Bored)
    }

    function drawGrid() {
        gridCtx.clearRect(0, 0, 250, 250);
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                gridCtx.fillStyle = getCellColor(visits[r][c]);
                gridCtx.fillRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);
                gridCtx.strokeStyle = '#e5e7eb';
                gridCtx.strokeRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);

                // Add text to show it's bored
                if (visits[r][c] >= 4) {
                    gridCtx.fillStyle = '#4b5563';
                    gridCtx.font = "10px sans-serif";
                    gridCtx.textAlign = "center";
                    gridCtx.fillText("Bored", c * CELL_SIZE + 25, r * CELL_SIZE + 28);
                }
            }
        }
    }

    // --- Chart Logic ---
    function drawChart() {
        chartCtx.clearRect(0, 0, 350, 250);

        // Draw axes
        chartCtx.strokeStyle = '#ccc';
        chartCtx.beginPath();
        chartCtx.moveTo(30, 10); chartCtx.lineTo(30, 220); // Y axis
        chartCtx.lineTo(340, 220); // X axis
        chartCtx.stroke();

        // Y-axis labels
        chartCtx.fillStyle = '#888';
        chartCtx.font = "10px sans-serif";
        chartCtx.fillText("1.0 -", 10, 25);
        chartCtx.fillText("0.5 -", 10, 120);
        chartCtx.fillText("0.0 -", 10, 220);

        if (rewardHistory.length === 0) return;

        // Draw line
        chartCtx.beginPath();
        chartCtx.strokeStyle = '#10b981';
        chartCtx.lineWidth = 2;

        const stepX = 300 / Math.max(10, rewardHistory.length);

        for (let i = 0; i < rewardHistory.length; i++) {
            const x = 30 + (i * stepX);
            const y = 220 - (rewardHistory[i] * 200); // Scale 0-1 to 0-200px

            if (i === 0) chartCtx.moveTo(x, y);
            else chartCtx.lineTo(x, y);

            // Draw points
            chartCtx.fillStyle = '#10b981';
            chartCtx.fillRect(x - 2, y - 2, 4, 4);
        }
        chartCtx.stroke();
    }

    // --- Interaction ---
    document.getElementById('gridCanvas').addEventListener('mousedown', (e) => {
        const rect = e.target.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const c = Math.floor(x / CELL_SIZE);
        const r = Math.floor(y / CELL_SIZE);

        if (r >= 0 && r < ROWS && c >= 0 && c < COLS) {
            visits[r][c]++;
            const reward = getReward(visits[r][c]);

            rewardHistory.push(reward);
            // Keep chart from squeezing too much (show last 25 clicks)
            if (rewardHistory.length > 25) rewardHistory.shift(); 

            totalScore += reward;
            scoreEl.innerText = totalScore.toFixed(2);

            drawGrid();
            drawChart();
        }
    });

    resetBtn.addEventListener('click', () => {
        visits = Array(ROWS).fill().map(() => Array(COLS).fill(0));
        rewardHistory = [];
        totalScore = 0;
        scoreEl.innerText = "0.00";
        drawGrid();
        drawChart();
    });

    // Init
    drawGrid();
    drawChart();
    </script>
    </body>
    </html>"""

    # 3. Stack the markdown and the widget together
    mo.vstack([
        intro_text,
        mo.iframe(html2, height="400px")
    ])
    return


@app.cell(hide_code=True)
def _(json, mo):
    # 1. The Narrative Text (Explaining the Trap and the Filter)
    act2_text = mo.md(
        """
        ### A fatal flaw: The Noisy TV Problem

        If you try to predict raw pixels, the agent gets distracted by irrelevant stochasticity (a flickering leaf, a TV screen with random noise) that has nothing to do with the agent's actions. Nor does it affect the agent in any way. It's novel, but it's not relevant. This is sometimes called the "noisy TV problem"

        How do we remove this noise, so agent can focus on what matters? How do we teach the agent to ignore this noise?

        Rather than making predictions in raw sensory space (e.g. pixels), the paper transforms sensory input into a feature space where only information relevant to the agent's actions is represented. The core insight is to predict only those changes in the environment that could possibly be due to the actions of the agent or could affect the agent — and ignore everything else.


        > "We learn this feature space using self-supervision – training a neural network on a proxy inverse dynamics task of predicting the agent’s action given its current and next states. Since the neural network is only required to predict the action, it has no incentive to represent within its feature embedding space the factors of variation in the environment that do not affect the agent itself."

        The Inverse Model is trained to predict the *agent's own actions*. Because the agent's actions cannot control the random TV static, the neural network learns to completely ignore the TV when creating the latent vector. The noise is mathematically filtered out.

        **Run the agents below** to see why Raw Pixel Prediction fails, and how the Latent Filter (ICM) saves the agent.
        """
    )

    # 2. Hardcode the pedagogical paths with UNIQUE variables
    grid_h_tv, grid_w_tv = 7, 8
    start_tv = (1, 1)
    goal_tv = (1, 6)
    noisy_tv_tile = (5, 1)

    path_down_hallway = [(1,1), (2,1), (3,1), (4,1), (5,1)]

    # RAW AGENT: Gets stuck at the TV forever (infinite prediction error)
    hist_raw_tv = path_down_hallway + [(5,1)] * 40 

    # ICM AGENT: Stares for 5 frames, learns to filter it, turns around, finds goal
    hist_icm_tv = path_down_hallway + [(5,1)] * 5 + [(4,1), (4,2), (4,3), (4,4), (4,5), (4,6), (3,6), (2,6), (1,6)] + [(1,6)] * 20

    # 3. Build the Canvas HTML (Saved to html_tv)
    html_tv = f"""<!DOCTYPE html>
    <html>
    <head>
    <style>
      body {{ margin:0; font-family:sans-serif; background:white; padding:10px; }}
      .row {{ display:flex; gap:24px; justify-content:center; }}
      .panel {{ display:flex; flex-direction:column; align-items:center; }}
      .label {{ font-weight:bold; font-size:13px; margin-bottom:5px; }}
      .sublabel {{ font-size:11px; color:#666; margin-bottom:8px; max-width: 250px; text-align: center; height: 30px;}}
      canvas {{ border:1px solid #ccc; background: #f9f9f9; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}}
      .controls {{ margin-top:15px; display:flex; gap:14px; align-items:center; justify-content:center; }}
      button {{ padding:6px 24px; font-size:14px; cursor:pointer; border-radius:4px; border:1px solid #aaa; background:#f5f5f5; font-weight:bold; }}
      button:hover {{ background:#e0e0e0; }}
      .legend {{ font-size:12px; color:#555; margin-top:12px; text-align:center; }}
    </style>
    </head>
    <body>
    <div class="row">
      <div class="panel">
        <div class="label">Raw Pixel Prediction</div>
        <div class="sublabel">Unable to predict the TV's static, the agent gets permanently stuck by infinite "surprise".</div>
        <canvas id="c1" width="240" height="210"></canvas>
      </div>
      <div class="panel">
        <div class="label">Latent Feature Prediction (ICM)</div>
        <div class="sublabel">The Inverse Model filters out the TV noise. The agent gets bored and moves on.</div>
        <canvas id="c2" width="240" height="210"></canvas>
      </div>
    </div>
    <div class="controls">
      <button id="btn">▶ Deploy Agents</button>
      <span id="info" style="font-size:14px;color:#444;font-weight:bold;width:100px;">Step: 0</span>
    </div>
    <div class="legend">
      <span style="color:royalblue">■</span> Start &nbsp;&nbsp; 
      <span style="color:gold">★</span> Goal &nbsp;&nbsp; 
      <span style="color:magenta">▒</span> Noisy TV &nbsp;&nbsp; 
      <span style="color:#ef4444">●</span> Agent
    </div>

    <script>
    const GH = {grid_h_tv}, GW = {grid_w_tv}, SZ_W = 240, SZ_H = 210;
    const CELL = SZ_W / GW; // 30px per cell
    const pathA = {json.dumps(hist_raw_tv)};
    const pathB = {json.dumps(hist_icm_tv)};
    const startPos = [{start_tv[1]}, {start_tv[0]}];
    const goalPos = [{goal_tv[1]}, {goal_tv[0]}];
    const tvPos = [{noisy_tv_tile[1]}, {noisy_tv_tile[0]}];

    const c1 = document.getElementById('c1').getContext('2d');
    const c2 = document.getElementById('c2').getContext('2d');
    const btn = document.getElementById('btn');
    const info = document.getElementById('info');
    let frame = 0, playing = false, timer = null;

    function cc(col, row) {{ return [(col+0.5)*CELL, (row+0.5)*CELL]; }}

    // Draw standard walls for the maze
    const walls = [
        [0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],
        [0,2],[1,2],[2,2],[3,2],[6,2],[7,2],
        [0,6],[1,6],[2,6],[3,6],[4,6],[5,6],[6,6],[7,6],
        [0,1],[0,3],[0,4],[0,5],[2,3],[2,4],[2,5],[7,1],[7,3],[7,4],[7,5]
    ];

    function drawMaze(ctx, path, isRaw) {{
      ctx.clearRect(0,0,SZ_W,SZ_H);

      // Draw grid lines
      ctx.strokeStyle='#e0e0e0'; ctx.lineWidth=1;
      for(let i=0; i<=GW; i++) {{ ctx.beginPath(); ctx.moveTo(i*CELL,0); ctx.lineTo(i*CELL,SZ_H); ctx.stroke(); }}
      for(let i=0; i<=GH; i++) {{ ctx.beginPath(); ctx.moveTo(0,i*CELL); ctx.lineTo(SZ_W,i*CELL); ctx.stroke(); }}

      // Draw walls
      ctx.fillStyle = '#444';
      walls.forEach(([wx, wy]) => {{
          ctx.fillRect(wx*CELL, wy*CELL, CELL, CELL);
      }});

      // Draw Start
      const [sx,sy] = cc(...startPos);
      ctx.fillStyle='royalblue'; ctx.fillRect(sx-8,sy-8,16,16);

      // Draw Goal
      const [gx,gy] = cc(...goalPos);
      ctx.fillStyle='gold'; ctx.beginPath(); ctx.arc(gx,gy,8,0,Math.PI*2); ctx.fill(); ctx.stroke();

      // Draw Noisy TV (Flashes random colors)
      ctx.fillStyle = `rgb(${{Math.random()*255}},${{Math.random()*255}},${{Math.random()*255}})`;
      ctx.fillRect(tvPos[0]*CELL + 2, tvPos[1]*CELL + 2, CELL-4, CELL-4);

      // Draw Trail
      if (frame > 0) {{
        ctx.beginPath(); ctx.strokeStyle='rgba(100,100,100,0.3)'; ctx.lineWidth=3;
        for (let i=0; i<=frame; i++) {{
          if(!path[i]) continue;
          const [cx,cy] = cc(path[i][1],path[i][0]);
          i===0 ? ctx.moveTo(cx,cy) : ctx.lineTo(cx,cy);
        }}
        ctx.stroke();
      }}

      // Draw Agent
      if(path[frame]) {{
          const [ax,ay] = cc(path[frame][1],path[frame][0]);
          ctx.beginPath(); ctx.arc(ax,ay,7,0,Math.PI*2);
          ctx.fillStyle = isRaw ? '#ef4444' : '#10b981'; // Red for Raw, Green for ICM
          ctx.fill(); ctx.strokeStyle='#000'; ctx.lineWidth=1.5; ctx.stroke();
      }}
    }}

    function render() {{
      drawMaze(c1, pathA, true);
      drawMaze(c2, pathB, false);
      info.textContent = 'Step: ' + frame;
    }}

    function tick() {{
      render();
      if (playing && frame < Math.max(pathA.length, pathB.length) - 1) {{
        frame++;
        timer = setTimeout(tick, 150);
      }} else if (playing) {{
        playing = false;
        btn.textContent = '🔄 Reset';
      }}
    }}

    btn.onclick = () => {{
      if (btn.textContent === '🔄 Reset') {{
          frame = 0;
          playing = true;
          btn.textContent = '⏸ Pause';
          tick();
      }} else {{
          playing = !playing;
          btn.textContent = playing ? '⏸ Pause' : '▶ Deploy Agents';
          if (playing) tick();
      }}
    }};

    render(); // Initial draw
    </script>
    </body>
    </html>"""

    # 4. Render everything together
    mo.vstack([
        act2_text,
        mo.iframe(html_tv, height="380px")
    ])
    return


@app.cell
def _(mo):
    # The PyTorch Architecture Walkthrough
    architecture_text = mo.md(
        """
        ### The Architecture: Under the Hood of ICM

        Before we look at how this breaks down in late-stage training, we need to understand how the Intrinsic Curiosity Module is actually constructed. 

        The ICM is not a standalone agent; it is an independent subsystem that runs *alongside* any standard RL algorithm (like PPO or DQN). It consists of three distinct neural networks working together:

        1. **The Feature Encoder ($\phi$):** Compresses raw pixels/states into a dense, latent vector.
        2. **The Inverse Model:** Takes the current state $\phi(s_t)$ and the next state $\phi(s_{t+1})$ to predict the action $a_t$. This is the filter that ignores unpredictable noise.
        3. **The Forward Model:** Takes the current state $\phi(s_t)$ and the action $a_t$ to predict the next state $\hat{\phi}(s_{t+1})$. 

        Here is the core PyTorch implementation of the module:

        ```python
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class IntrinsicCuriosityModule(nn.Module):
            def __init__(self, state_dim, action_dim, latent_dim=256):
                super().__init__()

                # 1. Feature Encoder: Compresses state into latent space
                self.encoder = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, latent_dim),
                    nn.ReLU()
                )

                # 2. Inverse Model: Predicts action from (s_t, s_{t+1})
                self.inverse_model = nn.Sequential(
                    nn.Linear(latent_dim * 2, 128),
                    nn.ReLU(),
                    nn.Linear(128, action_dim) 
                )

                # 3. Forward Model: Predicts next state from (s_t, action)
                self.forward_model = nn.Sequential(
                    nn.Linear(latent_dim + action_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, latent_dim)
                )

            def forward(self, state, next_state, action_one_hot):
                # Encode states into the latent space
                phi_t = self.encoder(state)
                phi_t_plus_1 = self.encoder(next_state)

                # Inverse Model Prediction (for training the encoder to filter noise)
                inverse_input = torch.cat([phi_t, phi_t_plus_1], dim=1)
                pred_action_logits = self.inverse_model(inverse_input)

                # Forward Model Prediction (for generating Intrinsic Reward)
                forward_input = torch.cat([phi_t, action_one_hot], dim=1)
                pred_phi_t_plus_1 = self.forward_model(forward_input)

                # Calculate Intrinsic Reward (Mean Squared Error in Latent Space)
                intrinsic_reward = 0.5 * (pred_phi_t_plus_1 - phi_t_plus_1).pow(2).sum(dim=1)

                return pred_action_logits, pred_phi_t_plus_1, phi_t_plus_1, intrinsic_reward
        ```

        Notice the `intrinsic_reward` calculation at the very bottom. It is completely decoupled from the game's actual score. The agent trains this module simultaneously with its policy, constantly trying to minimize both the Forward and Inverse loss, while using the resulting error as an exploration bonus.
        """
    )

    architecture_text
    return


@app.cell
def _(json, mo):
    import re
    from pathlib import Path

    # 1. Narrative Text
    train_text = mo.md(
        """
        ### Live Training: Real ICM Data from Snake

        Below are **real training logs** from our Snake DQN agent (Dueling DQN + 3-Step Returns + PER + ICM)
        trained on a 10×10 board for 16,000 games. The `Intrinsic` value is the **actual forward model
        prediction error** — not a simulation.

        Two signals tell the story:
        * **Mean Score** (blue) — rolling average game score over the last 100 games
        * **Intrinsic Reward** (orange) — the forward model's live prediction error (the curiosity signal)
        * **Epsilon** (grey dashed) — exploration rate, decaying 0.95 → 0.01

        Watch what happens after **Game ~2,000** (when ε hits its floor and the agent goes fully greedy):
        intrinsic reward keeps spiking on every novel death. The agent never stops being "surprised" —
        but that persistent surprise doesn't translate into better performance. The next sections show why.
        """
    )

    # 2. Parse the real log file
    _log_path = Path(__file__).parent.parent / "dqn_per_icm_10x10.log"
    _games, _mean_scores, _intrinsics, _epsilons = [], [], [], []
    with open(_log_path) as _f:
        for _line in _f:
            _m = re.match(
                r"Game (\d+).*?Mean Score:\s*([\d.]+).*?Epsilon:\s*([\d.]+).*?Intrinsic:\s*([\d.]+)",
                _line
            )
            if _m:
                _games.append(int(_m.group(1)))
                _mean_scores.append(float(_m.group(2)))
                _epsilons.append(float(_m.group(3)))
                _intrinsics.append(float(_m.group(4)))

    _max_score = max(_mean_scores) if _mean_scores else 1.0
    _max_intrinsic = max(_intrinsics) if _intrinsics else 1.0
    _epsilon_floor_game = next((g for g, e in zip(_games, _epsilons) if e <= 0.011), None)

    # 3. Code accordion
    code_accordion = mo.accordion({
        "🔍 View Log Parser": mo.md(
            """
            ```python
            import re
            from pathlib import Path

            log_path = Path(__file__).parent.parent / "dqn_per_icm_10x10.log"
            games, mean_scores, intrinsics, epsilons = [], [], [], []

            with open(log_path) as f:
                for line in f:
                    m = re.match(
                        r"Game (\\d+).*?Mean Score:\\s*([\\d.]+).*?Epsilon:\\s*([\\d.]+).*?Intrinsic:\\s*([\\d.]+)",
                        line
                    )
                    if m:
                        games.append(int(m.group(1)))
                        mean_scores.append(float(m.group(2)))
                        epsilons.append(float(m.group(3)))
                        intrinsics.append(float(m.group(4)))
            ```
            """
        )
    })

    # 4. Inject real data into the animated chart
    _json_data = json.dumps({
        "games": _games,
        "scores": [s / _max_score for s in _mean_scores],
        "intrinsics": [min(i / _max_intrinsic, 1.0) for i in _intrinsics],
        "epsilons": _epsilons,
        "max_score": _max_score,
        "epsilon_floor_game": _epsilon_floor_game,
        "last_game": _games[-1] if _games else 16000,
    })

    html_train = f"""<!DOCTYPE html>
    <html>
    <head>
    <style>
      body {{ margin:0; font-family:sans-serif; background:white; padding:10px; display:flex; flex-direction:column; align-items:center; }}
      .panel {{ border:1px solid #ccc; background:#f9f9f9; border-radius:6px; padding:15px; width:640px; box-shadow:0 2px 5px rgba(0,0,0,0.05); }}
      .header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }}
      .title {{ font-weight:bold; font-size:14px; color:#333; }}
      button {{ padding:8px 24px; font-size:14px; cursor:pointer; border-radius:4px; border:none; background:#10b981; color:white; font-weight:bold; transition:background 0.2s; }}
      button:hover {{ background:#059669; }}
      button:disabled {{ background:#a7f3d0; cursor:not-allowed; }}
      canvas {{ background:white; border:1px solid #eee; border-radius:4px; }}
      .legend {{ display:flex; gap:18px; font-size:12px; margin-top:10px; justify-content:center; font-weight:bold; }}
      .leg-item {{ display:flex; align-items:center; gap:5px; }}
      .dot {{ width:10px; height:10px; border-radius:50%; }}
      .note {{ font-size:11px; color:#6b7280; margin-top:6px; text-align:center; font-style:italic; }}
    </style>
    </head>
    <body>
    <div class="panel">
      <div class="header">
        <div class="title">Real ICM Training Logs — DQN + PER + ICM, 10×10 Snake ({len(_games)} checkpoints)</div>
        <button id="trainBtn">▶ Animate</button>
      </div>
      <canvas id="lossChart" width="640" height="260"></canvas>
      <div class="legend">
        <div class="leg-item"><div class="dot" style="background:#3b82f6;"></div>Mean Score</div>
        <div class="leg-item"><div class="dot" style="background:#f97316;"></div>Intrinsic Reward (scaled)</div>
        <div class="leg-item"><div class="dot" style="background:#9ca3af;"></div>Epsilon</div>
      </div>
      <div class="note">Orange spikes = real forward model surprise events on novel states and deaths. Does persistent surprise correlate with better performance? See the next section.</div>
    </div>
    <script>
    const canvas = document.getElementById('lossChart');
    const ctx = canvas.getContext('2d');
    const btn = document.getElementById('trainBtn');
    const W = 640, H = 260;
    const PL = 52, PR = 20, PT = 24, PB = 38;
    const PW = W - PL - PR, PH = H - PT - PB;
    const data = {_json_data};
    const N = data.games.length;
    let cur = 0, animId = null;

    const efx = data.epsilon_floor_game
        ? PL + (data.epsilon_floor_game / data.last_game) * PW : null;

    function drawAxes() {{
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#f9fafb';
        ctx.fillRect(PL, PT, PW, PH);
        ctx.strokeStyle = '#e5e7eb'; ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {{
            const y = PT + (i / 4) * PH;
            ctx.beginPath(); ctx.moveTo(PL, y); ctx.lineTo(PL + PW, y); ctx.stroke();
        }}
        ctx.strokeStyle = '#9ca3af'; ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(PL, PT); ctx.lineTo(PL, PT + PH); ctx.lineTo(PL + PW, PT + PH);
        ctx.stroke();
        ctx.fillStyle = '#6b7280'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {{
            const val = ((4 - i) / 4) * data.max_score;
            ctx.fillText(val.toFixed(1), PL - 4, PT + (i / 4) * PH + 4);
        }}
        // Rotated y-axis title
        ctx.save();
        ctx.translate(12, PT + PH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center'; ctx.font = '10px sans-serif'; ctx.fillStyle = '#9ca3af';
        ctx.fillText('Mean Score', 0, 0);
        ctx.restore();
        ctx.textAlign = 'center';
        for (let i = 0; i <= 4; i++) {{
            const g = Math.round((i / 4) * data.last_game);
            ctx.fillText('G' + g.toLocaleString(), PL + (i / 4) * PW, PT + PH + 14);
        }}
        if (efx) {{
            ctx.save();
            ctx.strokeStyle = 'rgba(239,68,68,0.5)'; ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 3]);
            ctx.beginPath(); ctx.moveTo(efx, PT); ctx.lineTo(efx, PT + PH); ctx.stroke();
            ctx.restore();
            ctx.fillStyle = '#ef4444'; ctx.font = 'bold 9px sans-serif'; ctx.textAlign = 'center';
            ctx.fillText('ε→0.01', efx, PT - 8);
        }}
    }}

    function drawLine(arr, color, maxIdx, dashed) {{
        if (maxIdx < 2) return;
        ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = dashed ? 1.5 : 2;
        if (dashed) ctx.setLineDash([5, 3]);
        for (let i = 0; i < maxIdx; i++) {{
            const x = PL + (i / (N - 1)) * PW;
            const y = PT + PH - (arr[i] * PH);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }}
        ctx.stroke();
        if (dashed) ctx.setLineDash([]);
    }}

    function render(maxIdx) {{
        drawAxes();
        drawLine(data.epsilons, '#9ca3af', maxIdx, true);
        drawLine(data.intrinsics, '#f97316', maxIdx, false);
        drawLine(data.scores, '#3b82f6', maxIdx, false);
    }}

    function animate() {{
        render(cur);
        if (cur < N) {{ cur += 2; animId = requestAnimationFrame(animate); }}
        else {{ btn.textContent = '🔄 Reset'; btn.disabled = false; }}
    }}

    btn.onclick = () => {{
        if (btn.textContent === '🔄 Reset') {{
            cur = 0; render(0); btn.textContent = '▶ Animate';
        }} else {{
            btn.disabled = true; btn.textContent = 'Animating...';
            animate();
        }}
    }};

    render(N);
    </script>
    </body>
    </html>"""

    mo.vstack([train_text, code_accordion, mo.iframe(html_train, height="430px")])
    return


@app.cell
def _(mo):
    # The Transition and Literature Review Text
    bridge_text = mo.md(
        """
        ### The Paradigm Shift: Self-Supervised vs. Hand-Coded Rewards

        Before we look at where this architecture breaks down, we must acknowledge why Pathak et al.'s approach was so revolutionary. The title of the paper specifies **"Self-Supervised Prediction."** What does this mean?

        Historically, to solve sparse environments, engineers relied on **Hand-Coded Reward Shaping**. A human programmer would manually inject domain knowledge: *"If the agent moves closer to the goal, give it +0.1 points. If it moves away, -0.1 points."* While reward shaping works, it is brittle. It requires a human to hand-hold the agent through every new game, and it frequently leads to "reward hacking" (where the agent finds a loophole to farm points without actually winning the game).

        **Self-Supervised** curiosity requires zero human intervention. The agent creates its own dense reward signal natively from the environment's raw pixels. It learns the physics of the world first, and uses that knowledge to systematically explore until it naturally stumbles upon the true goal.

        ---

        ### The Mechanics of Memory: Prioritized Experience Replay (PER)

        The paper itself anticipated this problem. In the final paragraphs, Pathak et al. write:

        > *“While the rich and diverse real world provides ample opportunities for interaction, reward signals are sparse. Our approach excels in this setting. However our approach does not directly extend to the scenarios where ‘opportunities for interactions’ are also rare. In theory, **one could save such events in a replay memory and use them to guide exploration**. However, we leave this extension for future work.”*

        Snake on a 10×10 board is precisely this setting. Early in training, apples are rare, interactions are thin, and the agent needs a way to revisit its most informative experiences. **We implemented the paper’s suggested extension** — a Prioritized Experience Replay (PER) buffer that samples memories based on their **Temporal Difference (TD) Error** — how “wrong” the agent was about that specific memory.

        * In a standard buffer, memories are sampled purely at random.
        * In **PER**, high-error memories are sampled more frequently, forcing the agent to study its biggest mistakes.

        But what happens when we combine **Self-Supervised Curiosity**, **PER**, and an environment with terminal failure states (like *Snake*)?
        """
    )

    bridge_text
    return


@app.cell
def _(mo):
    # 1. The Narrative Text
    act3_text = mo.md(
        """
        ### The Edge Case: The Death Oversampling Trap (PER + ICM)

        ICM is brilliant in sparse environments, but when deployed in environments with terminal failure states (like *Snake*), it introduces a fatal mathematical flaw. 

        When an agent dies, the environment resets abruptly. The Forward Model cannot predict a "Game Over" screen from a standard movement action. This results in a massive spike in Prediction Error, which translates to a massive Intrinsic Reward. **The agent learns that suicide is incredibly rewarding.**

        This becomes catastrophic if you are using **Prioritized Experience Replay (PER)**. PER samples memories from the replay buffer based on their error magnitude. Because "deaths" have the highest error, PER oversamples them. 

        **The Trap:** The buffer becomes poisoned. The agent's training batches become flooded with death sequences, completely drowning out the "normal" steps and successful apple captures. The agent unlearns how to play and optimizes for fast deaths.

        *Adjust the toggle below to see how adding ICM to a PER buffer poisons the training batch.*
        """
    )

    # 2. The Interactive UI (FIXED: Using a flat list to prevent KeyErrors)
    agent_type = mo.ui.radio(
        options=["Standard Agent (PER Only)", "Curious Agent (PER + ICM)"],
        value="Standard Agent (PER Only)",
        label="**Select Agent Architecture:**"
    )

    sample_btn = mo.ui.button(
        label="🎲 Sample Training Batch (n=32)", 
        kind="success"
    )
    return act3_text, agent_type, sample_btn


@app.cell
def _(act3_text, agent_type, mo, np, plt, sample_btn):
    # 3. The Visualization Logic
    def render_batch_viz(agent_val, btn_click):
        # Base Replay Buffer Composition (100 memories total)
        # 80 Normal Steps, 15 Apples, 5 Deaths
        types = ['Normal'] * 80 + ['Apple'] * 15 + ['Death'] * 5
        colors = ['#B0BEC5'] * 80 + ['#4CAF50'] * 15 + ['#F44336'] * 5

        # Base Temporal Difference (TD) errors
        base_td = np.array([0.1] * 80 + [0.4] * 15 + [0.8] * 5)

        # Check the exact string from the UI toggle
        if agent_val == "Curious Agent (PER + ICM)":
            # ICM adds massive intrinsic surprise to terminal states
            intrinsic_surprise = np.array([0.0] * 80 + [0.0] * 15 + [4.0] * 5)
            priorities = base_td + intrinsic_surprise
        else:
            priorities = base_td

        probabilities = priorities / np.sum(priorities)

        # Sample the Batch
        batch_size = 32
        sampled_indices = np.random.choice(
            len(types), size=batch_size, p=probabilities, replace=True
        )

        sampled_types = [types[i] for i in sampled_indices]
        counts = {
            'Normal Steps': sampled_types.count('Normal'),
            'Apples (Reward)': sampled_types.count('Apple'),
            'Deaths (Penalty)': sampled_types.count('Death')
        }

        # Render
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [1.2, 1]})

        # Left Plot: Memory Pool Bubble Chart
        np.random.seed(42) # Fixed seed for positions so bubbles don't jump around
        x_pos, y_pos = np.random.rand(100), np.random.rand(100)
        np.random.seed(None)

        sizes = priorities * 150 
        ax1.scatter(x_pos, y_pos, s=sizes, c=colors, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax1.set_title("Replay Buffer (Bubble Size = Selection Priority)", fontweight='bold')
        ax1.axis('off')

        # Right Plot: Bar Chart
        labels = list(counts.keys())
        values = list(counts.values())
        bar_colors = ['#B0BEC5', '#4CAF50', '#F44336']

        bars = ax2.bar(labels, values, color=bar_colors, edgecolor='black')
        ax2.set_title(f"Sampled Training Batch (Size: {batch_size})", fontweight='bold')
        ax2.set_ylabel("Number of Samples in Batch")
        ax2.set_ylim(0, batch_size)

        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        return mo.as_html(fig)

    # 4. Stack the UI elements
    ui_stack = mo.vstack([
        act3_text,
        mo.hstack([agent_type, sample_btn], justify="start", align="center", gap=2)
    ])

    # 5. Bind the render function to the UI state
    # Marimo will automatically re-run this when the button or radio changes
    chart = render_batch_viz(agent_type.value, sample_btn.value)

    mo.vstack([ui_stack, chart])
    return


@app.cell
def _(mo, plt):
    import re as _re
    from pathlib import Path as _Path

    # 1. Narrative
    fix_text = mo.md(
        """
        ### The Paper's Stated Limitations

        Before we reveal the fix, it's worth noting what Pathak et al. themselves acknowledged.
        The Inverse Model perfectly filters noise that is *independent* of the agent's actions
        (the TV with random static). But it fails when the noise is **action-dependent**:

        > If the environment contains a remote control that changes the TV to a random channel,
        > the Inverse Model *cannot* filter it out. Because the agent's action directly caused
        > the unpredictable state change, the agent will sit and press the button forever.

        The paper flags this as an open problem. Our Snake failure is a different but related
        variant: **terminal states** are action-dependent (the death is caused by the agent's
        move), and the physics of a game-over reset is maximally unpredictable. The paper's
        own framework predicts this should be a problem — and it is.

        ---

        ### The Fix: One Line Changes Everything

        The death trap has a clean solution — **zero out the intrinsic reward on terminal steps**.
        When the game ends (`done=True`), `(1 − done) = 0`, so the massive death-surprise is
        suppressed before it ever enters the replay buffer. PER can no longer oversample deaths
        with inflated priorities.

        ```python
        intrinsic_reward = agent.train_icm(state_old, final_move, state_new)
        intrinsic_reward *= (1.0 - float(done))  # ← The entire fix
        reward += intrinsic_reward
        ```

        The chart below overlays the **actual Mean Score curves** from both runs on the same
        10×10 board for 16,000 games.
        """
    )

    # 2. Parse both logs
    def _parse(path):
        games, scores = [], []
        try:
            with open(path) as f:
                for line in f:
                    m = _re.match(r"Game (\d+).*?Mean Score:\s*([\d.]+)", line)
                    if m:
                        games.append(int(m.group(1)))
                        scores.append(float(m.group(2)))
        except FileNotFoundError:
            pass
        return games, scores

    _root = _Path(__file__).parent.parent
    _g_poi, _s_poi = _parse(_root / "dqn_per_icm_10x10.log")
    _g_fix, _s_fix = _parse(_root / "dqn_per_v3_icm_10x10.log")

    # 3. Plot
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.plot(_g_poi, _s_poi, color='#ef4444', linewidth=2,
             label='PER + ICM — unmasked (poisoned)', alpha=0.9)
    _ax.plot(_g_fix, _s_fix, color='#10b981', linewidth=2,
             label='PER + ICM + terminal mask — fixed', alpha=0.9)

    if _s_poi:
        _ax.annotate(f"plateau ~{_s_poi[-1]:.1f}",
                     xy=(_g_poi[-1], _s_poi[-1]),
                     xytext=(-70, -18), textcoords='offset points',
                     color='#ef4444', fontweight='bold', fontsize=10,
                     arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.2))
    if _s_fix:
        _ax.annotate(f"~{_s_fix[-1]:.1f}",
                     xy=(_g_fix[-1], _s_fix[-1]),
                     xytext=(-55, 8), textcoords='offset points',
                     color='#10b981', fontweight='bold', fontsize=10,
                     arrowprops=dict(arrowstyle='->', color='#10b981', lw=1.2))

    _ax.set_xlabel("Game", fontsize=11)
    _ax.set_ylabel("Mean Score (100-game rolling avg)", fontsize=11)
    _ax.set_title("Terminal Reward Masking: r_i × (1 − done)", fontweight='bold', fontsize=13)
    _ax.legend(loc='upper left', fontsize=10)
    _ax.spines['top'].set_visible(False)
    _ax.spines['right'].set_visible(False)
    _ax.grid(axis='y', alpha=0.3)
    _ax.set_xlim(0)
    _ax.set_ylim(0)
    plt.tight_layout()

    mo.vstack([fix_text, mo.as_html(_fig)])
    return


@app.cell(hide_code=True)
def _(mo):
    conclusion_text = mo.md(
        """
        ---

        ### Conclusion: The Legacy of ICM

        The Intrinsic Curiosity Module (Pathak et al., 2017) fundamentally changed how we approach
        exploration in Reinforcement Learning. By equating **surprise with reward**, it proved that
        agents do not need hand-coded guidance to navigate sparse environments — they just need a
        model of the world's physics and a drive to be surprised by it.

        But as we demonstrated with the Snake edge-case, purely relying on forward prediction error
        has real limits. Action-dependent noise and terminal failure states can silently hijack the
        curiosity signal. The fix — zeroing intrinsic reward at terminal steps — is one line of code,
        but finding it required understanding the full interaction between ICM, PER, and game-over
        dynamics. That understanding is what this notebook was built to give you.

        ---

        ### Beyond ICM: Where the Research Went Next

        ICM was the first step. Here is the lineage of work that followed, in order:

        * **Large-Scale Study of Curiosity-Driven Learning** *(Burda, Edwards, Pathak et al., 2018)*  
          A follow-up proving that agents equipped with *only* intrinsic curiosity — zero extrinsic
          game score — could successfully learn 54 Atari environments purely by seeking novelty.
          It validated ICM's core hypothesis at scale.

        * **Exploration by Random Network Distillation (RND)** *(Burda et al., 2018)*  
          The definitive fix to the Noisy TV problem. Instead of predicting the *next state*
          (which stochastic noise corrupts), RND passes the current state through a fixed, randomly
          initialized network and trains a second network to match its output. Novelty = prediction
          gap between the two networks. No physics prediction, no noise vulnerability.

        * **Episodic Curiosity through Reachability** *(Savinov et al., 2018)*  
          Solves the within-episode "couch potato" problem: an agent that finds a novel state
          and stays there, farming curiosity reward in one spot. Episodic Curiosity stores a memory
          of states visited *this episode* and only rewards states that are hard to reach from
          anything in memory — forcing the agent to keep moving.

        * **Dream to Control / DreamerV2 / DreamerV3** *(Hafner et al., DeepMind, 2019–2023)*  
          The natural successor to ICM's core insight. Where ICM's Forward Model predicts only
          the **next state**, Dreamer learns a complete **world model** — a compact latent
          representation of the environment's dynamics, rewards, and terminations. The policy then
          trains entirely inside imagined rollouts ("dreams"), never touching the real environment
          during learning. DreamerV3 (2023) achieved human-level performance across 150+ tasks —
          including Minecraft diamond collection — without any task-specific tuning. The lineage
          from ICM's single forward model to Dreamer's full generative world model is direct.

        * **Agent57: Outperforming the Atari Human Benchmark** *(Badia et al., DeepMind, 2020)*  
          The convergence point of intrinsic exploration research. Agent57 combined episodic
          novelty (short-term) with a lifelong novelty metric (long-term) and became the first
          agent to achieve above-human performance on all 57 classic Atari games — a benchmark
          that had stood for nearly a decade.

        ---
        *Built with Python, PyTorch, and Marimo. Trained on a 10×10 Snake board.*
        """
    )

    conclusion_text
    return


if __name__ == "__main__":
    app.run()
