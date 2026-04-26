# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Curiosity Killed the Snake: The Trap That Taught an Agent to Die
    *An interactive exploration of "Curiosity-driven Exploration by Self-supervised Prediction" (Pathak et al., 2017).*

    [![Open in marimo](https://marimo.io/shield.svg)](https://molab.marimo.io/github/Saheb/rl-snake/blob/main/notebooks/curiosity.py/wasm)

    ### The Problem: Sparse Reward Trap
    In reinforcement learning, an agent learns by maximizing reward. Usually that reward is **external**: it comes from the environment, like `+1` for eating an apple in Snake or `-1` for dying. But what happens when the environment is vast, the board is empty, and the external reward is incredibly hard to find by pure chance? What happens if the environment gives no useful reward signal for a long time?

    When an $\epsilon$-greedy agent, such as a standard **Deep Q-Network (DQN)**, faces a sparse environment, its exploration is mostly random. It can repeatedly revisit familiar states, spinning in circles rather than systematically mapping the environment.

    **Play with the agent below to see how it performs when the reward is 13 steps away.**
    """)
    return


@app.cell(hide_code=True)
def _(json, mo, random, wasm_iframe):
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
    <div class="sublabel">Curiosity proxy — seeks lowest-visited tiles</div>
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
        wasm_iframe(html, height="500px"),
        mo.callout(
            mo.md(
                "**Visualization note:** The right-hand agent uses a **count-based heuristic** "
                "(always move to the lowest visit-count neighbour), not a live neural network. "
                "In this discrete finite grid, visit counts serve as a practical stand-in for "
                "the Intrinsic Curiosity Module (ICM)'s forward-model prediction error — both signal how *novel* a state is, "
                "and the outward-seeking behaviour looks similar. However, this equivalence "
                "**only holds in tabular settings** with an enumerable state space. "
                "The Intrinsic Curiosity Module's core contribution is generalizing curiosity to **continuous, high-dimensional "
                "observation spaces** (raw pixels, sensor arrays) where visit counts are undefined — "
                "and where a learned forward model is the only tractable novelty signal."
            ),
            kind="info"
        )
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    eta_slider = mo.ui.slider(
        start=0.01, stop=1.0, step=0.01, value=0.5,
        label="Curiosity Weight η (Eta)",
        show_value=True
    )
    return (eta_slider,)


@app.cell(hide_code=True)
def _(mo, wasm_iframe):
    # 1. The Narrative Text (Explaining the Forward Model)
    intro_text = mo.md(
        """
        ### The Concept: Mathematically Defining "Curiosity"

        If the environment is not giving the agent useful rewards, the agent has to generate its own. Pathak et al. achieved this by giving the agent a **Forward Model**: a neural network inside the agent that learns to predict what the next observation should look like after the agent takes an action.

        Before the agent takes a step, the Forward Model predicts the next state. The agent then takes the step, observes the actual next state, and compares prediction to reality. The difference is the **Prediction Error** (Mean Squared Error). 

        $$Intrinsic\\ Reward = \\frac{\\eta}{2} (Predicted\\ State - Actual\\ State)^2$$

        Where **$\\eta$ (Eta)** is the **Curiosity Weight**. It scales how much intrinsic reward the agent gets from being surprised. The key insight of this equation is that **Prediction Error = Surprise = Reward**.

        More precisely, ICM computes prediction error in a learned latent space:

        $$r_t^i = \\frac{\\eta}{2}\\left\\|\\hat{\\phi}(s_{t+1}) - \\phi(s_{t+1})\\right\\|_2^2$$

        The squared error is not arbitrary. It is the standard regression loss for a continuous target and corresponds to a Gaussian negative log-likelihood assumption: large prediction mistakes are penalized quadratically, while small residual errors fade smoothly. The latent-space term matters because raw pixels contain action-irrelevant noise. ICM first maps observations through $\\phi(\\cdot)$, then rewards errors only in the learned representation that is useful for predicting the agent's own actions.

        The Forward Model is trained from the agent's own experience: each transition `(state, action, next_state)` becomes a supervised learning example. After many updates on familiar transitions, its predictions become accurate and the intrinsic reward shrinks.

        * If the agent repeatedly visits the same tile and observes the same transition, its Forward Model learns that transition. The prediction error becomes small. The agent is bored.
        * If the agent reaches a new tile or a hard-to-predict transition, the Forward Model is wrong. The error is large. The agent experiences a spike of surprise.

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
        <button id="resetBtn" style="padding:6px 16px; cursor:pointer; background:#f0f0f0; border:1px solid #aaa; border-radius:4px; font-weight:bold;">↺ Reset Environment</button>
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
        wasm_iframe(html2, height="400px"),
        mo.callout(
            mo.md(
                "**Simplification note:** The reward curve above uses a hardcoded 3-step decay "
                "(1st visit → 1.0, 2nd → 0.4, 3rd → 0.1) as a pedagogical proxy, not a real forward model. "
                "A live ICM computes $\\frac{\\eta}{2}(\\hat{\\phi}_{t+1} - \\phi_{t+1})^2$ in latent space — "
                "the qualitative shape is the same but the exact decay depends on network training. "
                "See the **Mini-Experiment** section for real gradient-descent-based decay. "
                "The slider below shows how η scales the reward magnitude in the real formula."
            ),
            kind="info"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(eta_slider, mo):
    mo.hstack(
        [eta_slider, mo.md(f"**η = {eta_slider.value:.2f}** → max intrinsic reward on first visit ≈ **{eta_slider.value / 2:.3f}** (η/2 × 1²)")],
        justify="start", align="center", gap=2
    )
    return


@app.cell(hide_code=True)
def _(json, mo, wasm_iframe):
    # 1. The Narrative Text (Explaining the Trap and the Filter)
    act2_text = mo.md(
        """
        ### A fatal flaw: The Noisy TV Problem

        If you try to predict raw pixels, the agent gets distracted by irrelevant stochasticity (a flickering leaf, a TV screen with random noise) that has nothing to do with the agent's actions. Nor does it affect the agent in any way. It's novel, but it's not relevant. This is sometimes called the "noisy TV problem"

        How do we remove this noise, so agent can focus on what matters? How do we teach the agent to ignore this noise?

        Rather than making predictions in raw sensory space (e.g. pixels), the paper transforms sensory input into a feature space where only information relevant to the agent's actions is represented. The core insight is to predict only those changes in the environment that could possibly be due to the actions of the agent or could affect the agent — and ignore everything else.


        > "We learn this feature space using self-supervision – training a neural network on a proxy inverse dynamics task of predicting the agent’s action given its current and next states. Since the neural network is only required to predict the action, it has no incentive to represent within its feature embedding space the factors of variation in the environment that do not affect the agent itself."
        >
        > — Pathak et al., *"Curiosity-driven Exploration by Self-supervised Prediction"*, ICML 2017. [arXiv:1705.05363](https://arxiv.org/abs/1705.05363)

        The Inverse Model is trained to predict the *agent’s own actions*. Because the agent's actions cannot control the random TV static, the neural network learns to completely ignore the TV when creating the latent vector. The noise is mathematically filtered out.

        **Run the agents below** to see why raw-pixel prediction fails, and how the latent filter in the Intrinsic Curiosity Module (ICM) saves the agent.
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
        [0,6],[1,6],[2,6],[3,6],[4,6],[5,6],[6,6],[7,6],
        [0,1],[0,2],[0,3],[0,4],[0,5],[7,1],[7,2],[7,3],[7,4],[7,5],
        [3,1],[4,1],[5,1],[3,2],[4,2],[5,2],
        [2,3],[3,3],[4,3],[2,5],[3,5],[4,5],[5,5]
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
      ctx.fillStyle = 'rgb(' + Math.floor(Math.random()*255) + ',' + Math.floor(Math.random()*255) + ',' + Math.floor(Math.random()*255) + ')';
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
        btn.textContent = '↺ Reset';
      }}
    }}

    btn.onclick = () => {{
      if (btn.textContent === '↺ Reset') {{
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
        wasm_iframe(html_tv, height="380px"),
        mo.callout(
            mo.md(
                "**Visualization note:** The agent paths above are **scripted pedagogical illustrations**, "
                "not live neural network runs. The Raw agent is hardcoded to stay fixed at the TV tile; "
                "the ICM agent is hardcoded to move on after 5 steps. The qualitative behavior is "
                "grounded in the paper's theoretical predictions — a real ICM agent learns to filter "
                "action-independent noise through its inverse model training objective — but the paths "
                "themselves are hand-authored for clarity."
            ),
            kind="info"
        )
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    # The PyTorch Architecture Walkthrough
    architecture_text = mo.md(
        r"""
        ### The Architecture: Under the Hood of ICM

        Before we look at how this breaks down in late-stage training, we need to understand how the Intrinsic Curiosity Module is actually constructed. 

        The ICM is not a standalone agent; it is an auxiliary subsystem that runs *inside the training loop* of a standard reinforcement-learning algorithm such as Proximal Policy Optimization (PPO) or Deep Q-Network (DQN). The policy still chooses actions and learns from rewards. ICM adds an extra reward term and trains its own predictive models from the same transitions.

        1. **The Feature Encoder ($\phi$):** Compresses raw pixels/states into a dense, latent vector.
        2. **The Inverse Model:** Takes the current state $\phi(s_t)$ and the next state $\phi(s_{t+1})$ to predict the action $a_t$. This is the filter that ignores unpredictable noise.
        3. **The Forward Model:** Takes the current state $\phi(s_t)$ and the action $a_t$ to predict the next state $\hat{\phi}(s_{t+1})$. 

        Formally, the inverse model is a classifier and the forward model is a regressor:

        $$L_I = -\\log p(a_t \\mid \\phi(s_t), \\phi(s_{t+1}))$$

        $$L_F = \\frac{1}{2}\\left\\|\\hat{\\phi}(s_{t+1}) - \\phi(s_{t+1})\\right\\|_2^2$$

        $$L_{ICM} = (1 - \\beta)L_I + \\beta L_F$$

        $L_I$ trains the encoder to preserve action-relevant information: if a visual feature does not help infer which action moved the agent from $s_t$ to $s_{t+1}$, the inverse objective has little reason to keep it. $L_F$ trains the dynamics predictor in that filtered feature space. The intrinsic reward is proportional to $L_F$, but the encoder is shaped by both losses.

        In a DQN-style loop, the call site looks like this:

        ```python
        action = dqn_policy.select_action(state)
        next_state, external_reward, done = env.step(action)

        inverse_loss, forward_loss, intrinsic_reward = icm(state, next_state, action)
        reward_for_dqn = external_reward + eta * intrinsic_reward

        replay_buffer.add(state, action, reward_for_dqn, next_state, done)

        dqn_loss = train_dqn_from_replay(replay_buffer)
        icm_loss = (1 - beta) * inverse_loss + beta * forward_loss
        ```

        So the ICM is called immediately after the environment step. Its prediction error becomes an exploration bonus, while its own networks are trained on the observed transition.

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

        Notice the `intrinsic_reward` calculation at the very bottom. It is decoupled from the game's actual score, then added to the environment reward before the policy update. The agent trains this module simultaneously with its policy, constantly trying to minimize both the Forward and Inverse loss, while using the current Forward Model error as an exploration bonus.
        """
    )

    architecture_text
    return


@app.cell(hide_code=True)
def _(mo):
    _state_dim, _action_dim, _batch, _latent_dim = 20, 4, 8, 256

    mo.callout(
        mo.md(f"""
        **WASM-compatible forward-pass shape check** — same tensor contract as the PyTorch module above:

        | Tensor | Shape | Role |
        |--------|-------|------|
        | `pred_action_logits` | `{[_batch, _action_dim]}` | Inverse model output — predicted action |
        | `pred_phi_next` | `{[_batch, _latent_dim]}` | Forward model output — predicted next latent |
        | `phi_next` | `{[_batch, _latent_dim]}` | Encoder output — actual next latent |
        | `intrinsic_reward` | `{[_batch]}` | Per-step curiosity signal (MSE in latent space) |

        The executable PyTorch demo was removed from the notebook runtime because `torch` has no Pyodide/WASM build.
        The architecture above remains the implementation used in the repository; the browser cells below use NumPy
        proxies to preserve the same math and interactivity.
        """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _train_intro = mo.md(r"""
    ### Mini-Experiment: Does the Forward Model Actually Learn?

    The cell above defined the ICM tensor contract. This experiment shows the core learning signal directly.

    We train a tiny **NumPy forward model** on a **5×5 grid** with a **purely random policy**
    (ε = 1.0, zero extrinsic reward). It predicts the next grid position from the current position
    and action, then uses its own prediction error as the intrinsic reward.

    This is "boredom by gradient descent." Same phenomenon as the boredom simulator — now produced by
    a browser-compatible model that can run inside Pyodide.
    """)
    train_btn = mo.ui.button(
        label="▶ Train NumPy forward model on 5×5 grid",
        kind="success",
        value=0,
        on_click=lambda v: v + 1,
    )
    mo.vstack([_train_intro, train_btn])
    return (train_btn,)


@app.cell(hide_code=True)
def _(mo, np, plt, random, train_btn):
    mo.stop(
        np is None or plt is None,
        mo.callout(
            mo.md("**NumPy/Matplotlib are unavailable in this runtime.** In WASM they are installed from the notebook metadata."),
            kind="warn",
        ),
    )
    if not train_btn.value:
        _fig0, _ax0 = plt.subplots(figsize=(9, 3.5))
        _ax0.set_xlabel("Episode", fontsize=11)
        _ax0.set_ylabel("Mean Intrinsic Reward per Step (log scale)", fontsize=11)
        _ax0.set_title("ICM Forward Model Learning Curve — 5×5 Grid, Random Policy", fontweight="bold", fontsize=13)
        _ax0.set_xlim(1, 300)
        _ax0.spines["top"].set_visible(False)
        _ax0.spines["right"].set_visible(False)
        _ax0.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        _out = mo.as_html(_fig0)
    else:
        class _GridEnv:
            def __init__(self, size=5):
                self.size = size
                self.pos = [size // 2, size // 2]
            def reset(self):
                self.pos = [self.size // 2, self.size // 2]
                return np.array(self.pos, dtype=np.float32) / self.size
            def step(self, action):
                dy, dx = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
                self.pos[0] = max(0, min(self.size - 1, self.pos[0] + dy))
                self.pos[1] = max(0, min(self.size - 1, self.pos[1] + dx))
                return np.array(self.pos, dtype=np.float32) / self.size, 0.0, False

        _env = _GridEnv(5)
        _rng = np.random.default_rng(7)
        _w = _rng.normal(0.0, 0.15, size=(6, 2))
        _b = np.zeros(2)
        _lr = 0.2
        _ep_rewards = []

        for _ep in range(300):
            _s = _env.reset()
            _features, _targets = [], []
            for _ in range(25):
                _a = random.randint(0, 3)
                _ns, _, _ = _env.step(_a)
                _action_one_hot = np.eye(4, dtype=np.float32)[_a]
                _features.append(np.concatenate([_s, _action_one_hot]))
                _targets.append(_ns)
                _s = _ns

            _x = np.asarray(_features)
            _y = np.asarray(_targets)
            _pred = _x @ _w + _b
            _err = _pred - _y
            _r_i = 0.5 * np.sum(_err * _err, axis=1)
            _ep_rewards.append(float(np.mean(_r_i)))

            _grad_pred = _err / len(_x)
            _w -= _lr * (_x.T @ _grad_pred)
            _b -= _lr * np.sum(_grad_pred, axis=0)

        _episodes = list(range(1, 301))
        _floor = 1e-6
        _rewards_floored = [max(r, _floor) for r in _ep_rewards]
        _smooth = np.maximum(np.convolve(_rewards_floored, np.ones(10) / 10, mode='valid'), _floor)
        _fig, _ax = plt.subplots(figsize=(9, 3.5))
        _ax.plot(_episodes, _rewards_floored, color='#93c5fd', linewidth=1, alpha=0.5, label='Per-episode')
        _ax.plot(_episodes[9:], _smooth, color='#2563eb', linewidth=2.5, label='10-episode rolling avg')
        _ax.set_yscale('log')
        _ax.set_xlabel("Episode", fontsize=11)
        _ax.set_ylabel("Mean Intrinsic Reward per Step (log scale)", fontsize=11)
        _ax.set_title("ICM Forward Model Learning Curve — 5×5 Grid, Random Policy", fontweight='bold', fontsize=13)
        _ax.legend(fontsize=10)
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)
        _ax.grid(axis='y', alpha=0.3, which='both')
        _ax.set_xlim(1, 300)
        plt.tight_layout()
        _first5 = sum(_ep_rewards[:5]) / 5
        _last5 = sum(_ep_rewards[-5:]) / 5
        _drop = int(_first5 / (_last5 + 1e-9))

        _out = mo.vstack([
            mo.as_html(_fig),
            mo.callout(
                mo.md(
                    f"Intrinsic reward decayed from **{_first5:.4f}** (first 5 episodes) → "
                    f"**{_last5:.5f}** (last 5 episodes) — a **{_drop:,}× drop**. "
                    "The forward model has learned the grid's transition structure. There is little left to be curious about. "
                    "The same phenomenon the boredom simulator illustrated — now shown by gradient descent."
                ),
                kind="success"
            )
        ])
    _out
    return


@app.cell(hide_code=True)
def _(json, mo, wasm_iframe):
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

        Methodological caveat: this is a mechanistic case study, not a benchmark claim. A research-grade evaluation would run multiple random seeds and report confidence intervals for at least four ablations: DQN + PER, DQN + ICM, DQN + PER + ICM, and DQN + PER + ICM with terminal intrinsic rewards masked. The single-run logs here are still useful because they expose a concrete failure mode in the replay distribution.
        """
    )

    # 2. Inlined training data (parsed from dqn_per_icm_10x10.log; inlined so the notebook
    #    runs without local files in molab/WASM environments)
    _games = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900, 11000, 11100, 11200, 11300, 11400, 11500, 11600, 11700, 11800, 11900, 12000, 12100, 12200, 12300, 12400, 12500, 12600, 12700, 12800, 12900, 13000, 13100, 13200, 13300, 13400, 13500, 13600, 13700, 13800, 13900, 14000, 14100, 14200, 14300, 14400, 14500, 14600, 14700, 14800, 14900, 15000, 15100, 15200, 15300, 15400, 15500, 15600, 15700, 15800, 15900, 16000]
    _mean_scores = [0.11, 0.13, 0.13, 0.14, 0.15, 0.16, 0.19, 0.23, 0.27, 0.3, 0.34, 0.37, 0.42, 0.46, 0.49, 0.54, 0.6, 0.66, 0.74, 0.84, 0.96, 1.07, 1.23, 1.37, 1.49, 1.61, 1.71, 1.79, 1.88, 1.95, 2.05, 2.17, 2.27, 2.4, 2.53, 2.65, 2.76, 2.88, 2.95, 3.03, 3.12, 3.19, 3.26, 3.38, 3.45, 3.53, 3.61, 3.68, 3.77, 3.85, 3.91, 4.05, 4.15, 4.21, 4.3, 4.39, 4.43, 4.48, 4.6, 4.72, 4.84, 4.88, 4.97, 5.04, 5.13, 5.18, 5.25, 5.3, 5.35, 5.4, 5.47, 5.54, 5.59, 5.63, 5.67, 5.71, 5.75, 5.8, 5.84, 5.92, 5.96, 6.0, 6.04, 6.1, 6.15, 6.19, 6.21, 6.25, 6.3, 6.34, 6.38, 6.42, 6.44, 6.48, 6.51, 6.54, 6.6, 6.64, 6.66, 6.67, 6.67, 6.7, 6.73, 6.77, 6.81, 6.85, 6.86, 6.89, 6.92, 6.95, 6.99, 7.04, 7.08, 7.12, 7.15, 7.18, 7.19, 7.23, 7.25, 7.29, 7.31, 7.33, 7.36, 7.37, 7.39, 7.4, 7.42, 7.45, 7.48, 7.49, 7.51, 7.53, 7.57, 7.6, 7.63, 7.65, 7.66, 7.68, 7.68, 7.69, 7.72, 7.74, 7.76, 7.79, 7.8, 7.84, 7.86, 7.89, 7.9, 7.92, 7.94, 7.93, 7.96, 7.98, 7.99, 8.01, 8.03, 8.03, 8.05, 8.06]
    _epsilons = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    _intrinsics = [0.003, 0.003, 0.001, 0.001, 0.002, 0.003, 0.002, 0.003, 0.002, 0.001, 0.007, 0.005, 0.002, 0.014, 0.004, 0.001, 0.002, 0.011, 0.003, 0.003, 0.045, 0.011, 0.002, 0.047, 0.004, 0.004, 0.001, 0.001, 0.002, 0.001, 0.043, 0.002, 0.047, 0.001, 0.008, 0.054, 0.02, 0.012, 0.007, 0.013, 0.015, 0.006, 0.006, 0.013, 0.008, 0.006, 0.003, 0.004, 0.066, 0.107, 0.002, 0.004, 0.127, 0.008, 0.034, 0.003, 0.011, 0.067, 0.026, 0.02, 0.082, 0.004, 0.119, 0.11, 0.005, 0.034, 0.017, 0.158, 0.016, 0.002, 0.068, 0.012, 0.006, 0.085, 0.007, 0.042, 0.003, 0.003, 0.011, 0.034, 0.036, 0.112, 0.005, 0.064, 0.082, 0.009, 0.004, 0.125, 0.196, 0.009, 0.083, 0.007, 0.005, 0.011, 0.085, 0.011, 0.072, 0.007, 0.105, 0.089, 0.021, 0.09, 0.124, 0.014, 0.091, 0.158, 0.095, 0.005, 0.11, 0.132, 0.047, 0.044, 0.044, 0.096, 0.008, 0.01, 0.006, 0.058, 0.006, 0.169, 0.133, 0.03, 0.016, 0.04, 0.01, 0.007, 0.005, 0.064, 0.121, 0.007, 0.019, 0.07, 0.003, 0.143, 0.009, 0.168, 0.008, 0.01, 0.107, 0.026, 0.014, 0.135, 0.005, 0.015, 0.006, 0.092, 0.004, 0.106, 0.002, 0.081, 0.129, 0.022, 0.078, 0.108, 0.002, 0.079, 0.046, 0.002, 0.067, 0.008]

    _max_score = max(_mean_scores) if _mean_scores else 1.0
    _max_intrinsic = max(_intrinsics) if _intrinsics else 1.0
    _epsilon_floor_game = next((g for g, e in zip(_games, _epsilons) if e <= 0.011), None)

    # 3. Code accordion
    code_accordion = mo.accordion({
        "🔍 View Original Log Format": mo.md(
            """
            Each checkpoint line in `dqn_per_icm_10x10.log` looked like:

            ```
            Game 2000 | Score: 1 | Record: 12 | Mean Score: 0.84 | Epsilon: 0.01 | Intrinsic: 0.003 | Eta: 0.01000
            ```

            Parsed with:

            ```python
            import re
            m = re.match(
                r"Game (\\d+).*?Mean Score:\\s*([\\d.]+).*?Epsilon:\\s*([\\d.]+).*?Intrinsic:\\s*([\\d.]+)",
                line
            )
            ```

            The 160 checkpoint values (every 100 games) are inlined above so the notebook
            runs without local files in molab / WASM environments.
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
        if (cur < N) {{
            cur += 1;
            const isEpsFloor = data.epsilon_floor_game &&
                Math.abs(data.games[cur] - data.epsilon_floor_game) < 150;
            animId = setTimeout(animate, isEpsFloor ? 600 : 20);
        }} else {{
            btn.textContent = '↺ Reset'; btn.disabled = false;
        }}
    }}

    btn.onclick = () => {{
        if (btn.textContent === '↺ Reset') {{
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

    mo.vstack([train_text, code_accordion, wasm_iframe(html_train, height="430px")])
    return


@app.cell(hide_code=True)
def _(mo):
    # The Transition and Literature Review Text
    bridge_text = mo.md(
        """
        ### Self-Supervised vs. Hand-Coded Rewards

        Before we look at where this architecture breaks down, it helps to understand what Pathak et al. mean by **"Self-Supervised Prediction."** The paper was not the first work on intrinsic motivation or self-supervised signals, but it gave a clear and influential recipe for turning prediction error into an exploration bonus for deep reinforcement learning.

        Historically, to solve sparse environments, engineers relied on **Hand-Coded Reward Shaping**. A human programmer would manually inject domain knowledge: *"If the agent moves closer to the goal, give it +0.1 points. If it moves away, -0.1 points."* While reward shaping works, it is brittle. It requires a human to hand-hold the agent through every new game, and it frequently leads to "reward hacking" (where the agent finds a loophole to farm points without actually winning the game).

        **Self-supervised** curiosity reduces the amount of human reward engineering. The agent creates a dense auxiliary reward from its own observations, learns a predictive model of environment dynamics, and uses prediction error to explore until it encounters task rewards.

        ---

        ### The Mechanics of Memory: Prioritized Experience Replay (PER)

        We now have a new problem. Curiosity helps the agent discover rare events, but rare events are easy to forget if the training loop samples old experience uniformly. In Snake, an apple capture might be only a tiny fraction of the replay buffer. If the agent rarely replays those transitions, it does not learn much from them.

        The paper anticipated this memory problem. In the final paragraphs, Pathak et al. write:

        > *”While the rich and diverse real world provides ample opportunities for interaction, reward signals are sparse. Our approach excels in this setting. However our approach does not directly extend to the scenarios where ‘opportunities for interactions’ are also rare. In theory, **one could save such events in a replay memory and use them to guide exploration**. However, we leave this extension for future work.”*
        >
        > — Pathak et al., *”Curiosity-driven Exploration by Self-supervised Prediction”*, ICML 2017. [arXiv:1705.05363](https://arxiv.org/abs/1705.05363)

        Snake on a 10×10 board is precisely this setting. Early in training, apples are rare, interactions are thin, and the agent needs a way to revisit its most informative experiences. **We implemented the paper’s suggested extension** using a Prioritized Experience Replay (PER) buffer that samples memories based on their **Temporal Difference (TD) Error** — how wrong the agent's Q-value estimate was for that memory.

        * In a standard buffer, memories are sampled purely at random.
        * In **PER**, high-error memories are sampled more frequently, forcing the agent to study its biggest mistakes.

        But what happens when we combine **self-supervised curiosity**, **PER**, and an environment with terminal failure states (like *Snake*)?
        """
    )

    bridge_text
    return


@app.cell(hide_code=True)
def _(mo):
    # 1. The Narrative Text
    act3_text = mo.md(
        """
        ### The Edge Case: The Death Oversampling Trap (PER + ICM)

        ICM is brilliant in sparse environments, but when deployed in environments with terminal failure states (like *Snake*), it introduces a fatal mathematical flaw. 

        When an agent dies, the environment resets abruptly. The Forward Model cannot predict a "Game Over" screen from a standard movement action. This results in a massive spike in Prediction Error, which translates to a massive Intrinsic Reward. **The agent learns that suicide is incredibly rewarding.**

        This becomes catastrophic if you are using **Prioritized Experience Replay (PER)**. PER samples memories from the replay buffer based on their error magnitude. Because "deaths" have the highest error, PER oversamples them. 

        **The Trap:** The buffer becomes poisoned. The agent's training batches become flooded with death sequences, completely drowning out the "normal" steps and successful apple captures. The agent unlearns how to play and optimizes for fast deaths.

        Ablation intuition:

        | Variant | Expected behavior |
        |---|---|
        | PER only | Replays high-TD-error events, but death transitions are not additionally inflated by curiosity. |
        | ICM only | Death may be intrinsically surprising, but uniform replay limits how often that transition dominates training. |
        | PER + ICM | Death transitions get both high TD error and high intrinsic error, so replay probability compounds. |
        | PER + ICM + terminal mask | Keeps curiosity for non-terminal novelty while preventing terminal resets from becoming a reward source. |

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


@app.cell(hide_code=True)
def _(act3_text, agent_type, mo, np, plt, sample_btn):
    mo.stop(
        np is None or plt is None,
        mo.callout(
            mo.md("**NumPy/Matplotlib are unavailable in this runtime.** In WASM they are installed from the notebook metadata."),
            kind="warn",
        ),
    )

    # 3. The Visualization Logic
    def render_batch_viz(agent_val, btn_click):
        # Base Replay Buffer Composition (100 memories total)
        # 80 Normal Steps, 15 Apples, 5 Deaths
        types = ['Normal'] * 80 + ['Apple'] * 15 + ['Death'] * 5
        colors = ['#B0BEC5'] * 80 + ['#4CAF50'] * 15 + ['#F44336'] * 5

        # TD error ordering approximates empirical training: normal < apple < death.
        # Exact values are ordinal stand-ins; the qualitative point holds for any
        # ordering where deaths > apples > normal steps.
        base_td = np.array([0.1] * 80 + [0.4] * 15 + [0.8] * 5)

        if agent_val == "Curious Agent (PER + ICM)":
            # ICM adds a forward-model MSE spike at terminal transitions.
            # In the training logs, per-step intrinsic reward at game-over frames
            # ran ~10–50× normal-step values before η scaling (η=0.01).
            # The 4.0 approximates raw latent-space MSE on terminal transitions.
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


@app.cell(hide_code=True)
def _(json, mo, wasm_iframe):
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
        10×10 board for 16,000 games. Watch the red curve plateau while the green one climbs.
        """
    )

    # 2. Inlined score curves (parsed from training logs; inlined for molab/WASM compatibility)
    _g_poi = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900, 11000, 11100, 11200, 11300, 11400, 11500, 11600, 11700, 11800, 11900, 12000, 12100, 12200, 12300, 12400, 12500, 12600, 12700, 12800, 12900, 13000, 13100, 13200, 13300, 13400, 13500, 13600, 13700, 13800, 13900, 14000, 14100, 14200, 14300, 14400, 14500, 14600, 14700, 14800, 14900, 15000, 15100, 15200, 15300, 15400, 15500, 15600, 15700, 15800, 15900, 16000]
    _s_poi = [0.11, 0.13, 0.13, 0.14, 0.15, 0.16, 0.19, 0.23, 0.27, 0.3, 0.34, 0.37, 0.42, 0.46, 0.49, 0.54, 0.6, 0.66, 0.74, 0.84, 0.96, 1.07, 1.23, 1.37, 1.49, 1.61, 1.71, 1.79, 1.88, 1.95, 2.05, 2.17, 2.27, 2.4, 2.53, 2.65, 2.76, 2.88, 2.95, 3.03, 3.12, 3.19, 3.26, 3.38, 3.45, 3.53, 3.61, 3.68, 3.77, 3.85, 3.91, 4.05, 4.15, 4.21, 4.3, 4.39, 4.43, 4.48, 4.6, 4.72, 4.84, 4.88, 4.97, 5.04, 5.13, 5.18, 5.25, 5.3, 5.35, 5.4, 5.47, 5.54, 5.59, 5.63, 5.67, 5.71, 5.75, 5.8, 5.84, 5.92, 5.96, 6.0, 6.04, 6.1, 6.15, 6.19, 6.21, 6.25, 6.3, 6.34, 6.38, 6.42, 6.44, 6.48, 6.51, 6.54, 6.6, 6.64, 6.66, 6.67, 6.67, 6.7, 6.73, 6.77, 6.81, 6.85, 6.86, 6.89, 6.92, 6.95, 6.99, 7.04, 7.08, 7.12, 7.15, 7.18, 7.19, 7.23, 7.25, 7.29, 7.31, 7.33, 7.36, 7.37, 7.39, 7.4, 7.42, 7.45, 7.48, 7.49, 7.51, 7.53, 7.57, 7.6, 7.63, 7.65, 7.66, 7.68, 7.68, 7.69, 7.72, 7.74, 7.76, 7.79, 7.8, 7.84, 7.86, 7.89, 7.9, 7.92, 7.94, 7.93, 7.96, 7.98, 7.99, 8.01, 8.03, 8.03, 8.05, 8.06]
    _g_fix = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900, 11000, 11100, 11200, 11300, 11400, 11500, 11600, 11700, 11800, 11900, 12000, 12100, 12200, 12300, 12400, 12500, 12600, 12700, 12800, 12900, 13000, 13100, 13200, 13300, 13400, 13500, 13600, 13700, 13800, 13900, 14000, 14100, 14200, 14300, 14400, 14500, 14600, 14700, 14800, 14900, 15000, 15100, 15200, 15300, 15400, 15500, 15600, 15700, 15800, 15900, 16000]
    _s_fix = [0.12, 0.13, 0.14, 0.18, 0.19, 0.2, 0.23, 0.26, 0.29, 0.35, 0.4, 0.43, 0.46, 0.52, 0.59, 0.71, 0.82, 0.99, 1.17, 1.47, 1.83, 2.24, 2.63, 2.87, 3.15, 3.44, 3.73, 3.92, 4.15, 4.36, 4.57, 4.81, 4.94, 5.1, 5.27, 5.43, 5.57, 5.74, 5.86, 5.94, 6.07, 6.19, 6.34, 6.46, 6.55, 6.65, 6.73, 6.85, 6.94, 7.01, 7.11, 7.19, 7.27, 7.35, 7.44, 7.5, 7.56, 7.58, 7.65, 7.71, 7.81, 7.84, 7.9, 7.95, 8.03, 8.07, 8.12, 8.18, 8.23, 8.25, 8.29, 8.32, 8.37, 8.4, 8.44, 8.47, 8.5, 8.55, 8.56, 8.62, 8.67, 8.72, 8.75, 8.79, 8.81, 8.83, 8.85, 8.88, 8.91, 8.95, 8.96, 8.97, 9.0, 9.04, 9.06, 9.1, 9.12, 9.16, 9.18, 9.21, 9.23, 9.25, 9.27, 9.29, 9.33, 9.33, 9.36, 9.38, 9.41, 9.43, 9.44, 9.44, 9.44, 9.47, 9.48, 9.49, 9.5, 9.49, 9.52, 9.54, 9.56, 9.57, 9.58, 9.6, 9.61, 9.62, 9.65, 9.65, 9.66, 9.66, 9.68, 9.7, 9.69, 9.71, 9.72, 9.72, 9.73, 9.75, 9.78, 9.8, 9.81, 9.83, 9.83, 9.85, 9.85, 9.88, 9.88, 9.89, 9.91, 9.93, 9.94, 9.94, 9.96, 9.99, 10.02, 10.03, 10.03, 10.05, 10.06, 10.06]

    _max_score = max(max(_s_poi), max(_s_fix))
    _fix_json = json.dumps({
        "games_poi": _g_poi, "scores_poi": [s / _max_score for s in _s_poi],
        "games_fix": _g_fix, "scores_fix": [s / _max_score for s in _s_fix],
        "max_score": _max_score,
        "poi_final": _s_poi[-1], "fix_final": _s_fix[-1],
        "last_game": _g_poi[-1],
    })

    # 3. Animated two-phase chart
    html_fix = f"""<!DOCTYPE html>
    <html>
    <head>
    <style>
      body {{ margin:0; font-family:sans-serif; background:white; padding:10px; display:flex; flex-direction:column; align-items:center; }}
      .panel {{ border:1px solid #ccc; background:#f9f9f9; border-radius:6px; padding:15px; width:640px; box-shadow:0 2px 5px rgba(0,0,0,0.05); }}
      .header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }}
      .title {{ font-weight:bold; font-size:14px; color:#333; }}
      button {{ padding:8px 24px; font-size:14px; cursor:pointer; border-radius:4px; border:none; background:#6366f1; color:white; font-weight:bold; transition:background 0.2s; }}
      button:hover {{ background:#4f46e5; }}
      button:disabled {{ background:#a5b4fc; cursor:not-allowed; }}
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
        <div class="title">Terminal Reward Masking: r_i × (1 − done) — 10×10 Snake, 16,000 games</div>
        <button id="fixBtn">▶ Animate</button>
      </div>
      <canvas id="fixChart" width="640" height="260"></canvas>
      <div class="legend">
        <div class="leg-item"><div class="dot" style="background:#ef4444;"></div>PER + ICM — unmasked (poisoned)</div>
        <div class="leg-item"><div class="dot" style="background:#10b981;"></div>PER + ICM + terminal mask — fixed</div>
      </div>
      <div class="note">Both curves trained identically — the only difference is one line of code.</div>
    </div>
    <script>
    const canvas = document.getElementById('fixChart');
    const ctx = canvas.getContext('2d');
    const btn = document.getElementById('fixBtn');
    const W = 640, H = 260;
    const PL = 52, PR = 20, PT = 24, PB = 38;
    const PW = W - PL - PR, PH = H - PT - PB;
    const data = {_fix_json};
    const N = data.scores_poi.length;
    let phase = 0, cur = 0, animId = null;

    function drawAxes() {{
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#f9fafb'; ctx.fillRect(PL, PT, PW, PH);
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
        ctx.save();
        ctx.translate(12, PT + PH / 2); ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center'; ctx.font = '10px sans-serif'; ctx.fillStyle = '#9ca3af';
        ctx.fillText('Mean Score', 0, 0); ctx.restore();
        ctx.textAlign = 'center';
        for (let i = 0; i <= 4; i++) {{
            const g = Math.round((i / 4) * data.last_game);
            ctx.fillText('G' + g.toLocaleString(), PL + (i / 4) * PW, PT + PH + 14);
        }}
    }}

    function drawLine(scores, color, maxIdx) {{
        if (maxIdx < 2) return;
        ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2.5;
        for (let i = 0; i < maxIdx; i++) {{
            const x = PL + (i / (N - 1)) * PW;
            const y = PT + PH - (scores[i] * PH);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }}
        ctx.stroke();
    }}

    function drawAnnotation(scores, color, label, side) {{
        const x = PL + PW;
        const y = PT + PH - (scores[N - 1] * PH);
        ctx.fillStyle = color; ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = side === 'left' ? 'right' : 'right';
        ctx.fillText(label, x - 6, y + (side === 'top' ? -6 : 14));
    }}

    function render(poiIdx, fixIdx) {{
        drawAxes();
        drawLine(data.scores_poi, '#ef4444', poiIdx);
        if (poiIdx >= N) drawAnnotation(data.scores_poi, '#ef4444', 'plateau ~' + data.poi_final.toFixed(1), 'bottom');
        drawLine(data.scores_fix, '#10b981', fixIdx);
        if (fixIdx >= N) drawAnnotation(data.scores_fix, '#10b981', '~' + data.fix_final.toFixed(1), 'top');
        // Pulsing gap annotation when green surpasses red plateau
        if (fixIdx > 20 && data.scores_fix[fixIdx - 1] > data.scores_poi[N - 1]) {{
            ctx.save();
            ctx.font = 'bold 13px sans-serif'; ctx.fillStyle = '#10b981';
            ctx.textAlign = 'center';
            ctx.fillText('+' + ((data.fix_final - data.poi_final) / data.poi_final * 100).toFixed(0) + '%', PL + PW * 0.75, PT + 18);
            ctx.restore();
        }}
    }}

    function phase1() {{
        render(cur, 0);
        if (cur < N) {{ cur += 1; animId = setTimeout(phase1, 20); }}
        else {{
            setTimeout(() => {{ phase = 2; cur = 0; phase2(); }}, 700);
        }}
    }}

    function phase2() {{
        render(N, cur);
        if (cur < N) {{ cur += 1; animId = setTimeout(phase2, 20); }}
        else {{
            btn.textContent = '↺ Reset'; btn.disabled = false;
        }}
    }}

    btn.onclick = () => {{
        if (btn.textContent === '↺ Reset') {{
            phase = 0; cur = 0; render(0, 0); btn.textContent = '▶ Animate';
        }} else {{
            btn.disabled = true; btn.textContent = 'Animating...';
            phase = 1; cur = 0; phase1();
        }}
    }};

    drawAxes();
    </script>
    </body>
    </html>"""

    source_accordion = mo.accordion({
        "📂 Source code & training scripts": mo.md("""
    **GitHub:** [github.com/Saheb/rl-snake](https://github.com/Saheb/rl-snake)

    | File | Purpose |
    |------|---------|
    | [`scripts/train_dqn.py`](https://github.com/Saheb/rl-snake/blob/main/scripts/train_dqn.py) | DQN + PER + ICM training loop — terminal mask fix at line 547 |
    | [`utils/icm.py`](https://github.com/Saheb/rl-snake/blob/main/utils/icm.py) | ICM module (encoder, inverse model, forward model) |
        """)
    })

    mo.vstack([fix_text, wasm_iframe(html_fix, height="430px"), source_accordion])
    return


@app.cell(hide_code=True)
def _(mo):
    _conclusion = mo.md(
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

        ICM opened the door. Every paper below is a direct response to one of its limitations.
        """
    )

    _lit_tabs = mo.ui.tabs({
        "ICM (2017)": mo.md("""
    **Pathak et al., ICML 2017** — [arXiv:1705.05363](https://arxiv.org/abs/1705.05363)

    **Problem solved:** Dense exploration in sparse-reward environments without hand-coded reward shaping.

    **Key result:** Self-supervised agents matched or exceeded hand-shaped baselines on VizDoom and Super Mario Bros — with zero domain knowledge injected.

    **Relation to this notebook:** The foundation. Every section above builds on or stress-tests this paper's core mechanism.
        """),
        "Large-Scale Curiosity (2018)": mo.md("""
    **Burda, Edwards, Pathak et al., ICLR 2019** — [arXiv:1808.04355](https://arxiv.org/abs/1808.04355)

    **Problem solved:** Does intrinsic curiosity *alone* (zero extrinsic reward) produce meaningful learning?

    **Key result:** Yes — agents trained purely on curiosity learned 54 Atari environments and navigated 3D mazes, with no game score signal whatsoever.

    **Relation:** Validates ICM's hypothesis at scale. Also found the Noisy TV problem is endemic, motivating the next entry.
        """),
        "RND (2018)": mo.md("""
    **Burda et al., ICLR 2019** — [arXiv:1810.12894](https://arxiv.org/abs/1810.12894)

    **Problem solved:** The Noisy TV problem — unpredictable environment stochasticity corrupts ICM's forward model prediction.

    **Key result:** State-of-the-art on Montezuma's Revenge by replacing forward prediction with *random network distillation* — a frozen random network whose output a second network learns to predict. Novelty = gap between the two. Stochasticity cannot corrupt a random target.

    **Relation:** RND discards ICM's forward model entirely. No physics prediction, no noise vulnerability.
        """),
        "Episodic Curiosity (2018)": mo.md("""
    **Savinov et al., ICLR 2019** — [arXiv:1810.02274](https://arxiv.org/abs/1810.02274)

    **Problem solved:** Within-episode reward collapse — an agent that finds one novel state and farms it forever instead of exploring further.

    **Key result:** Outperformed ICM on long-horizon navigation tasks by gating rewards through *episodic reachability*: only states that are hard to reach from anything seen this episode earn reward.

    **Relation:** ICM's reward decays only as the forward model learns. Episodic Curiosity forces continued movement regardless of model quality.
        """),
        "DreamerV3 (2023)": mo.md("""
    **Hafner et al., DeepMind 2023** — [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)

    **Problem solved:** Sample efficiency and generalization across radically different environments without any task-specific hyperparameter tuning.

    **Key result:** Human-level performance on 150+ tasks including Minecraft diamond collection — with a single fixed hyperparameter set.

    **Relation:** ICM's forward model predicts one step ahead in latent space. Dreamer learns a complete *world model* — dynamics, rewards, and terminations — and trains its policy entirely inside imagined rollouts. The lineage from ICM's single forward model to Dreamer's full generative world model is direct and unbroken.
        """),
        "Agent57 (2020)": mo.md("""
    **Badia et al., DeepMind 2020** — [arXiv:2003.13350](https://arxiv.org/abs/2003.13350)

    **Problem solved:** Achieving above-human performance on *all* 57 classic Atari games simultaneously — including the hard-exploration games that had resisted every prior method.

    **Key result:** First agent to surpass human baselines on the complete Atari-57 suite. Used a meta-controller to balance intrinsic vs. extrinsic reward, combining episodic novelty (short-term) with a lifelong novelty metric (long-term).

    **Relation:** Agent57 is the convergence point of the ICM lineage. It synthesizes episodic curiosity, RND-style novelty, and adaptive exploration weighting into a single agent that never stops exploring but knows when to exploit.
        """),
    })

    mo.vstack([
        _conclusion,
        _lit_tabs,
        mo.md("*Built with Python, PyTorch, and Marimo. Trained on a 10×10 Snake board.*")
    ])
    return


@app.cell(hide_code=True)
def _():
    import base64
    import marimo as mo
    import json
    import random
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    try:
        import numpy as np
    except ImportError:
        np = None
    def wasm_iframe(html: str, *, width: str = "100%", height: str = "400px"):
        encoded = base64.b64encode(html.encode("utf-8")).decode("ascii")
        return mo.Html(
            f'<iframe src="data:text/html;charset=utf-8;base64,{encoded}" '
            f'width="{width}" height="{height}" '
            'style="border:0; width:100%;" loading="lazy"></iframe>'
        )

    return json, mo, np, plt, random, wasm_iframe


if __name__ == "__main__":
    app.run()
