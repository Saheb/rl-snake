# Teaching a Snake to Explore: From Dead Ends to Curiosity-Driven Mastery

**How a reinforcement learning agent learned to navigate a 10×10 grid — and the surprisingly deep engineering required to make it work.**

---

## 🐣 The Setup: What is RL, Really?

Reinforcement Learning is learning by trial and error. An agent takes actions in an environment, receives rewards or punishments, and slowly figures out what works.

```
   Agent ──action──▶ Environment
     ▲                    │
     └──state, reward◀────┘
```

There are two fundamentally different families of RL algorithms:

**Value-Based — "The Accountant"**
The agent calculates the exact worth of every possible move. *"Is turning left worth 10 points or 2?"* The algorithm we use here is **DQN (Deep Q-Network)** — Q stands for "Quality," as in the quality of a move. It picks the move with the highest Q-value, every single time — no gambling.

**Policy-Gradient — "The Athlete"**
The agent learns general instincts and outputs a *probability distribution* over moves. *"There's a 90% chance I should go left."* The famous algorithm here is **PPO (Proximal Policy Optimization)**. The key word "proximal" means it makes small, careful policy updates to avoid catastrophically unlearning what it already knows.

Both approaches have their place. Our journey tested both — and the results were not what we expected.

---

## 📉 Phase 0: Starting Simple (5×5 Board)

We started with **Tabular Q-Learning** — the simplest possible "Accountant" approach. The agent keeps a literal lookup table: one row for every board configuration, one column for each action.

| Agent | Mean Score | Max Score | Table Size |
|---|---|---|---|
| **Q-Learning** | ~11.0 | 24 (Perfect!) | ~2,000 rows |
| **Double Q-Learning** | ~14.0 | 24 (More stable) | ~2,000 rows |

On a 5×5 board with a maximum possible score of 24, both agents perform impressively. The Q-table only needs ~2,000 entries to map the entire game.

*Double Q-Learning* is a clever improvement: instead of one table, it cross-checks two. This avoids the tendency to be overconfident about untested moves — a bias called "maximization bias."

Problem solved? Not quite.

---

## 🚧 The Wall: Why Tables Cannot Scale

The moment we tried a **10×10 board**, tabular methods collapsed completely. The reason is mathematical.

**State Space Explosion**

A 5×5 board has a manageable number of possible configurations. A 10×10 board with 100 tiles has roughly $2^{100}$ possible unique states — more than the number of atoms in the observable universe. There is literally not enough RAM on Earth to build that table.

**The Fix: Deep Q-Networks**

We replaced the lookup table with a **neural network**. Instead of memorizing every board configuration, the neural network *generalizes* — it learns patterns. If it learns that "walls are dangerous" in the top-left corner, it automatically applies that knowledge to the bottom-right corner without being explicitly taught. This is the "Deep" in DQN.

With DQN, the state explosion problem vanished. Our agent scaled cleanly to 8×8 and 10×10:

| Board | Algorithm | Mean Score | Best Score | Notes |
|---|---|---|---|---|
| 5×5 | Tabular Q-Learning | ~11.0 | 24 ✅ | Perfect score achievable |
| 5×5 | **DQN Baseline** | **6.94** | **24** | Neural net generalizes board state |
| 8×8 | **DQN Baseline** | **9.25** | **46** | Trained from scratch, no lookup table |
| 10×10 | Tabular Q-Learning | ❌ | ❌ | RAM overflow — unusable |
| 10×10 | **DQN Baseline** | **10.54** | **54** | Scales cleanly despite 100 tiles |

The 10×10 column tells the whole story: the method that completely breaks the tabular approach handles the larger board *better* (higher mean score) by learning generalizable patterns rather than memorizing states one by one.

We needed to solve exploration. Our first instinct was to try a completely different class of algorithm.

---

## 🎭 Phase 1: The PPO Experiments — Imitation Learning & Curriculum Learning

Before discovering the ICM, we spent significant time trying to fix the sparse reward problem using two classical techniques. These approaches worked — and taught us some of the project's most surprising lessons.

### Attempt 1: Switching to PPO

We hypothesized that maybe the problem wasn't the environment — it was the algorithm. PPO (Proximal Policy Optimization) is a Policy-Gradient method. Instead of learning Q-values for every action, it directly learns a *probability distribution* over actions. We thought its on-policy nature might handle exploration better.

It didn't. On a 10×10 board without any guidance, PPO achieved a mean score near zero. The exploration problem is not specific to DQN — it is a property of the environment.

### Attempt 2: Imitation Learning (The "Classroom")

Rather than letting the agent learn from scratch, what if it watched an expert first?

**Behavioral Cloning** is exactly this: we let an expert agent play thousands of games, record every `(state, action)` pair, and use supervised learning to train PPO to imitate those moves. The PPO Actor's Cross-Entropy Loss is minimized against the expert's choices until the clone accuracy reaches ~90%. Only then do we drop PPO into the real environment.

We implemented this in two variants:

**`train_ppo_with_demos.py`** — The expert was **REINFORCE**, a simple policy-gradient algorithm we trained first on the 5×5 board. We then pre-trained PPO's Critic network on the actual discounted returns from REINFORCE's best trajectories. This solved the "bootstrap trap" — the Critic starting completely blind — and gave PPO survival instincts from Day 1.

**`train_ppo_from_dqn.py`** — We used the trained **DQN** as the expert instead. DQN demonstrated excellent episode quality, and the behavioral cloning achieved **96% accuracy** at reproducing DQN's moves.

> **Surprising finding:** The DQN-expert gave 96% cloning accuracy. The REINFORCE-expert gave 82%. But PPO performed *worse* with the DQN expert.
>
> Why? REINFORCE and PPO are both policy-gradient methods — they share the same mathematical "language." DQN is value-based and makes rigid argmax decisions with a fundamentally different internal structure. When PPO tried to clone DQN's deterministic reasoning with a probabilistic policy head, the mismatch caused instability. **Algorithm compatibility matters more than imitation accuracy.**

### Attempt 3: Curriculum Learning (Training Wheels)

Instead of forcing the agent to tackle 10×10 immediately, we broke the difficulty into stages:

```
5×5 (Easy)  →  8×8 (Medium)  →  10×10 (Goal)
    ↑               ↑                ↑
 Imitation       Transfer         Transfer
  Learning       + Adapt          + Master
```

`train_ppo_curriculum.py` implements this pipeline: REINFORCE trains on 5×5, its best episodes are used to pre-train a PPO agent, and the trained weights are then directly transferred to the 8×8 environment for fine-tuning. The neural network's weights carry learned skills across board sizes.

This worked — the curriculum-trained PPO significantly outperformed naive PPO on 8×8. But it had two fundamental weaknesses:

1. **It required a hand-crafted expert at each stage.** The agent never truly learned to explore — it was bootstrapped by human-designed scaffolding the entire time.
2. **It still plateaued.** Even with curriculum training and imitation learning, PPO's mean score on 10×10 remained low. The replay buffer gap with DQN was still decisive.

These experiments taught us that the right question wasn't *"How do we bootstrap exploration?"* — it was *"How do we make exploration intrinsically rewarding?"*

That insight led directly to the ICM.

---

## 🔍 The Sparse Reward Problem (Revisited with DQN)

We circled back to DQN — which, even without the scaffolding, outperformed curriculum PPO on raw mean score — and focused on solving its exploration problem directly.

On a **10×10 board** (100 tiles), the apple occupies just 1% of the available space. The snake can wander for hundreds of steps without finding anything. No rewards → no learning signal → no improvement.

We measured this directly. After 16,000 games, our baseline DQN on 10×10 achieved a mean score of barely 3.

We needed to solve exploration without training wheels.

---



## 💡 The Breakthrough: Intrinsic Curiosity Module (ICM)

What if we gave the agent a reward simply for *exploring*? Not for finding food — just for seeing something it had never seen before?

This is the core idea behind the **Intrinsic Curiosity Module**, introduced by Pathak et al. (2017).

The ICM trains two small neural networks alongside the main DQN:

**The Forward Model** predicts: *"If I'm in state $s_t$ and take action $a_t$, what will the next state $s_{t+1}$ look like?"*

**The Inverse Model** predicts: *"If I moved from $s_t$ to $s_{t+1}$, what action did I take?"*

The **intrinsic reward** is the Forward Model's prediction error — its surprise. If the agent visits a new part of the board and the Forward Model fails to predict what happened next, it generates a large intrinsic reward. If the agent revisits familiar territory, the prediction error is small and the reward is minimal.

The key insight: **the agent gets paid for being surprised.** This transforms a sparse-reward maze into a rich landscape of curiosity, with the agent actively motivated to explore every corner of the board.

We scaled $\eta$ (the curiosity intensity weight) and found that even a tiny `eta=0.01` was enough to dramatically improve exploration on the 10×10 grid.

---

## ⚔️ The Empirical Battle: DQN vs PPO

We ran both algorithms under the same conditions — with and without ICM — across 5×5, 8×8, and 10×10 boards, tracking everything in **Weights & Biases**. The results were decisive.

| Run | Board | Mean Score | Record |
|---|---|---|---|
| PPO Baseline | 10×10 | ~0.2 | 3 |
| PPO + ICM | 5×5 | ~2.1 | — |
| DQN Baseline | 10×10 | ~3.0 | 36+ |
| DQN + ICM | 10×10 | ~8+ | 53+ |

PPO fundamentally struggles here for two structural reasons:

**The Fatal Environment Problem**

PPO's Actor head outputs *probabilities*. Even if it learns that going left has a 95% chance of being correct, it still has a 5% chance of randomly picking a fatal direction. In Snake — where a single wall collision ends the game — this stochastic slippage causes the agent to "slip" and die even after it has ostensibly "learned" the rules.

DQN uses a rigid `argmax`. It picks the action with the single highest Q-value, every time, with zero randomness (outside of explicit epsilon-greedy exploration). Once it learns a rule, it follows it 100% of the time.

**The Replay Buffer Supremacy**

DQN stores every experience in a large Replay Buffer. When it accidentally discovers a brilliant 10-step path to an apple, it saves that memory and replays it thousands of times during future training batches — squeezing every bit of gradient out of each discovery.

PPO is On-Policy. It uses each batch of experiences exactly once for a gradient update, then permanently discards them. The replay buffer is not just an optimization — it is the fundamental reason DQN can learn from rare events in sparse environments.

**The conclusion was clear: for a game with hard failure states and sparse rewards, the offline Value-Based DQN architecture significantly outperforms the online Policy-Gradient PPO architecture.**

---

## 🔥 The Hidden Problem: Double Catastrophic Forgetting

Even with ICM boosting DQN's performance, we observed a bizarre pathology in the training curves: the agent would reach a record score of 50+, and then immediately start dying on Turn 1 for hundreds of games in a row.

We called this **Double Catastrophic Forgetting**.

Here's what was happening:

**Stage 1 — The Buffer Flood:** When the snake achieves a high score (say, 50 on a 10×10 board), it generates thousands of late-game transitions — all involving a long snake navigating a nearly-full board. These transitions flood the 100,000-entry Replay Buffer, pushing out the early-game transitions (short snake on an empty board).

**Stage 2 — The Q-Network Forgets:** The DQN stops practicing early-game states. It starts making bad decisions on the first 5 moves of a new game, instantly hitting walls.

**Stage 3 — The ICM Forgets (The Fatal Blow):** The Forward Model *also* stops practicing early-game physics. It becomes an expert at predicting how a 50-length snake moves through a crowded board, but completely forgets what an empty board looks like. When the game resets and the snake is length 1 on a wide-open board, the Forward Model fails to predict any of the transitions. Its prediction error is massive. It generates an enormous intrinsic reward for... dying on Turn 1. The agent learns that suiciding is surprisingly profitable.

---

## 🏗️ The Engineering Fix: Dual-Buffer PER Hybrid

We designed a three-part solution to break the forgetting cycle.

### 1. Foundation Buffer (Structural Fix)
We maintained two separate replay buffers instead of one:

- **Foundation Buffer** (`maxlen=20,000`): Stores only the first 50 steps of every game. By definition, only early-game experiences with a short snake on a relatively empty board. This buffer is *never evicted by late-game states*.
- **Deep Game PER Buffer** (`capacity=100,000`): Stores all transitions after step 50, optimized with priority sampling.

During every training batch, we enforced a **75/25 split**: 750 samples from the PER buffer, 250 from the Foundation Buffer. This mathematically guarantees the network and the ICM will always practice both early-game *and* late-game physics in every single update.

### 2. Prioritized Experience Replay (Sampling Fix)
We replaced the standard uniform sampling in the Deep Game buffer with a **SumTree-based PER**. 

Every time the network makes a prediction error, we calculate the Temporal Difference (TD) error and use it as the sampling priority. States where the network's predictions were most wrong get sampled most often in future batches — the network is constantly drilling its hardest failures.

We implemented a **priority cap** of `1.0` to prevent death states (which generate extremely high TD errors) from monopolizing the training batch. Without this cap, the network would obsessively practice dying, dragging all Q-values negative.

### 3. Terminal Intrinsic Reward Masking (ICM Fix)
We zeroed out the ICM's intrinsic reward on the terminal step of every game:

```python
intrinsic_reward = intrinsic_reward * (1.0 - float(done))
```

The game-over transition is physically discontinuous — the Forward Model *cannot* learn to predict it from normal movement physics. Without this mask, the massive prediction error on a death step generates a large curiosity reward, and the agent accidentally learns that dying is "surprising and therefore exciting."

---

## 📊 Tracking It All: Weights & Biases

Every experiment in this project was tracked in real-time using **Weights & Biases**, logging per-game metrics including `Mean_Score`, `Record`, `Intrinsic_Reward`, and `Eta` (curiosity decay rate).

This allowed us to observe the pathologies as they appeared — the signature "Score: 0" spike after a high-score game, the intrinsic reward stabilizing after the terminal mask fix, and the steady mean-score climb after applying the PER priority cap.

---

## 🔑 Key Takeaways

1. **Tables don't scale.** Neural networks are essential once you leave small, enumerable state spaces.
2. **DQN beats PPO in fatal, sparse-reward environments.** The Replay Buffer's ability to learn from rare successes is decisive. PPO's stochastic policy fatally slips in hard-boundary games.
3. **Intrinsic Curiosity is a genuine breakthrough.** It converts a sparse grid into a dense exploration landscape without any manual reward engineering.
4. **Forgetting is double with ICM.** Both the Q-Network *and* the Forward Model can forget the early game. You must architect your buffer to prevent this.
5. **Every fix has a failure mode.** PER prioritizes important experiences — but without a priority cap, it prioritizes dying. The architecture is a system of counterweights.

---

## 🔭 What's Next

- **Convolutional Neural Networks:** Replace the 14-number state vector with the full 10×10 board matrix, allowing the agent to "see" traps forming across the entire grid before walking into them.
- **Behavioral Cloning:** Pre-train PPO on expert DQN gameplay to give it survival instincts before it ever touches the real environment — bypassing the sparse reward problem entirely.

---

*Built with PyTorch, Weights & Biases, and a lot of dead snakes. 🐍*
