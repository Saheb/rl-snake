# Scaling Snake AI: From Random Wiggles to Strategic Mastery

**The story of how we trained a Reinforcement Learning agent to master Snake on a 10x10 board.**

## 🐣 A Quick Primer: What is RL?
If you're new to AI, Reinforcement Learning (RL) is simply **learning by trial and error**. An agent (our snake) takes actions and gets rewards (food) or punishments (hitting a wall).

There are two ways to build an AI "brain":
1. **Value-Based (The Accountant)**: The AI tries to calculate the exact worth of every move. "Is turning left worth 10 points or 2?" 
   - Known as **DQN** (**Deep Q-Network**). 'Q' stands for 'Quality'—the AI asks, "What's the quality of this move?"
2. **Policy-Based (The Athlete)**: The AI learns general instincts. "If the ball is coming, catch it." It doesn't calculate value; it just knows the right response. 
   - Known as **PPO** (**Proximal Policy Optimization**). 'Proximal' means it makes small, safe updates to its strategy so it doesn't "crash" while learning.

---

## 📉 Phase 0: The Baseline (5x5 Board)
We started with **Classic Tabular Q-Learning**.
- **Tabular Q**: A simple table memorizing the board.
- **Double Q**: A smarter version that double-checks its own math to avoid over-confidence.
**Result**: Perfect on 5x5, but hit a wall as soon as the grid grew.

## 🚧 The Wall: Scaling Challenges
On a 10x10 board, the "Accountant" approach fails because there are too many combinations to count. Worse, rewards are **sparse**. A random snake might wander for 1000 steps without seeing a single piece of food.

## 💡 The Solution: A Triple Threat
To master the 10x10 board, we combined three advanced techniques:

### 1. Imitation Learning (The Instinct)
We used an expert agent to generate "demos." We then taught our main agent to mimic these demos. This gave it an "instinct" for survival immediately.

### 2. PPO (Proximal Policy Optimization)
We switched to **PPO**. Because it learns high-level strategies ("move toward food") rather than specific board values, it generalizes much better to different board sizes.

### 3. Curriculum Learning (The Growth)
We didn't start at 10x10. We mastered **5x5**, then transferred that brain to **8x8**, and finally to **10x10**.

## 🏆 The Result
Our final agent is a strategist:
- **Peak Score**: **62+** (filling over 60% of a large board!).
- **Efficiency**: Plans long-term paths to avoid trapping itself.

---
### 🎬 Proof of evolution
Check out **`snake_learning_journey.html`** to see the interactive transition from Stage 0 (Classic Tabular) to Stage 8 (Master PPO).

*Created by Antigravity AI*
