"""
================================================================================
CURRICULUM LEARNING: 5x5 → 8x8 → 10x10 (WITH INTERMEDIATE DEMOS)
================================================================================

Strategy:
1. Master 5x5 (IL + PPO)
2. Transfer to 8x8 (PPO)
3. **Collect expert demos from trained 8x8 agent**
4. Pretrain 10x10 agent on 8x8 demos
5. Train on 10x10

The key insight: We need fresh demos at each major scale jump!
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import collections
from collections import deque
from enum import Enum

# Import everything from the original curriculum script
from train_ppo_curriculum import (
    SnakeGame, GameState, ActorCritic, PPOAgent, 
    ExpertCollector, PolicyNetwork,
    pretrain_critic, pretrain_actor, train_phase, get_absolute_action
)

def collect_demos_from_ppo(agent, board_size, num_episodes=100, min_score=10, gamma=0.99):
    """Collect expert demonstrations from a trained PPO agent."""
    print(f"\n{'='*60}")
    print(f"🎓 Collecting Demos from PPO Agent on {board_size}x{board_size}")
    print(f"{'='*60}")
    
    game = SnakeGame(board_size=board_size)
    expert_data = []
    collected = 0
    attempts = 0
    
    while collected < num_episodes and attempts < num_episodes * 10:
        attempts += 1
        trajectory = []
        rewards = []
        
        game.reset()
        done = False
        steps = 0
        max_steps = board_size * board_size * 10
        
        while not done and steps < max_steps:
            state = agent.get_state(game)
            action_relative, action_idx, _, _ = agent.select_action(state)
            trajectory.append((state.copy(), action_idx))
            
            action = get_absolute_action(action_relative, game)
            _, reward, done, info = game.step(action)
            rewards.append(reward)
            steps += 1
        
        if info['score'] >= min_score:
            collected += 1
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            for (state, action), ret in zip(trajectory, returns):
                expert_data.append({'state': state, 'action': action, 'return': ret})
            
            if collected % 10 == 0:
                print(f"  Collected {collected}/{num_episodes} episodes (last score: {info['score']})")
    
    print(f"✅ Collected {len(expert_data)} pairs from {collected} episodes")
    return expert_data


def main():
    print("\n" + "=" * 70)
    print("🎓 CURRICULUM LEARNING: 5x5 → 8x8 → 10x10 (WITH DEMOS)")
    print("=" * 70)
    
    # ========================================
    # STAGE 1: Master 5x5
    # ========================================
    print("\n" + "=" * 60)
    print("📚 STAGE 1: MASTER 5x5 BOARD")
    print("=" * 60)
    
    collector = ExpertCollector(board_size=5)
    expert_data_5x5, _ = collector.collect_with_trained_reinforce(num_episodes=50, min_score=3)
    
    network = ActorCritic(14, 256, 3)
    pretrain_critic(network, expert_data_5x5, epochs=50)
    pretrain_actor(network, expert_data_5x5, epochs=50)
    
    agent = PPOAgent(network=network, lr=0.0003)
    train_phase(agent, board_size=5, num_games=2000, max_steps=500)
    
    # ========================================
    # STAGE 2: Transfer to 8x8
    # ========================================
    print("\n" + "=" * 60)
    print("📚 STAGE 2: TRANSFER TO 8x8 BOARD")
    print("=" * 60)
    
    train_phase(agent, board_size=8, num_games=5000, max_steps=2000)
    
    # ========================================
    # STAGE 3: Collect demos from 8x8 agent
    # ========================================
    print("\n" + "=" * 60)
    print("📚 STAGE 3: COLLECT 8x8 DEMOS FOR 10x10")
    print("=" * 60)
    
    expert_data_8x8 = collect_demos_from_ppo(
        agent, board_size=8, 
        num_episodes=100, 
        min_score=10  # Only keep games with score >= 10
    )
    
    # ========================================
    # STAGE 4: Create NEW network for 10x10 and pretrain
    # ========================================
    print("\n" + "=" * 60)
    print("📚 STAGE 4: PRETRAIN 10x10 NETWORK ON 8x8 DEMOS")
    print("=" * 60)
    
    # Create fresh network (or could use agent.network)
    # Using the agent's network preserves learned features
    network_10x10 = agent.network  # Transfer weights!
    
    # Additional pretraining on 8x8 demos
    pretrain_critic(network_10x10, expert_data_8x8, epochs=30)
    pretrain_actor(network_10x10, expert_data_8x8, epochs=30)
    
    # ========================================
    # STAGE 5: Train on 10x10
    # ========================================
    print("\n" + "=" * 60)
    print("📚 STAGE 5: TRAIN ON 10x10 BOARD")
    print("=" * 60)
    
    # Reset agent optimizer for fresh training
    agent.optimizer = optim.Adam(agent.network.parameters(), lr=0.0003)
    train_phase(agent, board_size=10, num_games=10000, max_steps=5000)


if __name__ == '__main__':
    main()
