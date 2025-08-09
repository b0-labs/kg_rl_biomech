#!/usr/bin/env python3
"""Minimal test to verify training works without getting stuck"""

import time
import numpy as np
import torch
import yaml
from src.knowledge_graph import KnowledgeGraph, BiologicalEntity, BiologicalRelationship, RelationType
from src.mdp import BiologicalMDP
from src.networks import PolicyNetwork, ValueNetwork
from src.reward import RewardFunction
from src.ppo_trainer import PPOTrainer
from src.synthetic_data import SyntheticDataGenerator, SystemType

def load_config():
    with open('config.yml', 'r') as f:
        return yaml.safe_load(f)

def main():
    print("Testing minimal training loop...")
    
    # Setup
    config = load_config()
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create minimal KG
    kg = KnowledgeGraph(config)
    kg.add_entity(BiologicalEntity("enzyme1", "Enzyme", "enzyme", {}, 1.0))
    kg.add_entity(BiologicalEntity("substrate1", "Substrate", "substrate", {}, 1.0))
    kg.add_relationship(BiologicalRelationship(
        "enzyme1", "substrate1", RelationType.CATALYSIS,
        {}, ["michaelis_menten"], 1.0
    ))
    
    # Create components
    mdp = BiologicalMDP(kg, config)
    policy_net = PolicyNetwork(kg, config)
    value_net = ValueNetwork(kg, config)
    reward_fn = RewardFunction(kg, config)
    
    print("Creating trainer...")
    start = time.time()
    trainer = PPOTrainer(mdp, policy_net, value_net, reward_fn, config)
    print(f"Trainer created in {time.time()-start:.2f}s")
    
    # Generate simple data
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print(f"\nRunning 20 episodes...")
    episode_times = []
    rewards = []
    
    for i in range(20):
        start = time.time()
        stats = trainer.train_episode(system.data_X, system.data_y)
        episode_time = time.time() - start
        episode_times.append(episode_time)
        rewards.append(stats['episode_reward'])
        
        # Show progress
        if i % 5 == 0:
            print(f"Episode {i}: {episode_time*1000:.1f}ms, "
                  f"Reward: {stats['episode_reward']:.2f}, "
                  f"Best: {stats['best_score']:.2f}")
    
    # Analysis
    print("\n" + "="*50)
    print("Results:")
    print(f"Average time: {np.mean(episode_times)*1000:.1f}ms")
    print(f"Max time: {np.max(episode_times)*1000:.1f}ms")
    print(f"Average reward: {np.mean(rewards):.2f}")
    
    # Check for issues
    slow_count = sum(1 for t in episode_times if t > 0.5)
    if slow_count > 0:
        print(f"⚠ WARNING: {slow_count} episodes took >500ms")
    else:
        print("✓ All episodes completed quickly")
    
    # Check best mechanism
    best = trainer.get_best_mechanism()
    if best and best.mechanism_tree.get_complexity() > 1:
        print(f"✓ Found mechanism with complexity {best.mechanism_tree.get_complexity()}")
    else:
        print("⚠ No good mechanism found")

if __name__ == "__main__":
    main()