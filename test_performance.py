#!/usr/bin/env python3
"""Test that performance improvements work"""

import time
import numpy as np
import torch
from train import *

def test_episode_speed():
    """Test that episodes run at consistent speed"""
    
    config = load_config('config.yml')
    
    # Create minimal KG
    kg = KnowledgeGraph(config)
    from src.knowledge_graph import BiologicalEntity, BiologicalRelationship, RelationType
    
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
    trainer = PPOTrainer(mdp, policy_net, value_net, reward_fn, config)
    
    # Generate data
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print("Testing episode speeds...")
    print("="*50)
    
    times = []
    for i in range(30):
        start = time.time()
        stats = trainer.train_episode(system.data_X, system.data_y)
        elapsed = time.time() - start
        times.append(elapsed)
        
        if i % 10 == 0:
            print(f"Episode {i}: {elapsed*1000:.1f}ms, Reward: {stats['episode_reward']:.2f}")
    
    times_ms = np.array(times) * 1000
    print("\n" + "="*50)
    print(f"Mean: {np.mean(times_ms):.1f}ms")
    print(f"Std: {np.std(times_ms):.1f}ms")
    print(f"Max: {np.max(times_ms):.1f}ms")
    print(f"Min: {np.min(times_ms):.1f}ms")
    
    # Check for outliers
    slow = times_ms > 500
    if np.any(slow):
        slow_episodes = np.where(slow)[0]
        print(f"\n⚠ WARNING: {len(slow_episodes)} slow episodes (>500ms): {slow_episodes}")
        print(f"Slow episode times: {times_ms[slow]}")
    else:
        print("\n✓ No slow episodes detected!")
    
    return times_ms

if __name__ == "__main__":
    times = test_episode_speed()