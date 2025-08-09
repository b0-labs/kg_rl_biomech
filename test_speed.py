#!/usr/bin/env python3
"""Test training speed improvements"""

import time
import torch
import numpy as np
from train import *

def test_training_speed():
    """Test if training runs at reasonable speed"""
    
    # Load config
    config = load_config('config.yml')
    logger = setup_logging(config)
    
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Create simple KG
    knowledge_graph = create_knowledge_graph(config, logger, use_cache=True)
    
    # Create components
    mdp = BiologicalMDP(knowledge_graph, config)
    policy_network = PolicyNetwork(knowledge_graph, config)
    value_network = ValueNetwork(knowledge_graph, config)
    reward_function = RewardFunction(knowledge_graph, config)
    
    print("\n" + "="*50)
    print("Testing Training Speed")
    print("="*50)
    
    # Create trainer (includes warmup)
    print("Creating trainer (with warmup)...")
    start = time.time()
    trainer = PPOTrainer(mdp, policy_network, value_network, reward_function, config)
    init_time = time.time() - start
    print(f"Initialization time: {init_time:.2f}s")
    
    # Generate test data
    optimizer = ParameterOptimizer(config)
    evaluator = EvaluationMetrics(config)
    data_generator = SyntheticDataGenerator(config)
    systems = data_generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    # Time several episodes
    print("\nTiming episodes after warmup:")
    episode_times = []
    
    for i in range(10):
        start = time.time()
        episode_stats = trainer.train_episode(system.data_X, system.data_y)
        episode_time = time.time() - start
        episode_times.append(episode_time)
        
        print(f"Episode {i+1}: {episode_time*1000:.1f}ms, "
              f"Reward: {episode_stats['episode_reward']:.2f}, "
              f"Steps: {episode_stats['episode_length']}")
    
    avg_time = np.mean(episode_times) * 1000
    std_time = np.std(episode_times) * 1000
    
    print(f"\nAverage episode time: {avg_time:.1f} ± {std_time:.1f} ms")
    print(f"Episodes per second: {1000/avg_time:.1f}")
    
    # Check for slow episodes
    slow_episodes = [i for i, t in enumerate(episode_times) if t > 0.1]  # >100ms
    if slow_episodes:
        print(f"⚠ Slow episodes (>100ms): {slow_episodes}")
    else:
        print("✓ No slow episodes detected")
    
    # Check GPU memory
    if torch.cuda.is_available():
        print(f"\nGPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

if __name__ == "__main__":
    test_training_speed()