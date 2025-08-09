#!/usr/bin/env python3
"""Profile to find exact bottlenecks in training"""

import time
import cProfile
import pstats
import io
from contextlib import contextmanager
import numpy as np
import torch
from train import *

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    print(f"{name}: {elapsed:.2f}ms")

def profile_training():
    """Profile training to find bottlenecks"""
    
    config = load_config('config.yml')
    
    # Create minimal KG for speed
    kg = KnowledgeGraph(config)
    from src.knowledge_graph import BiologicalEntity, BiologicalRelationship, RelationType
    
    kg.add_entity(BiologicalEntity("enzyme1", "Enzyme", "enzyme", {}, 1.0))
    kg.add_entity(BiologicalEntity("substrate1", "Substrate", "substrate", {}, 1.0))
    kg.add_relationship(BiologicalRelationship(
        "enzyme1", "substrate1", RelationType.CATALYSIS,
        {}, ["michaelis_menten"], 1.0
    ))
    
    print("Creating components...")
    with timer("MDP creation"):
        mdp = BiologicalMDP(kg, config)
    
    with timer("Network creation"):
        policy_net = PolicyNetwork(kg, config)
        value_net = ValueNetwork(kg, config)
    
    with timer("Trainer creation"):
        reward_fn = RewardFunction(kg, config)
        trainer = PPOTrainer(mdp, policy_net, value_net, reward_fn, config)
    
    # Generate data
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print("\n" + "="*60)
    print("PROFILING SINGLE EPISODE COMPONENTS")
    print("="*60)
    
    # Profile episode components
    trainer.reward_function.set_data(system.data_X, system.data_y)
    
    with timer("\n1. Create initial state"):
        state = mdp.create_initial_state()
    
    with timer("2. Get valid actions"):
        valid_actions = mdp.get_valid_actions(state)
    print(f"   Found {len(valid_actions)} valid actions")
    
    if valid_actions:
        with timer("3. Value network forward"):
            value = trainer.value_network(state).item()
        
        with timer("4. Policy network get_action"):
            action, log_prob = trainer.policy_network.get_action(state, valid_actions, 0.5)
        
        with timer("5. State transition"):
            next_state = mdp.transition(state, action)
        
        with timer("6. Compute reward"):
            reward = trainer.reward_function.compute_reward(state, action, next_state)
    
    print("\n" + "="*60)
    print("PROFILING FULL EPISODES")
    print("="*60)
    
    # Profile full episodes
    episode_times = []
    step_counts = []
    
    for i in range(10):
        start = time.perf_counter()
        stats = trainer.train_episode(system.data_X, system.data_y)
        elapsed = (time.perf_counter() - start) * 1000
        episode_times.append(elapsed)
        step_counts.append(stats['num_steps'])
        
        print(f"Episode {i}: {elapsed:.1f}ms, {stats['num_steps']} steps, "
              f"reward={stats['episode_reward']:.2f}, complexity={stats['final_complexity']}")
    
    print("\n" + "="*60)
    print("EPISODE STATISTICS")
    print("="*60)
    print(f"Mean time: {np.mean(episode_times):.1f}ms")
    print(f"Std time: {np.std(episode_times):.1f}ms")
    print(f"Max time: {np.max(episode_times):.1f}ms")
    print(f"Min time: {np.min(episode_times):.1f}ms")
    print(f"Mean steps: {np.mean(step_counts):.1f}")
    
    # Check for slow episodes
    slow_mask = np.array(episode_times) > 500
    if np.any(slow_mask):
        print(f"\n⚠ WARNING: {np.sum(slow_mask)} slow episodes detected!")
        slow_indices = np.where(slow_mask)[0]
        print(f"Slow episodes: {slow_indices}")
        print(f"Their times: {np.array(episode_times)[slow_mask]}")
    
    # Profile with cProfile for detailed breakdown
    print("\n" + "="*60)
    print("DETAILED PROFILING (Top 20 time consumers)")
    print("="*60)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run 5 episodes
    for _ in range(5):
        trainer.train_episode(system.data_X, system.data_y)
    
    profiler.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())
    
    # Check GPU utilization
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("GPU MEMORY USAGE")
        print("="*60)
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

if __name__ == "__main__":
    profile_training()