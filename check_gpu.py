#!/usr/bin/env python3
"""Check GPU utilization and debug performance issues"""

import torch
import time
import numpy as np
from src.knowledge_graph import KnowledgeGraph
from src.mdp import BiologicalMDP
from src.networks import PolicyNetwork, ValueNetwork
from src.reward import RewardFunction
from src.ppo_trainer import PPOTrainer
from src.synthetic_data import SyntheticDataGenerator, SystemType
import yaml

def load_config(path='config.yml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def check_gpu_usage():
    """Check if GPU is available and being used"""
    print("=" * 50)
    print("GPU Status Check")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name()}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    else:
        print("✗ CUDA not available")
    
    print("\nPyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "N/A")
    print()

def profile_episode(trainer, system, num_episodes=5):
    """Profile training episodes to find bottlenecks"""
    print("=" * 50)
    print("Profiling Training Episodes")
    print("=" * 50)
    
    trainer.reward_function.set_data(system.data_X, system.data_y)
    
    timings = []
    for i in range(num_episodes):
        start = time.time()
        
        # Time each major step
        step_times = {}
        
        # Initial state
        t0 = time.time()
        state = trainer.mdp.create_initial_state()
        step_times['init_state'] = time.time() - t0
        
        # Get valid actions
        t0 = time.time()
        valid_actions = trainer.mdp.get_valid_actions(state)
        step_times['get_actions'] = time.time() - t0
        
        if valid_actions:
            # Forward pass
            t0 = time.time()
            with torch.no_grad():
                value = trainer.value_network(state)
            step_times['value_forward'] = time.time() - t0
            
            t0 = time.time()
            action, log_prob = trainer.policy_network.get_action(state, valid_actions, 0.5)
            step_times['policy_forward'] = time.time() - t0
            
            # Transition
            t0 = time.time()
            next_state = trainer.mdp.transition(state, action)
            step_times['transition'] = time.time() - t0
            
            # Reward
            t0 = time.time()
            reward = trainer.reward_function.compute_reward(state, action, next_state)
            step_times['reward'] = time.time() - t0
        
        total_time = time.time() - start
        timings.append((total_time, step_times))
        
        print(f"\nEpisode {i+1} timing (ms):")
        print(f"  Total: {total_time*1000:.2f}")
        for step, t in step_times.items():
            print(f"  {step}: {t*1000:.2f}")
    
    avg_time = np.mean([t[0] for t in timings])
    print(f"\nAverage episode time: {avg_time*1000:.2f} ms")
    
    # Check which step is slowest
    all_steps = {}
    for _, steps in timings:
        for step, t in steps.items():
            if step not in all_steps:
                all_steps[step] = []
            all_steps[step].append(t)
    
    print("\nAverage step times (ms):")
    slowest_step = None
    slowest_time = 0
    for step, times in all_steps.items():
        avg = np.mean(times) * 1000
        print(f"  {step}: {avg:.2f}")
        if avg > slowest_time:
            slowest_time = avg
            slowest_step = step
    
    print(f"\n⚠ Slowest step: {slowest_step} ({slowest_time:.2f} ms)")

def main():
    config = load_config()
    
    # Check GPU
    check_gpu_usage()
    
    # Create minimal KG for testing
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
    
    # Check device placement
    print("=" * 50)
    print("Model Device Placement")
    print("=" * 50)
    
    policy_net = PolicyNetwork(kg, config)
    value_net = ValueNetwork(kg, config)
    
    device = torch.device(config['compute']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Target device: {device}")
    
    policy_net.to(device)
    value_net.to(device)
    
    # Check if models are on GPU
    for name, param in policy_net.named_parameters():
        print(f"Policy network {name}: {param.device}")
        break  # Just check first param
    
    for name, param in value_net.named_parameters():
        print(f"Value network {name}: {param.device}")
        break
    
    # Create trainer
    reward_fn = RewardFunction(kg, config)
    trainer = PPOTrainer(mdp, policy_net, value_net, reward_fn, config)
    
    # Generate test data
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    # Profile episodes
    profile_episode(trainer, system)
    
    # Check GPU memory after operations
    if torch.cuda.is_available():
        print("\n" + "=" * 50)
        print("GPU Memory After Training")
        print("=" * 50)
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()