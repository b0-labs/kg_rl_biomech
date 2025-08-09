#!/usr/bin/env python3
"""Debug script to understand why training isn't improving"""

import numpy as np
import torch
import yaml
from src.synthetic_data import SyntheticDataGenerator, SystemType
from src.knowledge_graph import KnowledgeGraph
from src.mdp import BiologicalMDP
from src.networks import PolicyNetwork, ValueNetwork
from src.reward import RewardFunction
from src.ppo_trainer import PPOTrainer
from src.parameter_optimization import ParameterOptimizer
from src.evaluation import EvaluationMetrics

def load_config(path='config.yml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def analyze_training_step(trainer, system, step_num):
    """Analyze what happens in a single training step"""
    
    # Get initial state
    state = trainer.mdp.create_initial_state()
    print(f"\n=== Step {step_num} ===")
    print(f"Initial complexity: {state.mechanism_tree.get_complexity()}")
    
    # Get valid actions
    valid_actions = trainer.mdp.get_valid_actions(state)
    print(f"Valid actions: {len(valid_actions)}")
    
    if valid_actions:
        # Take an action
        action, log_prob = trainer.policy_network.get_action(state, valid_actions, trainer.epsilon)
        print(f"Action taken: {action.action_type.name}")
        
        # Get next state
        next_state = trainer.mdp.transition(state, action)
        print(f"Next complexity: {next_state.mechanism_tree.get_complexity()}")
        
        # Compute reward
        reward = trainer.reward_function.compute_reward(state, action, next_state)
        print(f"Reward: {reward:.3f}")
        
        # Evaluate mechanism
        if next_state.mechanism_tree.get_complexity() > 1:
            score = trainer._evaluate_mechanism(next_state, system.data_X, system.data_y)
            print(f"Mechanism score: {score:.3f}")
            
            # Show the mechanism expression
            try:
                expr = next_state.mechanism_tree.to_expression()
                print(f"Expression: {expr[:100]}...")  # First 100 chars
            except:
                print("Expression: <failed to generate>")
    
    return state

def main():
    config = load_config()
    print("Loading configuration...")
    
    # Create simple KG
    kg = KnowledgeGraph(config)
    from src.knowledge_graph import BiologicalEntity, BiologicalRelationship, RelationType
    
    # Add minimal entities
    kg.add_entity(BiologicalEntity("enzyme1", "Enzyme", "enzyme", {}, 1.0))
    kg.add_entity(BiologicalEntity("substrate1", "Substrate", "substrate", {}, 1.0))
    kg.add_relationship(BiologicalRelationship(
        "enzyme1", "substrate1", RelationType.CATALYSIS,
        {}, ["michaelis_menten"], 1.0
    ))
    
    print(f"KG: {len(kg.entities)} entities, {len(kg.relationships)} relationships")
    
    # Create components
    mdp = BiologicalMDP(kg, config)
    policy_net = PolicyNetwork(kg, config)
    value_net = ValueNetwork(kg, config)
    reward_fn = RewardFunction(kg, config)
    trainer = PPOTrainer(mdp, policy_net, value_net, reward_fn, config)
    
    # Generate simple data
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print(f"\nSystem type: {system.system_type.value}")
    print(f"Data shape: X={system.data_X.shape}, y={system.data_y.shape}")
    print(f"True mechanism: {system.mechanism}")
    
    # Set data for reward function
    trainer.reward_function.set_data(system.data_X, system.data_y)
    
    # Analyze a few training steps
    print("\n" + "="*50)
    print("ANALYZING TRAINING STEPS")
    print("="*50)
    
    for i in range(5):
        state = analyze_training_step(trainer, system, i)
        print()
    
    # Run actual training episode
    print("\n" + "="*50)
    print("RUNNING FULL EPISODE")
    print("="*50)
    
    episode_stats = trainer.train_episode(system.data_X, system.data_y)
    print(f"Episode reward: {episode_stats['episode_reward']:.3f}")
    print(f"Episode length: {episode_stats['episode_length']}")
    print(f"Best score: {episode_stats['best_score']:.3f}")
    print(f"Final complexity: {episode_stats['final_complexity']}")
    
    # Check if mechanism was built
    best = trainer.get_best_mechanism()
    if best:
        print(f"\nBest mechanism complexity: {best.mechanism_tree.get_complexity()}")
        try:
            expr = best.mechanism_tree.to_expression()
            print(f"Best expression: {expr[:200]}...")
        except:
            print("Best expression: <failed to generate>")

if __name__ == "__main__":
    main()