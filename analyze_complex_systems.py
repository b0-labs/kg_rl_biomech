#!/usr/bin/env python3
"""Analyze why complex systems fail even with more training"""

import numpy as np
from train import load_config, setup_logging, create_knowledge_graph
from src.synthetic_data import SyntheticDataGenerator, SystemType
from src.mdp import BiologicalMDP
from src.ppo_trainer import PPOTrainer
from src.networks import PolicyNetwork, ValueNetwork
from src.reward import RewardFunction
from src.parameter_optimization import ParameterOptimizer
from src.evaluation import EvaluationMetrics

def analyze_complex_system():
    config = load_config('config.yml')
    logger = setup_logging(config)
    
    # Generate a complex system (multi-substrate)
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 5)
    
    # Find the complexity 4 system
    complex_system = None
    for system in systems:
        if system.complexity_level == 4:
            complex_system = system
            break
    
    if not complex_system:
        print("No complexity 4 system found")
        return
    
    print("="*70)
    print("ANALYZING COMPLEX SYSTEM (Multi-substrate with product inhibition)")
    print("="*70)
    print(f"True mechanism: {complex_system.mechanism}")
    print(f"True parameters: {complex_system.true_parameters}")
    print(f"Data shape: X={complex_system.data_X.shape}, y={complex_system.data_y.shape}")
    print(f"y range: [{np.min(complex_system.data_y):.4f}, {np.max(complex_system.data_y):.4f}]")
    
    # Setup training
    kg = create_knowledge_graph(config, logger, use_cache=True)
    mdp = BiologicalMDP(kg, config)
    policy_network = PolicyNetwork(kg, config)
    value_network = ValueNetwork(kg, config)
    reward_function = RewardFunction(kg, config)
    trainer = PPOTrainer(mdp, policy_network, value_network, reward_function, config)
    optimizer = ParameterOptimizer(config)
    evaluator = EvaluationMetrics(config)
    
    print("\n" + "="*70)
    print("TRAINING ON COMPLEX SYSTEM")
    print("="*70)
    
    # Train for different episode counts
    episode_counts = [10, 50, 100]
    
    for num_episodes in episode_counts:
        print(f"\nTraining for {num_episodes} episodes...")
        
        # Reset trainer
        trainer = PPOTrainer(mdp, policy_network, value_network, reward_function, config)
        best_mechanism = None
        best_score = float('-inf')
        
        for episode in range(num_episodes):
            stats = trainer.train_episode(complex_system.data_X, complex_system.data_y)
            
            if episode % 20 == 0:
                print(f"  Episode {episode}: Reward={stats['episode_reward']:.3f}, Best={stats['best_score']:.3f}")
            
            current_best = trainer.get_best_mechanism()
            if current_best and stats['best_score'] > best_score:
                best_score = stats['best_score']
                best_mechanism = current_best
        
        if best_mechanism:
            print(f"\nBest mechanism after {num_episodes} episodes:")
            expr = best_mechanism.mechanism_tree.to_expression()
            print(f"  Expression: {expr}")
            print(f"  Complexity: {best_mechanism.mechanism_tree.get_complexity()}")
            
            # Check if it's just a simple mechanism
            if "S1" not in expr and "S2" not in expr:
                print("  ⚠️ WARNING: Mechanism doesn't use multi-substrate inputs!")
                print("  This is likely a simple Michaelis-Menten, not multi-substrate")
            
            # Optimize parameters
            optimized_params, loss = optimizer.optimize_parameters(
                best_mechanism, complex_system.data_X, complex_system.data_y
            )
            
            print(f"\n  Optimization loss: {loss:.6f}")
            
            # Update mechanism
            for param_name, value in optimized_params.items():
                for child in best_mechanism.mechanism_tree.children:
                    if param_name.startswith(f"{child.node_id}_"):
                        base_param = param_name.replace(f"{child.node_id}_", "")
                        if base_param in child.parameters:
                            child.parameters[base_param] = value
            
            # Evaluate
            predictions = evaluator._evaluate_mechanism_predictions(best_mechanism, complex_system.data_X)
            
            if predictions is not None:
                errors = evaluator.evaluate_prediction_error(predictions, complex_system.data_y)
                param_recovery = evaluator.evaluate_parameter_recovery(
                    optimized_params, complex_system.true_parameters
                )
                
                print(f"\n  Results:")
                print(f"    RMSE: {errors['rmse']:.4f}")
                print(f"    R²: {errors['r2']:.4f}")
                print(f"    Parameter Recovery: {param_recovery:.3f}")
            else:
                print("  Could not evaluate mechanism")
        else:
            print(f"  No mechanism discovered after {num_episodes} episodes")
    
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print("\nThe issue with complex systems is likely that:")
    print("1. The RL agent discovers simple mechanisms (Michaelis-Menten) instead of complex ones")
    print("2. Simple mechanisms can't fit multi-dimensional data well")
    print("3. More training doesn't help if the agent is stuck with wrong mechanism structure")
    print("\nPossible solutions:")
    print("- Adjust reward to favor mechanisms that use all input dimensions")
    print("- Increase exploration to discover more complex mechanisms")
    print("- Add specific actions for multi-substrate mechanisms")

if __name__ == "__main__":
    analyze_complex_system()