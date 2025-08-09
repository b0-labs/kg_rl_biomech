#!/usr/bin/env python3
"""Final test to verify all fixes are working"""

import numpy as np
from train import load_config, setup_logging, create_knowledge_graph
from src.synthetic_data import SyntheticDataGenerator, SystemType
from src.mdp import BiologicalMDP
from src.ppo_trainer import PPOTrainer
from src.networks import PolicyNetwork, ValueNetwork
from src.reward import RewardFunction
from src.parameter_optimization import ParameterOptimizer
from src.evaluation import EvaluationMetrics

def test_single_system():
    """Test training on a single system to verify fixes"""
    config = load_config('config.yml')
    logger = setup_logging(config)
    
    print("="*60)
    print("Testing fixes on a single enzyme kinetics system")
    print("="*60)
    
    # Generate one system
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print(f"\nSystem details:")
    print(f"  Mechanism: {system.mechanism}")
    print(f"  True parameters: {system.true_parameters}")
    print(f"  Data shape: X={system.data_X.shape}, y={system.data_y.shape}")
    print(f"  y stats: mean={np.mean(system.data_y):.3f}, std={np.std(system.data_y):.3f}")
    
    # Setup training
    kg = create_knowledge_graph(config, logger, use_cache=True)
    mdp = BiologicalMDP(kg, config)
    policy_network = PolicyNetwork(kg, config)
    value_network = ValueNetwork(kg, config)
    reward_function = RewardFunction(kg, config)
    trainer = PPOTrainer(mdp, policy_network, value_network, reward_function, config)
    optimizer = ParameterOptimizer(config)
    evaluator = EvaluationMetrics(config)
    
    # Train for just 10 episodes
    print("\nTraining for 10 episodes...")
    best_mechanism = None
    best_score = float('-inf')
    
    for episode in range(10):
        stats = trainer.train_episode(system.data_X, system.data_y)
        print(f"  Episode {episode}: Reward={stats['episode_reward']:.3f}, Best={stats['best_score']:.3f}")
        
        current_best = trainer.get_best_mechanism()
        if current_best and stats['best_score'] > best_score:
            best_score = stats['best_score']
            best_mechanism = current_best
    
    if best_mechanism:
        print(f"\nBest mechanism found:")
        print(f"  Expression: {best_mechanism.mechanism_tree.to_expression()}")
        print(f"  Complexity: {best_mechanism.mechanism_tree.get_complexity()}")
        
        # Get initial parameters
        def get_all_params(node):
            params = dict(node.parameters)
            for child in node.children:
                params.update(get_all_params(child))
            return params
        
        initial_params = get_all_params(best_mechanism.mechanism_tree)
        print(f"  Initial parameters: {initial_params}")
        
        # Optimize parameters
        print("\nOptimizing parameters...")
        optimized_params, loss = optimizer.optimize_parameters(
            best_mechanism, system.data_X, system.data_y
        )
        print(f"  Optimization loss: {loss:.4f}")
        print(f"  Optimized parameters: {optimized_params}")
        
        # Update mechanism with optimized parameters
        def update_params(node, opt_params):
            for param_name, value in opt_params.items():
                if param_name in node.parameters:
                    node.parameters[param_name] = value
                elif f"{node.node_id}_{param_name}" in opt_params:
                    node.parameters[param_name] = opt_params[f"{node.node_id}_{param_name}"]
            for child in node.children:
                update_params(child, opt_params)
        
        update_params(best_mechanism.mechanism_tree, optimized_params)
        
        # Evaluate
        print("\nEvaluating mechanism...")
        predictions = evaluator._evaluate_mechanism_predictions(best_mechanism, system.data_X)
        
        if predictions is not None:
            errors = evaluator.evaluate_prediction_error(predictions, system.data_y)
            param_recovery = evaluator.evaluate_parameter_recovery(
                optimized_params, system.true_parameters
            )
            
            print(f"\nResults:")
            print(f"  RMSE: {errors['rmse']:.4f}")
            print(f"  MAE: {errors['mae']:.4f}")
            print(f"  R²: {errors['r2']:.4f}")
            print(f"  MAPE: {errors['mape']:.2f}%")
            print(f"  Parameter Recovery: {param_recovery:.3f}")
            
            # Check if results are reasonable
            if errors['r2'] < -100:
                print("  ⚠️ WARNING: R² is extremely negative!")
            elif errors['r2'] < -1:
                print("  ⚠️ Note: R² is negative (predictions worse than mean)")
            elif errors['r2'] > 0.5:
                print("  ✅ Good R² value!")
            else:
                print("  ℹ️ R² indicates moderate fit")
            
            if param_recovery > 0.5:
                print("  ✅ Good parameter recovery!")
            elif param_recovery > 0.1:
                print("  ℹ️ Moderate parameter recovery")
            else:
                print("  ⚠️ Poor parameter recovery")
        else:
            print("  ❌ Could not evaluate mechanism")
    else:
        print("  ❌ No mechanism discovered")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

if __name__ == "__main__":
    test_single_system()