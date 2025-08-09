#!/usr/bin/env python3
"""Debug why R² is still negative despite parameter recovery working"""

import numpy as np
from train import load_config, setup_logging, create_knowledge_graph
from src.synthetic_data import SyntheticDataGenerator, SystemType
from src.mdp import BiologicalMDP, Action, ActionType
from src.parameter_optimization import ParameterOptimizer
from src.evaluation import EvaluationMetrics

def debug_r2_issue():
    config = load_config('config.yml')
    logger = setup_logging(config)
    
    # Generate a system
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print("="*60)
    print("SYSTEM DATA:")
    print(f"  True mechanism: {system.mechanism}")
    print(f"  True parameters: {system.true_parameters}")
    print(f"  X shape: {system.data_X.shape}")
    print(f"  X range: [{np.min(system.data_X):.6f}, {np.max(system.data_X):.6f}]")
    print(f"  y shape: {system.data_y.shape}")
    print(f"  y range: [{np.min(system.data_y):.6f}, {np.max(system.data_y):.6f}]")
    print(f"  y mean: {np.mean(system.data_y):.6f}, std: {np.std(system.data_y):.6f}")
    
    # Create a mechanism
    kg = create_knowledge_graph(config, logger, use_cache=True)
    mdp = BiologicalMDP(kg, config)
    initial_state = mdp.create_initial_state()
    
    valid_actions = mdp.get_valid_actions(initial_state)
    add_actions = [a for a in valid_actions if a.action_type == ActionType.ADD_ENTITY]
    
    if not add_actions:
        print("No add actions available")
        return
        
    state = mdp.transition(initial_state, add_actions[0])
    
    print("\n" + "="*60)
    print("INITIAL MECHANISM:")
    print(f"  Expression: {state.mechanism_tree.to_expression()}")
    
    # Get initial parameters
    def get_all_params(node):
        params = {}
        for child in node.children:
            for param_name, value in child.parameters.items():
                params[f"{child.node_id}_{param_name}"] = value
        return params
    
    initial_params = get_all_params(state.mechanism_tree)
    print(f"  Initial params: {initial_params}")
    
    # Optimize parameters
    optimizer = ParameterOptimizer(config)
    optimized_params, loss = optimizer.optimize_parameters(
        state, system.data_X, system.data_y
    )
    
    print("\n" + "="*60)
    print("OPTIMIZATION:")
    print(f"  Optimized params: {optimized_params}")
    print(f"  Loss: {loss:.6f}")
    
    # Update state with optimized parameters
    for child in state.mechanism_tree.children:
        for param_name in child.parameters:
            key = f"{child.node_id}_{param_name}"
            if key in optimized_params:
                child.parameters[param_name] = optimized_params[key]
    
    # Evaluate predictions
    evaluator = EvaluationMetrics(config)
    predictions = evaluator._evaluate_mechanism_predictions(state, system.data_X)
    
    if predictions is None:
        print("ERROR: Could not evaluate mechanism")
        return
    
    print("\n" + "="*60)
    print("PREDICTIONS:")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Predictions range: [{np.min(predictions):.6f}, {np.max(predictions):.6f}]")
    print(f"  Predictions mean: {np.mean(predictions):.6f}, std: {np.std(predictions):.6f}")
    
    # Check if predictions are constant
    if np.std(predictions) < 1e-6:
        print("  ⚠️ WARNING: Predictions are essentially constant!")
    
    # Evaluate with true parameters
    print("\n" + "="*60)
    print("EVALUATION WITH TRUE PARAMETERS:")
    
    # Set true parameters
    for child in state.mechanism_tree.children:
        for param_name in system.true_parameters:
            if param_name in child.parameters:
                child.parameters[param_name] = system.true_parameters[param_name]
    
    true_param_predictions = evaluator._evaluate_mechanism_predictions(state, system.data_X)
    
    if true_param_predictions is not None:
        print(f"  Predictions with true params range: [{np.min(true_param_predictions):.6f}, {np.max(true_param_predictions):.6f}]")
        print(f"  Predictions with true params mean: {np.mean(true_param_predictions):.6f}, std: {np.std(true_param_predictions):.6f}")
        
        # Calculate R² with true parameters
        ss_res_true = np.sum((system.data_y - true_param_predictions) ** 2)
        ss_tot = np.sum((system.data_y - np.mean(system.data_y)) ** 2)
        r2_true = 1 - (ss_res_true / ss_tot)
        print(f"  R² with true parameters: {r2_true:.6f}")
    
    print("\n" + "="*60)
    print("R² CALCULATION:")
    
    # Calculate errors
    errors = evaluator.evaluate_prediction_error(predictions, system.data_y)
    
    print(f"  RMSE: {errors['rmse']:.6f}")
    print(f"  MAE: {errors['mae']:.6f}")
    print(f"  R²: {errors['r2']:.6f}")
    
    # Manual R² calculation
    ss_res = np.sum((system.data_y - predictions) ** 2)
    ss_tot = np.sum((system.data_y - np.mean(system.data_y)) ** 2)
    r2_manual = 1 - (ss_res / ss_tot)
    
    print(f"\n  Manual calculation:")
    print(f"    SS_res: {ss_res:.6f}")
    print(f"    SS_tot: {ss_tot:.6f}")
    print(f"    R² = 1 - {ss_res:.6f}/{ss_tot:.6f} = {r2_manual:.6f}")
    
    # Plot first 10 points
    print("\n" + "="*60)
    print("FIRST 10 POINTS COMPARISON:")
    print("  Index |      X      |   y_true    |   y_pred    |    Error")
    print("  ------|-------------|-------------|-------------|-------------")
    for i in range(min(10, len(system.data_y))):
        x_val = system.data_X[i, 0] if system.data_X.ndim > 1 else system.data_X[i]
        error = system.data_y[i] - predictions[i]
        print(f"    {i:3d} | {x_val:11.6f} | {system.data_y[i]:11.6f} | {predictions[i]:11.6f} | {error:11.6f}")
    
    # Check if X values are reasonable for enzyme kinetics
    print("\n" + "="*60)
    print("DATA SANITY CHECK:")
    print(f"  X values seem {'reasonable' if np.max(system.data_X) > 0.01 else 'very small'}")
    print(f"  y values seem {'reasonable' if np.std(system.data_y) > 0.001 else 'very small variation'}")
    
    if np.max(system.data_X) < 0.01:
        print("  ⚠️ WARNING: X values are very small, this might affect fitting")

if __name__ == "__main__":
    debug_r2_issue()