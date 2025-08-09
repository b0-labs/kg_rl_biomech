#!/usr/bin/env python3
"""Debug why R² values are so negative"""

import numpy as np
import sys
from train import load_config, setup_logging, create_knowledge_graph
from src.mdp import BiologicalMDP
from src.evaluation import EvaluationMetrics
from src.synthetic_data import SyntheticDataGenerator, SystemType

def test_r2_calculation():
    """Test R² calculation with known data"""
    config = load_config('config.yml')
    evaluator = EvaluationMetrics(config)
    
    # Test 1: Perfect predictions should give R² = 1
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    errors = evaluator.evaluate_prediction_error(y_pred, y_true)
    print(f"Test 1 - Perfect predictions: R² = {errors['r2']:.4f} (expected ~1.0)")
    assert abs(errors['r2'] - 1.0) < 0.01, f"Perfect predictions should give R²≈1, got {errors['r2']}"
    
    # Test 2: Constant predictions (mean) should give R² = 0
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.ones(5) * np.mean(y_true)
    errors = evaluator.evaluate_prediction_error(y_pred, y_true)
    print(f"Test 2 - Mean predictions: R² = {errors['r2']:.4f} (expected ~0.0)")
    assert abs(errors['r2']) < 0.01, f"Mean predictions should give R²≈0, got {errors['r2']}"
    
    # Test 3: Bad predictions should give negative R²
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([5, 4, 3, 2, 1])  # Completely wrong
    errors = evaluator.evaluate_prediction_error(y_pred, y_true)
    print(f"Test 3 - Bad predictions: R² = {errors['r2']:.4f} (expected negative)")
    assert errors['r2'] < 0, f"Bad predictions should give R²<0, got {errors['r2']}"
    
    print("✅ Basic R² tests passed")

def test_mechanism_evaluation():
    """Test mechanism evaluation on synthetic data"""
    config = load_config('config.yml')
    logger = setup_logging(config)
    
    # Generate synthetic data
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print(f"\nSystem mechanism: {system.mechanism}")
    print(f"True parameters: {system.true_parameters}")
    print(f"Data shape: X={system.data_X.shape}, y={system.data_y.shape}")
    print(f"y range: [{np.min(system.data_y):.3f}, {np.max(system.data_y):.3f}]")
    print(f"y mean: {np.mean(system.data_y):.3f}, std: {np.std(system.data_y):.3f}")
    
    # Create knowledge graph and MDP
    kg = create_knowledge_graph(config, logger, use_cache=True)
    mdp = BiologicalMDP(kg, config)
    
    # Get initial state
    initial_state = mdp.get_initial_state()
    
    # Add an entity to create a simple mechanism
    from src.mdp import Action, ActionType
    from src.knowledge_graph import RelationType
    
    valid_actions = mdp.get_valid_actions(initial_state)
    add_actions = [a for a in valid_actions if a.action_type == ActionType.ADD_ENTITY]
    
    if add_actions:
        # Try first add action
        state = mdp.transition(initial_state, add_actions[0])
        
        print(f"\nMechanism expression: {state.mechanism_tree.to_expression()}")
        print(f"Mechanism parameters: {state.mechanism_tree.parameters}")
        
        # Get all parameters from tree
        def get_all_params(node):
            params = dict(node.parameters)
            for child in node.children:
                params.update(get_all_params(child))
            return params
        
        all_params = get_all_params(state.mechanism_tree)
        print(f"All parameters: {all_params}")
        
        # Evaluate mechanism
        evaluator = EvaluationMetrics(config)
        predictions = evaluator._evaluate_mechanism_predictions(state, system.data_X)
        
        if predictions is not None:
            print(f"\nPredictions shape: {predictions.shape}")
            print(f"Predictions range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
            print(f"Predictions mean: {np.mean(predictions):.3f}, std: {np.std(predictions):.3f}")
            
            # Check if predictions are constant
            if np.std(predictions) < 1e-10:
                print("⚠️ WARNING: Predictions are essentially constant!")
                print(f"First 10 predictions: {predictions[:10]}")
            
            # Calculate errors
            errors = evaluator.evaluate_prediction_error(predictions, system.data_y)
            print(f"\nEvaluation metrics:")
            print(f"  RMSE: {errors['rmse']:.4f}")
            print(f"  MAE: {errors['mae']:.4f}")
            print(f"  R²: {errors['r2']:.4f}")
            
            # Debug R² calculation manually
            ss_res = np.sum((system.data_y - predictions) ** 2)
            ss_tot = np.sum((system.data_y - np.mean(system.data_y)) ** 2)
            r2_manual = 1 - (ss_res / ss_tot)
            print(f"\nManual R² calculation:")
            print(f"  SS_res: {ss_res:.4f}")
            print(f"  SS_tot: {ss_tot:.4f}")
            print(f"  R² = 1 - {ss_res:.4f}/{ss_tot:.4f} = {r2_manual:.4f}")
            
            if abs(r2_manual) > 100:
                print(f"⚠️ WARNING: R² is extremely negative, indicating very poor predictions")
                print(f"  This means predictions are {abs(r2_manual):.0f}x worse than using the mean")
        else:
            print("❌ Could not evaluate mechanism (returned None)")
    else:
        print("❌ No valid add actions available")

def test_expression_evaluation():
    """Test that expressions evaluate properly with parameters"""
    config = load_config('config.yml')
    
    # Test a simple Michaelis-Menten expression
    expression = "(v_max * S) / (k_m + S)"
    params = {"v_max": 10.0, "k_m": 0.5}
    
    # Create test data
    X = np.linspace(0.1, 10, 50).reshape(-1, 1)
    
    # Evaluate manually
    safe_dict = {
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'max': np.maximum,
        'min': np.minimum,
        'S': X[:, 0]
    }
    safe_dict.update(params)
    
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        result = np.array(result)
        
        print(f"Expression: {expression}")
        print(f"Parameters: {params}")
        print(f"X range: [{np.min(X):.3f}, {np.max(X):.3f}]")
        print(f"Result shape: {result.shape}")
        print(f"Result range: [{np.min(result):.3f}, {np.max(result):.3f}]")
        print(f"Result variation: std={np.std(result):.3f}")
        
        # Check if result varies with input
        if np.std(result) < 0.01:
            print("⚠️ WARNING: Result has very low variation!")
        else:
            print("✅ Result varies properly with input")
            
        # Plot the function
        print("\nFunction behavior (first 10 points):")
        for i in range(min(10, len(X))):
            print(f"  S={X[i,0]:.2f} -> y={result[i]:.3f}")
            
    except Exception as e:
        print(f"❌ Error evaluating expression: {e}")

if __name__ == "__main__":
    print("="*60)
    print("Debugging R² calculation issues")
    print("="*60)
    
    try:
        print("\n1. Testing basic R² calculations:")
        print("-"*40)
        test_r2_calculation()
        
        print("\n2. Testing expression evaluation:")
        print("-"*40)
        test_expression_evaluation()
        
        print("\n3. Testing mechanism evaluation on synthetic data:")
        print("-"*40)
        test_mechanism_evaluation()
        
        print("\n" + "="*60)
        print("✅ DEBUGGING COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)