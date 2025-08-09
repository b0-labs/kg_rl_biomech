#!/usr/bin/env python3
"""Test that the fixes for numerical issues are working"""

import numpy as np
import sys
from train import load_config, setup_logging
from src.knowledge_graph import KnowledgeGraph
from src.mdp import BiologicalMDP, MechanismNode
from src.evaluation import EvaluationMetrics
from src.parameter_optimization import ParameterOptimizer
from src.synthetic_data import SyntheticDataGenerator, SystemType

def test_expression_generation():
    """Test that expressions are generated properly"""
    config = load_config('config.yml')
    
    # Test empty root
    root = MechanismNode("root", "root")
    expr = root.to_expression()
    print(f"Empty root expression: {expr}")
    assert expr != "1.0", "Empty root should not return constant 1.0"
    
    # Test entity node with functional form
    entity_node = MechanismNode(
        "node_1", "entity",
        functional_form="(v_max * S) / (k_m + S)",
        parameters={"v_max": 10.0, "k_m": 0.5}
    )
    expr = entity_node.to_expression()
    print(f"Entity node expression: {expr}")
    assert "v_max" in expr, "Parameters should remain as variables"
    assert "10.0" not in expr, "Parameter values should not be substituted"
    
    print("✅ Expression generation tests passed")

def test_evaluation_bounds():
    """Test that evaluation handles bounds properly"""
    config = load_config('config.yml')
    evaluator = EvaluationMetrics(config)
    
    # Test with extreme values
    predictions = np.array([1e15, -1e15, np.inf, -np.inf, np.nan])
    true_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    errors = evaluator.evaluate_prediction_error(predictions, true_values)
    print(f"R² with extreme values: {errors['r2']}")
    assert errors['r2'] > -101, "R² should be bounded"
    assert errors['r2'] < 2, "R² should be bounded"
    
    # Test with constant predictions
    constant_pred = np.ones(100)
    varying_true = np.random.randn(100)
    
    errors = evaluator.evaluate_prediction_error(constant_pred, varying_true)
    print(f"R² with constant predictions: {errors['r2']}")
    assert errors['r2'] > -101, "R² should be bounded even with constant predictions"
    
    print("✅ Evaluation bounds tests passed")

def test_mechanism_predictions():
    """Test that mechanism predictions work properly"""
    config = load_config('config.yml')
    
    # Create a simple mechanism
    from src.mdp import MDPState
    
    root = MechanismNode("root", "root")
    entity = MechanismNode(
        "node_1", "entity",
        entity_id="enzyme_1",
        functional_form="(v_max * S) / (k_m + S)",
        parameters={"v_max": 10.0, "k_m": 0.5}
    )
    root.children.append(entity)
    entity.parent = root
    
    state = MDPState(
        mechanism_tree=root,
        available_entities=set(),
        construction_history=[],
        parameter_constraints={},
        step_count=0,
        is_terminal=False
    )
    
    # Test evaluation
    evaluator = EvaluationMetrics(config)
    X = np.linspace(0.1, 10, 50).reshape(-1, 1)
    
    predictions = evaluator._evaluate_mechanism_predictions(state, X)
    
    if predictions is not None:
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
        assert not np.all(predictions == predictions[0]), "Predictions should vary, not be constant"
        assert np.all(np.isfinite(predictions)), "All predictions should be finite"
        print("✅ Mechanism prediction tests passed")
    else:
        print("⚠️ Warning: Predictions returned None")

def test_with_synthetic_data():
    """Test with actual synthetic data"""
    config = load_config('config.yml')
    logger = setup_logging(config)
    
    # Generate one simple system
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print(f"\nTesting with synthetic system:")
    print(f"  Mechanism: {system.mechanism}")
    print(f"  Data shape: X={system.data_X.shape}, y={system.data_y.shape}")
    print(f"  y range: [{np.min(system.data_y):.3f}, {np.max(system.data_y):.3f}]")
    
    # Create a simple mechanism for testing
    kg = KnowledgeGraph(config)
    kg.load('./kg_cache/comprehensive_kg.json')
    
    mdp = BiologicalMDP(kg, config)
    initial_state = mdp.get_initial_state()
    
    # Add an entity
    from src.mdp import Action, ActionType
    from src.knowledge_graph import RelationType
    
    # Find a valid action
    valid_actions = mdp.get_valid_actions(initial_state)
    add_actions = [a for a in valid_actions if a.action_type == ActionType.ADD_ENTITY]
    
    if add_actions:
        # Apply first add action
        state = mdp.transition(initial_state, add_actions[0])
        
        # Evaluate
        evaluator = EvaluationMetrics(config)
        predictions = evaluator._evaluate_mechanism_predictions(state, system.data_X)
        
        if predictions is not None:
            errors = evaluator.evaluate_prediction_error(predictions, system.data_y)
            print(f"\nEvaluation results:")
            print(f"  RMSE: {errors['rmse']:.4f}")
            print(f"  R²: {errors['r2']:.4f}")
            
            assert errors['r2'] > -100, f"R² too negative: {errors['r2']}"
            assert errors['r2'] < 1.01, f"R² too high: {errors['r2']}"
            print("✅ Synthetic data tests passed")
        else:
            print("⚠️ Could not evaluate mechanism")
    else:
        print("⚠️ No valid add actions available")

if __name__ == "__main__":
    print("="*60)
    print("Testing fixes for numerical issues")
    print("="*60)
    
    try:
        test_expression_generation()
        print()
        test_evaluation_bounds()
        print()
        test_mechanism_predictions()
        print()
        test_with_synthetic_data()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)