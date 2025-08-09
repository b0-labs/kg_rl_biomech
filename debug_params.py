#!/usr/bin/env python3
"""Debug parameter recovery issue"""

import numpy as np
from train import load_config, setup_logging, create_knowledge_graph
from src.synthetic_data import SyntheticDataGenerator, SystemType
from src.mdp import BiologicalMDP, Action, ActionType
from src.parameter_optimization import ParameterOptimizer
from src.evaluation import EvaluationMetrics

def debug_parameter_issue():
    config = load_config('config.yml')
    logger = setup_logging(config)
    
    # Generate a system
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print("="*60)
    print("TRUE SYSTEM:")
    print(f"  Mechanism: {system.mechanism}")
    print(f"  True parameters: {system.true_parameters}")
    print(f"  Parameter names: {list(system.true_parameters.keys())}")
    print()
    
    # Create a simple mechanism
    kg = create_knowledge_graph(config, logger, use_cache=True)
    mdp = BiologicalMDP(kg, config)
    initial_state = mdp.create_initial_state()
    
    # Add an entity to create mechanism
    valid_actions = mdp.get_valid_actions(initial_state)
    add_actions = [a for a in valid_actions if a.action_type == ActionType.ADD_ENTITY]
    
    if add_actions:
        state = mdp.transition(initial_state, add_actions[0])
        
        print("="*60)
        print("DISCOVERED MECHANISM:")
        print(f"  Expression: {state.mechanism_tree.to_expression()}")
        
        # Get parameters from tree
        def get_all_params(node, prefix=""):
            params = {}
            node_prefix = f"{node.node_id}_" if node.node_id != "node_1" else ""
            for param_name, value in node.parameters.items():
                full_name = f"{node_prefix}{param_name}" if node_prefix else param_name
                params[full_name] = value
            for child in node.children:
                child_params = get_all_params(child, prefix)
                params.update(child_params)
            return params
        
        tree_params = get_all_params(state.mechanism_tree)
        print(f"  Tree parameters: {tree_params}")
        print(f"  Parameter names: {list(tree_params.keys())}")
        print()
        
        # Optimize parameters
        optimizer = ParameterOptimizer(config)
        optimized_params, loss = optimizer.optimize_parameters(
            state, system.data_X, system.data_y
        )
        
        print("="*60)
        print("OPTIMIZED PARAMETERS:")
        print(f"  Optimized params: {optimized_params}")
        print(f"  Parameter names: {list(optimized_params.keys())}")
        print(f"  Optimization loss: {loss:.4f}")
        print()
        
        # Check parameter recovery
        evaluator = EvaluationMetrics(config)
        
        # Test 1: Compare as-is
        recovery1 = evaluator.evaluate_parameter_recovery(
            optimized_params, system.true_parameters
        )
        print("="*60)
        print("PARAMETER RECOVERY TEST:")
        print(f"  Test 1 - Direct comparison: {recovery1:.4f}")
        
        # Test 2: Strip node prefixes from optimized params
        stripped_params = {}
        for key, value in optimized_params.items():
            # Remove node_X_ prefix
            if '_' in key:
                parts = key.split('_')
                if parts[0] == 'node' and parts[1].isdigit():
                    # This is a node-prefixed parameter
                    base_name = '_'.join(parts[2:])
                    stripped_params[base_name] = value
                else:
                    stripped_params[key] = value
            else:
                stripped_params[key] = value
        
        print(f"\n  Stripped params: {stripped_params}")
        recovery2 = evaluator.evaluate_parameter_recovery(
            stripped_params, system.true_parameters
        )
        print(f"  Test 2 - Stripped prefixes: {recovery2:.4f}")
        
        # Test 3: Check if parameter names match
        print(f"\n  True param names: {set(system.true_parameters.keys())}")
        print(f"  Optimized param names: {set(optimized_params.keys())}")
        print(f"  Stripped param names: {set(stripped_params.keys())}")
        print(f"  Intersection: {set(stripped_params.keys()) & set(system.true_parameters.keys())}")
        
        # Test 4: Manual calculation
        if stripped_params and system.true_parameters:
            errors = []
            for param_name in system.true_parameters:
                if param_name in stripped_params:
                    true_val = system.true_parameters[param_name]
                    est_val = stripped_params[param_name]
                    if true_val != 0:
                        relative_error = abs(est_val - true_val) / abs(true_val)
                    else:
                        relative_error = abs(est_val - true_val)
                    errors.append(relative_error)
                    print(f"\n  {param_name}:")
                    print(f"    True: {true_val:.6f}")
                    print(f"    Estimated: {est_val:.6f}")
                    print(f"    Relative error: {relative_error:.4f}")
                else:
                    print(f"\n  {param_name}: NOT FOUND in optimized params")
                    errors.append(1.0)
            
            if errors:
                mean_error = np.mean(errors)
                recovery_manual = 1.0 - min(mean_error, 1.0)
                print(f"\n  Test 4 - Manual calculation: {recovery_manual:.4f}")

if __name__ == "__main__":
    debug_parameter_issue()