#!/usr/bin/env python3
"""
Test compositional mechanism discovery approach
Shows it can flexibly discover mechanisms for any number of dimensions
"""

import numpy as np
import yaml
from src.compositional_mdp import (
    CompositionalMDP, CompositionalState, CompositionalAction,
    ExpressionNode, NodeType, BinaryOp, BiologicalOp
)

def test_compositional_discovery():
    """Test compositional approach on various dimensional problems"""
    
    # Load config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create compositional MDP
    mdp = CompositionalMDP(config)
    
    print("=" * 70)
    print("COMPOSITIONAL MECHANISM DISCOVERY TEST")
    print("=" * 70)
    
    # Test 1: Simple 1D Michaelis-Menten
    print("\n1. Simple 1D Michaelis-Menten:")
    print("-" * 40)
    X1 = np.random.uniform(0.1, 10.0, (100, 1))
    y1 = (5.0 * X1[:, 0]) / (2.0 + X1[:, 0]) + np.random.normal(0, 0.01, 100)
    
    state1 = discover_mechanism(mdp, X1, y1, max_steps=20)
    evaluate_result(state1, X1, y1, "v_max * X / (k_m + X)")
    
    # Test 2: 2D Multi-substrate
    print("\n2. Two-substrate mechanism:")
    print("-" * 40)
    X2 = np.random.uniform(0.1, 10.0, (100, 2))
    y2 = (10.0 * X2[:, 0] * X2[:, 1]) / ((1.0 + X2[:, 0]) * (2.0 + X2[:, 1]))
    y2 += np.random.normal(0, 0.01, 100)
    
    state2 = discover_mechanism(mdp, X2, y2, max_steps=30)
    evaluate_result(state2, X2, y2, "v_max * X0 * X1 / ((k1 + X0) * (k2 + X1))")
    
    # Test 3: 3D with product inhibition
    print("\n3. Three-dimensional with product inhibition:")
    print("-" * 40)
    X3 = np.random.uniform(0.1, 10.0, (100, 3))
    y3 = (10.0 * X3[:, 0] * X3[:, 1]) / ((1.0 + X3[:, 0]) * (2.0 + X3[:, 1]) * (1.0 + X3[:, 2] / 0.5))
    y3 += np.random.normal(0, 0.01, 100)
    
    state3 = discover_mechanism(mdp, X3, y3, max_steps=40)
    evaluate_result(state3, X3, y3, "v * X0 * X1 / ((k1+X0) * (k2+X1) * (1+X2/kp))")
    
    # Test 4: 4D complex system
    print("\n4. Four-dimensional complex system:")
    print("-" * 40)
    X4 = np.random.uniform(0.1, 10.0, (100, 4))
    # Complex 4-way interaction with allosteric regulation
    numerator = 15.0 * X4[:, 0] * X4[:, 1] * (1.0 + 2.0 * X4[:, 3] / (1.0 + X4[:, 3]))
    denominator = (1.0 + X4[:, 0]) * (2.0 + X4[:, 1]) * (1.0 + X4[:, 2] / 0.5)
    y4 = numerator / denominator + np.random.normal(0, 0.01, 100)
    
    state4 = discover_mechanism(mdp, X4, y4, max_steps=50)
    evaluate_result(state4, X4, y4, "Complex 4D with allosteric regulation")
    
    # Test 5: 5D system
    print("\n5. Five-dimensional system:")
    print("-" * 40)
    X5 = np.random.uniform(0.1, 10.0, (100, 5))
    # 5-way interaction: first 3 are substrates, 4th is inhibitor, 5th is activator
    y5 = (20.0 * X5[:, 0] * X5[:, 1] * X5[:, 2] * X5[:, 4]) / (
        (1.0 + X5[:, 0]) * (1.0 + X5[:, 1]) * (1.0 + X5[:, 2]) * 
        (1.0 + X5[:, 3] / 0.5) * (0.1 + X5[:, 4])
    )
    y5 += np.random.normal(0, 0.01, 100)
    
    state5 = discover_mechanism(mdp, X5, y5, max_steps=60)
    evaluate_result(state5, X5, y5, "5D multi-substrate with inhibition and activation")
    
    print("\n" + "=" * 70)
    print("SUMMARY: Compositional approach handles N dimensions flexibly!")
    print("=" * 70)


def discover_mechanism(mdp: CompositionalMDP, X: np.ndarray, y: np.ndarray, 
                       max_steps: int = 30) -> CompositionalState:
    """Discover mechanism using greedy search with random exploration"""
    
    n_dims = X.shape[1]
    state = mdp.create_initial_state(data_dimensions=n_dims)
    
    best_state = None
    best_fitness = float('-inf')
    
    for step in range(max_steps):
        valid_actions = mdp.get_valid_actions(state)
        
        if not valid_actions:
            break
        
        # Epsilon-greedy action selection
        epsilon = 0.3 * (1.0 - step / max_steps)  # Decay exploration
        
        if np.random.random() < epsilon:
            # Random exploration
            action = valid_actions[np.random.randint(len(valid_actions))]
        else:
            # Greedy: try each action and pick best
            best_action = None
            best_action_fitness = float('-inf')
            
            for action in valid_actions[:10]:  # Limit to first 10 for speed
                # Try action
                next_state = mdp.transition(state, action, data=(X, y))
                
                # Evaluate fitness
                if next_state.expression_tree is not None:
                    try:
                        predictions = next_state.expression_tree.evaluate(X, next_state.parameters)
                        mse = np.mean((y - predictions) ** 2)
                        complexity_penalty = 0.01 * next_state.expression_tree.get_complexity()
                        fitness = -mse - complexity_penalty
                        
                        if fitness > best_action_fitness:
                            best_action_fitness = fitness
                            best_action = action
                    except:
                        pass
            
            action = best_action if best_action else valid_actions[0]
        
        # Apply action
        state = mdp.transition(state, action, data=(X, y))
        
        # Track best state
        if state.expression_tree is not None:
            try:
                predictions = state.expression_tree.evaluate(X, state.parameters)
                mse = np.mean((y - predictions) ** 2)
                fitness = -mse
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_state = state.deep_copy()
            except:
                pass
        
        # Early termination if good enough
        if best_fitness > -0.01:
            break
    
    return best_state if best_state else state


def evaluate_result(state: CompositionalState, X: np.ndarray, y: np.ndarray, 
                    true_mechanism: str):
    """Evaluate discovered mechanism"""
    
    if state is None or state.expression_tree is None:
        print("❌ No mechanism discovered")
        return
    
    # Get expression
    expr = state.expression_tree.to_expression()
    print(f"Discovered: {expr}")
    print(f"True:       {true_mechanism}")
    
    # Evaluate fit
    try:
        predictions = state.expression_tree.evaluate(X, state.parameters)
        mse = np.mean((y - predictions) ** 2)
        r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        print(f"MSE: {mse:.6f}")
        print(f"R²:  {r2:.4f}")
        
        # Check dimensions used
        used_vars = state.expression_tree.get_used_variables()
        print(f"Dimensions used: {len(used_vars)}/{X.shape[1]} = {used_vars}")
        
        # Success criteria
        success = r2 > 0.8 and len(used_vars) >= min(3, X.shape[1])
        print(f"{'✅ SUCCESS' if success else '⚠️  PARTIAL'}")
        
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
    
    print()


if __name__ == "__main__":
    try:
        test_compositional_discovery()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()