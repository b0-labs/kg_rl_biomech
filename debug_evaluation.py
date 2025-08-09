#!/usr/bin/env python3
"""Debug why evaluation still returns -100"""

import numpy as np
from train import *
from src.mdp import ActionType

def debug_evaluation():
    """Debug the evaluation pipeline in detail"""
    
    config = load_config('config.yml')
    
    # Load fixed KG
    kg = KnowledgeGraph(config)
    kg.load('./kg_cache/comprehensive_kg_fixed.json')
    
    print("="*70)
    print("DEBUGGING EVALUATION PIPELINE")
    print("="*70)
    
    # Create components
    mdp = BiologicalMDP(kg, config)
    policy_net = PolicyNetwork(kg, config)
    value_net = ValueNetwork(kg, config)
    reward_fn = RewardFunction(kg, config)
    trainer = PPOTrainer(mdp, policy_net, value_net, reward_fn, config)
    
    # Generate data
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print(f"\nData shape: X={system.data_X.shape}, y={system.data_y.shape}")
    print(f"Data X sample: {system.data_X[:3, 0]}")
    print(f"Data y sample: {system.data_y[:3]}")
    
    # Build a mechanism step by step
    print("\n1. Building a mechanism:")
    state = mdp.create_initial_state()
    
    for i in range(5):
        valid_actions = mdp.get_valid_actions(state)
        add_actions = [a for a in valid_actions if a.action_type == ActionType.ADD_ENTITY]
        
        if add_actions:
            action = add_actions[0]
            state = mdp.transition(state, action)
            print(f"   Step {i+1}: Added {action.entity_id} via {action.relation_type.value}")
    
    # Mark as terminal
    terminate_action = next((a for a in mdp.get_valid_actions(state) if a.action_type == ActionType.TERMINATE), None)
    if terminate_action:
        state = mdp.transition(state, terminate_action)
    
    print(f"\n2. Final mechanism:")
    expr = state.mechanism_tree.to_expression()
    print(f"   Expression: {expr[:200]}...")
    print(f"   Complexity: {state.mechanism_tree.get_complexity()}")
    print(f"   Is terminal: {state.is_terminal}")
    
    # Now debug the evaluation
    print("\n3. Debugging _evaluate_mechanism:")
    
    # Check what the expression evaluates to
    if expr in ["1.0", "0", "unknown", ""]:
        print(f"   ✗ Expression is trivial: '{expr}'")
        print("   This triggers the -100 return in _evaluate_mechanism")
    else:
        print(f"   ✓ Expression is non-trivial")
    
    # Try manual evaluation
    print("\n4. Manual evaluation:")
    try:
        # Build eval dict
        safe_dict = {
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'max': np.maximum, 'min': np.minimum
        }
        
        # Add data columns
        for i in range(system.data_X.shape[1]):
            safe_dict[f'X{i}'] = system.data_X[:, i]
        safe_dict['S'] = system.data_X[:, 0]
        if system.data_X.shape[1] > 1:
            safe_dict['I'] = system.data_X[:, 1]
        if system.data_X.shape[1] > 2:
            safe_dict['A'] = system.data_X[:, 2]
        
        # Add all parameters from tree
        all_params = trainer._get_all_parameters(state.mechanism_tree)
        print(f"   Parameters found: {list(all_params.keys())[:10]}")
        safe_dict.update(all_params)
        
        print(f"   Variables available: {sorted(safe_dict.keys())}")
        
        # Try to evaluate
        print(f"   Evaluating: {expr[:100]}...")
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        result = np.array(result)
        
        print(f"   ✓ Evaluation successful!")
        print(f"   Result shape: {result.shape}")
        print(f"   Result sample: {result[:3]}")
        
        # Check if it's scalar
        if result.ndim == 0:
            print(f"   ⚠ Result is scalar: {result}")
            result = np.full_like(system.data_y, result)
            print(f"   Broadcasted to shape: {result.shape}")
        
        # Calculate MSE
        mse = np.mean((system.data_y - result) ** 2)
        print(f"   MSE: {mse:.3f}")
        
        # Calculate score as in _evaluate_mechanism
        score = -mse
        plausibility = kg.compute_plausibility_score(
            list(state.mechanism_tree.get_all_entities()),
            state.mechanism_tree.get_all_relations()
        )
        complexity_penalty = np.exp(-0.1 * state.mechanism_tree.get_complexity())
        
        final_score = score + 0.5 * plausibility + 0.3 * complexity_penalty
        print(f"\n   Score components:")
        print(f"     -MSE: {score:.3f}")
        print(f"     Plausibility: {plausibility:.3f}")
        print(f"     Complexity penalty: {complexity_penalty:.3f}")
        print(f"     Final score: {final_score:.3f}")
        
    except Exception as e:
        print(f"   ✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Now test with trainer's method
    print("\n5. Testing trainer._evaluate_mechanism:")
    try:
        trainer_score = trainer._evaluate_mechanism(state, system.data_X, system.data_y)
        print(f"   Trainer score: {trainer_score:.3f}")
        
        if trainer_score == -100:
            print("   ✗ PROBLEM: Trainer returns -100!")
            print("   This means the expression is being classified as trivial")
            print(f"   Expression was: {expr[:100]}...")
        elif trainer_score < -100:
            print(f"   ✗ PROBLEM: Score is very negative: {trainer_score}")
        else:
            print(f"   ✓ Score looks reasonable: {trainer_score}")
            
    except Exception as e:
        print(f"   ✗ Trainer evaluation failed: {e}")

if __name__ == "__main__":
    debug_evaluation()