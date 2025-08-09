#!/usr/bin/env python3
"""Debug why mechanisms aren't evaluating properly"""

import numpy as np
from train import *
import logging
from src.mdp import ActionType, CombineOperation

def test_mechanism_generation():
    """Test that mechanisms are generated and evaluated correctly"""
    
    # Setup
    config = load_config('config.yml')
    kg = KnowledgeGraph(config)
    
    # Add minimal entities
    from src.knowledge_graph import BiologicalEntity, BiologicalRelationship, RelationType
    kg.add_entity(BiologicalEntity("enzyme1", "Enzyme", "enzyme", {}, 1.0))
    kg.add_entity(BiologicalEntity("substrate1", "Substrate", "substrate", {}, 1.0))
    kg.add_relationship(BiologicalRelationship(
        "enzyme1", "substrate1", RelationType.CATALYSIS,
        {}, ["michaelis_menten"], 1.0
    ))
    
    # Create MDP
    mdp = BiologicalMDP(kg, config)
    
    # Generate test data
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print("="*70)
    print("TESTING MECHANISM GENERATION AND EVALUATION")
    print("="*70)
    
    # Test 1: Create initial state
    state = mdp.create_initial_state()
    print(f"\n1. Initial state complexity: {state.mechanism_tree.get_complexity()}")
    print(f"   Expression: {state.mechanism_tree.to_expression()}")
    
    # Test 2: Add an entity
    valid_actions = mdp.get_valid_actions(state)
    add_actions = [a for a in valid_actions if a.action_type == ActionType.ADD_ENTITY]
    
    if add_actions:
        print(f"\n2. Found {len(add_actions)} add entity actions")
        
        # Apply first add action
        action = add_actions[0]
        print(f"   Applying: Add {action.entity_id} with {action.relation_type}")
        
        new_state = mdp.transition(state, action)
        print(f"   New complexity: {new_state.mechanism_tree.get_complexity()}")
        
        expr = new_state.mechanism_tree.to_expression()
        print(f"   Expression: {expr}")
        
        # Test 3: Try to evaluate the expression
        print("\n3. Testing expression evaluation:")
        
        try:
            # Create evaluation dictionary
            eval_dict = {
                'exp': np.exp,
                'log': np.log,
                'sqrt': np.sqrt,
                'max': np.maximum,
                'min': np.minimum
            }
            
            # Add data as S
            eval_dict['S'] = system.data_X[:, 0]
            if system.data_X.shape[1] > 1:
                eval_dict['I'] = system.data_X[:, 1]
            if system.data_X.shape[1] > 2:
                eval_dict['A'] = system.data_X[:, 2]
            
            # Add parameters
            all_params = {}
            def collect_params(node):
                all_params.update(node.parameters)
                for child in node.children:
                    collect_params(child)
            
            collect_params(new_state.mechanism_tree)
            eval_dict.update(all_params)
            
            print(f"   Variables in eval_dict: {list(eval_dict.keys())}")
            print(f"   Expression to evaluate: {expr}")
            
            # Try evaluation
            result = eval(expr, {"__builtins__": {}}, eval_dict)
            result = np.array(result)
            
            print(f"   ✓ Evaluation successful! Result shape: {result.shape}")
            print(f"   Result sample: {result[:5]}")
            
            # Calculate MSE
            mse = np.mean((system.data_y - result) ** 2)
            print(f"   MSE: {mse:.3f}")
            
        except Exception as e:
            print(f"   ✗ Evaluation failed: {e}")
            print(f"   Expression was: {expr}")
    
    # Test 4: Build a more complex mechanism
    print("\n4. Building complex mechanism:")
    
    state = mdp.create_initial_state()
    for i in range(10):
        valid_actions = mdp.get_valid_actions(state)
        if not valid_actions:
            break
        
        # Prefer add actions
        add_actions = [a for a in valid_actions if a.action_type == ActionType.ADD_ENTITY]
        if add_actions:
            action = add_actions[0]
        else:
            action = valid_actions[0]
        
        state = mdp.transition(state, action)
        
        if i % 3 == 0:
            complexity = state.mechanism_tree.get_complexity()
            expr = state.mechanism_tree.to_expression()
            print(f"   Step {i}: Complexity={complexity}, Expr={expr[:50]}...")
    
    # Final evaluation
    print("\n5. Final mechanism evaluation:")
    final_expr = state.mechanism_tree.to_expression()
    print(f"   Final expression: {final_expr[:100]}...")
    print(f"   Final complexity: {state.mechanism_tree.get_complexity()}")
    
    # Try PPO trainer evaluation
    print("\n6. Testing PPO trainer evaluation:")
    
    policy_net = PolicyNetwork(kg, config)
    value_net = ValueNetwork(kg, config)
    reward_fn = RewardFunction(kg, config)
    trainer = PPOTrainer(mdp, policy_net, value_net, reward_fn, config)
    
    try:
        score = trainer._evaluate_mechanism(state, system.data_X, system.data_y)
        print(f"   PPO evaluation score: {score:.3f}")
        
        if score <= -100:
            print("   ⚠ Score is still -100 or worse, mechanism evaluation is failing!")
        else:
            print("   ✓ Mechanism evaluated successfully!")
            
    except Exception as e:
        print(f"   ✗ PPO evaluation failed: {e}")

if __name__ == "__main__":
    test_mechanism_generation()