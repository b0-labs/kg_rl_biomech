#!/usr/bin/env python3
"""
Test script to verify that the RL agent can discover multi-substrate mechanisms
after the fixes to knowledge graph, MDP, and reward function.
"""

import numpy as np
import yaml
from src.synthetic_data import SyntheticDataGenerator
from src.knowledge_graph import KnowledgeGraph, BiologicalEntity, BiologicalRelationship, RelationType
from src.mdp import BiologicalMDP
from src.networks import PolicyNetwork, ValueNetwork
from src.reward import RewardFunction
from src.ppo_trainer import PPOTrainer

def create_test_knowledge_graph(config):
    """Create a simple knowledge graph for testing multi-substrate mechanisms"""
    kg = KnowledgeGraph(config)
    
    # Add entities for multi-substrate reactions
    kg.add_entity(BiologicalEntity(
        id="substrate1",
        name="Substrate 1",
        entity_type="substrate",
        properties={"molecular_weight": 180.0},
        confidence_score=0.95
    ))
    
    kg.add_entity(BiologicalEntity(
        id="substrate2", 
        name="Substrate 2",
        entity_type="substrate",
        properties={"molecular_weight": 200.0},
        confidence_score=0.95
    ))
    
    kg.add_entity(BiologicalEntity(
        id="product",
        name="Product",
        entity_type="product",
        properties={"molecular_weight": 380.0},
        confidence_score=0.95
    ))
    
    kg.add_entity(BiologicalEntity(
        id="enzyme",
        name="Multi-substrate Enzyme",
        entity_type="enzyme",
        properties={"EC_number": "1.2.3.4"},
        confidence_score=0.95
    ))
    
    # Add relationships
    kg.add_relationship(BiologicalRelationship(
        source="substrate1",
        target="enzyme",
        relation_type=RelationType.SUBSTRATE_OF,
        properties={},
        mathematical_constraints=[],
        confidence_score=0.9
    ))
    
    kg.add_relationship(BiologicalRelationship(
        source="substrate2",
        target="enzyme",
        relation_type=RelationType.SUBSTRATE_OF,
        properties={},
        mathematical_constraints=[],
        confidence_score=0.9
    ))
    
    kg.add_relationship(BiologicalRelationship(
        source="enzyme",
        target="product",
        relation_type=RelationType.CATALYSIS,
        properties={},
        mathematical_constraints=[],
        confidence_score=0.9
    ))
    
    kg.add_relationship(BiologicalRelationship(
        source="product",
        target="enzyme",
        relation_type=RelationType.PRODUCT_OF,
        properties={},
        mathematical_constraints=[],
        confidence_score=0.9
    ))
    
    return kg

def test_multi_substrate_discovery():
    """Test if the agent can discover multi-substrate mechanisms"""
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate multi-substrate test data
    data_generator = SyntheticDataGenerator(config)
    
    # Generate a complex multi-substrate system
    np.random.seed(42)
    n_samples = 100
    
    # Three dimensions: substrate1, substrate2, product
    X = np.random.uniform(0.1, 10.0, (n_samples, 3))
    
    # True mechanism: v_max * S1 * S2 / ((k_m1 + S1) * (k_m2 + S2) * (1 + P/k_p))
    v_max = 10.0
    k_m1 = 1.0
    k_m2 = 2.0  
    k_p = 0.5
    
    y = (v_max * X[:, 0] * X[:, 1]) / ((k_m1 + X[:, 0]) * (k_m2 + X[:, 1]) * (1.0 + X[:, 2] / k_p))
    y += np.random.normal(0, 0.01 * np.std(y), n_samples)  # Add small noise
    
    print("=" * 60)
    print("Testing Multi-Substrate Mechanism Discovery")
    print("=" * 60)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"True mechanism: (v_max * S1 * S2) / ((k_m1 + S1) * (k_m2 + S2) * (1 + P/k_p))")
    print(f"True parameters: v_max={v_max}, k_m1={k_m1}, k_m2={k_m2}, k_p={k_p}")
    print()
    
    # Create knowledge graph
    kg = create_test_knowledge_graph(config)
    
    # Create MDP with input dimensions specified
    mdp = BiologicalMDP(kg, config, input_dimensions=3)
    
    # Create networks
    policy_net = PolicyNetwork(kg, config)
    value_net = ValueNetwork(kg, config)
    
    # Create reward function
    reward_fn = RewardFunction(kg, config)
    
    # Create trainer
    trainer = PPOTrainer(mdp, policy_net, value_net, reward_fn, config)
    
    # Train for a few episodes
    print("Training agent...")
    best_scores = []
    
    for episode in range(20):  # Reduced episodes for quick test
        stats = trainer.train_episode(X, y)
        
        if episode % 5 == 0:
            print(f"Episode {episode}: reward={stats['episode_reward']:.3f}, "
                  f"complexity={stats['final_complexity']}, "
                  f"best_score={stats['best_score']:.3f}")
        
        # Update networks periodically
        if episode > 0 and episode % 2 == 0:
            trainer.update_networks()
        
        best_scores.append(stats['best_score'])
    
    # Get best mechanism
    best_state = trainer.get_best_mechanism()
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    if best_state:
        mechanism_expr = best_state.mechanism_tree.to_expression()
        print(f"Discovered mechanism: {mechanism_expr}")
        
        # Check if it uses multiple dimensions
        uses_x0 = 'X0' in mechanism_expr
        uses_x1 = 'X1' in mechanism_expr  
        uses_x2 = 'X2' in mechanism_expr
        uses_s1s2 = 'S1' in mechanism_expr and 'S2' in mechanism_expr
        uses_p = 'P' in mechanism_expr or 'X2' in mechanism_expr
        
        print(f"Uses X0: {uses_x0}")
        print(f"Uses X1: {uses_x1}")
        print(f"Uses X2: {uses_x2}")
        print(f"Uses S1/S2: {uses_s1s2}")
        print(f"Uses P: {uses_p}")
        
        # Count dimensions used
        dims_used = sum([uses_x0, uses_x1, uses_x2])
        print(f"\nDimensions used: {dims_used}/3")
        
        # Evaluate final MSE
        try:
            safe_dict = {
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'max': np.maximum, 'min': np.minimum
            }
            
            # Map dimensions
            for i in range(X.shape[1]):
                safe_dict[f'X{i}'] = X[:, i]
            safe_dict['S'] = X[:, 0]
            safe_dict['S1'] = X[:, 0]
            safe_dict['S2'] = X[:, 1]
            safe_dict['P'] = X[:, 2]
            safe_dict['I'] = X[:, 1]
            safe_dict['A'] = X[:, 2]
            
            # Get parameters
            def get_all_params(node):
                params = dict(node.parameters)
                for child in node.children:
                    params.update(get_all_params(child))
                return params
            
            all_params = get_all_params(best_state.mechanism_tree)
            safe_dict.update(all_params)
            
            predictions = eval(mechanism_expr, {"__builtins__": {}}, safe_dict)
            predictions = np.array(predictions)
            
            mse = np.mean((y - predictions) ** 2)
            r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            print(f"\nFinal MSE: {mse:.6f}")
            print(f"Final R²: {r2:.4f}")
            
            # Success criteria
            success = dims_used >= 2 and r2 > 0.5
            print(f"\n{'✓ SUCCESS' if success else '✗ FAILED'}: Multi-substrate discovery "
                  f"{'successful' if success else 'needs improvement'}")
            
        except Exception as e:
            print(f"Error evaluating mechanism: {e}")
    else:
        print("No mechanism discovered")
    
    return best_scores

if __name__ == "__main__":
    try:
        best_scores = test_multi_substrate_discovery()
        
        # Check if scores are improving
        if len(best_scores) > 5:
            early_avg = np.mean(best_scores[:5])
            late_avg = np.mean(best_scores[-5:])
            improvement = late_avg - early_avg
            
            print(f"\nScore improvement: {improvement:.3f} "
                  f"(early avg: {early_avg:.3f}, late avg: {late_avg:.3f})")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()