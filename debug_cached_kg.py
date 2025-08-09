#!/usr/bin/env python3
"""Debug why cached KG causes mechanism evaluation to fail"""

import numpy as np
import os
from train import *
from src.mdp import ActionType

def debug_cached_kg():
    """Debug issues with cached knowledge graph"""
    
    config = load_config('config.yml')
    
    print("="*70)
    print("DEBUGGING CACHED KNOWLEDGE GRAPH ISSUES")
    print("="*70)
    
    # Test 1: Load cached KG if it exists
    cache_path = './kg_cache/comprehensive_kg.json'
    
    if os.path.exists(cache_path):
        print(f"\n1. Loading cached KG from: {cache_path}")
        kg_cached = KnowledgeGraph(config)
        try:
            kg_cached.load(cache_path)
            print(f"   ✓ Loaded: {len(kg_cached.entities)} entities, {len(kg_cached.relationships)} relationships")
        except Exception as e:
            print(f"   ✗ Failed to load: {e}")
            kg_cached = None
    else:
        print(f"\n1. Cache not found at {cache_path}, creating minimal KG")
        kg_cached = None
    
    # Test 2: Compare with minimal KG
    print("\n2. Creating minimal working KG for comparison:")
    kg_minimal = KnowledgeGraph(config)
    from src.knowledge_graph import BiologicalEntity, BiologicalRelationship, RelationType
    
    kg_minimal.add_entity(BiologicalEntity("enzyme1", "Enzyme", "enzyme", {}, 1.0))
    kg_minimal.add_entity(BiologicalEntity("substrate1", "Substrate", "substrate", {}, 1.0))
    kg_minimal.add_relationship(BiologicalRelationship(
        "enzyme1", "substrate1", RelationType.CATALYSIS,
        {}, ["michaelis_menten"], 1.0
    ))
    print(f"   Minimal KG: {len(kg_minimal.entities)} entities, {len(kg_minimal.relationships)} relationships")
    
    # Test 3: Check mathematical constraints
    print("\n3. Checking mathematical constraints:")
    
    def check_constraints(kg, name):
        print(f"\n   {name}:")
        if hasattr(kg, 'mathematical_constraints'):
            print(f"   - Has {len(kg.mathematical_constraints)} constraints")
            # Sample a few
            for i, (key, constraint) in enumerate(list(kg.mathematical_constraints.items())[:3]):
                print(f"     {i+1}. {key}: {constraint.functional_form[:50]}...")
        else:
            print("   - No mathematical_constraints attribute!")
    
    check_constraints(kg_minimal, "Minimal KG")
    if kg_cached:
        check_constraints(kg_cached, "Cached KG")
    
    # Test 4: Test mechanism generation with each KG
    print("\n4. Testing mechanism generation:")
    
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    def test_kg(kg, name):
        print(f"\n   Testing {name}:")
        try:
            mdp = BiologicalMDP(kg, config)
            state = mdp.create_initial_state()
            
            # Get valid actions
            valid_actions = mdp.get_valid_actions(state)
            add_actions = [a for a in valid_actions if a.action_type == ActionType.ADD_ENTITY]
            
            print(f"   - Initial state: {state.mechanism_tree.to_expression()}")
            print(f"   - Valid actions: {len(valid_actions)}")
            print(f"   - Add entity actions: {len(add_actions)}")
            
            if add_actions:
                # Apply first add action
                action = add_actions[0]
                print(f"   - Adding: {action.entity_id} via {action.relation_type}")
                
                new_state = mdp.transition(state, action)
                expr = new_state.mechanism_tree.to_expression()
                print(f"   - Expression: {expr[:100]}...")
                print(f"   - Complexity: {new_state.mechanism_tree.get_complexity()}")
                
                # Try to evaluate
                policy_net = PolicyNetwork(kg, config)
                value_net = ValueNetwork(kg, config)
                reward_fn = RewardFunction(kg, config)
                trainer = PPOTrainer(mdp, policy_net, value_net, reward_fn, config)
                
                score = trainer._evaluate_mechanism(new_state, system.data_X, system.data_y)
                print(f"   - Evaluation score: {score:.3f}")
                
                if score <= -100:
                    print(f"   ✗ PROBLEM: Score is {score}, evaluation failing!")
                    
                    # Debug the expression
                    print(f"   - Full expression: {expr}")
                    print(f"   - Parameters: {new_state.mechanism_tree.parameters}")
                    
                    # Check what's in the node
                    if new_state.mechanism_tree.children:
                        child = new_state.mechanism_tree.children[0]
                        print(f"   - Child node type: {child.node_type}")
                        print(f"   - Child functional form: {child.functional_form}")
                        print(f"   - Child parameters: {child.parameters}")
                else:
                    print(f"   ✓ Score is good: {score:.3f}")
                    
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    test_kg(kg_minimal, "Minimal KG")
    if kg_cached:
        test_kg(kg_cached, "Cached KG")
    
    # Test 5: Check for specific issues
    print("\n5. Checking for specific issues:")
    
    if kg_cached:
        print("\n   Cached KG structure:")
        
        # Check if entities have the right structure
        sample_entities = list(kg_cached.entities.keys())[:5]
        print(f"   - Sample entities: {sample_entities}")
        
        # Check relationships
        sample_rels = list(kg_cached.relationships.keys())[:5] if kg_cached.relationships else []
        print(f"   - Sample relationships: {sample_rels}")
        
        # Check graph structure
        if hasattr(kg_cached, 'graph'):
            print(f"   - Graph nodes: {kg_cached.graph.number_of_nodes()}")
            print(f"   - Graph edges: {kg_cached.graph.number_of_edges()}")
            
            # Sample some edges
            edges = list(kg_cached.graph.edges(data=True))[:3]
            for u, v, data in edges:
                print(f"     Edge: {u} -> {v}, data: {data}")
    
    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70)
    
    if kg_cached:
        print("The cached KG exists. Check the output above for issues.")
    else:
        print("No cached KG found. The training is probably using a different KG.")
        print("Try running: python train.py --use-kg-cache")

if __name__ == "__main__":
    debug_cached_kg()