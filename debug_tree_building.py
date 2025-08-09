#!/usr/bin/env python3
"""Debug why mechanism tree isn't building properly"""

import numpy as np
from train import *
from src.mdp import ActionType, Action, MechanismNode

def debug_tree_building():
    """Debug the mechanism tree construction"""
    
    config = load_config('config.yml')
    
    # Load fixed KG
    kg = KnowledgeGraph(config)
    kg.load('./kg_cache/comprehensive_kg_fixed.json')
    
    print("="*70)
    print("DEBUGGING MECHANISM TREE BUILDING")
    print("="*70)
    
    mdp = BiologicalMDP(kg, config)
    
    # Start with initial state
    state = mdp.create_initial_state()
    print("\n1. Initial state:")
    print(f"   Tree expression: {state.mechanism_tree.to_expression()}")
    print(f"   Tree type: {state.mechanism_tree.node_type}")
    print(f"   Children: {len(state.mechanism_tree.children)}")
    print(f"   Complexity: {state.mechanism_tree.get_complexity()}")
    
    # Get valid actions
    valid_actions = mdp.get_valid_actions(state)
    add_actions = [a for a in valid_actions if a.action_type == ActionType.ADD_ENTITY]
    
    print(f"\n2. Available add actions: {len(add_actions)}")
    if add_actions:
        action = add_actions[0]
        print(f"   First action: Add {action.entity_id} via {action.relation_type.value}")
        print(f"   Position: {action.position}")
        
        # Check constraints
        print(f"\n3. Checking constraints for {action.relation_type}:")
        constraints = kg.get_constraints_for_relation(action.relation_type)
        print(f"   Found {len(constraints)} constraints")
        if constraints:
            constraint = constraints[0]
            print(f"   First constraint: {constraint.name}")
            print(f"   Functional form: {constraint.functional_form}")
            print(f"   Parameters: {constraint.parameter_bounds}")
        
        # Apply the action
        print("\n4. Applying action...")
        new_state = mdp.transition(state, action)
        
        print(f"   New tree expression: {new_state.mechanism_tree.to_expression()}")
        print(f"   New tree type: {new_state.mechanism_tree.node_type}")
        print(f"   New children: {len(new_state.mechanism_tree.children)}")
        print(f"   New complexity: {new_state.mechanism_tree.get_complexity()}")
        
        # Check the children
        if new_state.mechanism_tree.children:
            print("\n5. Examining children:")
            for i, child in enumerate(new_state.mechanism_tree.children):
                print(f"   Child {i}:")
                print(f"     Type: {child.node_type}")
                print(f"     Entity: {child.entity_id}")
                print(f"     Functional form: {child.functional_form}")
                print(f"     Parameters: {child.parameters}")
                print(f"     Expression: {child.to_expression()[:100]}...")
        
        # Try adding another
        print("\n6. Adding another entity:")
        valid_actions2 = mdp.get_valid_actions(new_state)
        add_actions2 = [a for a in valid_actions2 if a.action_type == ActionType.ADD_ENTITY]
        
        if add_actions2:
            action2 = add_actions2[0]
            print(f"   Action: Add {action2.entity_id} via {action2.relation_type.value}")
            
            state2 = mdp.transition(new_state, action2)
            print(f"   Expression after 2 adds: {state2.mechanism_tree.to_expression()[:200]}...")
            print(f"   Complexity: {state2.mechanism_tree.get_complexity()}")
            
            # Check tree structure
            print("\n7. Tree structure after 2 adds:")
            def print_tree(node, indent=0):
                prefix = "  " * indent
                print(f"{prefix}Node: {node.node_type}, entity={node.entity_id}, form={node.functional_form is not None}")
                for child in node.children:
                    print_tree(child, indent + 1)
            
            print_tree(state2.mechanism_tree)
    
    # Now test the _apply_add_entity directly
    print("\n8. Testing _apply_add_entity directly:")
    
    test_state = mdp.create_initial_state()
    test_action = Action(
        action_type=ActionType.ADD_ENTITY,
        entity_id="enzyme_test",
        relation_type=RelationType.CATALYSIS,  # We know this has constraints
        position="root"
    )
    
    print(f"   Before: {test_state.mechanism_tree.to_expression()}")
    
    # Get constraints
    constraints = kg.get_constraints_for_relation(RelationType.CATALYSIS)
    print(f"   Constraints for CATALYSIS: {len(constraints)}")
    
    # Apply manually
    mdp._apply_add_entity(test_state, test_action)
    
    print(f"   After: {test_state.mechanism_tree.to_expression()}")
    print(f"   Children: {len(test_state.mechanism_tree.children)}")
    
    if test_state.mechanism_tree.children:
        child = test_state.mechanism_tree.children[0]
        print(f"   Child functional form: {child.functional_form}")
        print(f"   Child parameters: {child.parameters}")

if __name__ == "__main__":
    debug_tree_building()