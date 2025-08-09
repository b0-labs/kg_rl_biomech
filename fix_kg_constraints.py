#!/usr/bin/env python3
"""Fix the knowledge graph constraint mapping issue"""

import numpy as np
from train import *
from src.mdp import ActionType, RelationType

def diagnose_constraint_issue():
    """Find why cached KG doesn't generate proper expressions"""
    
    config = load_config('config.yml')
    
    # Load cached KG
    kg = KnowledgeGraph(config)
    kg.load('./kg_cache/comprehensive_kg.json')
    
    print("="*70)
    print("DIAGNOSING CONSTRAINT MAPPING ISSUE")
    print("="*70)
    
    # Check what constraints are available for SUBSTRATE_OF
    print("\n1. Checking SUBSTRATE_OF constraints:")
    if RelationType.SUBSTRATE_OF in kg.mathematical_constraints:
        constraints = kg.mathematical_constraints[RelationType.SUBSTRATE_OF]
        print(f"   Found {len(constraints)} constraints")
        for i, c in enumerate(constraints):
            print(f"   {i+1}. {c.name}: {c.functional_form}")
    else:
        print("   ✗ No constraints found for SUBSTRATE_OF!")
    
    # Check the actual MDP behavior
    print("\n2. Testing MDP transition with SUBSTRATE_OF:")
    mdp = BiologicalMDP(kg, config)
    state = mdp.create_initial_state()
    
    # Create a SUBSTRATE_OF action
    from src.mdp import Action
    action = Action(
        action_type=ActionType.ADD_ENTITY,
        entity_id="enzyme_HK1",
        relation_type=RelationType.SUBSTRATE_OF,
        position="root"
    )
    
    print(f"   Before transition: {state.mechanism_tree.to_expression()}")
    
    # Check what happens in _apply_add_entity
    print("\n3. Checking constraint retrieval:")
    constraints = kg.get_constraints_for_relation(RelationType.SUBSTRATE_OF)
    print(f"   get_constraints_for_relation returned: {constraints}")
    
    if not constraints:
        print("   ✗ PROBLEM: No constraints returned for SUBSTRATE_OF!")
        print("\n4. Checking available relation types with constraints:")
        for rel_type in RelationType:
            constraints = kg.get_constraints_for_relation(rel_type)
            if constraints:
                print(f"   ✓ {rel_type.value}: {len(constraints)} constraints")
                if constraints:
                    print(f"      First: {constraints[0].functional_form[:50]}...")
    
    # Try with CATALYSIS instead
    print("\n5. Testing with CATALYSIS (which works):")
    
    # Find an enzyme that can catalyze
    catalysis_actions = []
    for entity_id in list(kg.entities.keys())[:10]:
        # Check if this entity has CATALYSIS relationships
        if kg.graph.has_node(entity_id):
            for neighbor in kg.graph.neighbors(entity_id):
                edge_data = kg.graph.get_edge_data(entity_id, neighbor)
                if edge_data and edge_data.get('relation_type') == 'catalysis':
                    catalysis_actions.append((entity_id, neighbor))
                    break
    
    if catalysis_actions:
        print(f"   Found {len(catalysis_actions)} entities with catalysis")
        entity_id, target = catalysis_actions[0]
        print(f"   Testing with {entity_id} -> {target}")
        
        action = Action(
            action_type=ActionType.ADD_ENTITY,
            entity_id=entity_id,
            relation_type=RelationType.CATALYSIS,
            position="root"
        )
        
        new_state = mdp.transition(state, action)
        print(f"   After transition: {new_state.mechanism_tree.to_expression()}")
        print(f"   Complexity: {new_state.mechanism_tree.get_complexity()}")
    
    # Check the actual graph structure
    print("\n6. Analyzing graph structure for valid actions:")
    print(f"   Total nodes: {kg.graph.number_of_nodes()}")
    print(f"   Total edges: {kg.graph.number_of_edges()}")
    
    # Sample some edges to see their structure
    edges_by_type = {}
    for u, v, data in list(kg.graph.edges(data=True))[:50]:
        rel_type = data.get('relation_type', 'unknown')
        if rel_type not in edges_by_type:
            edges_by_type[rel_type] = []
        edges_by_type[rel_type].append((u, v))
    
    print("\n   Edge types found:")
    for rel_type, edges in edges_by_type.items():
        print(f"   - {rel_type}: {len(edges)} edges")
        
    # The real issue: Check if RelationType enum values match graph edge types
    print("\n7. Checking RelationType enum vs graph edge types:")
    
    # Get all unique relation types from graph
    graph_rel_types = set()
    for u, v, data in kg.graph.edges(data=True):
        graph_rel_types.add(data.get('relation_type'))
    
    print(f"   Graph relation types: {sorted(graph_rel_types)[:10]}...")
    
    # Get all RelationType enum values
    enum_values = [r.value for r in RelationType]
    print(f"   RelationType enum values: {enum_values[:10]}...")
    
    # Check for mismatches
    print("\n   Checking for mismatches:")
    for graph_type in list(graph_rel_types)[:20]:
        if graph_type not in enum_values:
            print(f"   ✗ Graph has '{graph_type}' but not in RelationType enum")
        else:
            # Check if it has constraints
            try:
                rel_enum = RelationType(graph_type)
                constraints = kg.get_constraints_for_relation(rel_enum)
                if not constraints:
                    print(f"   ⚠ '{graph_type}' in enum but no constraints")
            except:
                print(f"   ✗ Can't create RelationType from '{graph_type}'")

if __name__ == "__main__":
    diagnose_constraint_issue()