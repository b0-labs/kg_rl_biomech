#!/usr/bin/env python3
"""Fix missing constraints for all relation types"""

from src.knowledge_graph import KnowledgeGraph, MathematicalConstraint, RelationType
import json

def add_missing_constraints(kg):
    """Add mathematical constraints for all relation types that lack them"""
    
    print("Adding missing constraints...")
    
    # Add constraints for SUBSTRATE_OF (substrate binding/consumption)
    if RelationType.SUBSTRATE_OF not in kg.mathematical_constraints or not kg.mathematical_constraints[RelationType.SUBSTRATE_OF]:
        kg.mathematical_constraints[RelationType.SUBSTRATE_OF] = [
            MathematicalConstraint(
                "substrate_binding",
                "(v_max * S) / (k_m + S)",  # Simple Michaelis-Menten for substrate
                {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0)}
            ),
            MathematicalConstraint(
                "substrate_consumption",
                "k_cat * E * S / (k_m + S)",  # Enzyme-catalyzed consumption
                {"k_cat": (0.01, 1000.0), "E": (0.0001, 10.0), "k_m": (0.000001, 1000.0)}
            )
        ]
        print(f"  Added {len(kg.mathematical_constraints[RelationType.SUBSTRATE_OF])} constraints for SUBSTRATE_OF")
    
    # Add constraints for PRODUCT_OF (product formation)
    if RelationType.PRODUCT_OF not in kg.mathematical_constraints or not kg.mathematical_constraints[RelationType.PRODUCT_OF]:
        kg.mathematical_constraints[RelationType.PRODUCT_OF] = [
            MathematicalConstraint(
                "product_formation",
                "(v_max * S) / (k_m + S)",  # Product formation rate
                {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0)}
            ),
            MathematicalConstraint(
                "product_inhibition",
                "(v_max * S) / ((k_m + S) * (1.0 + P / k_p))",  # With product inhibition
                {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "k_p": (0.001, 100.0)}
            )
        ]
        print(f"  Added {len(kg.mathematical_constraints[RelationType.PRODUCT_OF])} constraints for PRODUCT_OF")
    
    # Add constraints for BINDS_TO (simple binding)
    if RelationType.BINDS_TO not in kg.mathematical_constraints or not kg.mathematical_constraints[RelationType.BINDS_TO]:
        kg.mathematical_constraints[RelationType.BINDS_TO] = [
            MathematicalConstraint(
                "simple_binding",
                "B_max * S / (k_d + S)",  # Simple binding equation
                {"B_max": (0.001, 100.0), "k_d": (0.000001, 100.0)}
            ),
            MathematicalConstraint(
                "cooperative_binding",
                "B_max * (S ** n) / ((k_d ** n) + (S ** n))",  # Cooperative binding
                {"B_max": (0.001, 100.0), "k_d": (0.000001, 100.0), "n": (1.0, 4.0)}
            )
        ]
        print(f"  Added {len(kg.mathematical_constraints[RelationType.BINDS_TO])} constraints for BINDS_TO")
    
    # Add constraints for TRANSPORTS
    if RelationType.TRANSPORTS not in kg.mathematical_constraints or not kg.mathematical_constraints[RelationType.TRANSPORTS]:
        kg.mathematical_constraints[RelationType.TRANSPORTS] = [
            MathematicalConstraint(
                "active_transport",
                "(v_max * S) / (k_m + S)",  # Active transport
                {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0)}
            )
        ]
        print(f"  Added {len(kg.mathematical_constraints[RelationType.TRANSPORTS])} constraints for TRANSPORTS")
    
    # Add constraints for BIOMARKER_FOR
    if RelationType.BIOMARKER_FOR not in kg.mathematical_constraints or not kg.mathematical_constraints[RelationType.BIOMARKER_FOR]:
        kg.mathematical_constraints[RelationType.BIOMARKER_FOR] = [
            MathematicalConstraint(
                "biomarker_response",
                "baseline + delta * S / (EC50 + S)",  # Biomarker response
                {"baseline": (0.0, 10.0), "delta": (0.1, 100.0), "EC50": (0.001, 100.0)}
            )
        ]
        print(f"  Added constraints for BIOMARKER_FOR")
    
    # Add constraints for CAUSES_DISEASE
    if RelationType.CAUSES_DISEASE not in kg.mathematical_constraints or not kg.mathematical_constraints[RelationType.CAUSES_DISEASE]:
        kg.mathematical_constraints[RelationType.CAUSES_DISEASE] = [
            MathematicalConstraint(
                "disease_progression",
                "severity * (1.0 - exp(-k * S))",  # Disease progression model
                {"severity": (0.1, 10.0), "k": (0.001, 1.0)}
            )
        ]
        print(f"  Added constraints for CAUSES_DISEASE")
    
    # Add constraints for TREATS
    if RelationType.TREATS not in kg.mathematical_constraints or not kg.mathematical_constraints[RelationType.TREATS]:
        kg.mathematical_constraints[RelationType.TREATS] = [
            MathematicalConstraint(
                "treatment_effect",
                "E_max * S / (EC50 + S)",  # Treatment effect
                {"E_max": (0.1, 1.0), "EC50": (0.001, 100.0)}
            )
        ]
        print(f"  Added constraints for TREATS")
    
    # Add constraints for LOCATED_IN, PART_OF (structural relationships - use constant)
    for rel in [RelationType.LOCATED_IN, RelationType.PART_OF]:
        if rel not in kg.mathematical_constraints or not kg.mathematical_constraints[rel]:
            kg.mathematical_constraints[rel] = [
                MathematicalConstraint(
                    "structural_constant",
                    "c",  # Simple constant for structural relationships
                    {"c": (0.1, 10.0)}
                )
            ]
            print(f"  Added constraints for {rel.value}")
    
    return kg

def fix_and_save_kg():
    """Load KG, fix constraints, and save"""
    from train import load_config
    
    config = load_config('config.yml')
    
    # Load the cached KG
    kg = KnowledgeGraph(config)
    try:
        kg.load('./kg_cache/comprehensive_kg.json')
        print(f"Loaded KG with {len(kg.entities)} entities")
    except:
        print("Creating new KG...")
        # If no cache, create minimal KG
        from src.knowledge_graph import BiologicalEntity, BiologicalRelationship
        kg.add_entity(BiologicalEntity("enzyme1", "Enzyme", "enzyme", {}, 1.0))
        kg.add_entity(BiologicalEntity("substrate1", "Substrate", "substrate", {}, 1.0))
        kg.add_relationship(BiologicalRelationship(
            "enzyme1", "substrate1", RelationType.CATALYSIS,
            {}, ["michaelis_menten"], 1.0
        ))
    
    # Add missing constraints
    kg = add_missing_constraints(kg)
    
    # Save the fixed KG
    kg.save('./kg_cache/comprehensive_kg_fixed.json')
    print("\nSaved fixed KG to ./kg_cache/comprehensive_kg_fixed.json")
    
    # Test that it works
    print("\nTesting fixed KG...")
    from src.mdp import BiologicalMDP, ActionType
    from src.synthetic_data import SyntheticDataGenerator, SystemType
    
    mdp = BiologicalMDP(kg, config)
    state = mdp.create_initial_state()
    
    # Get actions
    valid_actions = mdp.get_valid_actions(state)
    add_actions = [a for a in valid_actions if a.action_type == ActionType.ADD_ENTITY]
    
    print(f"Valid actions: {len(valid_actions)}")
    print(f"Add entity actions: {len(add_actions)}")
    
    # Test a few actions
    tested = set()
    for action in add_actions[:10]:
        if action.relation_type not in tested:
            tested.add(action.relation_type)
            new_state = mdp.transition(state, action)
            expr = new_state.mechanism_tree.to_expression()
            print(f"\n{action.relation_type.value}:")
            print(f"  Entity: {action.entity_id}")
            print(f"  Expression: {expr[:80]}...")
            print(f"  Complexity: {new_state.mechanism_tree.get_complexity()}")
            
            if expr == "1.0":
                print("  ⚠ WARNING: Still returning 1.0!")

if __name__ == "__main__":
    fix_and_save_kg()