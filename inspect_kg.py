#!/usr/bin/env python3
"""
Script to inspect the knowledge graph and show all entity IDs organized by type
"""

import yaml
import sys
sys.path.append('.')
from kg_builder import ComprehensiveKGBuilder
from collections import defaultdict

def inspect_knowledge_graph():
    # Load config
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Building knowledge graph...")
    builder = ComprehensiveKGBuilder(config)
    kg = builder.build_complete_kg()
    
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH ENTITY INSPECTION")
    print("="*80)
    
    # Group entities by type
    entities_by_type = defaultdict(list)
    for entity_id, entity in kg.entities.items():
        entities_by_type[entity.entity_type].append((entity_id, entity.name))
    
    # Print entities by type
    for entity_type in sorted(entities_by_type.keys()):
        entities = sorted(entities_by_type[entity_type])
        print(f"\n{entity_type.upper()} ({len(entities)} entities):")
        print("-" * 40)
        
        # Show first 10 and last 5 for each type
        if len(entities) <= 15:
            for eid, name in entities:
                print(f"  {eid:<50} -> {name}")
        else:
            # Show first 10
            for eid, name in entities[:10]:
                print(f"  {eid:<50} -> {name}")
            print(f"  ... ({len(entities) - 15} more entities)")
            # Show last 5
            for eid, name in entities[-5:]:
                print(f"  {eid:<50} -> {name}")
    
    print("\n" + "="*80)
    print("RELATIONSHIPS SUMMARY")
    print("="*80)
    
    # Count relationships by type
    rel_counts = defaultdict(int)
    rel_examples = defaultdict(list)
    
    for rel in kg.relationships:
        rel_type = rel.relation_type.value
        rel_counts[rel_type] += 1
        if len(rel_examples[rel_type]) < 3:
            rel_examples[rel_type].append(f"{rel.source} -> {rel.target}")
    
    print(f"\nTotal relationships: {len(kg.relationships)}")
    print("\nRelationship types:")
    for rel_type in sorted(rel_counts.keys()):
        count = rel_counts[rel_type]
        print(f"\n  {rel_type}: {count} relationships")
        for example in rel_examples[rel_type]:
            print(f"    Example: {example}")
    
    print("\n" + "="*80)
    print("ENTITY ID PATTERNS")
    print("="*80)
    
    # Show ID patterns for debugging
    print("\nSample entity IDs by type:")
    for entity_type in ['enzyme', 'substrate', 'product', 'inhibitor', 'drug']:
        print(f"\n{entity_type.upper()}:")
        samples = [eid for eid, e in kg.entities.items() if e.entity_type == entity_type][:5]
        for sample in samples:
            print(f"  {sample}")
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total entities: {len(kg.entities)}")
    print(f"Total relationships: {len(kg.relationships)}")
    print(f"Average relationships per entity: {len(kg.relationships) / len(kg.entities):.2f}")
    print(f"Entity types: {len(entities_by_type)}")
    print(f"Relationship types: {len(rel_counts)}")

if __name__ == "__main__":
    inspect_knowledge_graph()