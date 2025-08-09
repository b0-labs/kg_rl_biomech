#!/usr/bin/env python3
"""Rebuild the knowledge graph with all constraints properly set"""

import os
import sys
from train import load_config, setup_logging
from src.knowledge_graph import KnowledgeGraph, BiologicalEntity, BiologicalRelationship, RelationType

def rebuild_kg():
    """Rebuild the knowledge graph from scratch with all constraints"""
    
    config = load_config('config.yml')
    logger = setup_logging(config)
    
    logger.info("=" * 70)
    logger.info("REBUILDING KNOWLEDGE GRAPH WITH ALL CONSTRAINTS")
    logger.info("=" * 70)
    
    # Create cache directory if it doesn't exist
    os.makedirs('./kg_cache', exist_ok=True)
    
    # Try to build using kg_builder if available
    try:
        # Try enhanced builder first
        try:
            from kg_builder_enhanced import EnhancedKGBuilder
            logger.info("Using enhanced KG builder...")
            builder = EnhancedKGBuilder(config)
            kg = builder.build_complete_graph()
        except ImportError:
            # Try standard builder
            try:
                from kg_builder import ComprehensiveKGBuilder
                logger.info("Using standard KG builder...")
                builder = ComprehensiveKGBuilder(config)
                kg = builder.build_complete_kg()
            except ImportError:
                logger.info("No kg_builder found, creating default KG...")
                kg = None
    except Exception as e:
        logger.warning(f"Builder failed: {e}, creating default KG...")
        kg = None
    
    # If no builder worked, create a default KG with proper constraints
    if kg is None:
        logger.info("Creating default knowledge graph with basic entities...")
        kg = KnowledgeGraph(config)
        
        # Add basic entities for testing
        
        # Enzymes
        for i in range(1, 6):
            kg.add_entity(BiologicalEntity(
                f"enzyme_{i}", f"Enzyme {i}", "enzyme", 
                {"catalytic_efficiency": 1e6}, 1.0
            ))
        
        # Substrates
        for i in range(1, 11):
            kg.add_entity(BiologicalEntity(
                f"substrate_{i}", f"Substrate {i}", "substrate",
                {"molecular_weight": 180}, 1.0
            ))
        
        # Products
        for i in range(1, 6):
            kg.add_entity(BiologicalEntity(
                f"product_{i}", f"Product {i}", "product",
                {"molecular_weight": 162}, 1.0
            ))
        
        # Inhibitors
        for i in range(1, 4):
            kg.add_entity(BiologicalEntity(
                f"inhibitor_{i}", f"Inhibitor {i}", "inhibitor",
                {"ki": 0.1}, 0.9
            ))
        
        # Add relationships
        kg.add_relationship(BiologicalRelationship(
            "enzyme_1", "substrate_1", RelationType.CATALYSIS,
            {}, ["michaelis_menten"], 1.0
        ))
        
        kg.add_relationship(BiologicalRelationship(
            "substrate_1", "enzyme_1", RelationType.SUBSTRATE_OF,
            {}, ["substrate_binding"], 1.0
        ))
        
        kg.add_relationship(BiologicalRelationship(
            "product_1", "enzyme_1", RelationType.PRODUCT_OF,
            {}, ["product_formation"], 1.0
        ))
        
        kg.add_relationship(BiologicalRelationship(
            "inhibitor_1", "enzyme_1", RelationType.COMPETITIVE_INHIBITION,
            {}, ["competitive_mm"], 0.9
        ))
    
    # Verify constraints are present
    logger.info("\nVerifying mathematical constraints:")
    constraint_counts = {}
    for rel_type in RelationType:
        constraints = kg.get_constraints_for_relation(rel_type)
        if constraints:
            constraint_counts[rel_type.value] = len(constraints)
            logger.info(f"  {rel_type.value}: {len(constraints)} constraints")
    
    if not constraint_counts:
        logger.error("NO CONSTRAINTS FOUND! This is a problem.")
        return False
    
    # Save the KG
    cache_path = config.get('kg_builder', {}).get('cache_path', './kg_cache/comprehensive_kg.json')
    kg.save(cache_path)
    logger.info(f"\nSaved knowledge graph to: {cache_path}")
    logger.info(f"  Entities: {len(kg.entities)}")
    logger.info(f"  Relationships: {len(kg.relationships)}")
    logger.info(f"  Constraint types: {len(constraint_counts)}")
    
    # Test that it loads correctly
    logger.info("\nTesting reload...")
    kg_test = KnowledgeGraph(config)
    kg_test.load(cache_path)
    
    # Verify constraints after reload
    test_rel = RelationType.SUBSTRATE_OF
    test_constraints = kg_test.get_constraints_for_relation(test_rel)
    if test_constraints:
        logger.info(f"✓ After reload, {test_rel.value} has {len(test_constraints)} constraints")
        logger.info(f"  First constraint: {test_constraints[0].functional_form[:50]}...")
    else:
        logger.error(f"✗ After reload, {test_rel.value} has NO constraints!")
        return False
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ KNOWLEDGE GRAPH REBUILT SUCCESSFULLY")
    logger.info("=" * 70)
    return True

if __name__ == "__main__":
    success = rebuild_kg()
    sys.exit(0 if success else 1)