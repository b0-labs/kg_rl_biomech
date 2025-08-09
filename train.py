#!/usr/bin/env python3

import os
import sys
import yaml
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional

from src.knowledge_graph import KnowledgeGraph, BiologicalEntity, BiologicalRelationship, RelationType
from src.mdp import BiologicalMDP
from src.networks import PolicyNetwork, ValueNetwork
from src.reward import RewardFunction
from src.ppo_trainer import PPOTrainer
from src.parameter_optimization import ParameterOptimizer
from src.synthetic_data import SyntheticDataGenerator, SystemType
from src.evaluation import EvaluationMetrics
# Use unified knowledge graph loader
from src.kg_loader_unified import KnowledgeGraphLoader, KnowledgeGraphBuilder, DataSourceType, DATA_SOURCES
from src.baselines import RandomSearchBaseline, GeneticProgrammingBaseline, UnconstrainedRLBaseline
from sklearn.model_selection import KFold

def setup_logging(config: Dict) -> logging.Logger:
    log_level = config['logging']['level']
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def _update_mechanism_parameters(node, optimized_params: Dict[str, float]):
    """Recursively update parameters in mechanism tree"""
    # Update this node's parameters
    for param_name, value in optimized_params.items():
        # Check if this parameter belongs to this node
        if param_name in node.parameters:
            node.parameters[param_name] = value
        # Also check with node_id prefix
        elif f"{node.node_id}_{param_name}" in optimized_params:
            node.parameters[param_name] = optimized_params[f"{node.node_id}_{param_name}"]
    
    # Recursively update children
    for child in node.children:
        _update_mechanism_parameters(child, optimized_params)

def create_knowledge_graph(config: Dict, logger: logging.Logger, sources: Optional[List[str]] = None, 
                          use_cache: bool = False, cache_path: Optional[str] = None) -> KnowledgeGraph:
    logger.info("Creating knowledge graph...")
    
    # Check for cached KG if requested
    if use_cache:
        default_cache_path = config.get('kg_builder', {}).get('cache_path', './kg_cache/comprehensive_kg.json')
        cache_file = cache_path or default_cache_path
        
        if os.path.exists(cache_file):
            logger.info(f"Loading pre-built knowledge graph from cache: {cache_file}")
            kg = KnowledgeGraph(config)
            kg.load(cache_file)
            logger.info(f"Loaded cached KG with {len(kg.entities)} entities and {len(kg.relationships)} relationships")
            return kg
        else:
            logger.warning(f"Cache file not found at {cache_file}, building from scratch...")
            # Try to build using kg_builder_enhanced.py (or fallback to kg_builder.py)
            try:
                import sys
                sys.path.append('.')
                try:
                    from kg_builder_enhanced import EnhancedKGBuilder
                    logger.info("Building comprehensive knowledge graph using kg_builder_enhanced.py...")
                    builder = EnhancedKGBuilder(config)
                    kg = builder.build_complete_graph()
                except ImportError:
                    from kg_builder import ComprehensiveKGBuilder
                    logger.info("Building comprehensive knowledge graph using kg_builder.py...")
                    builder = ComprehensiveKGBuilder(config)
                    kg = builder.build_complete_kg()
                
                # Save to cache for future use
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                kg.save(cache_file)
                logger.info(f"Saved knowledge graph to cache: {cache_file}")
                return kg
            except ImportError:
                logger.warning("kg_builder.py not found, falling back to other methods...")
    
    if sources:
        # Use builder pattern for specified sources
        builder = KnowledgeGraphBuilder(config)
        
        for source in sources:
            source_lower = source.lower()
            if source_lower == 'go':
                builder.with_gene_ontology()
            elif source_lower == 'kegg':
                builder.with_kegg_pathways()
            elif source_lower == 'drugbank':
                # Check for auth in config
                auth = config.get('drugbank_auth', {})
                builder.with_drugbank(auth.get('username'), auth.get('password'))
            elif source_lower == 'uniprot':
                builder.with_uniprot()
            elif source_lower == 'chembl':
                builder.with_chembl()
            elif source.endswith('.json'):
                builder.with_custom_json(source)
        
        kg = builder.build()
        logger.info(f"Loaded knowledge graph from sources: {sources}")
        logger.info(f"Available data sources: {', '.join(DATA_SOURCES.keys())}")
    else:
        # Use demo knowledge graph
        kg = KnowledgeGraph(config)
        entities = [
            BiologicalEntity("enzyme1", "Enzyme 1", "enzyme", {"catalytic_efficiency": 1e6}, 1.0),
            BiologicalEntity("substrate1", "Substrate 1", "substrate", {"molecular_weight": 180}, 1.0),
            BiologicalEntity("product1", "Product 1", "product", {"molecular_weight": 162}, 1.0),
            BiologicalEntity("inhibitor1", "Inhibitor 1", "inhibitor", {"ki": 0.1}, 0.9),
            BiologicalEntity("allosteric1", "Allosteric Modulator", "allosteric", {"ka": 0.05}, 0.8),
            BiologicalEntity("drug1", "Drug 1", "drug", {"bioavailability": 0.7}, 0.95),
            BiologicalEntity("receptor1", "Receptor 1", "receptor", {"density": 1e-6}, 1.0),
            BiologicalEntity("transporter1", "Transporter 1", "transporter", {"capacity": 100}, 0.9),
        ]
        
        for entity in entities:
            kg.add_entity(entity)
        
        relationships = [
            BiologicalRelationship("enzyme1", "substrate1", RelationType.CATALYSIS, 
                                  {"reversible": False}, ["michaelis_menten"], 1.0),
            BiologicalRelationship("inhibitor1", "enzyme1", RelationType.COMPETITIVE_INHIBITION,
                                  {"reversible": True}, ["competitive_mm"], 0.9),
            BiologicalRelationship("allosteric1", "enzyme1", RelationType.ALLOSTERIC_REGULATION,
                                  {"positive": True}, ["allosteric_hill"], 0.8),
            BiologicalRelationship("drug1", "receptor1", RelationType.BINDING,
                                  {"affinity": 1e-9}, ["simple_binding"], 0.95),
            BiologicalRelationship("transporter1", "substrate1", RelationType.TRANSPORT,
                                  {"direction": "bidirectional"}, ["facilitated_diffusion"], 0.85),
        ]
        
        for rel in relationships:
            kg.add_relationship(rel)
    
    logger.info(f"Knowledge graph created with {len(kg.entities)} entities and {len(kg.relationships)} relationships")
    return kg

def train_on_synthetic_system(trainer: PPOTrainer, system, optimizer: ParameterOptimizer,
                            evaluator: EvaluationMetrics, num_episodes: int,
                            logger: logging.Logger) -> Dict:
    
    logger.info(f"Training on {system.system_type.value} system (complexity: {system.complexity_level})")
    
    best_mechanism = None
    best_score = float('-inf')
    
    for episode in range(num_episodes):
        episode_stats = trainer.train_episode(system.data_X, system.data_y)
        
        if episode % 10 == 0 and episode > 0:
            trainer.update_networks()
        
        if episode % 100 == 0:
            logger.info(f"Episode {episode}: Reward={episode_stats['episode_reward']:.3f}, "
                       f"Best Score={episode_stats['best_score']:.3f}")
        
        current_best = trainer.get_best_mechanism()
        if current_best and episode_stats['best_score'] > best_score:
            best_score = episode_stats['best_score']
            best_mechanism = current_best
            
            optimized_params, loss = optimizer.optimize_parameters(
                best_mechanism, system.data_X, system.data_y
            )
            
            # Properly update parameters in the mechanism tree
            _update_mechanism_parameters(best_mechanism.mechanism_tree, optimized_params)
        
        # Check convergence
        trainer.update_performance_history(episode_stats['episode_reward'])
        if episode % 10 == 0 and trainer.check_convergence():
            logger.info(f"Converged at episode {episode}")
            break
    
    if best_mechanism:
        predictions = evaluator._evaluate_mechanism_predictions(best_mechanism, system.data_X)
        if predictions is not None:
            errors = evaluator.evaluate_prediction_error(predictions, system.data_y)
            param_recovery = evaluator.evaluate_parameter_recovery(
                optimized_params if 'optimized_params' in locals() else {},
                system.true_parameters
            )
            
            return {
                'mechanism': best_mechanism,
                'errors': errors,
                'param_recovery': param_recovery,
                'true_system': system
            }
    
    return None

def run_cross_validation(systems: List, config: Dict, n_folds: int, logger: logging.Logger) -> Dict:
    """Run k-fold cross validation"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(systems)):
        logger.info(f"Running fold {fold+1}/{n_folds}")
        
        train_systems = [systems[i] for i in train_idx]
        test_systems = [systems[i] for i in test_idx]
        
        # Train on fold  
        # Use cache if available for CV to speed up
        knowledge_graph = create_knowledge_graph(config, logger, use_cache=True)
        mdp = BiologicalMDP(knowledge_graph, config)
        policy_network = PolicyNetwork(knowledge_graph, config)
        value_network = ValueNetwork(knowledge_graph, config)
        reward_function = RewardFunction(knowledge_graph, config)
        trainer = PPOTrainer(mdp, policy_network, value_network, reward_function, config)
        optimizer = ParameterOptimizer(config)
        evaluator = EvaluationMetrics(config)
        
        discovered_mechanisms = []
        for system in train_systems:
            result = train_on_synthetic_system(
                trainer, system, optimizer, evaluator, 
                config['mdp']['max_steps_per_episode'], logger
            )
            if result:
                discovered_mechanisms.append(result['mechanism'])
        
        # Evaluate on test fold
        test_predictions = []
        for system in test_systems:
            if discovered_mechanisms:
                # Use best mechanism for this system type
                predictions = evaluator._evaluate_mechanism_predictions(
                    discovered_mechanisms[0], system.data_X
                )
                if predictions is not None:
                    errors = evaluator.evaluate_prediction_error(predictions, system.data_y)
                    test_predictions.append(errors)
        
        cv_results.append({
            'fold': fold,
            'train_size': len(train_systems),
            'test_size': len(test_systems),
            'test_errors': test_predictions
        })
    
    return {'cv_results': cv_results}

def run_ablation_study(systems: List, config: Dict, logger: logging.Logger) -> Dict:
    """Run ablation study by removing different components"""
    ablation_results = {}
    
    # Full model
    logger.info("Training full model...")
    # Use cache if available for ablation to speed up
    knowledge_graph = create_knowledge_graph(config, logger, use_cache=True)
    mdp = BiologicalMDP(knowledge_graph, config)
    policy_network = PolicyNetwork(knowledge_graph, config)
    value_network = ValueNetwork(knowledge_graph, config)
    reward_function = RewardFunction(knowledge_graph, config)
    trainer = PPOTrainer(mdp, policy_network, value_network, reward_function, config)
    optimizer = ParameterOptimizer(config)
    evaluator = EvaluationMetrics(config)
    
    # Train full model
    full_results = []
    for system in systems[:5]:  # Use subset for faster ablation
        result = train_on_synthetic_system(
            trainer, system, optimizer, evaluator, 
            config['mdp']['max_steps_per_episode'] // 2, logger
        )
        if result:
            full_results.append(result)
    
    ablation_results['full'] = full_results
    
    # Ablate knowledge graph
    logger.info("Ablating knowledge graph...")
    config_no_kg = config.copy()
    config_no_kg['reward']['lambda_plausibility'] = 0.0
    trainer_no_kg = PPOTrainer(mdp, policy_network, value_network, 
                               RewardFunction(knowledge_graph, config_no_kg), config_no_kg)
    
    no_kg_results = []
    for system in systems[:5]:
        result = train_on_synthetic_system(
            trainer_no_kg, system, optimizer, evaluator, 
            config['mdp']['max_steps_per_episode'] // 2, logger
        )
        if result:
            no_kg_results.append(result)
    
    ablation_results['no_kg'] = no_kg_results
    
    # Ablate GNN (use simple MLP instead)
    logger.info("Ablating GNN architecture...")
    # This would require a simpler network architecture
    # For now, we'll reduce attention heads
    config_no_gnn = config.copy()
    config_no_gnn['policy_network']['num_attention_heads'] = 1
    config_no_gnn['policy_network']['num_gnn_layers'] = 1
    policy_simple = PolicyNetwork(knowledge_graph, config_no_gnn)
    value_simple = ValueNetwork(knowledge_graph, config_no_gnn)
    trainer_no_gnn = PPOTrainer(mdp, policy_simple, value_simple, reward_function, config_no_gnn)
    
    no_gnn_results = []
    for system in systems[:5]:
        result = train_on_synthetic_system(
            trainer_no_gnn, system, optimizer, evaluator, 
            config['mdp']['max_steps_per_episode'] // 2, logger
        )
        if result:
            no_gnn_results.append(result)
    
    ablation_results['no_gnn'] = no_gnn_results
    
    return ablation_results

def main():
    parser = argparse.ArgumentParser(description='Train KG-RL for Biological Mechanism Discovery')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config file')
    parser.add_argument('--system-type', type=str, default='enzyme_kinetics',
                       choices=['enzyme_kinetics', 'multi_scale', 'disease_state', 
                               'drug_interaction', 'hierarchical'],
                       help='Type of synthetic system to train on')
    parser.add_argument('--num-systems', type=int, default=10, help='Number of synthetic systems')
    parser.add_argument('--num-episodes', type=int, default=1000, help='Episodes per system')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--kg-sources', nargs='+', 
                       help='Knowledge graph sources to load. Available: GO, KEGG, DrugBank, UniProt, ChEMBL, Reactome, STRING, or path to .json/.obo files')
    parser.add_argument('--use-kg-cache', action='store_true', 
                       help='Use pre-built knowledge graph from cache')
    parser.add_argument('--kg-cache-path', type=str, 
                       help='Path to pre-built knowledge graph cache file')
    parser.add_argument('--build-kg-cache', action='store_true',
                       help='Build comprehensive KG and save to cache (requires kg_builder.py)')
    parser.add_argument('--run-cv', action='store_true', help='Run cross-validation')
    parser.add_argument('--cv-folds', type=int, default=3, help='Number of CV folds')
    parser.add_argument('--run-ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--compare-baselines', action='store_true', help='Compare with baseline methods')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    config = load_config(args.config)
    logger = setup_logging(config)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    logger.info("="*50)
    logger.info("Knowledge Graph-Guided RL for Mechanism Discovery")
    logger.info("="*50)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"System Type: {args.system_type}")
    logger.info(f"Number of Systems: {args.num_systems}")
    logger.info(f"Episodes per System: {args.num_episodes}")
    
    # Handle KG cache building
    if args.build_kg_cache:
        logger.info("Building comprehensive knowledge graph cache...")
        try:
            import sys
            sys.path.append('.')
            try:
                from kg_builder_enhanced import EnhancedKGBuilder
                logger.info("Using enhanced KG builder with comprehensive relationships...")
                builder = EnhancedKGBuilder(config)
                kg = builder.build_complete_graph()
            except ImportError:
                from kg_builder import ComprehensiveKGBuilder
                logger.info("Using standard KG builder...")
                builder = ComprehensiveKGBuilder(config)
                kg = builder.build_complete_kg()
            
            cache_path = args.kg_cache_path or config.get('kg_builder', {}).get('cache_path', './kg_cache/comprehensive_kg.json')
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            kg.save(cache_path)
            logger.info(f"Knowledge graph cache saved to: {cache_path}")
            logger.info(f"Cache contains {len(kg.entities)} entities and {len(kg.relationships)} relationships")
            sys.exit(0)  # Exit after building cache
        except ImportError as e:
            logger.error(f"Failed to import kg_builder: {e}")
            logger.error("Make sure kg_builder.py is in the current directory")
            sys.exit(1)
    
    knowledge_graph = create_knowledge_graph(config, logger, 
                                            sources=args.kg_sources,
                                            use_cache=args.use_kg_cache,
                                            cache_path=args.kg_cache_path)
    
    mdp = BiologicalMDP(knowledge_graph, config)
    
    policy_network = PolicyNetwork(knowledge_graph, config)
    value_network = ValueNetwork(knowledge_graph, config)
    
    reward_function = RewardFunction(knowledge_graph, config)
    
    trainer = PPOTrainer(mdp, policy_network, value_network, reward_function, config)
    
    optimizer = ParameterOptimizer(config)
    
    evaluator = EvaluationMetrics(config)
    
    data_generator = SyntheticDataGenerator(config)
    
    system_type = SystemType[args.system_type.upper()]
    synthetic_systems = data_generator.generate_dataset(system_type, args.num_systems)
    
    logger.info(f"Generated {len(synthetic_systems)} synthetic systems")
    
    # Run cross-validation if requested
    if args.run_cv:
        logger.info(f"\nRunning {args.cv_folds}-fold cross-validation...")
        cv_results = run_cross_validation(synthetic_systems, config, args.cv_folds, logger)
        
        cv_path = os.path.join(args.results_dir, 
                              f'cv_results_{args.system_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(cv_path, 'w') as f:
            json.dump(cv_results, f, indent=2, default=str)
        logger.info(f"CV results saved to {cv_path}")
    
    # Run ablation study if requested
    if args.run_ablation:
        logger.info("\nRunning ablation study...")
        ablation_results = run_ablation_study(synthetic_systems, config, logger)
        
        ablation_path = os.path.join(args.results_dir, 
                                    f'ablation_{args.system_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(ablation_path, 'w') as f:
            json.dump(ablation_results, f, indent=2, default=str)
        logger.info(f"Ablation results saved to {ablation_path}")
    
    all_results = []
    discovered_mechanisms = []
    
    for i, system in enumerate(synthetic_systems):
        logger.info(f"\n{'='*40}")
        logger.info(f"Training on System {i+1}/{len(synthetic_systems)}")
        logger.info(f"{'='*40}")
        
        result = train_on_synthetic_system(
            trainer, system, optimizer, evaluator, 
            args.num_episodes, logger
        )
        
        if result:
            all_results.append(result)
            discovered_mechanisms.append(result['mechanism'])
            
            logger.info(f"System {i+1} Results:")
            logger.info(f"  RMSE: {result['errors']['rmse']:.4f}")
            logger.info(f"  R2: {result['errors']['r2']:.4f}")
            logger.info(f"  Param Recovery: {result['param_recovery']:.3f}")
        
        if (i + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_system_{i+1}.pt')
            trainer.save_checkpoint(checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    overall_metrics = evaluator.evaluate_on_dataset(
        discovered_mechanisms, synthetic_systems, knowledge_graph
    )
    
    logger.info("\n" + "="*50)
    logger.info("Overall Performance Metrics:")
    logger.info("="*50)
    for metric, value in overall_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    summary_stats = evaluator.get_summary_statistics()
    
    results = {
        'config': config,
        'args': vars(args),
        'overall_metrics': overall_metrics,
        'summary_statistics': summary_stats,
        'training_stats': trainer.get_training_stats(),
        'individual_results': [
            {
                'errors': r['errors'],
                'param_recovery': r['param_recovery'],
                'true_mechanism': r['true_system'].mechanism,
                'complexity': r['true_system'].complexity_level
            }
            for r in all_results
        ]
    }
    
    results_path = os.path.join(args.results_dir, 
                               f'results_{args.system_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {results_path}")
    
    final_checkpoint = os.path.join(args.checkpoint_dir, 'final_checkpoint.pt')
    trainer.save_checkpoint(final_checkpoint)
    logger.info(f"Final checkpoint saved to {final_checkpoint}")
    
    logger.info("\nTraining completed successfully!")

if __name__ == "__main__":
    main()