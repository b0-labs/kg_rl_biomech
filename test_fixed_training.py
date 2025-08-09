#!/usr/bin/env python3
"""Test training with the fixed knowledge graph"""

import time
import numpy as np
from train import *
from tqdm import tqdm

def test_with_fixed_kg():
    """Test that training works with fixed KG"""
    
    print("="*70)
    print("TESTING TRAINING WITH FIXED KNOWLEDGE GRAPH")
    print("="*70)
    
    config = load_config('config.yml')
    logger = setup_logging(config)
    
    # Load the FIXED KG
    print("\nLoading fixed knowledge graph...")
    kg = create_knowledge_graph(config, logger, use_cache=True)
    print(f"Loaded KG with {len(kg.entities)} entities")
    
    # Create components
    print("Creating training components...")
    mdp = BiologicalMDP(kg, config)
    policy_net = PolicyNetwork(kg, config)
    value_net = ValueNetwork(kg, config)
    reward_fn = RewardFunction(kg, config)
    trainer = PPOTrainer(mdp, policy_net, value_net, reward_fn, config)
    
    # Generate data
    print("Generating synthetic data...")
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print("\n" + "="*70)
    print("RUNNING TRAINING TEST")
    print("="*70)
    
    # Track performance
    episode_times = []
    best_scores = []
    complexities = []
    improvements = 0
    
    # Run episodes
    num_episodes = 50
    pbar = tqdm(range(num_episodes), desc="Training")
    
    for episode in pbar:
        start = time.perf_counter()
        stats = trainer.train_episode(system.data_X, system.data_y)
        elapsed = (time.perf_counter() - start) * 1000
        
        episode_times.append(elapsed)
        best_scores.append(stats['best_score'])
        complexities.append(stats.get('final_complexity', 0))
        
        # Check for improvements
        if episode > 0 and stats['best_score'] > best_scores[episode-1]:
            improvements += 1
            print(f"\n✅ Episode {episode}: Improved! Score: {stats['best_score']:.3f}")
        
        # Update networks occasionally
        if episode > 0 and episode % 30 == 0:
            trainer.update_networks()
        
        # Update progress bar
        best_cmplx = 0
        if hasattr(trainer, 'best_mechanism') and trainer.best_mechanism:
            best_cmplx = trainer.best_mechanism.mechanism_tree.get_complexity()
        
        pbar.set_postfix({
            'best': f"{stats['best_score']:.3f}",
            'cmplx': best_cmplx,
            'ms': f"{elapsed:.0f}"
        })
    
    # Results
    print("\n" + "="*70)
    print("RESULTS WITH FIXED KG")
    print("="*70)
    
    final_best = trainer.best_score
    print(f"\n📊 Final Results:")
    print(f"  • Final best score: {final_best:.3f}")
    print(f"  • Score improvements: {improvements}/{num_episodes}")
    print(f"  • Mean episode time: {np.mean(episode_times):.1f}ms")
    print(f"  • Max complexity reached: {max(complexities)}")
    
    # Check if it actually worked
    success = True
    issues = []
    
    if final_best <= -100:
        success = False
        issues.append(f"Best score stuck at {final_best}")
    
    if improvements == 0:
        success = False
        issues.append("No improvements during training")
    
    if max(complexities) <= 1:
        success = False
        issues.append("No complex mechanisms built")
    
    if np.mean(episode_times) > 200:
        success = False
        issues.append(f"Too slow: {np.mean(episode_times):.0f}ms avg")
    
    print("\n" + "="*70)
    if success:
        print("✅ SUCCESS! Training with fixed KG works properly!")
        print(f"   • Best score improved from -1000 to {final_best:.3f}")
        print(f"   • Built mechanisms with complexity up to {max(complexities)}")
        print(f"   • Fast execution: {np.mean(episode_times):.0f}ms average")
    else:
        print("❌ ISSUES REMAIN:")
        for issue in issues:
            print(f"   • {issue}")
    print("="*70)
    
    return success

if __name__ == "__main__":
    success = test_with_fixed_kg()
    exit(0 if success else 1)