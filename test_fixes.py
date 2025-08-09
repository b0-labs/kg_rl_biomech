#!/usr/bin/env python3
"""Test that all fixes work properly"""

import time
import numpy as np
from train import *
from tqdm import tqdm

def test_fixed_training():
    """Test the fixed training system"""
    
    print("Loading configuration...")
    config = load_config('config.yml')
    
    print("Creating minimal knowledge graph...")
    kg = KnowledgeGraph(config)
    from src.knowledge_graph import BiologicalEntity, BiologicalRelationship, RelationType
    
    # Add entities
    entities = [
        BiologicalEntity("enzyme1", "Enzyme", "enzyme", {}, 1.0),
        BiologicalEntity("substrate1", "Substrate", "substrate", {}, 1.0),
        BiologicalEntity("product1", "Product", "product", {}, 0.95),
        BiologicalEntity("inhibitor1", "Inhibitor", "inhibitor", {}, 0.9),
    ]
    for entity in entities:
        kg.add_entity(entity)
    
    # Add relationships
    kg.add_relationship(BiologicalRelationship(
        "enzyme1", "substrate1", RelationType.CATALYSIS,
        {}, ["michaelis_menten"], 1.0
    ))
    kg.add_relationship(BiologicalRelationship(
        "inhibitor1", "enzyme1", RelationType.COMPETITIVE_INHIBITION,
        {}, ["competitive_mm"], 0.9
    ))
    
    print("Creating training components...")
    mdp = BiologicalMDP(kg, config)
    policy_net = PolicyNetwork(kg, config)
    value_net = ValueNetwork(kg, config)
    reward_fn = RewardFunction(kg, config)
    trainer = PPOTrainer(mdp, policy_net, value_net, reward_fn, config)
    
    print("Generating synthetic data...")
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print("\n" + "="*70)
    print("TESTING FIXED TRAINING SYSTEM")
    print("="*70)
    
    # Track metrics
    episode_times = []
    best_scores = []
    complexities = []
    stuck_counter = 0
    last_best = -1000
    
    # Run training with progress bar
    num_episodes = 100
    pbar = tqdm(range(num_episodes), desc="Testing Fixed Training")
    
    for episode in pbar:
        # Time the episode
        start = time.perf_counter()
        stats = trainer.train_episode(system.data_X, system.data_y)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        
        episode_times.append(elapsed)
        best_scores.append(stats['best_score'])
        complexities.append(stats.get('final_complexity', 0))
        
        # Check if stuck
        if abs(stats['best_score'] - last_best) < 1e-6:
            stuck_counter += 1
        else:
            stuck_counter = 0
        last_best = stats['best_score']
        
        # Update networks less frequently
        if episode > 0 and episode % 50 == 0 and len(trainer.replay_buffer) >= 100:
            update_start = time.perf_counter()
            trainer.update_networks()
            update_time = (time.perf_counter() - update_start) * 1000
            print(f"\n→ Network update at episode {episode} took {update_time:.0f}ms")
        
        # Get best complexity from trainer
        best_cmplx = 0
        if hasattr(trainer, 'best_mechanism') and trainer.best_mechanism:
            best_cmplx = trainer.best_mechanism.mechanism_tree.get_complexity()
        
        # Update progress bar
        pbar.set_postfix({
            'best': f"{stats['best_score']:.3f}",
            'cmplx': best_cmplx,
            'stuck': stuck_counter,
            'ms': f"{elapsed:.0f}"
        })
        
        # Log significant improvements
        if stats['best_score'] > -900 and stats['best_score'] > best_scores[max(0, episode-1)]:
            print(f"\n✅ Episode {episode}: New best score = {stats['best_score']:.3f}, complexity = {best_cmplx}")
    
    pbar.close()
    
    # Calculate statistics
    times_ms = np.array(episode_times)
    slow_episodes = np.where(times_ms > 500)[0]
    very_slow_episodes = np.where(times_ms > 1000)[0]
    
    print("\n" + "="*70)
    print("TRAINING RESULTS")
    print("="*70)
    
    print(f"\n📊 Performance Statistics:")
    print(f"  • Mean episode time: {np.mean(times_ms):.1f}ms")
    print(f"  • Std episode time: {np.std(times_ms):.1f}ms")
    print(f"  • Max episode time: {np.max(times_ms):.1f}ms")
    print(f"  • Min episode time: {np.min(times_ms):.1f}ms")
    
    print(f"\n🎯 Training Results:")
    print(f"  • Final best score: {trainer.best_score:.3f}")
    print(f"  • Best complexity: {best_cmplx}")
    print(f"  • Score improved: {trainer.best_score > -1000}")
    
    print(f"\n⚠️  Performance Issues:")
    print(f"  • Slow episodes (>500ms): {len(slow_episodes)}/{num_episodes}")
    print(f"  • Very slow episodes (>1000ms): {len(very_slow_episodes)}/{num_episodes}")
    
    if len(slow_episodes) > 0:
        print(f"  • Slow episodes occurred at: {slow_episodes[:10]}...")
    
    # Check if training was successful
    success = True
    issues = []
    
    if trainer.best_score <= -900:
        success = False
        issues.append("Best score never improved significantly")
    
    if len(slow_episodes) > num_episodes * 0.2:  # More than 20% slow
        success = False
        issues.append(f"Too many slow episodes ({len(slow_episodes)})")
    
    if np.mean(times_ms) > 100:
        success = False
        issues.append(f"Average episode time too high ({np.mean(times_ms):.0f}ms)")
    
    print("\n" + "="*70)
    if success:
        print("✅ TRAINING SYSTEM IS WORKING PROPERLY!")
        print(f"   Best score: {trainer.best_score:.3f}")
        print(f"   Avg time: {np.mean(times_ms):.0f}ms")
        print(f"   Mechanisms found: {best_cmplx > 0}")
    else:
        print("❌ TRAINING SYSTEM STILL HAS ISSUES:")
        for issue in issues:
            print(f"   • {issue}")
    print("="*70)
    
    return success

if __name__ == "__main__":
    success = test_fixed_training()
    exit(0 if success else 1)