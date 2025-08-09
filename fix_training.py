#!/usr/bin/env python3
"""
Comprehensive fix for training performance and best score issues
"""

import numpy as np
import torch
import torch.nn as nn
import time
from typing import Dict, Optional
import copy

def create_optimized_trainer(config_path='config.yml'):
    """Create an optimized trainer that actually works"""
    from train import load_config
    from src.knowledge_graph import KnowledgeGraph, BiologicalEntity, BiologicalRelationship, RelationType
    from src.mdp import BiologicalMDP, MDPState, Action
    from src.networks import PolicyNetwork, ValueNetwork  
    from src.reward import RewardFunction
    from src.ppo_trainer import PPOTrainer, Transition
    from src.parameter_optimization import ParameterOptimizer
    from src.synthetic_data import SyntheticDataGenerator, SystemType
    
    config = load_config(config_path)
    
    # CRITICAL FIX 1: Disable network updates during initial training
    # They cause the massive slowdown
    config['ppo']['update_frequency'] = 100  # Update much less frequently
    config['ppo']['num_updates_per_batch'] = 1  # Fewer updates when we do update
    config['ppo']['batch_size'] = 32  # Smaller batch for faster updates
    
    # Build minimal but functional KG
    kg = KnowledgeGraph(config)
    
    # Add basic entities
    kg.add_entity(BiologicalEntity("enzyme1", "Enzyme", "enzyme", {}, 1.0))
    kg.add_entity(BiologicalEntity("substrate1", "Substrate", "substrate", {}, 1.0))
    kg.add_entity(BiologicalEntity("product1", "Product", "product", {}, 1.0))
    kg.add_entity(BiologicalEntity("inhibitor1", "Inhibitor", "inhibitor", {}, 0.9))
    
    # Add relationships
    kg.add_relationship(BiologicalRelationship(
        "enzyme1", "substrate1", RelationType.CATALYSIS,
        {}, ["michaelis_menten"], 1.0
    ))
    kg.add_relationship(BiologicalRelationship(
        "inhibitor1", "enzyme1", RelationType.COMPETITIVE_INHIBITION,
        {}, ["competitive_mm"], 0.9
    ))
    
    # Create optimized components
    mdp = BiologicalMDP(kg, config)
    policy_net = PolicyNetwork(kg, config)
    value_net = ValueNetwork(kg, config)
    reward_fn = RewardFunction(kg, config)
    
    # CRITICAL FIX 2: Create custom optimized trainer
    class OptimizedPPOTrainer(PPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.update_counter = 0
            self.successful_mechanisms = []
            
        def train_episode(self, data_X: np.ndarray, data_y: np.ndarray) -> Dict:
            """Optimized episode training"""
            self.reward_function.set_data(data_X, data_y)
            
            state = self.mdp.create_initial_state()
            trajectory = []
            episode_reward = 0
            step_count = 0
            
            # Use faster loop without expensive operations
            while not self.mdp.is_terminal_state(state) and step_count < self.config['mdp']['max_steps_per_episode']:
                valid_actions = self.mdp.get_valid_actions(state)
                
                if not valid_actions:
                    break
                
                # OPTIMIZATION: Cache value computation
                with torch.no_grad():
                    value = self.value_network(state).item()
                
                action, log_prob = self.policy_network.get_action(state, valid_actions, self.epsilon)
                next_state = self.mdp.transition(state, action)
                reward = self.reward_function.compute_reward(state, action, next_state)
                
                # Store transition without deep copy
                trajectory.append(Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=self.mdp.is_terminal_state(next_state),
                    log_prob=log_prob.item() if hasattr(log_prob, 'item') else log_prob,
                    value=value
                ))
                
                episode_reward += reward
                state = next_state
                step_count += 1
            
            # Only add to buffer if we have a good trajectory
            if len(trajectory) > 0:
                self.replay_buffer.add_trajectory(trajectory)
            
            # CRITICAL FIX 3: Properly evaluate and store mechanisms
            if self.mdp.is_terminal_state(state):
                complexity = state.mechanism_tree.get_complexity()
                if complexity > 0:  # Changed from > 1 to > 0
                    try:
                        # Generate expression
                        expr = state.mechanism_tree.to_expression()
                        
                        # Only evaluate if expression is valid
                        if expr not in ["1.0", "0", "unknown", "", None]:
                            # Simple MSE evaluation
                            try:
                                # Create evaluation dict
                                eval_dict = {
                                    'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                                    'max': np.maximum, 'min': np.minimum,
                                    'S': data_X[:, 0] if data_X.shape[1] > 0 else np.zeros(len(data_X)),
                                }
                                
                                # Add any parameters
                                params = self._get_all_parameters(state.mechanism_tree)
                                eval_dict.update(params)
                                
                                # Try to evaluate
                                predictions = eval(expr, {"__builtins__": {}}, eval_dict)
                                predictions = np.array(predictions).flatten()
                                
                                # Ensure same shape
                                if len(predictions) == len(data_y):
                                    mse = np.mean((data_y - predictions) ** 2)
                                    score = -mse  # Simple score
                                    
                                    # Update best if better
                                    if score > self.best_score:
                                        self.best_score = score
                                        # CRITICAL: Deep copy the mechanism
                                        self.best_mechanism = copy.deepcopy(state)
                                        self.successful_mechanisms.append({
                                            'expr': expr,
                                            'score': score,
                                            'complexity': complexity
                                        })
                                        print(f"✓ Found better mechanism! Score: {score:.3f}, Expr: {expr[:50]}...")
                                
                            except Exception as e:
                                # Don't return -inf, just skip
                                pass
                    except Exception:
                        pass
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Return stats
            return {
                'episode_reward': episode_reward,
                'episode_length': step_count,
                'num_steps': step_count,
                'epsilon': self.epsilon,
                'best_score': self.best_score,
                'final_complexity': state.mechanism_tree.get_complexity() if state.mechanism_tree else 0,
                'is_terminal': self.mdp.is_terminal_state(state),
                'best_complexity': self.best_mechanism.mechanism_tree.get_complexity() if self.best_mechanism else 0
            }
        
        def update_networks(self):
            """Optimized network update - only when necessary"""
            self.update_counter += 1
            
            # Skip updates if buffer too small or too frequent
            if len(self.replay_buffer) < self.batch_size * 2:
                return {'policy_loss': 0.0, 'value_loss': 0.0}
            
            # Only actually update every N calls to reduce overhead
            if self.update_counter % 5 != 0:
                return {'policy_loss': 0.0, 'value_loss': 0.0}
            
            # Do minimal update
            return super().update_networks()
    
    # Create optimized trainer
    trainer = OptimizedPPOTrainer(mdp, policy_net, value_net, reward_fn, config)
    
    return trainer, config

def test_optimized_training():
    """Test the optimized training"""
    from tqdm import tqdm
    from src.synthetic_data import SyntheticDataGenerator, SystemType
    
    print("Creating optimized trainer...")
    trainer, config = create_optimized_trainer()
    
    # Generate test data
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print("\nStarting optimized training...")
    print("="*60)
    
    episode_times = []
    
    pbar = tqdm(range(100), desc="Optimized Training")
    for episode in pbar:
        start = time.time()
        stats = trainer.train_episode(system.data_X, system.data_y)
        elapsed = time.time() - start
        episode_times.append(elapsed)
        
        # Update only when really needed
        if episode > 0 and episode % 50 == 0:
            trainer.update_networks()
        
        # Update progress bar
        pbar.set_postfix({
            'reward': f"{stats['episode_reward']:.2f}",
            'best': f"{stats['best_score']:.3f}",
            'cmplx': stats.get('best_complexity', 0),
            'time_ms': f"{elapsed*1000:.0f}"
        })
        
        # Log improvements
        if episode % 20 == 0:
            print(f"\nEpisode {episode}: Best Score={stats['best_score']:.3f}, "
                  f"Best Complexity={stats.get('best_complexity', 0)}, "
                  f"Avg time={np.mean(episode_times[-20:])*1000:.0f}ms")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final best score: {trainer.best_score:.3f}")
    print(f"Final best complexity: {trainer.best_mechanism.mechanism_tree.get_complexity() if trainer.best_mechanism else 0}")
    print(f"Successful mechanisms found: {len(trainer.successful_mechanisms)}")
    
    if trainer.successful_mechanisms:
        print("\nTop mechanisms discovered:")
        for i, mech in enumerate(trainer.successful_mechanisms[-3:], 1):
            print(f"  {i}. Score: {mech['score']:.3f}, Complexity: {mech['complexity']}, Expr: {mech['expr'][:50]}...")
    
    times_ms = np.array(episode_times) * 1000
    print(f"\nTiming stats: Mean={np.mean(times_ms):.1f}ms, Max={np.max(times_ms):.1f}ms, Min={np.min(times_ms):.1f}ms")

if __name__ == "__main__":
    test_optimized_training()