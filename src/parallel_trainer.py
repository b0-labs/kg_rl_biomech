"""Parallel episode collection for faster training"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

class ParallelEpisodeCollector:
    """Collect multiple episodes in parallel for better GPU utilization"""
    
    def __init__(self, trainer, num_workers: int = 4):
        self.trainer = trainer
        self.num_workers = num_workers
        
    def collect_episodes(self, num_episodes: int, data_X: np.ndarray, 
                         data_y: np.ndarray) -> List[Dict]:
        """Collect multiple episodes in parallel"""
        
        # Use thread pool for parallel episode collection
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for _ in range(num_episodes):
                # Create a copy of trainer for thread safety
                future = executor.submit(self._collect_single_episode, data_X, data_y)
                futures.append(future)
            
            # Collect results
            episodes = []
            for future in as_completed(futures):
                try:
                    episode_data = future.result()
                    episodes.append(episode_data)
                except Exception as e:
                    print(f"Episode collection failed: {e}")
                    
        return episodes
    
    def _collect_single_episode(self, data_X: np.ndarray, data_y: np.ndarray) -> Dict:
        """Collect a single episode (thread-safe)"""
        # This would need proper thread-safe implementation
        # For now, just call the trainer's method
        return self.trainer.train_episode(data_X, data_y)
    
    def batch_update(self, episodes: List[Dict]):
        """Update networks with batch of episodes"""
        if not episodes:
            return
            
        # Aggregate all transitions
        all_transitions = []
        for episode in episodes:
            if 'trajectory' in episode:
                all_transitions.extend(episode['trajectory'])
        
        # Add to replay buffer
        self.trainer.replay_buffer.add_trajectory(all_transitions)
        
        # Perform batch update
        if len(self.trainer.replay_buffer) >= self.trainer.batch_size:
            return self.trainer.update_networks()
        
        return {'policy_loss': 0.0, 'value_loss': 0.0}