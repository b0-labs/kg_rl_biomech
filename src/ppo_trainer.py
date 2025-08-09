import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import copy

from .mdp import BiologicalMDP, MDPState, Action
from .networks import PolicyNetwork, ValueNetwork
from .reward import RewardFunction, CumulativeRewardTracker
from .knowledge_graph import KnowledgeGraph

@dataclass
class Transition:
    state: MDPState
    action: Action
    reward: float
    next_state: MDPState
    done: bool
    log_prob: float
    value: float

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, transition: Transition):
        self.buffer.append(transition)
    
    def add_trajectory(self, trajectory: List[Transition]):
        for transition in trajectory:
            self.add(transition)
    
    def sample(self, batch_size: int) -> List[Transition]:
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)

class PPOTrainer:
    def __init__(self, mdp: BiologicalMDP, policy_network: PolicyNetwork, 
                 value_network: ValueNetwork, reward_function: RewardFunction, 
                 config: Dict):
        
        self.mdp = mdp
        self.policy_network = policy_network
        self.value_network = value_network
        self.reward_function = reward_function
        self.config = config
        self.knowledge_graph = mdp.knowledge_graph  # Get KG reference from MDP
        
        self.device = torch.device(config['compute']['device'] if torch.cuda.is_available() else 'cpu')
        self.policy_network.to(self.device)
        self.value_network.to(self.device)
        
        # Ensure learning rates are floats (handle string scientific notation)
        policy_lr = float(config['policy_network']['learning_rate'])
        value_lr = float(config['value_network']['learning_rate'])
        
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=policy_lr
        )
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=value_lr
        )
        
        self.clip_epsilon = config['ppo']['clip_epsilon']
        self.lambda_gae = config['ppo']['lambda_gae']
        self.gamma = config['mdp']['discount_factor']
        self.num_updates = config['ppo']['num_updates_per_batch']
        self.batch_size = config['ppo']['batch_size']
        
        buffer_capacity = config['ppo']['replay_buffer_capacity']
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.epsilon = config['exploration']['epsilon_init']
        self.epsilon_min = config['exploration']['epsilon_min']
        self.epsilon_decay = config['exploration']['epsilon_decay']
        
        self.reward_tracker = CumulativeRewardTracker()
        
        self.best_mechanism = None
        # Initialize with a reasonable negative value instead of -inf
        self.best_score = -1000.0  # Changed from float('-inf')
        
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'best_scores': []
        }
        
        # Convergence tracking
        self.convergence_window = 100
        self.convergence_threshold = config['mdp'].get('convergence_threshold', 1e-4)
        self.performance_history = []
        
        # Cache for mechanism expressions to avoid repeated generation
        self.expression_cache = {}
        
        # Skip warmup for now - it might be causing issues
        # self._warmup_networks()
    
    def train_episode(self, data_X: np.ndarray, data_y: np.ndarray) -> Dict:
        self.reward_function.set_data(data_X, data_y)
        
        state = self.mdp.create_initial_state()
        trajectory = []
        episode_reward = 0
        step_count = 0
        
        while not self.mdp.is_terminal_state(state) and step_count < self.config['mdp']['max_steps_per_episode']:
            valid_actions = self.mdp.get_valid_actions(state)
            
            if not valid_actions:
                break
            
            with torch.no_grad():
                value = self.value_network(state).item()
            
            action, log_prob = self.policy_network.get_action(state, valid_actions, self.epsilon)
            
            next_state = self.mdp.transition(state, action)
            
            reward = self.reward_function.compute_reward(state, action, next_state)
            
            transition = Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=self.mdp.is_terminal_state(next_state),
                log_prob=log_prob.item(),
                value=value
            )
            
            trajectory.append(transition)
            episode_reward += reward
            
            state = next_state
            step_count += 1
        
        self.replay_buffer.add_trajectory(trajectory)
        
        self.reward_tracker.add_reward(episode_reward)
        self.reward_tracker.end_episode()
        
        # Evaluate ANY terminal mechanism with complexity > 0
        if self.mdp.is_terminal_state(state) and state.mechanism_tree:
            complexity = state.mechanism_tree.get_complexity()
            if complexity > 0:  # Changed from > 1 to > 0
                try:
                    score = self._evaluate_mechanism(state, data_X, data_y)
                    # Only update if score is valid and better
                    if score > -900 and score > self.best_score:  # Check for valid score
                        self.best_score = score
                        self.best_mechanism = copy.deepcopy(state)  # Deep copy to preserve
                        print(f"✓ New best: score={score:.3f}, complexity={complexity}")
                except Exception as e:
                    # Log the actual error for debugging
                    print(f"⚠ Evaluation failed: {e}")
                    pass
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        episode_stats = {
            'episode_reward': episode_reward,
            'episode_length': step_count,
            'num_steps': step_count,
            'epsilon': self.epsilon,
            'best_score': self.best_score,
            'final_complexity': state.mechanism_tree.get_complexity() if state.mechanism_tree else 0,
            'is_terminal': self.mdp.is_terminal_state(state)
        }
        
        self.training_stats['episode_rewards'].append(episode_reward)
        self.training_stats['episode_lengths'].append(step_count)
        self.training_stats['best_scores'].append(self.best_score)
        
        return episode_stats
    
    def update_networks(self):
        # Skip if not enough data
        if len(self.replay_buffer) < self.batch_size * 2:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        # Use smaller batch for faster updates
        actual_batch_size = min(self.batch_size, 32)
        batch = self.replay_buffer.sample(actual_batch_size)
        
        # Compute advantages and returns more efficiently
        with torch.no_grad():
            advantages = self._compute_advantages(batch)
            returns = self._compute_returns(batch)
        
        old_log_probs = torch.tensor([t.log_prob for t in batch], dtype=torch.float32, device=self.device)
        old_values = torch.tensor([t.value for t in batch], dtype=torch.float32, device=self.device)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        # Limit updates to avoid slowdown
        actual_updates = min(self.num_updates, 2)
        for _ in range(actual_updates):
            current_log_probs = []
            current_values = []
            
            for transition in batch:
                valid_actions = self.mdp.get_valid_actions(transition.state)
                action_idx = valid_actions.index(transition.action) if transition.action in valid_actions else 0
                
                action_probs = self.policy_network(transition.state, valid_actions)
                log_probs = torch.log(action_probs + 1e-8)
                current_log_probs.append(log_probs[action_idx])
                
                value = self.value_network(transition.state)
                current_values.append(value)
            
            current_log_probs = torch.stack(current_log_probs)
            current_values = torch.stack(current_values)
            
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
            value_loss = nn.MSELoss()(current_values, returns_tensor)
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        avg_policy_loss = total_policy_loss / self.num_updates
        avg_value_loss = total_value_loss / self.num_updates
        
        self.training_stats['policy_losses'].append(avg_policy_loss)
        self.training_stats['value_losses'].append(avg_value_loss)
        
        return {'policy_loss': avg_policy_loss, 'value_loss': avg_value_loss}
    
    def _warmup_networks(self):
        """Warmup networks to trigger CUDA kernel compilation"""
        try:
            # Create a dummy state for warmup
            dummy_state = self.mdp.create_initial_state()
            
            # Warmup value network
            with torch.no_grad():
                for _ in range(3):  # Run a few times to ensure warmup
                    _ = self.value_network(dummy_state)
            
            # Warmup policy network
            valid_actions = self.mdp.get_valid_actions(dummy_state)
            if valid_actions:
                with torch.no_grad():
                    for _ in range(3):
                        _ = self.policy_network(dummy_state, valid_actions)
            
            # Clear any GPU cache from warmup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception:
            # Silently fail if warmup has issues
            pass
    
    def _compute_advantages(self, batch: List[Transition]) -> np.ndarray:
        advantages = []
        
        for i, transition in enumerate(batch):
            if transition.done:
                next_value = 0
            else:
                with torch.no_grad():
                    next_value = self.value_network(transition.next_state).item()
            
            delta = transition.reward + self.gamma * next_value - transition.value
            
            advantage = delta
            for j in range(i + 1, min(i + 10, len(batch))):
                if batch[j-1].done:
                    break
                advantage = advantage * self.gamma * self.lambda_gae + (
                    batch[j].reward + self.gamma * self.value_network(batch[j].next_state).item() - batch[j].value
                )
            
            advantages.append(advantage)
        
        advantages = np.array(advantages)
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def _compute_returns(self, batch: List[Transition]) -> np.ndarray:
        returns = []
        
        for i, transition in enumerate(batch):
            G = transition.reward
            discount = self.gamma
            
            for j in range(i + 1, len(batch)):
                if batch[j-1].done:
                    break
                G += discount * batch[j].reward
                discount *= self.gamma
            
            returns.append(G)
        
        return np.array(returns)
    
    def _evaluate_mechanism(self, state: MDPState, data_X: np.ndarray, 
                           data_y: np.ndarray) -> float:
        
        try:
            # Don't cache expressions - just generate them
            mechanism_expr = state.mechanism_tree.to_expression()
            
            # Quick validation - if expression is too simple, return low score
            if mechanism_expr in ["1.0", "0", "unknown", ""]:
                return -100.0
            safe_dict = {
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'max': np.maximum, 'min': np.minimum
            }
            
            for i in range(data_X.shape[1]):
                safe_dict[f'X{i}'] = data_X[:, i]
                safe_dict['S'] = data_X[:, 0]
                if data_X.shape[1] > 1:
                    safe_dict['I'] = data_X[:, 1]
                if data_X.shape[1] > 2:
                    safe_dict['A'] = data_X[:, 2]
            
            all_params = self._get_all_parameters(state.mechanism_tree)
            safe_dict.update(all_params)
            
            predictions = eval(mechanism_expr, {"__builtins__": {}}, safe_dict)
            predictions = np.array(predictions)
            # Ensure predictions are the right shape
            if predictions.ndim == 0:
                predictions = np.full_like(data_y, predictions)
            elif len(predictions) != len(data_y):
                return -500.0  # Shape mismatch
            
            mse = np.mean((data_y - predictions) ** 2)
            
            plausibility = self.knowledge_graph.compute_plausibility_score(
                list(state.mechanism_tree.get_all_entities()),
                state.mechanism_tree.get_all_relations()
            )
            
            complexity_penalty = np.exp(-0.1 * state.mechanism_tree.get_complexity())
            
            score = -mse + 0.5 * plausibility + 0.3 * complexity_penalty
            
            return score
            
        except Exception as e:
            # Return a bad but not infinitely bad score
            return -999.0  # Changed from -inf to allow some updates
    
    def _get_all_parameters(self, node) -> Dict[str, float]:
        params = dict(node.parameters)
        for child in node.children:
            params.update(self._get_all_parameters(child))
        return params
    
    def get_best_mechanism(self) -> Optional[MDPState]:
        # Return a copy only when requested, not during training
        if self.best_mechanism:
            return copy.deepcopy(self.best_mechanism)
        return None
    
    def get_training_stats(self) -> Dict:
        return self.training_stats
    
    def save_checkpoint(self, filepath: str):
        checkpoint = {
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_score': self.best_score,
            'training_stats': self.training_stats
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.best_score = checkpoint['best_score']
        self.training_stats = checkpoint['training_stats']
    
    def check_convergence(self) -> bool:
        """Check if training has converged based on performance history"""
        if len(self.performance_history) < self.convergence_window:
            return False
        
        recent_performance = self.performance_history[-self.convergence_window:]
        
        # Check mean and variance stability
        mean_first_half = np.mean(recent_performance[:self.convergence_window//2])
        mean_second_half = np.mean(recent_performance[self.convergence_window//2:])
        
        relative_change = abs(mean_second_half - mean_first_half) / (abs(mean_first_half) + 1e-8)
        
        # Check if variance is low
        variance = np.var(recent_performance)
        
        converged = relative_change < self.convergence_threshold and variance < 0.01
        
        return converged
    
    def update_performance_history(self, episode_reward: float):
        """Update performance history for convergence checking"""
        self.performance_history.append(episode_reward)
        
        # Keep only recent history to save memory
        if len(self.performance_history) > self.convergence_window * 2:
            self.performance_history = self.performance_history[-self.convergence_window * 2:]