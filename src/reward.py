import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from .mdp import MDPState, Action, ActionType
from .knowledge_graph import KnowledgeGraph
import scipy.optimize as opt

class RewardFunction:
    def __init__(self, knowledge_graph: KnowledgeGraph, config: Dict):
        self.knowledge_graph = knowledge_graph
        self.config = config
        
        self.lambda_likelihood = config['reward']['lambda_likelihood']
        self.lambda_plausibility = config['reward']['lambda_plausibility']
        self.lambda_interpretability = config['reward']['lambda_interpretability']
        self.lambda_action_penalty = config['reward']['lambda_action_penalty']
        self.lambda_violation_penalty = config['reward']['lambda_violation_penalty']
        self.complexity_penalty = config['reward']['complexity_penalty']
        
        self.data = None
        self.previous_likelihood = None
    
    def set_data(self, X: np.ndarray, y: np.ndarray):
        self.data = {'X': X, 'y': y}
        self.previous_likelihood = None
    
    def compute_reward(self, state: MDPState, action: Action, 
                      next_state: MDPState) -> float:
        
        likelihood_improvement = self._compute_likelihood_improvement(state, next_state)
        
        plausibility_score = self._compute_plausibility_score(next_state)
        
        interpretability_score = self._compute_interpretability_score(next_state)
        
        action_penalty = self._compute_action_penalty(action)
        
        violation_penalty = self._compute_violation_penalty(next_state)
        
        # Add exploration bonus for early steps to encourage building
        exploration_bonus = 0.0
        if state.mechanism_tree.get_complexity() < 3:
            exploration_bonus = 0.2  # Small bonus for taking actions early
        
        reward = (
            self.lambda_likelihood * likelihood_improvement +
            self.lambda_plausibility * plausibility_score +
            self.lambda_interpretability * interpretability_score -
            self.lambda_action_penalty * action_penalty -
            self.lambda_violation_penalty * violation_penalty +
            exploration_bonus
        )
        
        if next_state.is_terminal:
            if next_state.mechanism_tree.get_complexity() > 1:
                terminal_bonus = self._compute_terminal_bonus(next_state)
                reward += terminal_bonus
        
        return reward
    
    def _compute_likelihood_improvement(self, state: MDPState, 
                                       next_state: MDPState) -> float:
        if self.data is None:
            return 0.0
        
        current_likelihood = self._evaluate_likelihood(state)
        next_likelihood = self._evaluate_likelihood(next_state)
        
        improvement = next_likelihood - current_likelihood
        
        # Add a small bonus for increasing complexity early on
        complexity_diff = next_state.mechanism_tree.get_complexity() - state.mechanism_tree.get_complexity()
        if complexity_diff > 0 and state.mechanism_tree.get_complexity() < 3:
            improvement += 0.5 * complexity_diff
        
        # Use a softer normalization to preserve more signal
        normalized_improvement = np.tanh(improvement / 5.0)  # Changed from 10.0 to 5.0
        
        return normalized_improvement
    
    def _evaluate_likelihood(self, state: MDPState) -> float:
        complexity = state.mechanism_tree.get_complexity()
        
        # For very simple mechanisms, return a small negative value that scales with complexity
        # This encourages building more complex mechanisms without harsh penalties
        if complexity <= 1:
            # Return a small negative value that improves as complexity approaches 1
            return -10.0 * (2.0 - complexity)  # -20 for complexity=0, -10 for complexity=1
        
        try:
            mechanism_expr = state.mechanism_tree.to_expression()
            
            predictions = self._evaluate_mechanism(mechanism_expr, self.data['X'])
            
            if predictions is None:
                # Return a penalty that scales with complexity (less harsh for simple mechanisms)
                return -50.0 / max(1, complexity)
            
            residuals = self.data['y'] - predictions
            neg_log_likelihood = 0.5 * np.sum(residuals ** 2) / len(residuals)
            
            # Add a small complexity bonus to encourage exploration
            complexity_bonus = 0.1 * np.log(complexity + 1)
            
            return -neg_log_likelihood + complexity_bonus
            
        except Exception:
            # Return a penalty that scales with complexity
            return -50.0 / max(1, complexity)
    
    def _evaluate_mechanism(self, expression: str, X: np.ndarray) -> Optional[np.ndarray]:
        try:
            safe_dict = {
                'exp': np.exp,
                'log': np.log,
                'sqrt': np.sqrt,
                'sin': np.sin,
                'cos': np.cos,
                'max': np.maximum,
                'min': np.minimum,
                'abs': np.abs,
                'tanh': np.tanh,
                'sigmoid': lambda x: 1 / (1 + np.exp(-x))
            }
            
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            for i in range(X.shape[1]):
                safe_dict[f'X{i}'] = X[:, i]
                safe_dict[f'S'] = X[:, 0]
                safe_dict[f'I'] = X[:, 1] if X.shape[1] > 1 else np.zeros_like(X[:, 0])
                safe_dict[f'A'] = X[:, 2] if X.shape[1] > 2 else np.zeros_like(X[:, 0])
                safe_dict[f'D'] = X[:, 0]
                safe_dict[f'R'] = X[:, 1] if X.shape[1] > 1 else np.ones_like(X[:, 0])
            
            safe_dict.update({
                'v_max': 1.0, 'k_m': 0.1, 'k_i': 0.1, 'k_a': 0.1,
                'n': 1.5, 'alpha': 1.0, 'k_on': 1e6, 'k_off': 1.0,
                'k_d': 1e-6, 'IC50': 1e-6, 'EC50': 1e-6
            })
            
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            
            if isinstance(result, (int, float)):
                result = np.full(len(X), result)
            
            result = np.array(result)
            
            if not np.all(np.isfinite(result)):
                return None
            
            return result
            
        except Exception:
            return None
    
    def _compute_plausibility_score(self, state: MDPState) -> float:
        entities = list(state.mechanism_tree.get_all_entities())
        relations = state.mechanism_tree.get_all_relations()
        
        if not entities:
            return 0.0
        
        plausibility = self.knowledge_graph.compute_plausibility_score(
            entities, relations
        )
        
        if self._check_biological_constraints(state):
            plausibility *= 1.2
        
        return plausibility
    
    def _check_biological_constraints(self, state: MDPState) -> bool:
        all_params = self._get_all_parameters(state.mechanism_tree)
        
        for param_name, value in all_params.items():
            base_param = param_name.split('_')[-1] if '_' in param_name else param_name
            
            if base_param in state.parameter_constraints:
                min_val, max_val = state.parameter_constraints[base_param]
                if not (min_val <= value <= max_val):
                    return False
        
        return True
    
    def _get_all_parameters(self, node) -> Dict[str, float]:
        params = dict(node.parameters)
        for child in node.children:
            params.update(self._get_all_parameters(child))
        return params
    
    def _compute_interpretability_score(self, state: MDPState) -> float:
        complexity = state.mechanism_tree.get_complexity()
        
        interpretability = np.exp(-self.complexity_penalty * complexity)
        
        num_params = len(self._get_all_parameters(state.mechanism_tree))
        param_penalty = np.exp(-0.05 * num_params)
        
        tree_depth = self._get_tree_depth(state.mechanism_tree)
        depth_penalty = np.exp(-0.1 * tree_depth)
        
        return interpretability * param_penalty * depth_penalty
    
    def _get_tree_depth(self, node, current_depth=0) -> int:
        if not node.children:
            return current_depth
        
        max_child_depth = 0
        for child in node.children:
            child_depth = self._get_tree_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _compute_action_penalty(self, action: Action) -> float:
        if action.action_type == ActionType.TERMINATE:
            return 0.0
        elif action.action_type == ActionType.ADD_ENTITY:
            return 0.1
        elif action.action_type == ActionType.MODIFY_PARAMETER:
            return 0.05
        elif action.action_type == ActionType.COMBINE_SUBTREES:
            return 0.15
        else:
            return 0.0
    
    def _compute_violation_penalty(self, state: MDPState) -> float:
        penalty = 0.0
        
        if not self._check_biological_constraints(state):
            penalty += 1.0
        
        if state.mechanism_tree.get_complexity() > 50:
            penalty += (state.mechanism_tree.get_complexity() - 50) * 0.1
        
        tree_depth = self._get_tree_depth(state.mechanism_tree)
        if tree_depth > 10:
            penalty += (tree_depth - 10) * 0.2
        
        if not self.knowledge_graph.validate_mechanism_consistency(
            self._tree_to_dict(state.mechanism_tree)
        ):
            penalty += 0.5
        
        return penalty
    
    def _tree_to_dict(self, node) -> Dict:
        result = {
            'entity': node.entity_id,
            'relation': node.relation_type
        }
        
        if node.parent and node.parent.entity_id:
            result['parent'] = node.parent.entity_id
        
        if node.children:
            result['children'] = [self._tree_to_dict(child) for child in node.children]
        
        return result
    
    def _compute_terminal_bonus(self, state: MDPState) -> float:
        bonus = 0.0
        
        if self._check_biological_constraints(state):
            bonus += 1.0
        
        plausibility = self._compute_plausibility_score(state)
        if plausibility > 0.8:
            bonus += 2.0
        
        interpretability = self._compute_interpretability_score(state)
        if interpretability > 0.7:
            bonus += 1.0
        
        if self.data is not None:
            likelihood = self._evaluate_likelihood(state)
            if likelihood > -10.0:
                bonus += 3.0
        
        return bonus

class CumulativeRewardTracker:
    def __init__(self):
        self.episode_rewards = []
        self.current_episode = []
        self.total_rewards = []
    
    def add_reward(self, reward: float):
        self.current_episode.append(reward)
        self.total_rewards.append(reward)
    
    def end_episode(self):
        if self.current_episode:
            episode_total = sum(self.current_episode)
            self.episode_rewards.append(episode_total)
            self.current_episode = []
    
    def get_episode_return(self, gamma: float = 0.99) -> float:
        if not self.current_episode:
            return 0.0
        
        discounted_return = 0.0
        for t, reward in enumerate(self.current_episode):
            discounted_return += (gamma ** t) * reward
        
        return discounted_return
    
    def get_statistics(self) -> Dict[str, float]:
        if not self.episode_rewards:
            return {
                'mean_episode_reward': 0.0,
                'std_episode_reward': 0.0,
                'max_episode_reward': 0.0,
                'min_episode_reward': 0.0,
                'total_rewards': 0.0
            }
        
        return {
            'mean_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            'max_episode_reward': np.max(self.episode_rewards),
            'min_episode_reward': np.min(self.episode_rewards),
            'total_rewards': np.sum(self.total_rewards)
        }
    
    def reset(self):
        self.episode_rewards = []
        self.current_episode = []
        self.total_rewards = []