import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .mdp import MDPState
from .synthetic_data import SyntheticSystem

class EvaluationMetrics:
    def __init__(self, config: Dict):
        self.config = config
        self.results = {
            'discovery_accuracy': [],
            'biological_plausibility': [],
            'parameter_recovery': [],
            'convergence_speed': [],
            'interpretability': [],
            'prediction_error': []
        }
    
    def evaluate_discovery(self, discovered_mechanism: str, true_mechanism: str) -> float:
        discovered_components = self._extract_components(discovered_mechanism)
        true_components = self._extract_components(true_mechanism)
        
        if not true_components:
            return 0.0
        
        intersection = discovered_components.intersection(true_components)
        accuracy = len(intersection) / len(true_components)
        
        self.results['discovery_accuracy'].append(accuracy)
        return accuracy
    
    def evaluate_biological_plausibility(self, state: MDPState, 
                                        knowledge_graph) -> float:
        
        entities = list(state.mechanism_tree.get_all_entities())
        relations = state.mechanism_tree.get_all_relations()
        
        plausibility = knowledge_graph.compute_plausibility_score(entities, relations)
        
        self.results['biological_plausibility'].append(plausibility)
        return plausibility
    
    def evaluate_parameter_recovery(self, estimated_params: Dict[str, float],
                                   true_params: Dict[str, float]) -> float:
        
        if not true_params:
            return 0.0
        
        errors = []
        for param_name in true_params:
            if param_name in estimated_params:
                true_val = true_params[param_name]
                est_val = estimated_params[param_name]
                
                if true_val != 0:
                    relative_error = abs(est_val - true_val) / abs(true_val)
                else:
                    relative_error = abs(est_val - true_val)
                
                errors.append(relative_error)
            else:
                errors.append(1.0)
        
        mean_error = np.mean(errors) if errors else 1.0
        recovery_score = 1.0 - min(mean_error, 1.0)
        
        self.results['parameter_recovery'].append(recovery_score)
        return recovery_score
    
    def evaluate_convergence_speed(self, episode_rewards: List[float],
                                  threshold_fraction: float = 0.9) -> int:
        
        if not episode_rewards:
            return -1
        
        final_performance = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_rewards[-1]
        threshold = threshold_fraction * final_performance
        
        for i, reward in enumerate(episode_rewards):
            if reward >= threshold:
                self.results['convergence_speed'].append(i)
                return i
        
        self.results['convergence_speed'].append(len(episode_rewards))
        return len(episode_rewards)
    
    def evaluate_interpretability(self, state: MDPState) -> float:
        complexity = state.mechanism_tree.get_complexity()
        
        complexity_penalty = self.config['reward']['complexity_penalty']
        interpretability = np.exp(-complexity_penalty * complexity)
        
        num_params = len(self._get_all_parameters(state.mechanism_tree))
        param_penalty = np.exp(-0.05 * num_params)
        
        tree_depth = self._get_tree_depth(state.mechanism_tree)
        depth_penalty = np.exp(-0.1 * tree_depth)
        
        total_interpretability = interpretability * param_penalty * depth_penalty
        
        self.results['interpretability'].append(total_interpretability)
        return total_interpretability
    
    def evaluate_prediction_error(self, predictions: np.ndarray, 
                                 true_values: np.ndarray) -> Dict[str, float]:
        
        # Ensure inputs are proper arrays
        predictions = np.asarray(predictions)
        true_values = np.asarray(true_values)
        
        # Check dimensions
        if predictions.ndim == 0:
            predictions = predictions.reshape(1)
        if true_values.ndim == 0:
            true_values = true_values.reshape(1)
        
        # Ensure same length
        if len(predictions) != len(true_values):
            raise ValueError(f"Predictions length ({len(predictions)}) doesn't match true values length ({len(true_values)})")
        
        mse = mean_squared_error(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        
        if np.var(true_values) > 0:
            r2 = r2_score(true_values, predictions)
        else:
            r2 = 0.0
        
        rmse = np.sqrt(mse)
        
        if np.mean(true_values) != 0:
            mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
        else:
            mape = 100.0
        
        errors = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
        
        self.results['prediction_error'].append(errors)
        return errors
    
    def evaluate_on_dataset(self, discovered_mechanisms: List[MDPState],
                           synthetic_systems: List[SyntheticSystem],
                           knowledge_graph) -> Dict[str, float]:
        
        metrics = {
            'mean_discovery_accuracy': 0.0,
            'mean_plausibility': 0.0,
            'mean_parameter_recovery': 0.0,
            'mean_interpretability': 0.0,
            'mean_rmse': 0.0,
            'mean_r2': 0.0
        }
        
        num_systems = min(len(discovered_mechanisms), len(synthetic_systems))
        
        for i in range(num_systems):
            discovered = discovered_mechanisms[i]
            true_system = synthetic_systems[i]
            
            if discovered is not None:
                discovered_expr = discovered.mechanism_tree.to_expression()
                discovery_acc = self.evaluate_discovery(discovered_expr, true_system.mechanism)
                metrics['mean_discovery_accuracy'] += discovery_acc
                
                plausibility = self.evaluate_biological_plausibility(discovered, knowledge_graph)
                metrics['mean_plausibility'] += plausibility
                
                discovered_params = self._get_all_parameters(discovered.mechanism_tree)
                param_recovery = self.evaluate_parameter_recovery(
                    discovered_params, true_system.true_parameters
                )
                metrics['mean_parameter_recovery'] += param_recovery
                
                interpretability = self.evaluate_interpretability(discovered)
                metrics['mean_interpretability'] += interpretability
                
                predictions = self._evaluate_mechanism_predictions(
                    discovered, true_system.data_X
                )
                if predictions is not None:
                    errors = self.evaluate_prediction_error(predictions, true_system.data_y)
                    metrics['mean_rmse'] += errors['rmse']
                    metrics['mean_r2'] += errors['r2']
        
        if num_systems > 0:
            for key in metrics:
                metrics[key] /= num_systems
        
        return metrics
    
    def statistical_comparison(self, method1_results: Dict[str, List[float]],
                             method2_results: Dict[str, List[float]]) -> Dict[str, Dict]:
        
        comparison = {}
        
        for metric in method1_results:
            if metric in method2_results:
                values1 = np.array(method1_results[metric])
                values2 = np.array(method2_results[metric])
                
                if len(values1) > 1 and len(values2) > 1:
                    t_stat, p_value = stats.ttest_ind(values1, values2)
                    
                    wilcoxon_stat, wilcoxon_p = stats.mannwhitneyu(values1, values2)
                    
                    effect_size = (np.mean(values1) - np.mean(values2)) / np.sqrt(
                        (np.std(values1)**2 + np.std(values2)**2) / 2
                    )
                    
                    comparison[metric] = {
                        'mean_difference': np.mean(values1) - np.mean(values2),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'wilcoxon_statistic': wilcoxon_stat,
                        'wilcoxon_p_value': wilcoxon_p,
                        'effect_size': effect_size,
                        'significant': p_value < 0.05
                    }
        
        return comparison
    
    def cross_validation_evaluation(self, discovered_mechanisms: List[MDPState],
                                   data_splits: List[Tuple[np.ndarray, np.ndarray]],
                                   n_folds: int = 3) -> Dict[str, float]:
        
        cv_metrics = {
            'cv_rmse': [],
            'cv_mae': [],
            'cv_r2': []
        }
        
        for fold in range(min(n_folds, len(data_splits))):
            X_test, y_test = data_splits[fold]
            
            fold_rmse = []
            fold_mae = []
            fold_r2 = []
            
            for mechanism in discovered_mechanisms:
                if mechanism is not None:
                    predictions = self._evaluate_mechanism_predictions(mechanism, X_test)
                    if predictions is not None:
                        errors = self.evaluate_prediction_error(predictions, y_test)
                        fold_rmse.append(errors['rmse'])
                        fold_mae.append(errors['mae'])
                        fold_r2.append(errors['r2'])
            
            if fold_rmse:
                cv_metrics['cv_rmse'].append(np.mean(fold_rmse))
                cv_metrics['cv_mae'].append(np.mean(fold_mae))
                cv_metrics['cv_r2'].append(np.mean(fold_r2))
        
        results = {}
        for metric, values in cv_metrics.items():
            if values:
                results[f'{metric}_mean'] = np.mean(values)
                results[f'{metric}_std'] = np.std(values)
            else:
                results[f'{metric}_mean'] = 0.0
                results[f'{metric}_std'] = 0.0
        
        return results
    
    def ablation_study_metrics(self, full_model_results: Dict[str, float],
                              ablated_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict]:
        
        ablation_impact = {}
        
        for component, component_results in ablated_results.items():
            impact = {}
            
            for metric in full_model_results:
                if metric in component_results:
                    full_value = full_model_results[metric]
                    ablated_value = component_results[metric]
                    
                    if full_value != 0:
                        relative_change = (ablated_value - full_value) / abs(full_value) * 100
                    else:
                        relative_change = 0.0
                    
                    impact[metric] = {
                        'full_model': full_value,
                        'ablated_model': ablated_value,
                        'absolute_change': ablated_value - full_value,
                        'relative_change_%': relative_change
                    }
            
            ablation_impact[component] = impact
        
        return ablation_impact
    
    def _extract_components(self, mechanism: str) -> set:
        components = set()
        
        keywords = ['v_max', 'k_m', 'k_i', 'k_a', 'alpha', 'n', 
                   'k_on', 'k_off', 'EC50', 'IC50']
        
        for keyword in keywords:
            if keyword in mechanism:
                components.add(keyword)
        
        operations = ['/', '*', '+', '-', '**']
        for op in operations:
            if op in mechanism:
                components.add(f'op_{op}')
        
        return components
    
    def _get_all_parameters(self, node) -> Dict[str, float]:
        params = dict(node.parameters)
        for child in node.children:
            params.update(self._get_all_parameters(child))
        return params
    
    def _get_tree_depth(self, node, current_depth=0) -> int:
        if not node.children:
            return current_depth
        
        max_child_depth = 0
        for child in node.children:
            child_depth = self._get_tree_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _evaluate_mechanism_predictions(self, state: MDPState, 
                                       X: np.ndarray) -> Optional[np.ndarray]:
        
        try:
            mechanism_expr = state.mechanism_tree.to_expression()
            params = self._get_all_parameters(state.mechanism_tree)
            
            safe_dict = {
                'exp': np.exp,
                'log': np.log,
                'sqrt': np.sqrt,
                'max': np.maximum,
                'min': np.minimum
            }
            
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            n_samples = X.shape[0]
            
            for i in range(X.shape[1]):
                safe_dict[f'X{i}'] = X[:, i]
                safe_dict['S'] = X[:, 0]
                safe_dict['S1'] = X[:, 0]
                if X.shape[1] > 1:
                    safe_dict['I'] = X[:, 1]
                    safe_dict['S2'] = X[:, 1]
                if X.shape[1] > 2:
                    safe_dict['A'] = X[:, 2]
                    safe_dict['P'] = X[:, 2]
            
            safe_dict.update(params)
            
            result = eval(mechanism_expr, {"__builtins__": {}}, safe_dict)
            
            # Ensure result is an array with proper shape
            if isinstance(result, (int, float)):
                # If result is scalar, broadcast to all samples
                result = np.full(n_samples, result)
            else:
                result = np.array(result)
            
            # Ensure proper shape
            if result.shape != (n_samples,):
                if result.ndim == 0:
                    # Scalar case
                    result = np.full(n_samples, result.item())
                elif result.shape[0] != n_samples:
                    # Wrong number of samples
                    return None
            
            if not np.all(np.isfinite(result)):
                return None
            
            return result
            
        except Exception as e:
            # Log the error for debugging but don't crash
            return None
    
    def get_summary_statistics(self) -> Dict[str, Dict]:
        summary = {}
        
        for metric, values in self.results.items():
            if values and metric != 'prediction_error':
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
            elif metric == 'prediction_error' and values:
                error_types = ['mse', 'rmse', 'mae', 'mape', 'r2']
                for error_type in error_types:
                    error_values = [v[error_type] for v in values if error_type in v]
                    if error_values:
                        summary[f'{metric}_{error_type}'] = {
                            'mean': np.mean(error_values),
                            'std': np.std(error_values),
                            'min': np.min(error_values),
                            'max': np.max(error_values),
                            'median': np.median(error_values)
                        }
        
        return summary