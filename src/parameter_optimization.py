import numpy as np
import scipy.optimize as opt
from scipy.optimize import differential_evolution, minimize, least_squares
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

from .mdp import MDPState, MechanismNode

class ParameterOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.method = config['optimization']['method']
        self.max_iterations = config['optimization']['max_iterations']
        self.tolerance = config['optimization']['tolerance']
        self.lambda_regularization = config['optimization']['lambda_regularization']
        self.lambda_biological = config['optimization']['lambda_biological']
        
        self.biological_params = config['biological_params']
        
    def optimize_parameters(self, state: MDPState, data_X: np.ndarray, 
                           data_y: np.ndarray) -> Tuple[Dict[str, float], float]:
        
        all_params = self._extract_parameters(state.mechanism_tree)
        if not all_params:
            return {}, float('inf')
        
        param_names = list(all_params.keys())
        initial_values = np.array([all_params[name] for name in param_names])
        
        bounds = self._get_parameter_bounds(param_names, state.parameter_constraints)
        
        mechanism_expr = state.mechanism_tree.to_expression()
        
        def objective(params):
            param_dict = dict(zip(param_names, params))
            
            predictions = self._evaluate_mechanism(mechanism_expr, data_X, param_dict)
            if predictions is None:
                return 1e10
            
            mse = np.mean((data_y - predictions) ** 2)
            
            reg_term = self.lambda_regularization * np.sum(params ** 2)
            
            bio_term = self._compute_biological_penalty(param_names, params)
            
            return mse + reg_term + bio_term
        
        if self.method == 'L-BFGS-B':
            result = minimize(
                objective,
                initial_values,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance,
                    'gtol': self.tolerance
                }
            )
            optimized_params = result.x
            final_loss = result.fun
            
        elif self.method == 'differential_evolution':
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.max_iterations // 10,
                tol=self.tolerance,
                seed=42
            )
            optimized_params = result.x
            final_loss = result.fun
            
        elif self.method == 'least_squares':
            def residuals(params):
                param_dict = dict(zip(param_names, params))
                predictions = self._evaluate_mechanism(mechanism_expr, data_X, param_dict)
                if predictions is None:
                    return np.full_like(data_y, 1e5)
                return data_y - predictions
            
            lb = [b[0] for b in bounds]
            ub = [b[1] for b in bounds]
            
            result = least_squares(
                residuals,
                initial_values,
                bounds=(lb, ub),
                max_nfev=self.max_iterations,
                ftol=self.tolerance,
                xtol=self.tolerance
            )
            optimized_params = result.x
            final_loss = np.sum(result.fun ** 2)
            
        else:
            result = minimize(
                objective,
                initial_values,
                method='Nelder-Mead',
                options={
                    'maxiter': self.max_iterations,
                    'xatol': self.tolerance,
                    'fatol': self.tolerance
                }
            )
            optimized_params = result.x
            final_loss = result.fun
        
        optimized_dict = dict(zip(param_names, optimized_params))
        
        return optimized_dict, final_loss
    
    def optimize_with_constraints(self, state: MDPState, data_X: np.ndarray, 
                                 data_y: np.ndarray, 
                                 custom_constraints: Optional[List[Dict]] = None) -> Tuple[Dict[str, float], float]:
        
        all_params = self._extract_parameters(state.mechanism_tree)
        if not all_params:
            return {}, float('inf')
        
        param_names = list(all_params.keys())
        initial_values = np.array([all_params[name] for name in param_names])
        
        bounds = self._get_parameter_bounds(param_names, state.parameter_constraints)
        
        mechanism_expr = state.mechanism_tree.to_expression()
        
        def objective(params):
            param_dict = dict(zip(param_names, params))
            predictions = self._evaluate_mechanism(mechanism_expr, data_X, param_dict)
            if predictions is None:
                return 1e10
            
            mse = np.mean((data_y - predictions) ** 2)
            reg_term = self.lambda_regularization * np.sum(params ** 2)
            bio_term = self._compute_biological_penalty(param_names, params)
            
            return mse + reg_term + bio_term
        
        constraints = []
        
        if 'v_max' in param_names and 'k_m' in param_names:
            v_max_idx = param_names.index('v_max')
            k_m_idx = param_names.index('k_m')
            
            def catalytic_efficiency_constraint(params):
                return params[v_max_idx] / params[k_m_idx] - 1e-3
            
            constraints.append({
                'type': 'ineq',
                'fun': catalytic_efficiency_constraint
            })
        
        if 'k_on' in param_names and 'k_off' in param_names:
            k_on_idx = param_names.index('k_on')
            k_off_idx = param_names.index('k_off')
            
            def diffusion_limit_constraint(params):
                return 1e9 - params[k_on_idx]
            
            def equilibrium_constraint(params):
                return params[k_on_idx] / params[k_off_idx] - 1e3
            
            constraints.extend([
                {'type': 'ineq', 'fun': diffusion_limit_constraint},
                {'type': 'ineq', 'fun': equilibrium_constraint}
            ])
        
        if custom_constraints:
            for constraint_dict in custom_constraints:
                param_name = constraint_dict['param']
                if param_name in param_names:
                    idx = param_names.index(param_name)
                    constraint_type = constraint_dict['type']
                    value = constraint_dict['value']
                    
                    if constraint_type == 'eq':
                        constraints.append({
                            'type': 'eq',
                            'fun': lambda p, i=idx, v=value: p[i] - v
                        })
                    elif constraint_type == 'ineq':
                        constraints.append({
                            'type': 'ineq',
                            'fun': lambda p, i=idx, v=value: p[i] - v
                        })
        
        result = minimize(
            objective,
            initial_values,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance
            }
        )
        
        optimized_dict = dict(zip(param_names, result.x))
        
        return optimized_dict, result.fun
    
    def bayesian_optimization(self, state: MDPState, data_X: np.ndarray, 
                            data_y: np.ndarray, n_calls: int = 50) -> Tuple[Dict[str, float], float]:
        
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ImportError:
            print("scikit-optimize not installed, falling back to standard optimization")
            return self.optimize_parameters(state, data_X, data_y)
        
        all_params = self._extract_parameters(state.mechanism_tree)
        if not all_params:
            return {}, float('inf')
        
        param_names = list(all_params.keys())
        bounds = self._get_parameter_bounds(param_names, state.parameter_constraints)
        
        dimensions = [Real(low=b[0], high=b[1], name=name) for name, b in zip(param_names, bounds)]
        
        mechanism_expr = state.mechanism_tree.to_expression()
        
        def objective(params):
            param_dict = dict(zip(param_names, params))
            predictions = self._evaluate_mechanism(mechanism_expr, data_X, param_dict)
            if predictions is None:
                return 1e10
            
            mse = np.mean((data_y - predictions) ** 2)
            reg_term = self.lambda_regularization * np.sum(np.array(params) ** 2)
            bio_term = self._compute_biological_penalty(param_names, params)
            
            return mse + reg_term + bio_term
        
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            acq_func='EI',
            random_state=42
        )
        
        optimized_dict = dict(zip(param_names, result.x))
        
        return optimized_dict, result.fun
    
    def _extract_parameters(self, node: MechanismNode) -> Dict[str, float]:
        params = {}
        
        def collect_params(n: MechanismNode, prefix: str = ""):
            node_prefix = f"{prefix}{n.node_id}_" if prefix or n.node_id != "node_1" else ""
            
            for param_name, value in n.parameters.items():
                full_name = f"{node_prefix}{param_name}" if node_prefix else param_name
                params[full_name] = value
            
            for child in n.children:
                collect_params(child, prefix)
        
        collect_params(node)
        return params
    
    def _get_parameter_bounds(self, param_names: List[str], 
                             constraints: Dict[str, Tuple[float, float]]) -> List[Tuple[float, float]]:
        
        bounds = []
        
        for param_name in param_names:
            # First check if the full param_name is in constraints (e.g., node_2_v_max)
            if param_name in constraints:
                bounds.append(constraints[param_name])
            else:
                # Then try the base parameter name (last part after underscore)
                base_param = param_name.split('_')[-1] if '_' in param_name else param_name
                
                if base_param in constraints:
                    bounds.append(constraints[base_param])
                elif base_param in self.biological_params:
                    param_config = self.biological_params[base_param]
                    bounds.append((param_config['min'], param_config['max']))
                else:
                    bounds.append((1e-6, 1e3))
        
        return bounds
    
    def _evaluate_mechanism(self, expression: str, X: np.ndarray, 
                          params: Dict[str, float]) -> Optional[np.ndarray]:
        
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
                'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            }
            
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            for i in range(X.shape[1]):
                safe_dict[f'X{i}'] = X[:, i]
                safe_dict[f'S'] = X[:, 0]
                safe_dict[f'S1'] = X[:, 0]
                if X.shape[1] > 1:
                    safe_dict[f'I'] = X[:, 1]
                    safe_dict[f'S2'] = X[:, 1]
                    safe_dict[f'D'] = X[:, 0]
                    safe_dict[f'R'] = X[:, 1]
                if X.shape[1] > 2:
                    safe_dict[f'A'] = X[:, 2]
                    safe_dict[f'P'] = X[:, 2]
            
            safe_dict.update(params)
            
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            
            if isinstance(result, (int, float)):
                result = np.full(len(X), float(result))
            
            result = np.array(result, dtype=float)
            
            # Handle non-finite values FIRST
            if not np.all(np.isfinite(result)):
                # Replace non-finite values with reasonable defaults
                result = np.nan_to_num(result, nan=0.01, posinf=1e10, neginf=-1e10)
            
            # Then clip to prevent extreme values
            result = np.clip(result, -1e10, 1e10)
            
            return result
            
        except Exception as e:
            return None
    
    def _compute_biological_penalty(self, param_names: List[str], 
                                   params: np.ndarray) -> float:
        
        penalty = 0.0
        
        for i, (name, value) in enumerate(zip(param_names, params)):
            base_param = name.split('_')[-1] if '_' in name else name
            
            if base_param in self.biological_params:
                param_config = self.biological_params[base_param]
                
                if 'prior_mean' in param_config and 'prior_std' in param_config:
                    mean = param_config['prior_mean']
                    std = param_config['prior_std']
                    
                    z_score = (value - mean) / std
                    penalty += self.lambda_biological * z_score ** 2
                
                if value < param_config['min'] or value > param_config['max']:
                    penalty += 100.0
        
        if 'k_on' in param_names and 'k_off' in param_names:
            k_on_idx = param_names.index('k_on')
            k_off_idx = param_names.index('k_off')
            
            if k_on_idx < len(params) and k_off_idx < len(params):
                k_d = params[k_off_idx] / params[k_on_idx]
                
                if k_d < 1e-12 or k_d > 1e-3:
                    penalty += 50.0
        
        return penalty
    
    def grid_search(self, state: MDPState, data_X: np.ndarray, 
                   data_y: np.ndarray, grid_points: int = 10) -> Tuple[Dict[str, float], float]:
        
        all_params = self._extract_parameters(state.mechanism_tree)
        if not all_params:
            return {}, float('inf')
        
        param_names = list(all_params.keys())
        bounds = self._get_parameter_bounds(param_names, state.parameter_constraints)
        
        if len(param_names) > 3:
            print(f"Grid search not feasible for {len(param_names)} parameters, using standard optimization")
            return self.optimize_parameters(state, data_X, data_y)
        
        param_grids = []
        for (low, high) in bounds:
            if low > 0:
                grid = np.logspace(np.log10(low), np.log10(high), grid_points)
            else:
                grid = np.linspace(low, high, grid_points)
            param_grids.append(grid)
        
        mechanism_expr = state.mechanism_tree.to_expression()
        
        best_params = None
        best_loss = float('inf')
        
        import itertools
        for param_values in itertools.product(*param_grids):
            param_dict = dict(zip(param_names, param_values))
            
            predictions = self._evaluate_mechanism(mechanism_expr, data_X, param_dict)
            if predictions is None:
                continue
            
            mse = np.mean((data_y - predictions) ** 2)
            reg_term = self.lambda_regularization * np.sum(np.array(param_values) ** 2)
            bio_term = self._compute_biological_penalty(param_names, np.array(param_values))
            
            loss = mse + reg_term + bio_term
            
            if loss < best_loss:
                best_loss = loss
                best_params = param_dict
        
        if best_params is None:
            return {}, float('inf')
        
        return best_params, best_loss