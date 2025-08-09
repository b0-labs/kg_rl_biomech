"""
Compositional MDP for flexible mechanism discovery
Instead of fixed templates, builds mechanisms from primitive operations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from enum import Enum
import copy

class NodeType(Enum):
    """Types of nodes in the expression tree"""
    VARIABLE = "variable"      # Input variable (X0, X1, ...)
    PARAMETER = "parameter"     # Learnable parameter
    CONSTANT = "constant"       # Fixed constant
    BINARY_OP = "binary_op"     # Binary operation (+, *, /, ^)
    UNARY_OP = "unary_op"       # Unary operation (exp, log, sqrt, etc.)
    BIOLOGICAL = "biological"   # Biological primitive (saturation, hill, etc.)

class BinaryOp(Enum):
    """Binary operations"""
    ADD = "+"
    MULTIPLY = "*"
    DIVIDE = "/"
    POWER = "^"
    SUBTRACT = "-"

class UnaryOp(Enum):
    """Unary operations"""
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    SQUARE = "square"
    RECIPROCAL = "reciprocal"
    ABS = "abs"
    TANH = "tanh"
    SIGMOID = "sigmoid"

class BiologicalOp(Enum):
    """Biological primitive operations"""
    SATURATION = "saturation"      # x / (k + x)
    HILL = "hill"                   # x^n / (k^n + x^n)
    INHIBITION = "inhibition"       # 1 / (1 + x/k)
    ACTIVATION = "activation"       # x / (k + x) with cooperativity
    COMPETITIVE = "competitive"     # competitive binding

@dataclass
class ExpressionNode:
    """Node in the expression tree"""
    node_type: NodeType
    value: Optional[Union[int, float, str, Enum]] = None  # Variable index, parameter name, constant value, or operation
    children: List['ExpressionNode'] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)  # For biological ops with parameters
    
    def to_expression(self, variable_names: List[str] = None) -> str:
        """Convert tree to mathematical expression string"""
        if self.node_type == NodeType.VARIABLE:
            if variable_names and self.value < len(variable_names):
                return variable_names[self.value]
            return f"X{self.value}"
        
        elif self.node_type == NodeType.PARAMETER:
            return str(self.value)
        
        elif self.node_type == NodeType.CONSTANT:
            return str(self.value)
        
        elif self.node_type == NodeType.BINARY_OP:
            if len(self.children) != 2:
                return "0"
            left = self.children[0].to_expression(variable_names)
            right = self.children[1].to_expression(variable_names)
            
            if self.value == BinaryOp.ADD:
                return f"({left} + {right})"
            elif self.value == BinaryOp.MULTIPLY:
                return f"({left} * {right})"
            elif self.value == BinaryOp.DIVIDE:
                return f"({left} / ({right} + 1e-8))"  # Avoid division by zero
            elif self.value == BinaryOp.POWER:
                return f"({left} ** {right})"
            elif self.value == BinaryOp.SUBTRACT:
                return f"({left} - {right})"
        
        elif self.node_type == NodeType.UNARY_OP:
            if len(self.children) != 1:
                return "0"
            child = self.children[0].to_expression(variable_names)
            
            if self.value == UnaryOp.EXP:
                return f"exp({child})"
            elif self.value == UnaryOp.LOG:
                return f"log(abs({child}) + 1e-8)"
            elif self.value == UnaryOp.SQRT:
                return f"sqrt(abs({child}))"
            elif self.value == UnaryOp.SQUARE:
                return f"({child} ** 2)"
            elif self.value == UnaryOp.RECIPROCAL:
                return f"(1.0 / ({child} + 1e-8))"
            elif self.value == UnaryOp.ABS:
                return f"abs({child})"
            elif self.value == UnaryOp.TANH:
                return f"tanh({child})"
            elif self.value == UnaryOp.SIGMOID:
                return f"(1.0 / (1.0 + exp(-{child})))"
        
        elif self.node_type == NodeType.BIOLOGICAL:
            if self.value == BiologicalOp.SATURATION:
                if len(self.children) != 1:
                    return "0"
                x = self.children[0].to_expression(variable_names)
                k = self.parameters.get('k', 1.0)
                return f"({x} / ({k} + {x}))"
            
            elif self.value == BiologicalOp.HILL:
                if len(self.children) != 1:
                    return "0"
                x = self.children[0].to_expression(variable_names)
                k = self.parameters.get('k', 1.0)
                n = self.parameters.get('n', 2.0)
                return f"(({x} ** {n}) / (({k} ** {n}) + ({x} ** {n})))"
            
            elif self.value == BiologicalOp.INHIBITION:
                if len(self.children) != 1:
                    return "0"
                x = self.children[0].to_expression(variable_names)
                k = self.parameters.get('k', 1.0)
                return f"(1.0 / (1.0 + {x} / {k}))"
            
            elif self.value == BiologicalOp.COMPETITIVE:
                if len(self.children) != 2:
                    return "0"
                substrate = self.children[0].to_expression(variable_names)
                inhibitor = self.children[1].to_expression(variable_names)
                k_m = self.parameters.get('k_m', 1.0)
                k_i = self.parameters.get('k_i', 1.0)
                return f"({substrate} / ({k_m} * (1.0 + {inhibitor} / {k_i}) + {substrate}))"
        
        return "0"
    
    def evaluate(self, X: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Evaluate the expression tree on data"""
        if self.node_type == NodeType.VARIABLE:
            if X.ndim == 1:
                return X if self.value == 0 else np.zeros_like(X)
            return X[:, self.value] if self.value < X.shape[1] else np.zeros(len(X))
        
        elif self.node_type == NodeType.PARAMETER:
            param_value = params.get(str(self.value), 1.0)
            if isinstance(X, np.ndarray):
                return np.full(len(X) if X.ndim > 0 else 1, param_value)
            return param_value
        
        elif self.node_type == NodeType.CONSTANT:
            if isinstance(X, np.ndarray):
                return np.full(len(X) if X.ndim > 0 else 1, self.value)
            return self.value
        
        elif self.node_type == NodeType.BINARY_OP:
            if len(self.children) != 2:
                return np.zeros(len(X))
            
            left = self.children[0].evaluate(X, params)
            right = self.children[1].evaluate(X, params)
            
            if self.value == BinaryOp.ADD:
                return left + right
            elif self.value == BinaryOp.MULTIPLY:
                return left * right
            elif self.value == BinaryOp.DIVIDE:
                return left / (right + 1e-8)
            elif self.value == BinaryOp.POWER:
                return np.power(np.abs(left), right)
            elif self.value == BinaryOp.SUBTRACT:
                return left - right
        
        elif self.node_type == NodeType.UNARY_OP:
            if len(self.children) != 1:
                return np.zeros(len(X))
            
            child = self.children[0].evaluate(X, params)
            
            if self.value == UnaryOp.EXP:
                return np.exp(np.clip(child, -10, 10))
            elif self.value == UnaryOp.LOG:
                return np.log(np.abs(child) + 1e-8)
            elif self.value == UnaryOp.SQRT:
                return np.sqrt(np.abs(child))
            elif self.value == UnaryOp.SQUARE:
                return child ** 2
            elif self.value == UnaryOp.RECIPROCAL:
                return 1.0 / (child + 1e-8)
            elif self.value == UnaryOp.ABS:
                return np.abs(child)
            elif self.value == UnaryOp.TANH:
                return np.tanh(child)
            elif self.value == UnaryOp.SIGMOID:
                return 1.0 / (1.0 + np.exp(-np.clip(child, -10, 10)))
        
        elif self.node_type == NodeType.BIOLOGICAL:
            updated_params = {**params, **self.parameters}
            
            if self.value == BiologicalOp.SATURATION:
                if len(self.children) != 1:
                    return np.zeros(len(X))
                x = self.children[0].evaluate(X, params)
                k = updated_params.get('k', 1.0)
                return x / (k + x)
            
            elif self.value == BiologicalOp.HILL:
                if len(self.children) != 1:
                    return np.zeros(len(X))
                x = self.children[0].evaluate(X, params)
                k = updated_params.get('k', 1.0)
                n = updated_params.get('n', 2.0)
                return (x ** n) / ((k ** n) + (x ** n))
            
            elif self.value == BiologicalOp.INHIBITION:
                if len(self.children) != 1:
                    return np.zeros(len(X))
                x = self.children[0].evaluate(X, params)
                k = updated_params.get('k', 1.0)
                return 1.0 / (1.0 + x / k)
            
            elif self.value == BiologicalOp.COMPETITIVE:
                if len(self.children) != 2:
                    return np.zeros(len(X))
                substrate = self.children[0].evaluate(X, params)
                inhibitor = self.children[1].evaluate(X, params)
                k_m = updated_params.get('k_m', 1.0)
                k_i = updated_params.get('k_i', 1.0)
                return substrate / (k_m * (1.0 + inhibitor / k_i) + substrate)
        
        return np.zeros(len(X))
    
    def get_complexity(self) -> int:
        """Calculate tree complexity"""
        complexity = 1
        for child in self.children:
            complexity += child.get_complexity()
        return complexity
    
    def get_depth(self) -> int:
        """Calculate tree depth"""
        if not self.children:
            return 1
        return 1 + max(child.get_depth() for child in self.children)
    
    def get_used_variables(self) -> Set[int]:
        """Get set of variable indices used"""
        if self.node_type == NodeType.VARIABLE:
            return {self.value}
        
        used = set()
        for child in self.children:
            used.update(child.get_used_variables())
        return used
    
    def get_parameters(self) -> Set[str]:
        """Get set of parameter names"""
        params = set()
        
        if self.node_type == NodeType.PARAMETER:
            params.add(str(self.value))
        
        if self.node_type == NodeType.BIOLOGICAL:
            params.update(self.parameters.keys())
        
        for child in self.children:
            params.update(child.get_parameters())
        
        return params
    
    def deep_copy(self) -> 'ExpressionNode':
        """Create deep copy of the tree"""
        new_node = ExpressionNode(
            node_type=self.node_type,
            value=self.value,
            parameters=dict(self.parameters),
            children=[]
        )
        
        for child in self.children:
            new_node.children.append(child.deep_copy())
        
        return new_node


class CompositionalAction(Enum):
    """Actions for building expression trees"""
    ADD_VARIABLE = "add_variable"           # Add input variable node
    ADD_PARAMETER = "add_parameter"         # Add learnable parameter
    ADD_CONSTANT = "add_constant"           # Add constant value
    ADD_BINARY_OP = "add_binary_op"         # Add binary operation
    ADD_UNARY_OP = "add_unary_op"           # Add unary operation
    ADD_BIOLOGICAL = "add_biological"       # Add biological primitive
    REPLACE_SUBTREE = "replace_subtree"     # Replace a subtree
    SIMPLIFY = "simplify"                   # Simplify expression
    OPTIMIZE_PARAMS = "optimize_params"     # Optimize parameters
    TERMINATE = "terminate"                 # Complete mechanism


@dataclass
class CompositionalState:
    """State for compositional mechanism discovery"""
    expression_tree: Optional[ExpressionNode]
    parameters: Dict[str, float]
    step_count: int = 0
    is_terminal: bool = False
    fitness_score: float = 0.0
    data_dimensions: int = 1
    
    def get_state_representation(self) -> Dict[str, Any]:
        """Get state representation for neural networks"""
        if self.expression_tree is None:
            return {
                'complexity': 0,
                'depth': 0,
                'num_variables': 0,
                'num_parameters': 0,
                'fitness': self.fitness_score,
                'step_count': self.step_count
            }
        
        return {
            'complexity': self.expression_tree.get_complexity(),
            'depth': self.expression_tree.get_depth(),
            'num_variables': len(self.expression_tree.get_used_variables()),
            'num_parameters': len(self.expression_tree.get_parameters()),
            'fitness': self.fitness_score,
            'step_count': self.step_count
        }
    
    def deep_copy(self) -> 'CompositionalState':
        """Create deep copy of state"""
        return CompositionalState(
            expression_tree=self.expression_tree.deep_copy() if self.expression_tree else None,
            parameters=dict(self.parameters),
            step_count=self.step_count,
            is_terminal=self.is_terminal,
            fitness_score=self.fitness_score,
            data_dimensions=self.data_dimensions
        )


class CompositionalMDP:
    """MDP for compositional mechanism discovery"""
    
    def __init__(self, config: Dict, max_depth: int = 10, max_complexity: int = 50):
        self.config = config
        self.max_depth = max_depth
        self.max_complexity = max_complexity
        self.max_steps = config.get('mdp', {}).get('max_steps_per_episode', 100)
        self.param_counter = 0
        
    def create_initial_state(self, data_dimensions: int = 1) -> CompositionalState:
        """Create initial empty state"""
        self.param_counter = 0
        return CompositionalState(
            expression_tree=None,
            parameters={},
            data_dimensions=data_dimensions
        )
    
    def get_valid_actions(self, state: CompositionalState) -> List[Tuple[CompositionalAction, Dict]]:
        """Get valid actions for current state"""
        if state.is_terminal:
            return []
        
        valid_actions = []
        
        # If tree is empty, can only add initial nodes
        if state.expression_tree is None:
            # Add variables
            for i in range(state.data_dimensions):
                valid_actions.append((CompositionalAction.ADD_VARIABLE, {'index': i}))
            
            # Add parameters
            valid_actions.append((CompositionalAction.ADD_PARAMETER, {'name': f'p_{self.param_counter}'}))
            
            # Add constants
            for const in [0.1, 0.5, 1.0, 2.0]:
                valid_actions.append((CompositionalAction.ADD_CONSTANT, {'value': const}))
        
        else:
            # Can build on existing tree
            tree_complexity = state.expression_tree.get_complexity()
            tree_depth = state.expression_tree.get_depth()
            
            if tree_complexity < self.max_complexity and tree_depth < self.max_depth:
                # Add binary operations
                for op in BinaryOp:
                    valid_actions.append((CompositionalAction.ADD_BINARY_OP, {'operation': op}))
                
                # Add unary operations
                for op in UnaryOp:
                    valid_actions.append((CompositionalAction.ADD_UNARY_OP, {'operation': op}))
                
                # Add biological primitives
                for op in BiologicalOp:
                    valid_actions.append((CompositionalAction.ADD_BIOLOGICAL, {'operation': op}))
            
            # Can always optimize parameters if we have them
            if state.parameters:
                valid_actions.append((CompositionalAction.OPTIMIZE_PARAMS, {}))
            
            # Can always terminate
            valid_actions.append((CompositionalAction.TERMINATE, {}))
        
        return valid_actions
    
    def transition(self, state: CompositionalState, action: Tuple[CompositionalAction, Dict],
                  data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> CompositionalState:
        """Apply action to state"""
        new_state = state.deep_copy()
        new_state.step_count += 1
        
        action_type, action_params = action
        
        if action_type == CompositionalAction.TERMINATE:
            new_state.is_terminal = True
            return new_state
        
        if new_state.step_count >= self.max_steps:
            new_state.is_terminal = True
            return new_state
        
        # Apply different action types
        if action_type == CompositionalAction.ADD_VARIABLE:
            node = ExpressionNode(NodeType.VARIABLE, value=action_params['index'])
            if new_state.expression_tree is None:
                new_state.expression_tree = node
            else:
                new_state.expression_tree = self._wrap_with_random_op(new_state.expression_tree, node)
        
        elif action_type == CompositionalAction.ADD_PARAMETER:
            param_name = action_params['name']
            node = ExpressionNode(NodeType.PARAMETER, value=param_name)
            new_state.parameters[param_name] = np.random.uniform(0.1, 10.0)
            
            if new_state.expression_tree is None:
                new_state.expression_tree = node
            else:
                new_state.expression_tree = self._wrap_with_random_op(new_state.expression_tree, node)
        
        elif action_type == CompositionalAction.ADD_CONSTANT:
            node = ExpressionNode(NodeType.CONSTANT, value=action_params['value'])
            if new_state.expression_tree is None:
                new_state.expression_tree = node
            else:
                new_state.expression_tree = self._wrap_with_random_op(new_state.expression_tree, node)
        
        elif action_type == CompositionalAction.ADD_BINARY_OP:
            if new_state.expression_tree is not None:
                op = action_params['operation']
                # Create new variable or parameter for second operand
                if np.random.random() < 0.5 and state.data_dimensions > 1:
                    # Add a different variable
                    used_vars = new_state.expression_tree.get_used_variables()
                    available = [i for i in range(state.data_dimensions) if i not in used_vars]
                    if available:
                        second = ExpressionNode(NodeType.VARIABLE, value=np.random.choice(available))
                    else:
                        second = ExpressionNode(NodeType.CONSTANT, value=1.0)
                else:
                    # Add parameter
                    param_name = f'p_{self.param_counter}'
                    self.param_counter += 1
                    second = ExpressionNode(NodeType.PARAMETER, value=param_name)
                    new_state.parameters[param_name] = np.random.uniform(0.1, 10.0)
                
                new_node = ExpressionNode(NodeType.BINARY_OP, value=op, 
                                        children=[new_state.expression_tree, second])
                new_state.expression_tree = new_node
        
        elif action_type == CompositionalAction.ADD_UNARY_OP:
            if new_state.expression_tree is not None:
                op = action_params['operation']
                new_node = ExpressionNode(NodeType.UNARY_OP, value=op,
                                        children=[new_state.expression_tree])
                new_state.expression_tree = new_node
        
        elif action_type == CompositionalAction.ADD_BIOLOGICAL:
            if new_state.expression_tree is not None:
                op = action_params['operation']
                
                # Add biological parameters
                if op == BiologicalOp.SATURATION or op == BiologicalOp.INHIBITION:
                    k_param = f'k_{self.param_counter}'
                    self.param_counter += 1
                    bio_params = {'k': k_param}
                    new_state.parameters[k_param] = np.random.uniform(0.1, 10.0)
                
                elif op == BiologicalOp.HILL:
                    k_param = f'k_{self.param_counter}'
                    n_param = f'n_{self.param_counter}'
                    self.param_counter += 1
                    bio_params = {'k': k_param, 'n': n_param}
                    new_state.parameters[k_param] = np.random.uniform(0.1, 10.0)
                    new_state.parameters[n_param] = np.random.uniform(1.0, 4.0)
                
                elif op == BiologicalOp.COMPETITIVE:
                    # Need two inputs for competitive
                    if state.data_dimensions > 1:
                        used_vars = new_state.expression_tree.get_used_variables()
                        available = [i for i in range(state.data_dimensions) if i not in used_vars]
                        if available:
                            inhibitor = ExpressionNode(NodeType.VARIABLE, value=available[0])
                        else:
                            inhibitor = ExpressionNode(NodeType.CONSTANT, value=1.0)
                    else:
                        inhibitor = ExpressionNode(NodeType.CONSTANT, value=1.0)
                    
                    k_m_param = f'k_m_{self.param_counter}'
                    k_i_param = f'k_i_{self.param_counter}'
                    self.param_counter += 1
                    bio_params = {'k_m': k_m_param, 'k_i': k_i_param}
                    new_state.parameters[k_m_param] = np.random.uniform(0.1, 10.0)
                    new_state.parameters[k_i_param] = np.random.uniform(0.1, 10.0)
                    
                    new_node = ExpressionNode(NodeType.BIOLOGICAL, value=op,
                                            children=[new_state.expression_tree, inhibitor],
                                            parameters=bio_params)
                    new_state.expression_tree = new_node
                    return new_state
                
                new_node = ExpressionNode(NodeType.BIOLOGICAL, value=op,
                                        children=[new_state.expression_tree],
                                        parameters=bio_params)
                new_state.expression_tree = new_node
        
        elif action_type == CompositionalAction.OPTIMIZE_PARAMS:
            # Optimize parameters using gradient descent if data is provided
            if data is not None and new_state.expression_tree is not None:
                X, y = data
                new_state.parameters = self._optimize_parameters(
                    new_state.expression_tree, new_state.parameters, X, y
                )
                # Update fitness after optimization
                predictions = new_state.expression_tree.evaluate(X, new_state.parameters)
                mse = np.mean((y - predictions) ** 2)
                new_state.fitness_score = -mse
        
        return new_state
    
    def _wrap_with_random_op(self, tree1: ExpressionNode, tree2: ExpressionNode) -> ExpressionNode:
        """Wrap two trees with a random binary operation"""
        op = np.random.choice(list(BinaryOp))
        return ExpressionNode(NodeType.BINARY_OP, value=op, children=[tree1, tree2])
    
    def _optimize_parameters(self, tree: ExpressionNode, params: Dict[str, float],
                            X: np.ndarray, y: np.ndarray, max_iter: int = 50) -> Dict[str, float]:
        """Optimize parameters using gradient descent"""
        param_names = list(params.keys())
        if not param_names:
            return params
        
        # Convert to torch tensors
        X_torch = torch.tensor(X, dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.float32)
        
        # Create parameter tensors
        param_values = torch.tensor([params[name] for name in param_names], 
                                   dtype=torch.float32, requires_grad=True)
        
        optimizer = optim.Adam([param_values], lr=0.1)
        
        for _ in range(max_iter):
            optimizer.zero_grad()
            
            # Update parameter dict
            current_params = {name: param_values[i].item() 
                            for i, name in enumerate(param_names)}
            
            # Evaluate tree
            predictions = tree.evaluate(X, current_params)
            pred_torch = torch.tensor(predictions, dtype=torch.float32)
            
            # Compute loss
            loss = nn.MSELoss()(pred_torch, y_torch)
            
            # Approximate gradient by finite differences
            grad = torch.zeros_like(param_values)
            eps = 1e-4
            
            for i, name in enumerate(param_names):
                # Forward difference
                perturbed_params = current_params.copy()
                perturbed_params[name] += eps
                pred_plus = tree.evaluate(X, perturbed_params)
                
                # Backward difference  
                perturbed_params[name] = current_params[name] - eps
                pred_minus = tree.evaluate(X, perturbed_params)
                
                # Gradient approximation
                grad[i] = np.mean((pred_plus - pred_minus) * (predictions - y)) / (2 * eps)
            
            param_values.grad = grad
            optimizer.step()
            
            # Ensure positive parameters where needed
            with torch.no_grad():
                param_values.clamp_(min=0.001, max=100.0)
        
        # Return optimized parameters
        return {name: param_values[i].item() for i, name in enumerate(param_names)}
    
    def is_terminal_state(self, state: CompositionalState) -> bool:
        """Check if state is terminal"""
        return state.is_terminal or state.step_count >= self.max_steps