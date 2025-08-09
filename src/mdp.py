import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any
from enum import Enum
import copy
from .knowledge_graph import KnowledgeGraph, RelationType, MathematicalConstraint

class ActionType(Enum):
    ADD_ENTITY = "add_entity"
    MODIFY_PARAMETER = "modify_parameter"
    COMBINE_SUBTREES = "combine_subtrees"
    TERMINATE = "terminate"

class CombineOperation(Enum):
    ADD = "+"
    MULTIPLY = "*"
    COMPOSE = "compose"
    MAX = "max"
    MIN = "min"

@dataclass
class MechanismNode:
    node_id: str
    node_type: str
    entity_id: Optional[str] = None
    relation_type: Optional[RelationType] = None
    functional_form: Optional[str] = None
    parameters: Dict[str, float] = field(default_factory=dict)
    children: List['MechanismNode'] = field(default_factory=list)
    parent: Optional['MechanismNode'] = None
    
    def to_expression(self) -> str:
        # Handle root node specially - combine children
        if self.node_type == "root":
            if len(self.children) == 0:
                # Return a basic Michaelis-Menten with default parameters for empty root
                return "(0.1 * S) / (0.01 + S)"  # Basic MM with small default values
            elif len(self.children) == 1:
                return self.children[0].to_expression()
            else:
                # Combine multiple children with addition by default
                child_exprs = [child.to_expression() for child in self.children]
                return f"({' + '.join(child_exprs)})"
        
        # Handle entity nodes with functional forms
        if self.functional_form and self.node_type == "entity":
            # Keep the functional form - DO NOT replace parameter names with values
            # The parameters will be provided during evaluation
            return self.functional_form
        
        if self.node_type == "combine":
            if len(self.children) == 0:
                return "0.01"  # Small non-zero value instead of 0
            elif len(self.children) == 1:
                return self.children[0].to_expression()
            else:
                child_exprs = [child.to_expression() for child in self.children]
                if self.relation_type == CombineOperation.ADD:
                    return f"({' + '.join(child_exprs)})"
                elif self.relation_type == CombineOperation.MULTIPLY:
                    return f"({' * '.join(child_exprs)})"
                elif self.relation_type == CombineOperation.MAX:
                    return f"max({', '.join(child_exprs)})"
                elif self.relation_type == CombineOperation.MIN:
                    return f"min({', '.join(child_exprs)})"
                else:
                    return f"compose({', '.join(child_exprs)})"
        
        # Return basic functional form if nothing else works
        return "(0.1 * S) / (0.01 + S)"
    
    def get_complexity(self) -> int:
        complexity = 1
        complexity += len(self.parameters)
        for child in self.children:
            complexity += child.get_complexity()
        return complexity
    
    def get_all_entities(self) -> Set[str]:
        entities = set()
        if self.entity_id:
            entities.add(self.entity_id)
        for child in self.children:
            entities.update(child.get_all_entities())
        return entities
    
    def get_all_relations(self) -> List[RelationType]:
        relations = []
        if self.relation_type and isinstance(self.relation_type, RelationType):
            relations.append(self.relation_type)
        for child in self.children:
            relations.extend(child.get_all_relations())
        return relations
    
    def deep_copy(self) -> 'MechanismNode':
        new_node = MechanismNode(
            node_id=self.node_id,
            node_type=self.node_type,
            entity_id=self.entity_id,
            relation_type=self.relation_type,
            functional_form=self.functional_form,
            parameters=dict(self.parameters),  # Shallow copy is sufficient
            children=[],
            parent=None
        )
        
        for child in self.children:
            new_child = child.deep_copy()
            new_child.parent = new_node
            new_node.children.append(new_child)
        
        return new_node

@dataclass
class MDPState:
    mechanism_tree: MechanismNode
    available_entities: Set[str]
    construction_history: List[Tuple[ActionType, Any]]
    parameter_constraints: Dict[str, Tuple[float, float]]
    step_count: int = 0
    is_terminal: bool = False
    
    def get_state_representation(self) -> Dict[str, Any]:
        return {
            'tree_structure': self._tree_to_dict(self.mechanism_tree),
            'entities_used': list(self.mechanism_tree.get_all_entities()),
            'relations_used': [r.value for r in self.mechanism_tree.get_all_relations()],
            'complexity': self.mechanism_tree.get_complexity(),
            'num_parameters': len(self._get_all_parameters()),
            'step_count': self.step_count,
            'is_terminal': self.is_terminal
        }
    
    def _tree_to_dict(self, node: MechanismNode) -> Dict:
        return {
            'node_id': node.node_id,
            'node_type': node.node_type,
            'entity_id': node.entity_id,
            'relation_type': node.relation_type.value if isinstance(node.relation_type, RelationType) else str(node.relation_type),
            'functional_form': node.functional_form,
            'parameters': node.parameters,
            'children': [self._tree_to_dict(child) for child in node.children]
        }
    
    def _get_all_parameters(self) -> Dict[str, float]:
        params = {}
        
        def collect_params(node: MechanismNode):
            params.update(node.parameters)
            for child in node.children:
                collect_params(child)
        
        collect_params(self.mechanism_tree)
        return params
    
    def deep_copy(self) -> 'MDPState':
        # OPTIMIZED: Use shallow copies and limit history size
        # Limit construction history to last 20 items to prevent unbounded growth
        history = self.construction_history[-20:] if len(self.construction_history) > 20 else self.construction_history
        
        return MDPState(
            mechanism_tree=self.mechanism_tree.deep_copy(),
            available_entities=set(self.available_entities),  # Shallow copy
            construction_history=list(history),  # Limited shallow copy
            parameter_constraints=dict(self.parameter_constraints),  # Shallow copy
            step_count=self.step_count,
            is_terminal=self.is_terminal
        )

@dataclass
class Action:
    action_type: ActionType
    entity_id: Optional[str] = None
    relation_type: Optional[RelationType] = None
    position: Optional[str] = None
    parameter_name: Optional[str] = None
    parameter_delta: Optional[float] = None
    combine_operation: Optional[CombineOperation] = None
    subtree_ids: Optional[Tuple[str, str]] = None

class BiologicalMDP:
    def __init__(self, knowledge_graph: KnowledgeGraph, config: Dict, input_dimensions: int = 1):
        self.knowledge_graph = knowledge_graph
        self.config = config
        self.max_steps = config['mdp']['max_steps_per_episode']
        self.discount_factor = config['mdp']['discount_factor']
        self.node_counter = 0
        self.input_dimensions = input_dimensions  # Track number of input dimensions
        
    def create_initial_state(self) -> MDPState:
        self.node_counter = 0
        root_node = MechanismNode(
            node_id=self._generate_node_id(),
            node_type="root",
            functional_form=None  # Root doesn't have its own form, it combines children
        )
        
        all_entities = set(self.knowledge_graph.entities.keys())
        
        parameter_constraints = {}
        for param_config in ['v_max', 'k_m', 'k_i', 'hill_coefficient', 'k_on', 'k_off']:
            if param_config in self.config['biological_params']:
                bounds = self.config['biological_params'][param_config]
                # Ensure bounds are floats
                parameter_constraints[param_config] = (float(bounds['min']), float(bounds['max']))
        
        return MDPState(
            mechanism_tree=root_node,
            available_entities=all_entities,
            construction_history=[],
            parameter_constraints=parameter_constraints,
            step_count=0,
            is_terminal=False
        )
    
    def _generate_node_id(self) -> str:
        self.node_counter += 1
        return f"node_{self.node_counter}"
    
    def get_valid_actions(self, state: MDPState) -> List[Action]:
        if state.is_terminal:
            return []
        
        valid_actions = []
        
        valid_actions.extend(self._get_add_entity_actions(state))
        valid_actions.extend(self._get_modify_parameter_actions(state))
        valid_actions.extend(self._get_combine_actions(state))
        
        valid_actions.append(Action(action_type=ActionType.TERMINATE))
        
        return valid_actions
    
    def _get_add_entity_actions(self, state: MDPState) -> List[Action]:
        actions = []
        
        entities_used = state.mechanism_tree.get_all_entities()
        
        kg_valid_actions = self.knowledge_graph.get_valid_actions(entities_used)
        
        for source_entity, relation_type, target_entity in kg_valid_actions:
            if target_entity in state.available_entities:
                actions.append(Action(
                    action_type=ActionType.ADD_ENTITY,
                    entity_id=target_entity,
                    relation_type=relation_type,
                    position=source_entity
                ))
        
        # Only allow KG-validated root additions if no entities used yet
        if not entities_used:
            # Find high-confidence starting entities
            for entity_id, entity in self.knowledge_graph.entities.items():
                if entity.confidence_score >= 0.9 and entity_id in state.available_entities:
                    # Check if this entity has outgoing edges in KG
                    if entity_id in self.knowledge_graph.graph and self.knowledge_graph.graph.out_degree(entity_id) > 0:
                        # Get valid relation types for this entity
                        for neighbor in self.knowledge_graph.graph.neighbors(entity_id):
                            edge_data = self.knowledge_graph.graph.get_edge_data(entity_id, neighbor)
                            if edge_data:
                                rel_type = RelationType(edge_data['relation_type'])
                                actions.append(Action(
                                    action_type=ActionType.ADD_ENTITY,
                                    entity_id=entity_id,
                                    relation_type=rel_type,
                                    position="root"
                                ))
                                break  # One relation per entity at root
        
        return actions[:50]
    
    def _get_modify_parameter_actions(self, state: MDPState) -> List[Action]:
        actions = []
        
        all_params = self._get_all_parameters_from_tree(state.mechanism_tree)
        
        for param_name, current_value in all_params.items():
            if param_name in state.parameter_constraints:
                min_val, max_val = state.parameter_constraints[param_name]
                # Ensure values are floats for comparison
                min_val = float(min_val)
                max_val = float(max_val)
                current_value = float(current_value)
                
                for delta_factor in [0.5, 0.8, 1.2, 2.0]:
                    new_value = current_value * delta_factor
                    if min_val <= new_value <= max_val:
                        actions.append(Action(
                            action_type=ActionType.MODIFY_PARAMETER,
                            parameter_name=param_name,
                            parameter_delta=new_value - current_value
                        ))
        
        return actions[:20]
    
    def _get_combine_actions(self, state: MDPState) -> List[Action]:
        actions = []
        
        nodes = self._get_all_nodes(state.mechanism_tree)
        
        if len(nodes) >= 2:
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes):
                    if i < j and node1.parent == node2.parent:
                        for operation in [CombineOperation.ADD, CombineOperation.MULTIPLY]:
                            actions.append(Action(
                                action_type=ActionType.COMBINE_SUBTREES,
                                combine_operation=operation,
                                subtree_ids=(node1.node_id, node2.node_id)
                            ))
        
        return actions[:10]
    
    def _get_all_nodes(self, node: MechanismNode) -> List[MechanismNode]:
        nodes = [node]
        for child in node.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes
    
    def _get_all_parameters_from_tree(self, node: MechanismNode) -> Dict[str, float]:
        params = dict(node.parameters)
        for child in node.children:
            params.update(self._get_all_parameters_from_tree(child))
        return params
    
    def transition(self, state: MDPState, action: Action) -> MDPState:
        # OPTIMIZED: Only copy when we actually modify the state
        if action.action_type == ActionType.TERMINATE:
            # For terminate, just mark terminal without deep copy
            new_state = MDPState(
                mechanism_tree=state.mechanism_tree,  # Share reference
                available_entities=state.available_entities,  # Share reference
                construction_history=state.construction_history,  # Share reference
                parameter_constraints=state.parameter_constraints,  # Share reference
                step_count=state.step_count + 1,
                is_terminal=True
            )
            return new_state
        
        # Only do deep copy for actions that modify the tree
        new_state = state.deep_copy()
        new_state.step_count += 1
        
        if new_state.step_count >= self.max_steps:
            new_state.is_terminal = True
            return new_state
        
        if action.action_type == ActionType.ADD_ENTITY:
            self._apply_add_entity(new_state, action)
        elif action.action_type == ActionType.MODIFY_PARAMETER:
            self._apply_modify_parameter(new_state, action)
        elif action.action_type == ActionType.COMBINE_SUBTREES:
            self._apply_combine_subtrees(new_state, action)
        
        new_state.construction_history.append((action.action_type, action))
        
        return new_state
    
    def _apply_add_entity(self, state: MDPState, action: Action):
        constraints = self.knowledge_graph.get_constraints_for_relation(action.relation_type)
        
        if constraints:
            # Select appropriate constraint based on input dimensions
            selected_constraint = None
            
            # Prioritize multi-substrate forms for multi-dimensional data
            if self.input_dimensions > 1:
                # Look for multi-substrate forms first
                multi_substrate_keywords = ['multi_substrate', 'X0', 'X1', 'X2', 'S1', 'S2']
                for constraint in constraints:
                    if any(keyword in constraint.functional_form for keyword in multi_substrate_keywords):
                        selected_constraint = constraint
                        break
            
            # Fallback to first constraint if no multi-substrate found or single dimension
            if selected_constraint is None:
                selected_constraint = constraints[0]
            
            # Use log-scale midpoint for parameters that vary over orders of magnitude
            def get_initial_value(param_name, bounds):
                min_val, max_val = float(bounds[0]), float(bounds[1])
                # Use geometric mean for parameters with wide ranges
                if max_val / min_val > 100:  # Wide range
                    return np.sqrt(min_val * max_val)
                else:
                    return (min_val + max_val) / 2
            
            new_node = MechanismNode(
                node_id=self._generate_node_id(),
                node_type="entity",
                entity_id=action.entity_id,
                relation_type=action.relation_type,
                functional_form=selected_constraint.functional_form,
                parameters={param: get_initial_value(param, bounds)
                          for param, bounds in selected_constraint.parameter_bounds.items()}
            )
            
            for param, bounds in selected_constraint.parameter_bounds.items():
                state.parameter_constraints[f"{new_node.node_id}_{param}"] = bounds
            
            if action.position == "root":
                state.mechanism_tree.children.append(new_node)
                new_node.parent = state.mechanism_tree
            else:
                target_node = self._find_node_by_entity(state.mechanism_tree, action.position)
                if target_node:
                    target_node.children.append(new_node)
                    new_node.parent = target_node
        
        if action.entity_id in state.available_entities:
            state.available_entities.remove(action.entity_id)
    
    def _apply_modify_parameter(self, state: MDPState, action: Action):
        def modify_params(node: MechanismNode):
            if action.parameter_name in node.parameters:
                node.parameters[action.parameter_name] += action.parameter_delta
            for child in node.children:
                modify_params(child)
        
        modify_params(state.mechanism_tree)
    
    def _apply_combine_subtrees(self, state: MDPState, action: Action):
        node1 = self._find_node_by_id(state.mechanism_tree, action.subtree_ids[0])
        node2 = self._find_node_by_id(state.mechanism_tree, action.subtree_ids[1])
        
        if node1 and node2 and node1.parent == node2.parent:
            parent = node1.parent
            
            combine_node = MechanismNode(
                node_id=self._generate_node_id(),
                node_type="combine",
                relation_type=action.combine_operation,
                children=[node1, node2]
            )
            
            node1.parent = combine_node
            node2.parent = combine_node
            
            if parent:
                parent.children.remove(node1)
                parent.children.remove(node2)
                parent.children.append(combine_node)
                combine_node.parent = parent
    
    def _find_node_by_entity(self, node: MechanismNode, entity_id: str) -> Optional[MechanismNode]:
        if node.entity_id == entity_id:
            return node
        for child in node.children:
            result = self._find_node_by_entity(child, entity_id)
            if result:
                return result
        return None
    
    def _find_node_by_id(self, node: MechanismNode, node_id: str) -> Optional[MechanismNode]:
        if node.node_id == node_id:
            return node
        for child in node.children:
            result = self._find_node_by_id(child, node_id)
            if result:
                return result
        return None
    
    def is_terminal_state(self, state: MDPState) -> bool:
        return state.is_terminal or state.step_count >= self.max_steps