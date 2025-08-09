import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sympy as sp
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

from .mdp import MDPState, MechanismNode, ActionType, Action
from .knowledge_graph import KnowledgeGraph
from .parameter_optimization import ParameterOptimizer

@dataclass
class BaselineResult:
    mechanism_expression: str
    parameters: Dict[str, float]
    fitness: float
    complexity: int

class RandomSearchBaseline:
    def __init__(self, knowledge_graph: KnowledgeGraph, config: Dict):
        self.knowledge_graph = knowledge_graph
        self.config = config
        self.optimizer = ParameterOptimizer(config)
        
    def search(self, data_X: np.ndarray, data_y: np.ndarray, 
               num_iterations: int = 1000) -> BaselineResult:
        
        best_result = None
        best_fitness = float('inf')
        
        for iteration in range(num_iterations):
            mechanism = self._generate_random_mechanism()
            
            initial_params = self._initialize_parameters(mechanism)
            
            optimized_params, loss = self.optimizer.optimize_parameters(
                self._create_state_from_mechanism(mechanism, initial_params),
                data_X, data_y
            )
            
            fitness = self._evaluate_fitness(mechanism, optimized_params, data_X, data_y)
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_result = BaselineResult(
                    mechanism_expression=mechanism,
                    parameters=optimized_params,
                    fitness=fitness,
                    complexity=self._compute_complexity(mechanism)
                )
        
        return best_result
    
    def _generate_random_mechanism(self) -> str:
        templates = [
            "v_max * S / (k_m + S)",
            "v_max * S / (k_m * (1 + I/k_i) + S)",
            "v_max * S**n / (k_m**n + S**n)",
            "v_max * S / ((k_m + S) * (1 + I/k_i))",
            "v_max * S**n / (k_m**n + S**n) * (1 + alpha * A/k_a) / (1 + A/k_a)",
        ]
        
        return random.choice(templates)
    
    def _initialize_parameters(self, mechanism: str) -> Dict[str, float]:
        params = {}
        
        param_names = ['v_max', 'k_m', 'k_i', 'k_a', 'n', 'alpha']
        for param in param_names:
            if param in mechanism:
                if param in self.config['biological_params']:
                    bounds = self.config['biological_params'][param]
                    params[param] = np.random.uniform(bounds['min'], bounds['max'])
                else:
                    params[param] = np.random.uniform(0.01, 10.0)
        
        return params
    
    def _create_state_from_mechanism(self, mechanism: str, params: Dict[str, float]) -> MDPState:
        root = MechanismNode(
            node_id="baseline_root",
            node_type="root",
            functional_form=mechanism,
            parameters=params
        )
        
        from .mdp import MDPState
        state = MDPState(
            mechanism_tree=root,
            available_entities=set(),
            construction_history=[],
            parameter_constraints={k: (v['min'], v['max']) 
                                 for k, v in self.config['biological_params'].items()},
            step_count=0,
            is_terminal=True
        )
        
        return state
    
    def _evaluate_fitness(self, mechanism: str, params: Dict[str, float],
                         data_X: np.ndarray, data_y: np.ndarray) -> float:
        try:
            predictions = self._evaluate_mechanism(mechanism, params, data_X)
            if predictions is None:
                return float('inf')
            
            mse = np.mean((data_y - predictions) ** 2)
            complexity_penalty = 0.1 * self._compute_complexity(mechanism)
            
            return mse + complexity_penalty
            
        except Exception:
            return float('inf')
    
    def _evaluate_mechanism(self, mechanism: str, params: Dict[str, float],
                           X: np.ndarray) -> Optional[np.ndarray]:
        try:
            safe_dict = {
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt
            }
            
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            safe_dict['S'] = X[:, 0]
            if X.shape[1] > 1:
                safe_dict['I'] = X[:, 1]
            if X.shape[1] > 2:
                safe_dict['A'] = X[:, 2]
            
            safe_dict.update(params)
            
            result = eval(mechanism, {"__builtins__": {}}, safe_dict)
            return np.array(result)
            
        except Exception:
            return None
    
    def _compute_complexity(self, mechanism: str) -> int:
        return len(mechanism.split()) + len([c for c in mechanism if c in '*/+-**'])

class GeneticProgrammingBaseline:
    def __init__(self, config: Dict, population_size: int = 100):
        self.config = config
        self.population_size = population_size
        self.optimizer = ParameterOptimizer(config)
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.tournament_size = 5
        
    def evolve(self, data_X: np.ndarray, data_y: np.ndarray,
               num_generations: int = 50) -> BaselineResult:
        
        population = self._initialize_population()
        
        best_individual = None
        best_fitness = float('inf')
        
        for generation in range(num_generations):
            fitnesses = []
            for individual in population:
                fitness = self._evaluate_individual(individual, data_X, data_y)
                fitnesses.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = individual
            
            new_population = []
            
            elite_size = int(0.1 * self.population_size)
            elite_indices = np.argsort(fitnesses)[:elite_size]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    parent1 = self._tournament_selection(population, fitnesses)
                    parent2 = self._tournament_selection(population, fitnesses)
                    offspring = self._crossover(parent1, parent2)
                else:
                    offspring = self._tournament_selection(population, fitnesses)
                
                if random.random() < self.mutation_rate:
                    offspring = self._mutate(offspring)
                
                new_population.append(offspring)
            
            population = new_population
        
        if best_individual:
            params = self._extract_parameters(best_individual)
            optimized_params, _ = self.optimizer.optimize_parameters(
                self._create_state(best_individual, params), data_X, data_y
            )
            
            return BaselineResult(
                mechanism_expression=self._to_expression(best_individual),
                parameters=optimized_params,
                fitness=best_fitness,
                complexity=self._compute_tree_complexity(best_individual)
            )
        
        return None
    
    def _initialize_population(self) -> List[Dict]:
        population = []
        
        for _ in range(self.population_size):
            depth = random.randint(2, 5)
            tree = self._generate_random_tree(depth)
            population.append(tree)
        
        return population
    
    def _generate_random_tree(self, max_depth: int, current_depth: int = 0) -> Dict:
        if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
            terminal_types = ['variable', 'constant']
            terminal_type = random.choice(terminal_types)
            
            if terminal_type == 'variable':
                return {'type': 'variable', 'name': random.choice(['S', 'I', 'A'])}
            else:
                return {'type': 'constant', 'value': random.uniform(0.01, 10.0)}
        
        operators = ['+', '-', '*', '/', '**']
        operator = random.choice(operators)
        
        if operator in ['**']:
            return {
                'type': 'operator',
                'op': operator,
                'left': self._generate_random_tree(max_depth, current_depth + 1),
                'right': {'type': 'constant', 'value': random.uniform(0.5, 3.0)}
            }
        else:
            return {
                'type': 'operator',
                'op': operator,
                'left': self._generate_random_tree(max_depth, current_depth + 1),
                'right': self._generate_random_tree(max_depth, current_depth + 1)
            }
    
    def _evaluate_individual(self, tree: Dict, data_X: np.ndarray,
                           data_y: np.ndarray) -> float:
        try:
            expression = self._to_expression(tree)
            params = self._extract_parameters(tree)
            
            predictions = self._evaluate_expression(expression, params, data_X)
            if predictions is None:
                return float('inf')
            
            mse = np.mean((data_y - predictions) ** 2)
            complexity_penalty = 0.05 * self._compute_tree_complexity(tree)
            
            return mse + complexity_penalty
            
        except Exception:
            return float('inf')
    
    def _to_expression(self, tree: Dict) -> str:
        if tree['type'] == 'variable':
            return tree['name']
        elif tree['type'] == 'constant':
            return f"param_{id(tree)}"
        elif tree['type'] == 'operator':
            left_expr = self._to_expression(tree['left'])
            right_expr = self._to_expression(tree['right'])
            return f"({left_expr} {tree['op']} {right_expr})"
        return "1.0"
    
    def _extract_parameters(self, tree: Dict) -> Dict[str, float]:
        params = {}
        
        def extract(node):
            if node['type'] == 'constant':
                params[f"param_{id(node)}"] = node['value']
            elif node['type'] == 'operator':
                extract(node['left'])
                extract(node['right'])
        
        extract(tree)
        return params
    
    def _tournament_selection(self, population: List[Dict],
                            fitnesses: List[float]) -> Dict:
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]
        return population[winner_idx]
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        import copy
        offspring = copy.deepcopy(parent1)
        
        def get_random_node(tree, prob=0.9):
            if tree['type'] != 'operator' or random.random() > prob:
                return tree
            
            if random.random() < 0.5:
                return get_random_node(tree['left'], prob * 0.9)
            else:
                return get_random_node(tree['right'], prob * 0.9)
        
        def replace_random_node(tree, replacement, prob=0.9):
            if tree['type'] != 'operator' or random.random() > prob:
                return replacement
            
            if random.random() < 0.5:
                tree['left'] = replace_random_node(tree['left'], replacement, prob * 0.9)
            else:
                tree['right'] = replace_random_node(tree['right'], replacement, prob * 0.9)
            
            return tree
        
        subtree = get_random_node(parent2)
        offspring = replace_random_node(offspring, copy.deepcopy(subtree))
        
        return offspring
    
    def _mutate(self, tree: Dict) -> Dict:
        import copy
        mutated = copy.deepcopy(tree)
        
        def mutate_node(node):
            if node['type'] == 'constant':
                node['value'] *= random.uniform(0.5, 2.0)
            elif node['type'] == 'variable':
                if random.random() < 0.3:
                    node['name'] = random.choice(['S', 'I', 'A'])
            elif node['type'] == 'operator':
                if random.random() < 0.2:
                    node['op'] = random.choice(['+', '-', '*', '/'])
                mutate_node(node['left'])
                mutate_node(node['right'])
        
        mutate_node(mutated)
        return mutated
    
    def _compute_tree_complexity(self, tree: Dict) -> int:
        if tree['type'] in ['variable', 'constant']:
            return 1
        return 1 + self._compute_tree_complexity(tree['left']) + \
               self._compute_tree_complexity(tree['right'])
    
    def _evaluate_expression(self, expression: str, params: Dict[str, float],
                           X: np.ndarray) -> Optional[np.ndarray]:
        try:
            safe_dict = {'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt}
            
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            safe_dict['S'] = X[:, 0]
            if X.shape[1] > 1:
                safe_dict['I'] = X[:, 1]
            if X.shape[1] > 2:
                safe_dict['A'] = X[:, 2]
            
            safe_dict.update(params)
            
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return np.array(result)
            
        except Exception:
            return None
    
    def _create_state(self, tree: Dict, params: Dict[str, float]) -> MDPState:
        from .mdp import MDPState, MechanismNode
        
        root = MechanismNode(
            node_id="gp_root",
            node_type="root",
            functional_form=self._to_expression(tree),
            parameters=params
        )
        
        state = MDPState(
            mechanism_tree=root,
            available_entities=set(),
            construction_history=[],
            parameter_constraints={k: (v['min'], v['max']) 
                                 for k, v in self.config['biological_params'].items()},
            step_count=0,
            is_terminal=True
        )
        
        return state

class UnconstrainedRLBaseline:
    def __init__(self, mdp, policy_network, value_network, reward_function, config: Dict):
        self.mdp = mdp
        self.original_kg = mdp.knowledge_graph
        self.policy_network = policy_network
        self.value_network = value_network
        self.reward_function = reward_function
        self.config = config
        
    def train(self, data_X: np.ndarray, data_y: np.ndarray,
              num_episodes: int = 1000) -> BaselineResult:
        
        self.mdp.knowledge_graph = self._create_unconstrained_graph()
        
        from .ppo_trainer import PPOTrainer
        trainer = PPOTrainer(
            self.mdp, self.policy_network, self.value_network,
            self.reward_function, self.config
        )
        
        for episode in range(num_episodes):
            trainer.train_episode(data_X, data_y)
            
            if episode % 10 == 0 and episode > 0:
                trainer.update_networks()
        
        best_mechanism = trainer.get_best_mechanism()
        
        self.mdp.knowledge_graph = self.original_kg
        
        if best_mechanism:
            from .parameter_optimization import ParameterOptimizer
            optimizer = ParameterOptimizer(self.config)
            
            optimized_params, loss = optimizer.optimize_parameters(
                best_mechanism, data_X, data_y
            )
            
            return BaselineResult(
                mechanism_expression=best_mechanism.mechanism_tree.to_expression(),
                parameters=optimized_params,
                fitness=loss,
                complexity=best_mechanism.mechanism_tree.get_complexity()
            )
        
        return None
    
    def _create_unconstrained_graph(self) -> KnowledgeGraph:
        unconstrained_kg = KnowledgeGraph(self.config)
        
        for i in range(20):
            entity = BiologicalEntity(
                f"entity_{i}",
                f"Entity {i}",
                "generic",
                {},
                1.0
            )
            unconstrained_kg.add_entity(entity)
        
        from itertools import combinations
        for entity1, entity2 in combinations(range(10), 2):
            for rel_type in RelationType:
                relationship = BiologicalRelationship(
                    f"entity_{entity1}",
                    f"entity_{entity2}",
                    rel_type,
                    {},
                    [],
                    0.5
                )
                unconstrained_kg.add_relationship(relationship)
        
        return unconstrained_kg