import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum

class RelationType(Enum):
    # Catalytic Relationships
    CATALYSIS = "catalysis"
    SUBSTRATE_OF = "substrate_of"
    PRODUCT_OF = "product_of"
    
    # Inhibition Relationships
    COMPETITIVE_INHIBITION = "competitive_inhibition"
    NON_COMPETITIVE_INHIBITION = "non_competitive_inhibition"
    ALLOSTERIC_REGULATION = "allosteric_regulation"
    
    # Binding Relationships
    BINDING = "binding"
    BINDS_TO = "binds_to"
    TRANSPORT = "transport"
    TRANSPORTS = "transports"
    
    # Regulatory Relationships
    INDUCES = "induces"
    REPRESSES = "represses"
    INHIBITS = "inhibits"
    PHOSPHORYLATES = "phosphorylates"
    ENZYME_INDUCTION = "enzyme_induction"
    
    # Multi-Scale Relationships
    EXPRESSED_IN = "expressed_in"
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    
    # Clinical Relationships
    CAUSES_DISEASE = "causes_disease"
    BIOMARKER_FOR = "biomarker_for"
    TREATS = "treats"
    
    # Drug Interaction Relationships
    DRUG_DRUG_INTERACTION = "drug_drug_interaction"
    
    # Legacy support
    INHIBITION = "inhibition"

@dataclass
class BiologicalEntity:
    id: str
    name: str
    entity_type: str
    properties: Dict[str, Any]
    confidence_score: float = 1.0

@dataclass
class BiologicalRelationship:
    source: str
    target: str
    relation_type: RelationType
    properties: Dict[str, Any]
    mathematical_constraints: List[str]
    confidence_score: float = 1.0

class MathematicalConstraint:
    def __init__(self, constraint_type: str, functional_form: str, parameter_bounds: Dict[str, Tuple[float, float]]):
        self.constraint_type = constraint_type
        self.functional_form = functional_form
        self.parameter_bounds = parameter_bounds
    
    def validate_parameters(self, params: Dict[str, float]) -> bool:
        for param_name, value in params.items():
            if param_name in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param_name]
                if not (min_val <= value <= max_val):
                    return False
        return True
    
    def get_functional_form(self) -> str:
        return self.functional_form

class KnowledgeGraph:
    def __init__(self, config: Dict):
        self.config = config
        self.graph = nx.DiGraph()
        self.entities: Dict[str, BiologicalEntity] = {}
        self.relationships: List[BiologicalRelationship] = []
        self.mathematical_constraints: Dict[RelationType, List[MathematicalConstraint]] = {}
        self._initialize_constraints()
        
    def _initialize_constraints(self):
        # Enzyme Kinetics Constraints
        self.mathematical_constraints[RelationType.CATALYSIS] = [
            MathematicalConstraint(
                "michaelis_menten",
                "(v_max * S) / (k_m + S)",
                {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0)}
            ),
            MathematicalConstraint(
                "hill_equation",
                "(v_max * (S ** n)) / ((k_m ** n) + (S ** n))",
                {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "n": (0.5, 4.0)}
            ),
            MathematicalConstraint(
                "multi_substrate",
                "(v_max * S1 * S2) / ((k_m1 + S1) * (k_m2 + S2))",
                {"v_max": (0.001, 1000.0), "k_m1": (0.000001, 1000.0), "k_m2": (0.000001, 1000.0)}
            ),
            MathematicalConstraint(
                "product_inhibition",
                "((v_max * S) / (k_m + S)) * (1.0 / (1.0 + P / k_p))",
                {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "k_p": (0.000001, 1000.0)}
            )
        ]
        
        # Inhibition Constraints
        self.mathematical_constraints[RelationType.COMPETITIVE_INHIBITION] = [
            MathematicalConstraint(
                "competitive_mm",
                "(v_max * S) / (k_m * (1.0 + I / k_i) + S)",
                {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "k_i": (0.000001, 1000.0)}
            ),
            MathematicalConstraint(
                "competitive_binding",
                "L / (k_d * (1.0 + C / k_c) + L)",
                {"k_d": (0.000000001, 0.001), "k_c": (0.000000001, 0.001)}
            )
        ]
        
        self.mathematical_constraints[RelationType.NON_COMPETITIVE_INHIBITION] = [
            MathematicalConstraint(
                "non_competitive_mm",
                "(v_max * S) / ((k_m + S) * (1.0 + I / k_i))",
                {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "k_i": (0.000001, 1000.0)}
            )
        ]
        
        self.mathematical_constraints[RelationType.ALLOSTERIC_REGULATION] = [
            MathematicalConstraint(
                "allosteric_hill",
                "((v_max * (S ** n)) / ((k_m ** n) + (S ** n))) * ((1.0 + alpha * A / k_a) / (1.0 + A / k_a))",
                {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "n": (0.5, 4.0), 
                 "alpha": (0.1, 10.0), "k_a": (0.000001, 1000.0)}
            )
        ]
        
        # Binding Constraints
        self.mathematical_constraints[RelationType.BINDING] = [
            MathematicalConstraint(
                "simple_binding",
                "(k_on * D * R) - (k_off * DR)",
                {"k_on": (1000.0, 1000000000.0), "k_off": (0.001, 1000.0)}
            ),
            MathematicalConstraint(
                "receptor_occupancy",
                "DR / (DR + k_d)",
                {"k_d": (0.000000001, 0.001)}
            ),
            MathematicalConstraint(
                "cooperative_binding",
                "(L ** n) / ((k_d ** n) + (L ** n))",
                {"k_d": (0.000000001, 0.001), "n": (1.0, 4.0)}
            ),
            MathematicalConstraint(
                "two_site_binding",
                "((B_max1 * L) / (k_d1 + L)) + ((B_max2 * L) / (k_d2 + L))",
                {"B_max1": (0.1, 100.0), "B_max2": (0.1, 100.0), "k_d1": (0.000000001, 0.001), "k_d2": (0.000000001, 0.001)}
            )
        ]
        
        # Transport Constraints
        self.mathematical_constraints[RelationType.TRANSPORT] = [
            MathematicalConstraint(
                "facilitated_diffusion",
                "(v_max * (S_out - S_in)) / (k_m + S_out + S_in)",
                {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0)}
            ),
            MathematicalConstraint(
                "active_transport",
                "((v_max * S) / (k_m + S)) * (ATP / (k_atp + ATP))",
                {"v_max": (0.001, 1000.0), "k_m": (0.000001, 1000.0), "k_atp": (0.000001, 0.001)}
            ),
            MathematicalConstraint(
                "ion_channel",
                "g * P_open * (V - E_rev)",
                {"g": (0.000000000001, 0.000001), "P_open": (0.0, 1.0), "E_rev": (-100.0, 100.0)}
            )
        ]
        
        # Regulatory Constraints
        self.mathematical_constraints[RelationType.ENZYME_INDUCTION] = [
            MathematicalConstraint(
                "induction_hill",
                "v_max_0 * (1.0 + (I_max * (S ** n)) / ((IC50 ** n) + (S ** n)))",
                {"v_max_0": (0.001, 1000.0), "I_max": (1.0, 10.0), 
                 "IC50": (0.000000001, 0.001), "n": (0.5, 4.0)}
            )
        ]
        
        self.mathematical_constraints[RelationType.INDUCES] = [
            MathematicalConstraint(
                "fold_induction",
                "(fold * (S ** n)) / ((EC50 ** n) + (S ** n))",
                {"fold": (1.0, 100.0), "EC50": (0.000000001, 0.001), "n": (0.5, 4.0)}
            )
        ]
        
        self.mathematical_constraints[RelationType.REPRESSES] = [
            MathematicalConstraint(
                "gene_repression",
                "1.0 / (1.0 + ((S / IC50) ** n))",
                {"IC50": (0.000000001, 0.001), "n": (0.5, 4.0)}
            )
        ]

        # General inhibitory regulation
        self.mathematical_constraints[RelationType.INHIBITS] = [
            MathematicalConstraint(
                "inhibitory_regulation",
                "1.0 / (1.0 + ((S / IC50) ** n))",
                {"IC50": (0.000000001, 0.001), "n": (0.5, 4.0)}
            )
        ]
        
        self.mathematical_constraints[RelationType.PHOSPHORYLATES] = [
            MathematicalConstraint(
                "phosphorylation_kinetics",
                "k_cat * E * S / (k_m + S) * ATP / (k_atp + ATP)",
                {"k_cat": (0.1, 1000), "k_m": (1e-6, 1e-3), "k_atp": (1e-6, 1e-3)}
            )
        ]
        
        # Multi-Scale Constraints
        self.mathematical_constraints[RelationType.EXPRESSED_IN] = [
            MathematicalConstraint(
                "expression_level",
                "basal + induced * S^n / (k_50^n + S^n)",
                {"basal": (0, 100), "induced": (0, 1000), "k_50": (1e-9, 1e-3), "n": (0.5, 4.0)}
            )
        ]
        
        # Drug Interaction Constraints
        self.mathematical_constraints[RelationType.DRUG_DRUG_INTERACTION] = [
            MathematicalConstraint(
                "drug_interaction_competitive",
                "v_i = v_max * S_i / (k_m * (1 + sum(S_j/k_ij)) + S_i)",
                {"v_max": (1e-3, 1e3), "k_m": (1e-6, 1e3), "k_ij": (1e-6, 1e3)}
            ),
            MathematicalConstraint(
                "drug_interaction_synergistic",
                "(1 + alpha * D1 * D2 / (IC50_1 * IC50_2))",
                {"alpha": (0.1, 10.0), "IC50_1": (1e-9, 1e-3), "IC50_2": (1e-9, 1e-3)}
            )
        ]
    
    def add_entity(self, entity: BiologicalEntity):
        self.entities[entity.id] = entity
        self.graph.add_node(entity.id, **entity.properties)
    
    def get_entity(self, entity_id: str) -> Optional[BiologicalEntity]:
        """Get an entity by its ID"""
        return self.entities.get(entity_id)
    
    def add_relationship(self, relationship: BiologicalRelationship):
        self.relationships.append(relationship)
        self.graph.add_edge(
            relationship.source,
            relationship.target,
            relation_type=relationship.relation_type.value,
            **relationship.properties
        )
    
    def get_valid_actions(self, current_entities: Set[str]) -> List[Tuple[str, RelationType, str]]:
        valid_actions = []
        for entity in current_entities:
            if entity in self.graph:
                for neighbor in self.graph.neighbors(entity):
                    edge_data = self.graph.get_edge_data(entity, neighbor)
                    if edge_data:
                        relation_type = RelationType(edge_data['relation_type'])
                        valid_actions.append((entity, relation_type, neighbor))
        return valid_actions
    
    def get_constraints_for_relation(self, relation_type: RelationType) -> List[MathematicalConstraint]:
        return self.mathematical_constraints.get(relation_type, [])
    
    def compute_graph_distance(self, source: str, target: str) -> float:
        try:
            path_length = nx.shortest_path_length(self.graph, source, target)
            return float(path_length)
        except nx.NetworkXNoPath:
            return float('inf')
    
    def compute_plausibility_score(self, mechanism_components: List[str], 
                                  relationships_used: List[RelationType]) -> float:
        alpha = self.config['knowledge_graph']['alpha_distance_penalty']
        core_threshold = self.config['knowledge_graph']['core_concepts_threshold']
        
        core_entities = [e_id for e_id, e in self.entities.items() 
                         if e.confidence_score >= core_threshold]
        
        if not core_entities:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for component in mechanism_components:
            if component in self.entities:
                entity = self.entities[component]
                weight = entity.confidence_score
                
                min_distance = min([self.compute_graph_distance(component, core) 
                                   for core in core_entities])
                
                if min_distance != float('inf'):
                    distance_score = np.exp(-alpha * min_distance)
                    total_score += weight * distance_score
                    total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        return 0.0
    
    def get_hierarchical_structure(self) -> Dict[int, List[str]]:
        if not self.graph.nodes():
            return {}
        
        root_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        if not root_nodes:
            root_nodes = list(self.graph.nodes())[:1]
        
        hierarchy = {}
        visited = set()
        current_level = root_nodes
        level = 0
        
        while current_level and level < 10:
            hierarchy[level] = current_level
            visited.update(current_level)
            
            next_level = []
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        next_level.append(neighbor)
            
            current_level = list(set(next_level))
            level += 1
        
        return hierarchy
    
    def validate_mechanism_consistency(self, mechanism_tree: Dict) -> bool:
        def check_node(node):
            if 'entity' in node and node['entity'] not in self.entities:
                return False
            
            if 'relation' in node and 'parent' in node:
                parent_entity = node['parent']
                child_entity = node['entity']
                
                if not self.graph.has_edge(parent_entity, child_entity):
                    return False
                
                edge_data = self.graph.get_edge_data(parent_entity, child_entity)
                if edge_data['relation_type'] != node['relation'].value:
                    return False
            
            if 'children' in node:
                for child in node['children']:
                    if not check_node(child):
                        return False
            
            return True
        
        return check_node(mechanism_tree)
    
    def get_subgraph(self, entities: Set[str], max_distance: int = 2) -> nx.DiGraph:
        subgraph_nodes = set(entities)
        
        for entity in entities:
            if entity in self.graph:
                for node in self.graph.nodes():
                    try:
                        distance = nx.shortest_path_length(self.graph, entity, node)
                        if distance <= max_distance:
                            subgraph_nodes.add(node)
                    except nx.NetworkXNoPath:
                        continue
        
        return self.graph.subgraph(subgraph_nodes).copy()
    
    def load_from_sources(self, sources: List[str]):
        """
        Load knowledge graph from external sources using KnowledgeGraphLoader.
        
        Args:
            sources: List of source names (e.g., ['GO', 'KEGG', 'DrugBank']) or file paths
        """
        try:
            from .kg_loader_unified import KnowledgeGraphBuilder
            
            builder = KnowledgeGraphBuilder(self.config)
            
            for source in sources:
                source_lower = source.lower()
                if source_lower == 'go':
                    builder.with_gene_ontology()
                elif source_lower == 'kegg':
                    builder.with_kegg_pathways()
                elif source_lower == 'drugbank':
                    # Check for auth in config
                    auth = self.config.get('kg_loader', {}).get('auth_tokens', {})
                    builder.with_drugbank(
                        auth.get('drugbank_username'),
                        auth.get('drugbank_password')
                    )
                elif source_lower == 'uniprot':
                    builder.with_uniprot()
                elif source_lower == 'chembl':
                    builder.with_chembl()
                elif source.endswith(('.json', '.obo', '.xml')):
                    builder.with_custom_json(source)
                else:
                    print(f"Warning: Unknown source '{source}', skipping...")
            
            # Build and merge the loaded graph
            loaded_kg = builder.build()
            
            # Merge entities
            for entity_id, entity in loaded_kg.entities.items():
                if entity_id not in self.entities:
                    self.add_entity(entity)
            
            # Merge relationships
            for rel in loaded_kg.relationships:
                # Check if relationship already exists
                existing = False
                for existing_rel in self.relationships:
                    if (existing_rel.source == rel.source and 
                        existing_rel.target == rel.target and
                        existing_rel.relation_type == rel.relation_type):
                        existing = True
                        break
                if not existing:
                    self.add_relationship(rel)
                    
        except ImportError:
            print("Warning: KnowledgeGraphLoader not available, load_from_sources is disabled")
            print("To enable, ensure kg_loader_unified.py is properly installed")
    
    def save(self, filepath: str):
        # Serialize relationships with a JSON-friendly relation_type
        relationships_serialized = []
        for r in self.relationships:
            rel_dict = dict(r.__dict__)
            if isinstance(r.relation_type, RelationType):
                rel_dict['relation_type'] = r.relation_type.value
            else:
                # Fallback: stringify
                rel_dict['relation_type'] = str(r.relation_type)
            relationships_serialized.append(rel_dict)

        graph_data = {
            'entities': {k: v.__dict__ for k, v in self.entities.items()},
            'relationships': relationships_serialized,
            'graph': nx.node_link_data(self.graph)
        }
        import json
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2, default=str)
    
    def load(self, filepath: str):
        import json
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        
        for entity_id, entity_data in graph_data['entities'].items():
            entity = BiologicalEntity(**entity_data)
            self.add_entity(entity)
        
        for rel_data in graph_data['relationships']:
            raw = rel_data.get('relation_type')
            parsed_type = None
            # Already an enum
            if isinstance(raw, RelationType):
                parsed_type = raw
            # Strings from earlier caches or different serializers
            elif isinstance(raw, str):
                # Handle formats like "RelationType.SUBSTRATE_OF"
                if raw.startswith('RelationType.'):
                    name = raw.split('.', 1)[1]
                    try:
                        parsed_type = RelationType[name]
                    except KeyError:
                        # Fall back to enum value if name lookup fails
                        try:
                            parsed_type = RelationType(name.lower())
                        except Exception:
                            parsed_type = None
                if parsed_type is None:
                    # Try enum NAME (e.g., SUBSTRATE_OF)
                    try:
                        parsed_type = RelationType[raw]
                    except KeyError:
                        # Try enum VALUE (e.g., "substrate_of")
                        parsed_type = RelationType(raw)
            else:
                # Unknown format – try direct construction
                try:
                    parsed_type = RelationType(raw)
                except Exception:
                    pass

            if parsed_type is None:
                raise ValueError(f"Unrecognized relation_type in cache: {raw}")
            rel_data['relation_type'] = parsed_type
            relationship = BiologicalRelationship(**rel_data)
            self.add_relationship(relationship)
        
        self.graph = nx.node_link_graph(graph_data['graph'])