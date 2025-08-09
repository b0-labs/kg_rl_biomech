import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional
from .mdp import MDPState, Action, ActionType
from .knowledge_graph import KnowledgeGraph

class GraphAttentionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, num_heads: int = 8, dropout: float = 0.1):
        super(GraphAttentionEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.gat_layers = nn.ModuleList()
        
        current_dim = input_dim
        for i in range(num_layers):
            if i == num_layers - 1:
                self.gat_layers.append(
                    GATConv(current_dim, output_dim, heads=1, dropout=dropout)
                )
                current_dim = output_dim
            else:
                self.gat_layers.append(
                    GATConv(current_dim, hidden_dim, heads=num_heads, dropout=dropout)
                )
                current_dim = hidden_dim * num_heads
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * num_heads if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        for i, (gat_layer, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_residual = x
            
            x = gat_layer(x, edge_index)
            x = norm(x)
            
            if i < self.num_layers - 1:
                x = F.leaky_relu(x, negative_slope=0.2)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
                if x_residual.shape == x.shape:
                    x = x + x_residual
        
        if batch is not None:
            x_global_mean = global_mean_pool(x, batch)
            x_global_max = global_max_pool(x, batch)
            x = torch.cat([x_global_mean, x_global_max], dim=-1)
        
        return x

class StateEncoder(nn.Module):
    def __init__(self, knowledge_graph: KnowledgeGraph, hidden_dim: int, 
                 num_gnn_layers: int, num_heads: int, dropout: float):
        super(StateEncoder, self).__init__()
        
        self.knowledge_graph = knowledge_graph
        
        self.entity_embedding = nn.Embedding(
            num_embeddings=len(knowledge_graph.entities) + 1,
            embedding_dim=hidden_dim
        )
        
        self.relation_embedding = nn.Embedding(
            num_embeddings=len(knowledge_graph.mathematical_constraints) + 1,
            embedding_dim=hidden_dim
        )
        
        self.parameter_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        self.tree_position_encoding = nn.Embedding(10, hidden_dim // 4)
        
        node_feature_dim = hidden_dim * 2 + hidden_dim // 4
        
        self.graph_encoder = GraphAttentionEncoder(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.state_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.entity_to_idx = {entity_id: idx for idx, entity_id in enumerate(knowledge_graph.entities.keys())}
        self.relation_to_idx = {}
        idx = 0
        for rel_type, constraints in knowledge_graph.mathematical_constraints.items():
            self.relation_to_idx[rel_type] = idx
            idx += 1
    
    def state_to_graph(self, state: MDPState) -> Data:
        nodes = []
        edges = []
        node_features = []
        node_depths = []
        
        def traverse_tree(node, depth=0, parent_idx=None):
            node_idx = len(nodes)
            nodes.append(node)
            node_depths.append(depth)
            
            if parent_idx is not None:
                edges.append([parent_idx, node_idx])
                edges.append([node_idx, parent_idx])
            
            entity_idx = 0
            if node.entity_id and node.entity_id in self.entity_to_idx:
                entity_idx = self.entity_to_idx[node.entity_id] + 1
            entity_emb = self.entity_embedding(torch.tensor([entity_idx]))
            
            relation_idx = 0
            if node.relation_type and node.relation_type in self.relation_to_idx:
                relation_idx = self.relation_to_idx[node.relation_type] + 1
            relation_emb = self.relation_embedding(torch.tensor([relation_idx]))
            
            avg_param_value = 0.0
            if node.parameters:
                avg_param_value = np.mean(list(node.parameters.values()))
            param_emb = self.parameter_encoder(torch.tensor([[avg_param_value]], dtype=torch.float32))
            
            depth_emb = self.tree_position_encoding(torch.tensor([min(depth, 9)]))
            
            node_feature = torch.cat([
                entity_emb.squeeze(0),
                relation_emb.squeeze(0),
                param_emb.squeeze(0),
                depth_emb.squeeze(0)
            ])
            node_features.append(node_feature)
            
            for child in node.children:
                traverse_tree(child, depth + 1, node_idx)
        
        traverse_tree(state.mechanism_tree)
        
        if not edges:
            edges = [[0, 0]]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.stack(node_features)
        
        return Data(x=x, edge_index=edge_index)
    
    def forward(self, state: MDPState) -> torch.Tensor:
        graph_data = self.state_to_graph(state)
        
        graph_embedding = self.graph_encoder(
            graph_data.x, 
            graph_data.edge_index
        )
        
        state_info = torch.tensor([
            state.mechanism_tree.get_complexity() / 100.0,
            state.step_count / 100.0,
            float(state.is_terminal)
        ], dtype=torch.float32)
        
        combined_features = torch.cat([graph_embedding, state_info])
        
        state_representation = self.state_projector(combined_features)
        
        return state_representation

class PolicyNetwork(nn.Module):
    def __init__(self, knowledge_graph: KnowledgeGraph, config: Dict):
        super(PolicyNetwork, self).__init__()
        
        self.knowledge_graph = knowledge_graph
        self.config = config
        
        hidden_dim = config['policy_network']['hidden_dim']
        num_gnn_layers = config['policy_network']['num_gnn_layers']
        num_heads = config['policy_network']['num_attention_heads']
        dropout = config['policy_network']['dropout_rate']
        
        self.state_encoder = StateEncoder(
            knowledge_graph, hidden_dim, num_gnn_layers, num_heads, dropout
        )
        
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4)
        )
        
        self.entity_selection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(knowledge_graph.entities) + 1)
        )
        
        self.relation_selection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(knowledge_graph.mathematical_constraints) + 1)
        )
        
        self.parameter_modification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 20)
        )
        
        self.combine_operation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5)
        )
    
    def forward(self, state: MDPState, valid_actions: List[Action]) -> torch.Tensor:
        state_repr = self.state_encoder(state)
        
        action_type_logits = self.action_type_head(state_repr)
        entity_logits = self.entity_selection_head(state_repr)
        relation_logits = self.relation_selection_head(state_repr)
        param_logits = self.parameter_modification_head(state_repr)
        combine_logits = self.combine_operation_head(state_repr)
        
        action_probs = []
        
        for action in valid_actions:
            if action.action_type == ActionType.ADD_ENTITY:
                action_type_prob = F.softmax(action_type_logits, dim=-1)[0]
                
                entity_idx = 0
                if action.entity_id and action.entity_id in self.state_encoder.entity_to_idx:
                    entity_idx = self.state_encoder.entity_to_idx[action.entity_id] + 1
                entity_prob = F.softmax(entity_logits, dim=-1)[entity_idx]
                
                relation_idx = 0
                if action.relation_type and action.relation_type in self.state_encoder.relation_to_idx:
                    relation_idx = self.state_encoder.relation_to_idx[action.relation_type] + 1
                relation_prob = F.softmax(relation_logits, dim=-1)[relation_idx]
                
                action_prob = action_type_prob * entity_prob * relation_prob
                
            elif action.action_type == ActionType.MODIFY_PARAMETER:
                action_type_prob = F.softmax(action_type_logits, dim=-1)[1]
                param_prob = F.softmax(param_logits, dim=-1)[0]
                action_prob = action_type_prob * param_prob
                
            elif action.action_type == ActionType.COMBINE_SUBTREES:
                action_type_prob = F.softmax(action_type_logits, dim=-1)[2]
                combine_prob = F.softmax(combine_logits, dim=-1)[0]
                action_prob = action_type_prob * combine_prob
                
            elif action.action_type == ActionType.TERMINATE:
                action_prob = F.softmax(action_type_logits, dim=-1)[3]
                
            else:
                action_prob = torch.tensor(1e-8)
            
            action_probs.append(action_prob)
        
        action_probs = torch.stack(action_probs)
        action_probs = action_probs / (action_probs.sum() + 1e-8)
        
        return action_probs
    
    def get_action(self, state: MDPState, valid_actions: List[Action], 
                   epsilon: float = 0.0) -> Tuple[Action, torch.Tensor]:
        
        if np.random.random() < epsilon:
            action_idx = np.random.choice(len(valid_actions))
            action = valid_actions[action_idx]
            log_prob = torch.log(torch.tensor(1.0 / len(valid_actions)))
        else:
            with torch.no_grad():
                action_probs = self.forward(state, valid_actions)
            
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample()
            
            action = valid_actions[action_idx.item()]
            log_prob = action_dist.log_prob(action_idx)
        
        return action, log_prob

class ValueNetwork(nn.Module):
    def __init__(self, knowledge_graph: KnowledgeGraph, config: Dict):
        super(ValueNetwork, self).__init__()
        
        self.knowledge_graph = knowledge_graph
        self.config = config
        
        hidden_dim = config['value_network']['hidden_dim']
        num_gnn_layers = config['value_network']['num_gnn_layers']
        dropout = config['policy_network']['dropout_rate']
        
        self.state_encoder = StateEncoder(
            knowledge_graph, hidden_dim, num_gnn_layers, 
            num_heads=4, dropout=dropout
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: MDPState) -> torch.Tensor:
        state_repr = self.state_encoder(state)
        value = self.value_head(state_repr)
        return value.squeeze(-1)