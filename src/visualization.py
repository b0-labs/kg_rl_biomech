import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import networkx as nx
from pathlib import Path
import json

from .mdp import MDPState, MechanismNode
from .synthetic_data import SyntheticSystem

class Visualizer:
    def __init__(self, save_dir: str = "./figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        self.figure_size = (10, 6)
        self.dpi = 300
    
    def plot_training_curves(self, training_stats: Dict, save_name: str = "training_curves"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if 'episode_rewards' in training_stats:
            axes[0, 0].plot(training_stats['episode_rewards'], alpha=0.7)
            axes[0, 0].plot(pd.Series(training_stats['episode_rewards']).rolling(50).mean(), 
                          linewidth=2, label='Moving Average (50)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Episode Reward')
            axes[0, 0].set_title('Training Rewards')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        if 'episode_lengths' in training_stats:
            axes[0, 1].plot(training_stats['episode_lengths'], alpha=0.7, color='orange')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Episode Length')
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].grid(True, alpha=0.3)
        
        if 'policy_losses' in training_stats:
            axes[1, 0].plot(training_stats['policy_losses'], alpha=0.7, color='green')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Policy Loss')
            axes[1, 0].set_title('Policy Network Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'value_losses' in training_stats:
            axes[1, 1].plot(training_stats['value_losses'], alpha=0.7, color='red')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Value Loss')
            axes[1, 1].set_title('Value Network Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_mechanism_comparison(self, true_system: SyntheticSystem,
                                 discovered_state: MDPState,
                                 predictions: np.ndarray,
                                 save_name: str = "mechanism_comparison"):
        
        fig, axes = plt.subplots(1, 2, figsize=self.figure_size)
        
        data_X = true_system.data_X
        data_y = true_system.data_y
        
        if data_X.shape[1] == 1:
            axes[0].scatter(data_X[:, 0], data_y, alpha=0.6, label='True Data', s=30)
            axes[0].scatter(data_X[:, 0], predictions, alpha=0.6, label='Predictions', s=30)
            axes[0].set_xlabel('Substrate Concentration')
            axes[0].set_ylabel('Response')
            axes[0].set_xscale('log')
            axes[0].legend()
            axes[0].set_title('Data Fit')
            axes[0].grid(True, alpha=0.3)
        else:
            im = axes[0].scatter(data_X[:, 0], data_X[:, 1], c=data_y, 
                               cmap='viridis', s=50, alpha=0.7)
            axes[0].set_xlabel('Variable 1')
            axes[0].set_ylabel('Variable 2')
            axes[0].set_title('True Response Surface')
            plt.colorbar(im, ax=axes[0])
        
        residuals = data_y - predictions
        axes[1].scatter(predictions, residuals, alpha=0.6, s=30)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        std_residuals = np.std(residuals)
        axes[1].axhline(y=2*std_residuals, color='orange', linestyle='--', alpha=0.3)
        axes[1].axhline(y=-2*std_residuals, color='orange', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_parameter_recovery(self, true_params: Dict[str, float],
                               estimated_params: Dict[str, float],
                               save_name: str = "parameter_recovery"):
        
        common_params = set(true_params.keys()).intersection(set(estimated_params.keys()))
        
        if not common_params:
            return None
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        param_names = list(common_params)
        true_values = [true_params[p] for p in param_names]
        estimated_values = [estimated_params[p] for p in param_names]
        
        x = np.arange(len(param_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, true_values, width, label='True', alpha=0.8)
        bars2 = ax.bar(x + width/2, estimated_values, width, label='Estimated', alpha=0.8)
        
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Values')
        ax.set_title('Parameter Recovery Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (true_val, est_val) in enumerate(zip(true_values, estimated_values)):
            if true_val > 0:
                recovery_pct = (est_val / true_val) * 100
                ax.text(i, max(true_val, est_val) * 1.05, f'{recovery_pct:.1f}%',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_mechanism_tree(self, state: MDPState, save_name: str = "mechanism_tree"):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        G = nx.DiGraph()
        
        def add_nodes_edges(node: MechanismNode, parent_id: Optional[str] = None):
            node_label = f"{node.node_type}\n{node.entity_id or ''}"
            if node.relation_type:
                node_label += f"\n{str(node.relation_type)}"
            
            G.add_node(node.node_id, label=node_label)
            
            if parent_id:
                G.add_edge(parent_id, node.node_id)
            
            for child in node.children:
                add_nodes_edges(child, node.node_id)
        
        add_nodes_edges(state.mechanism_tree)
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=2000, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, ax=ax)
        
        labels = {node: data['label'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f"Mechanism Tree (Complexity: {state.mechanism_tree.get_complexity()})")
        ax.axis('off')
        
        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_evaluation_metrics(self, metrics: Dict[str, float], 
                               save_name: str = "evaluation_metrics"):
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metric_names)))
        
        bars = ax.bar(range(len(metric_names)), metric_values, color=colors, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Evaluation Metrics Summary')
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_convergence_analysis(self, episode_rewards_list: List[List[float]],
                                 labels: List[str], save_name: str = "convergence_analysis"):
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for rewards, label in zip(episode_rewards_list, labels):
            episodes = np.arange(len(rewards))
            axes[0].plot(episodes, rewards, alpha=0.6, label=label)
            
            moving_avg = pd.Series(rewards).rolling(50).mean()
            axes[0].plot(episodes, moving_avg, linewidth=2)
        
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Episode Reward')
        axes[0].set_title('Convergence Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        convergence_speeds = []
        for rewards, label in zip(episode_rewards_list, labels):
            final_performance = np.mean(rewards[-10:]) if len(rewards) >= 10 else rewards[-1]
            threshold = 0.9 * final_performance
            
            for i, reward in enumerate(rewards):
                if reward >= threshold:
                    convergence_speeds.append(i)
                    break
            else:
                convergence_speeds.append(len(rewards))
        
        x_pos = np.arange(len(labels))
        bars = axes[1].bar(x_pos, convergence_speeds, alpha=0.8)
        
        for bar, speed in zip(bars, convergence_speeds):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(speed)}', ha='center', va='bottom')
        
        axes[1].set_xlabel('Method')
        axes[1].set_ylabel('Episodes to 90% Performance')
        axes[1].set_title('Convergence Speed Comparison')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_ablation_study(self, ablation_results: Dict[str, Dict],
                          save_name: str = "ablation_study"):
        
        components = list(ablation_results.keys())
        metrics = list(next(iter(ablation_results.values())).keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            values = []
            for component in components:
                if metric in ablation_results[component]:
                    values.append(ablation_results[component][metric]['relative_change_%'])
                else:
                    values.append(0)
            
            colors = ['green' if v >= 0 else 'red' for v in values]
            bars = axes[idx].bar(range(len(components)), values, color=colors, alpha=0.7)
            
            axes[idx].set_xlabel('Ablated Component')
            axes[idx].set_ylabel('Relative Change (%)')
            axes[idx].set_title(f'{metric}')
            axes[idx].set_xticks(range(len(components)))
            axes[idx].set_xticklabels(components, rotation=45, ha='right')
            axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                y_pos = height if height >= 0 else height - 2
                axes[idx].text(bar.get_x() + bar.get_width()/2., y_pos,
                             f'{value:.1f}%', ha='center', 
                             va='bottom' if height >= 0 else 'top', fontsize=8)
        
        plt.suptitle('Ablation Study Results', fontsize=14, y=1.02)
        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_complexity_vs_performance(self, complexities: List[int],
                                      performances: List[float],
                                      save_name: str = "complexity_performance"):
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        scatter = ax.scatter(complexities, performances, 
                           c=performances, cmap='coolwarm',
                           s=100, alpha=0.7, edgecolors='black')
        
        z = np.polyfit(complexities, performances, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(complexities), max(complexities), 100)
        ax.plot(x_smooth, p(x_smooth), 'r--', alpha=0.5, label='Trend')
        
        ax.set_xlabel('Mechanism Complexity')
        ax.set_ylabel('Performance (1 - RMSE)')
        ax.set_title('Complexity vs Performance Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='Performance')
        
        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_summary_report(self, results: Dict, save_name: str = "summary_report"):
        from matplotlib.backends.backend_pdf import PdfPages
        
        pdf_path = self.save_dir / f"{save_name}.pdf"
        
        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle('KG-RL Biological Mechanism Discovery - Summary Report', 
                        fontsize=16, fontweight='bold')
            
            text = []
            text.append("Overall Performance Metrics:")
            if 'overall_metrics' in results:
                for metric, value in results['overall_metrics'].items():
                    text.append(f"  {metric}: {value:.4f}")
            
            text.append("\nSummary Statistics:")
            if 'summary_statistics' in results:
                for stat_type, stats in results['summary_statistics'].items():
                    if isinstance(stats, dict):
                        text.append(f"  {stat_type}:")
                        for key, val in stats.items():
                            text.append(f"    {key}: {val:.4f}")
            
            text.append("\nConfiguration:")
            if 'args' in results:
                for key, val in results['args'].items():
                    text.append(f"  {key}: {val}")
            
            plt.text(0.1, 0.9, '\n'.join(text), fontsize=10, 
                    verticalalignment='top', fontfamily='monospace')
            plt.axis('off')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        return pdf_path