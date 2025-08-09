# Knowledge Graph-Guided Reinforcement Learning for Biological Mechanism Discovery

A comprehensive framework that combines biological knowledge graphs with reinforcement learning to automatically discover interpretable and biologically plausible mechanisms from experimental data.

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install -r requirements.txt
```

### End-to-End Workflow

#### 1. Build the Comprehensive Knowledge Graph

First, create the biological knowledge graph with 500+ entities covering enzymes, substrates, products, inhibitors, drugs, receptors, and disease states:

```bash
# Build and cache the comprehensive knowledge graph
python train.py --build-kg-cache

# This creates a cached KG at ./kg_cache/comprehensive_kg.json
# Contains: 50+ enzymes, 100 substrates, 50 products, 30 inhibitors, 
#          20 allosteric regulators, 30+ drugs, and more
```

#### 2. Train on Synthetic Systems

Train the RL agent to discover mechanisms from synthetic biological data:

```bash
# Train on enzyme kinetics systems using cached KG
python train.py --use-kg-cache \
                --system-type enzyme_kinetics \
                --num-systems 10 \
                --num-episodes 1000

# Other system types available:
# - multi_scale: Molecular to tissue-level coupling
# - disease_state: Biomarker-dependent mechanisms  
# - drug_interaction: Multi-drug interaction networks
# - hierarchical: Progressive complexity systems
```

#### 3. Run Experiments with Different Configurations

```bash
# Full experimental run with cross-validation
python train.py --use-kg-cache \
                --system-type enzyme_kinetics \
                --num-systems 50 \
                --num-episodes 2000 \
                --run-cv \
                --cv-folds 5

# Run ablation study to evaluate component contributions
python train.py --use-kg-cache \
                --system-type multi_scale \
                --num-systems 20 \
                --run-ablation

# Compare with baseline methods
python train.py --use-kg-cache \
                --system-type drug_interaction \
                --num-systems 30 \
                --compare-baselines
```

#### 4. Load External Knowledge Sources (Optional)

If you have access to biological databases:

```bash
# Load from multiple sources
python train.py --kg-sources GO KEGG UniProt ChEMBL \
                --system-type enzyme_kinetics \
                --num-systems 10

# Load from custom JSON file
python train.py --kg-sources ./my_custom_kg.json \
                --system-type disease_state \
                --num-systems 20
```

## 📊 System Architecture

### Core Components

1. **Knowledge Graph (KG)**
   - 500+ biological entities from `kg_builder.py`
   - 23 relationship types with mathematical constraints
   - Hierarchical organization from molecular to organism level

2. **Reinforcement Learning Agent**
   - Graph Neural Network (GNN) policy network
   - Proximal Policy Optimization (PPO) training
   - Biologically-constrained action space

3. **Mechanism Discovery Process**
   - Sequential mechanism construction via MDP
   - KG-guided valid action filtering
   - Multi-objective reward (accuracy + plausibility + interpretability)

### Knowledge Graph Structure

```
Entities (500+):
├── Enzymes (50): HK1, PFK1, CS, IDH1, ...
├── Substrates (100): Glucose, ATP, NAD+, amino acids, ...
├── Products (50): Lactate, CO2, NADH, ...
├── Inhibitors (30): Metformin, 2-DG, CB-839, ...
├── Regulators (20): AMP, ATP, Ca2+, cAMP, ...
├── Receptors (30): EGFR, INSR, GPCR, ...
├── Drugs (30): Aspirin, Ibuprofen, Statins, ...
└── Disease markers (50+): HbA1c, CRP, TNF-α, ...

Relationships (23 types):
├── Catalytic: CATALYSIS, SUBSTRATE_OF, PRODUCT_OF
├── Inhibitory: COMPETITIVE/NON_COMPETITIVE_INHIBITION
├── Regulatory: ALLOSTERIC_REGULATION, INDUCES, REPRESSES
├── Binding: BINDS_TO, TRANSPORTS
└── Clinical: BIOMARKER_FOR, TREATS, CAUSES_DISEASE
```

## 🔧 Configuration

Edit `config.yml` to customize:

```yaml
# Knowledge graph settings
knowledge_graph:
  min_confidence_score: 0.8
  core_concepts_threshold: 0.95
  
# RL training parameters  
mdp:
  max_steps_per_episode: 100
  discount_factor: 0.99
  
# Reward weights
reward:
  lambda_likelihood: 1.0      # Data fit
  lambda_plausibility: 0.5    # Biological consistency
  lambda_interpretability: 0.3 # Mechanism simplicity
```

## 📈 Output and Results

Training produces:

1. **Discovered Mechanisms**: Mathematical expressions with biological interpretation
2. **Performance Metrics**: RMSE, R², parameter recovery, plausibility scores
3. **Checkpoints**: Saved models in `./checkpoints/`
4. **Results JSON**: Detailed metrics in `./results/`

Example discovered mechanism:
```
Competitive Inhibition of Hexokinase:
v = (V_max * [Glucose]) / (K_m * (1 + [2-DG]/K_i) + [Glucose])
Parameters: V_max=158 s⁻¹, K_m=0.1 mM, K_i=5.0 mM
Plausibility Score: 0.92
```

## 🏃 Running Batch Experiments

```bash
#!/bin/bash
# experiment.sh - Run comprehensive experiments

# Build KG once
python train.py --build-kg-cache

# Run across all system types
for system in enzyme_kinetics multi_scale disease_state drug_interaction; do
    echo "Testing $system..."
    python train.py --use-kg-cache \
                    --system-type $system \
                    --num-systems 50 \
                    --num-episodes 2000 \
                    --run-cv \
                    --results-dir ./results/$system/
done

# Analyze results (create analyze_results.py if needed)
python analyze_results.py --results-dir ./results/
```

## 📚 Advanced Usage

### Custom Knowledge Graph

Create your own domain-specific KG:

```python
from src.knowledge_graph import KnowledgeGraph, BiologicalEntity, BiologicalRelationship, RelationType

# Initialize KG
kg = KnowledgeGraph(config)

# Add custom entities
entity = BiologicalEntity(
    id="my_enzyme",
    name="My Custom Enzyme",
    entity_type="enzyme",
    properties={"ec_number": "1.1.1.1", "k_cat": 100},
    confidence_score=0.95
)
kg.add_entity(entity)

# Add relationships
rel = BiologicalRelationship(
    source="my_enzyme",
    target="my_substrate",
    relation_type=RelationType.CATALYSIS,
    properties={"k_m": 0.1},
    mathematical_constraints=["michaelis_menten"],
    confidence_score=0.9
)
kg.add_relationship(rel)

# Save custom KG
kg.save("./custom_kg.json")
```

### Extending the Framework

Add new mechanism types in `src/mdp.py`:

```python
def _get_custom_actions(self, state: MDPState) -> List[Action]:
    """Add custom action types for specialized mechanisms"""
    actions = []
    # Your custom logic here
    return actions
```

## 📁 Project Structure

```
kg_rl_biomech/
├── config.yml                    # Main configuration
├── train.py                      # Training script
├── kg_builder.py                 # Comprehensive KG builder
├── requirements.txt              # Dependencies
├── .gitignore                    # Git ignore rules
├── CLAUDE.md                     # Theoretical methodology
├── docs/
│   └── kg_builder.md            # KG specification (500+ entities)
├── src/
│   ├── knowledge_graph.py      # KG implementation
│   ├── mdp.py                   # MDP formulation
│   ├── networks.py              # GNN architectures
│   ├── reward.py                # Reward functions
│   ├── ppo_trainer.py           # PPO algorithm
│   ├── parameter_optimization.py # Parameter fitting
│   ├── synthetic_data.py        # Data generation
│   ├── evaluation.py            # Metrics
│   ├── baselines.py             # Baseline methods
│   ├── visualization.py         # Plotting tools
│   ├── kg_loader.py             # KG loader wrapper
│   └── kg_loader_unified.py    # Unified KG loader
└── kg_cache/                    # Cached knowledge graphs
```

## 🎯 Key Features

- **Biologically Constrained**: All discovered mechanisms respect biological relationships
- **Interpretable**: Produces human-readable mathematical expressions
- **Scalable**: Handles graphs with 500+ entities efficiently  
- **Flexible**: Supports multiple biological system types
- **Validated**: Includes cross-validation and ablation studies
- **Cached**: Pre-built KG for fast experimentation

## 📖 Documentation

- **Methodology**: See `CLAUDE.md` for complete theoretical foundation
- **KG Specification**: See `docs/kg_builder.md` for entity details
- **API Reference**: Docstrings in source files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check existing documentation in `/docs`
- Review the methodology in `CLAUDE.md`

## 🙏 Acknowledgments

This framework integrates knowledge from:
- Gene Ontology (GO)
- KEGG Pathways  
- DrugBank
- UniProt
- ChEMBL

---

**Ready to discover biological mechanisms?** Start with the Quick Start guide above!