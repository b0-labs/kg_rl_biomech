#!/bin/bash
# Comprehensive experiment runner for KG-RL biological mechanism discovery

echo "================================================"
echo "KG-RL Biological Mechanism Discovery Experiments"
echo "================================================"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Create necessary directories
mkdir -p kg_cache
mkdir -p checkpoints
mkdir -p results
mkdir -p logs

echo ""
echo "Step 1: Building Comprehensive Knowledge Graph"
echo "----------------------------------------------"
if [ ! -f "kg_cache/comprehensive_kg.json" ]; then
    echo "Building and caching knowledge graph..."
    python train.py --build-kg-cache
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build knowledge graph"
        exit 1
    fi
    echo "✓ Knowledge graph built successfully"
else
    echo "✓ Using existing cached knowledge graph"
fi

echo ""
echo "Step 2: Running Experiments Across System Types"
echo "-----------------------------------------------"

# System types to test
SYSTEMS=("enzyme_kinetics" "multi_scale" "disease_state" "drug_interaction")

# Default parameters (can be overridden by command line args)
NUM_SYSTEMS=${1:-20}
NUM_EPISODES=${2:-1000}
CV_FOLDS=${3:-3}

echo "Configuration:"
echo "  - Number of systems per type: $NUM_SYSTEMS"
echo "  - Episodes per system: $NUM_EPISODES"
echo "  - Cross-validation folds: $CV_FOLDS"
echo ""

# Run experiments for each system type
for system in "${SYSTEMS[@]}"; do
    echo "Testing $system systems..."
    echo "=========================="
    
    # Create results directory for this system type
    mkdir -p results/$system
    
    # Run training with cross-validation
    python train.py \
        --use-kg-cache \
        --system-type $system \
        --num-systems $NUM_SYSTEMS \
        --num-episodes $NUM_EPISODES \
        --run-cv \
        --cv-folds $CV_FOLDS \
        --results-dir ./results/$system/ \
        --checkpoint-dir ./checkpoints/$system/ \
        2>&1 | tee logs/${system}_experiment.log
    
    if [ $? -eq 0 ]; then
        echo "✓ $system experiment completed successfully"
    else
        echo "✗ Warning: $system experiment encountered issues"
    fi
    echo ""
done

echo ""
echo "Step 3: Running Ablation Study"
echo "------------------------------"
echo "Testing component contributions..."

python train.py \
    --use-kg-cache \
    --system-type enzyme_kinetics \
    --num-systems 10 \
    --num-episodes 500 \
    --run-ablation \
    --results-dir ./results/ablation/ \
    2>&1 | tee logs/ablation_study.log

if [ $? -eq 0 ]; then
    echo "✓ Ablation study completed"
else
    echo "✗ Warning: Ablation study encountered issues"
fi

echo ""
echo "Step 4: Baseline Comparisons"
echo "----------------------------"
echo "Comparing with baseline methods..."

python train.py \
    --use-kg-cache \
    --system-type enzyme_kinetics \
    --num-systems 10 \
    --num-episodes $NUM_EPISODES \
    --compare-baselines \
    --results-dir ./results/baselines/ \
    2>&1 | tee logs/baseline_comparison.log

if [ $? -eq 0 ]; then
    echo "✓ Baseline comparison completed"
else
    echo "✗ Warning: Baseline comparison encountered issues"
fi

echo ""
echo "================================================"
echo "Experiment Summary"
echo "================================================"

# Count result files
TOTAL_RESULTS=$(find results -name "*.json" | wc -l)
echo "Total result files generated: $TOTAL_RESULTS"

# Show result locations
echo ""
echo "Results saved in:"
for system in "${SYSTEMS[@]}"; do
    if [ -d "results/$system" ]; then
        FILE_COUNT=$(find results/$system -name "*.json" | wc -l)
        echo "  - results/$system/ ($FILE_COUNT files)"
    fi
done

if [ -d "results/ablation" ]; then
    echo "  - results/ablation/ (ablation study)"
fi

if [ -d "results/baselines" ]; then
    echo "  - results/baselines/ (baseline comparisons)"
fi

echo ""
echo "Logs saved in:"
echo "  - logs/"

echo ""
echo "Checkpoints saved in:"
echo "  - checkpoints/"

echo ""
echo "================================================"
echo "All experiments completed!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Review results in ./results/"
echo "2. Check logs for detailed training information"
echo "3. Use saved checkpoints for further analysis"
echo "4. Run 'python analyze_results.py' for visualization (if available)"