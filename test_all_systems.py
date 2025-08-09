#!/usr/bin/env python3
"""Test all enzyme kinetics system types to see which ones work"""

import numpy as np
from train import load_config
from src.synthetic_data import SyntheticDataGenerator, SystemType

def test_all_enzyme_systems():
    config = load_config('config.yml')
    generator = SyntheticDataGenerator(config)
    
    print("Testing 5 enzyme kinetics systems with different complexity levels:")
    print("="*70)
    
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 5)
    
    for i, system in enumerate(systems):
        print(f"\nSystem {i+1} (complexity {system.complexity_level}):")
        print(f"  Mechanism: {system.mechanism}")
        print(f"  Parameters: {system.true_parameters}")
        print(f"  Data shape: X={system.data_X.shape}, y={system.data_y.shape}")
        
        # Check X range
        if system.data_X.ndim == 1:
            x_min, x_max = np.min(system.data_X), np.max(system.data_X)
            print(f"  X range: [{x_min:.6f}, {x_max:.6f}]")
        else:
            for j in range(system.data_X.shape[1]):
                x_min, x_max = np.min(system.data_X[:, j]), np.max(system.data_X[:, j])
                print(f"  X[{j}] range: [{x_min:.6f}, {x_max:.6f}]")
        
        # Check y range and statistics
        y_min, y_max = np.min(system.data_y), np.max(system.data_y)
        y_mean, y_std = np.mean(system.data_y), np.std(system.data_y)
        print(f"  y range: [{y_min:.6f}, {y_max:.6f}]")
        print(f"  y stats: mean={y_mean:.6f}, std={y_std:.6f}")
        
        # Check for issues
        issues = []
        
        # Check if X values are too small
        if system.data_X.ndim == 1:
            if np.max(system.data_X) < 0.01:
                issues.append("X values too small (max < 0.01)")
        else:
            for j in range(system.data_X.shape[1]):
                if np.max(system.data_X[:, j]) < 0.01:
                    issues.append(f"X[{j}] values too small (max < 0.01)")
        
        # Check if y has low variation
        if y_std < 0.01:
            issues.append("y has very low variation (std < 0.01)")
        
        # Check if y is mostly negative (could be noise-dominated)
        if y_mean < 0:
            issues.append("y mean is negative (possible noise issue)")
        
        # Check for reasonable parameter values
        for param_name, param_value in system.true_parameters.items():
            if param_value <= 0:
                issues.append(f"Parameter {param_name}={param_value} is non-positive")
            elif param_value > 1000:
                issues.append(f"Parameter {param_name}={param_value} is very large")
        
        if issues:
            print("  ⚠️ ISSUES:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  ✅ No obvious issues")
        
        print("-"*70)

if __name__ == "__main__":
    test_all_enzyme_systems()