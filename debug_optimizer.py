#!/usr/bin/env python3
"""Debug why optimizer returns 1e10 loss"""

import numpy as np
from train import load_config
from src.synthetic_data import SyntheticDataGenerator, SystemType

def test_optimizer_evaluation():
    config = load_config('config.yml')
    
    # Generate a system
    generator = SyntheticDataGenerator(config)
    systems = generator.generate_dataset(SystemType.ENZYME_KINETICS, 1)
    system = systems[0]
    
    print("System data:")
    print(f"  X shape: {system.data_X.shape}")
    print(f"  y shape: {system.data_y.shape}")
    print(f"  X range: [{np.min(system.data_X):.3f}, {np.max(system.data_X):.3f}]")
    print(f"  y range: [{np.min(system.data_y):.3f}, {np.max(system.data_y):.3f}]")
    
    # Test manual evaluation
    expression = "(v_max * S) / (k_m + S)"
    params = {"node_2_v_max": 1.0, "node_2_k_m": 0.03}
    
    # Create safe dict for evaluation
    X = system.data_X
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    safe_dict = {
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'max': np.maximum,
        'min': np.minimum,
        'S': X[:, 0]
    }
    
    # Test with node-prefixed params
    safe_dict.update(params)
    
    print("\nAttempt 1: With node-prefixed params")
    print(f"  Safe dict keys: {list(safe_dict.keys())}")
    
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        print(f"  Result: {result[:5]}...")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test with stripped params
    safe_dict2 = {
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'max': np.maximum,
        'min': np.minimum,
        'S': X[:, 0],
        'v_max': 1.0,
        'k_m': 0.03
    }
    
    print("\nAttempt 2: With proper param names")
    print(f"  Safe dict keys: {list(safe_dict2.keys())}")
    
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict2)
        result = np.array(result)
        print(f"  Result shape: {result.shape}")
        print(f"  Result range: [{np.min(result):.3f}, {np.max(result):.3f}]")
        print(f"  First 5 values: {result[:5]}")
        
        # Calculate MSE
        mse = np.mean((system.data_y - result) ** 2)
        print(f"  MSE: {mse:.6f}")
    except Exception as e:
        print(f"  ERROR: {e}")

if __name__ == "__main__":
    test_optimizer_evaluation()