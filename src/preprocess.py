import torch
import numpy as np
import random

def preprocess_data():
    """
    Preprocess synthetic data for DMS-LMA 2.0 experiments.
    This function generates synthetic datasets for the three experiments.
    """
    print("Starting data preprocessing for DMS-LMA 2.0 experiments...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    num_samples = 100
    input_complexities = torch.rand(num_samples)
    hardware_conditions = torch.tensor(np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6]))
    
    quantization_levels = np.linspace(0.1, 1.0, num=10)
    
    num_tasks = 5
    
    print(f"Generated {num_samples} samples for dynamic adaptation experiment")
    print(f"Generated {len(quantization_levels)} quantization levels for robustness experiment")
    print(f"Generated {num_tasks} tasks for continual learning experiment")
    
    return {
        'dynamic_adaptation': {
            'input_complexities': input_complexities,
            'hardware_conditions': hardware_conditions
        },
        'quantization_robustness': {
            'quantization_levels': quantization_levels
        },
        'continual_learning': {
            'num_tasks': num_tasks
        }
    }

if __name__ == "__main__":
    data = preprocess_data()
    print("Data preprocessing completed successfully!")
