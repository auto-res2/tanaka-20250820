#!/usr/bin/env python3
"""
DMS-LMA 2.0: Dynamically Modularized Small Language Model Architecture
Main experimental script for comprehensive evaluation.

This script orchestrates the entire experimental pipeline:
1. Data preprocessing
2. Model training 
3. Comprehensive evaluation with three experiments
4. Results visualization and analysis
"""

import os
import sys
import time
import torch
import json
from datetime import datetime

from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model

def setup_experiment_environment():
    """
    Set up the experimental environment and check system capabilities.
    """
    print("=" * 60)
    print("DMS-LMA 2.0 EXPERIMENTAL FRAMEWORK")
    print("=" * 60)
    print(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory >= 15.0:  # Tesla T4 has ~16GB
            print("✓ GPU memory sufficient for Tesla T4 compatibility")
        else:
            print("⚠ GPU memory may be limited for some experiments")
    else:
        print("Running on CPU - experiments will be adapted accordingly")
    
    os.makedirs('.research/iteration1/images', exist_ok=True)
    print("✓ Output directory prepared")
    
    print("=" * 60)
    return torch.cuda.is_available()

def run_complete_experiment():
    """
    Run the complete DMS-LMA 2.0 experimental pipeline.
    """
    experiment_start_time = time.time()
    
    cuda_available = setup_experiment_environment()
    device = 'cuda' if cuda_available else 'cpu'
    
    try:
        print("\nSTEP 1: DATA PREPROCESSING")
        print("-" * 30)
        data = preprocess_data()
        
        print("\nSTEP 2: MODEL TRAINING")
        print("-" * 30)
        model, training_losses = train_model(data, device=device)
        
        print("\nSTEP 3: COMPREHENSIVE EVALUATION")
        print("-" * 30)
        evaluation_results = evaluate_model(model, save_dir='.research/iteration1/images')
        
        print("\nSTEP 4: GENERATING SUMMARY REPORT")
        print("-" * 30)
        
        total_experiment_time = time.time() - experiment_start_time
        
        summary_report = {
            'experiment_info': {
                'framework': 'DMS-LMA 2.0',
                'timestamp': datetime.now().isoformat(),
                'device': device,
                'total_execution_time': total_experiment_time,
                'pytorch_version': torch.__version__,
                'cuda_available': cuda_available
            },
            'training_results': {
                'final_loss': training_losses[-1] if training_losses else None,
                'training_epochs': len(training_losses),
                'loss_progression': training_losses
            },
            'evaluation_results': evaluation_results,
            'status_enum': 'stopped'  # Set status as required
        }
        
        with open('.research/iteration1/experiment_results.json', 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        print("=" * 60)
        print("EXPERIMENT COMPLETION SUMMARY")
        print("=" * 60)
        print(f"✓ Total execution time: {total_experiment_time:.2f} seconds")
        print(f"✓ Device used: {device}")
        print(f"✓ Training completed with final loss: {training_losses[-1]:.4f}")
        print(f"✓ All three experiments executed successfully")
        print(f"✓ Results saved to: .research/iteration1/")
        print(f"✓ Status: {summary_report['status_enum']}")
        
        print("\nGenerated files:")
        image_files = [
            'training_accuracy.pdf',
            'compute_cost.pdf', 
            'quantization_robustness.pdf',
            'continual_learning.pdf'
        ]
        
        for img_file in image_files:
            img_path = f'.research/iteration1/images/{img_file}'
            if os.path.exists(img_path):
                print(f"  ✓ {img_path}")
            else:
                print(f"  ✗ {img_path} (not found)")
        
        print(f"  ✓ .research/iteration1/experiment_results.json")
        
        print("\n" + "=" * 60)
        print("DMS-LMA 2.0 EXPERIMENTAL FRAMEWORK COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return summary_report
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        error_report = {
            'experiment_info': {
                'framework': 'DMS-LMA 2.0',
                'timestamp': datetime.now().isoformat(),
                'device': device,
                'error': str(e),
                'error_type': type(e).__name__
            },
            'status_enum': 'stopped'
        }
        
        with open('.research/iteration1/experiment_error.json', 'w') as f:
            json.dump(error_report, f, indent=2)
        
        raise

def test_quick_functionality():
    """
    Quick test to verify all components work correctly.
    This is a shorter version for testing purposes.
    """
    print("Running quick functionality test...")
    
    try:
        data = preprocess_data()
        print("✓ Data preprocessing works")
        
        from evaluate import experiment_dynamic_adaptation
        result = experiment_dynamic_adaptation(plot_results=False)
        print("✓ Evaluation functions work")
        
        print("✓ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_quick_functionality()
    else:
        results = run_complete_experiment()
