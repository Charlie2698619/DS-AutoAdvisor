#!/usr/bin/env python3
"""
🎯 Training Stage Plugin Integration
===================================

Extends the training stage to support Optuna HPO plugin with backward compatibility.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import json

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "utils"))

def enhance_training_with_hpo(training_function, config_manager, mode: str = "custom"):
    """
    Decorator/wrapper to enhance training with HPO capabilities.
    
    This is designed to work with your existing training functions while adding HPO support.
    """
    
    def enhanced_training_wrapper(X, y, output_dir: str = "pipeline_outputs", **kwargs):
        """Enhanced training function with HPO support"""
        
        # Import here to avoid circular imports
        from src.plugins.integration_helper import get_plugin_helper
        
        plugin_helper = get_plugin_helper(config_manager)
        
        # Print plugin status
        plugin_helper.print_plugin_status()
        
        results = {
            'original_results': None,
            'hpo_results': None,
            'comparison': None,
            'recommended_config': None
        }
        
        # Check if HPO should be used
        if plugin_helper.should_use_optuna_hpo(mode):
            print("\n🎯 HPO-Enhanced Training Mode")
            print("="*40)
            
            try:
                # Generate auto config if enabled
                hpo_config_path = plugin_helper.generate_hpo_config_if_needed(X, y, mode, output_dir)
                if hpo_config_path:
                    results['recommended_config'] = str(hpo_config_path)
                
                # Get HPO manager
                hpo_manager = plugin_helper.get_optuna_hpo_manager(output_dir)
                if hpo_manager:
                    # Run HPO-enhanced training
                    results['hpo_results'] = run_hpo_training(
                        hpo_manager, X, y, mode, output_dir, **kwargs
                    )
                    
                    print(f"✅ HPO training completed")
                    
                    # Also run original training for comparison if requested
                    plugin_config = plugin_helper.get_plugin_config('optuna_hpo')
                    if plugin_config.get('fallback_to_default', True):
                        print("\n📊 Running default training for comparison...")
                        results['original_results'] = training_function(X, y, output_dir, **kwargs)
                        
                        # Compare results
                        results['comparison'] = compare_hpo_vs_default(
                            results['hpo_results'], results['original_results']
                        )
                    
                    return results
                    
            except Exception as e:
                print(f"❌ HPO training failed: {e}")
                print("🔄 Falling back to default training...")
                logging.warning(f"HPO training failed, using fallback: {e}")
        
        # Run original training (default behavior)
        print("\n🏋️ Standard Training Mode")
        print("="*30)
        results['original_results'] = training_function(X, y, output_dir, **kwargs)
        
        return results
    
    return enhanced_training_wrapper

def run_hpo_training(hpo_manager, X, y, mode: str, output_dir: str, **kwargs) -> Dict[str, Any]:
    """Run HPO-enhanced training"""
    
    try:
        from src.plugins.optuna_hpo import HPOConfigGenerator, ModelOptimizer
        
        # Get HPO configuration
        hpo_config = hpo_manager.get_hpo_config(mode)
        
        # Initialize components
        config_gen = HPOConfigGenerator(output_dir)
        optimizer = ModelOptimizer(output_dir=output_dir)
        
        # Analyze data and recommend models
        data_analysis = config_gen.analyze_data_characteristics(X, y)
        recommended_models = config_gen.recommend_models_for_data(data_analysis)
        
        print(f"🤖 Recommended models: {', '.join(recommended_models)}")
        print(f"🔧 HPO iterations: {hpo_config['iterations']}")
        
        # Define parameter spaces (simplified for demo)
        model_parameter_spaces = {}
        
        # Add parameter spaces for recommended models
        for model_name in recommended_models[:3]:  # Limit to top 3 for efficiency
            if model_name in config_gen.parameter_spaces:
                model_parameter_spaces[model_name] = config_gen.parameter_spaces[model_name]
        
        if not model_parameter_spaces:
            # Fallback to RandomForest if no other models available
            model_parameter_spaces = {
                'RandomForestClassifier': config_gen.parameter_spaces.get(
                    'RandomForestClassifier', {
                        'n_estimators': {'type': 'int', 'low': 50, 'high': 200, 'step': 50},
                        'max_depth': {'type': 'int', 'low': 3, 'high': 15}
                    }
                )
            }
        
        # Run optimization for each model
        optimization_results = optimizer.optimize_multiple_models(
            model_parameter_spaces, X, y, 
            n_trials=hpo_config['iterations'], 
            cv_folds=hpo_config['cv_folds']
        )
        
        # Get best model across all optimizations
        best_model, best_params, best_score = optimizer.get_best_model_across_all()
        
        # Save results
        results_path = optimizer.save_optimization_results(
            f"hpo_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # Save study results for each model
        study_results = {}
        for model_name, study in optimization_results.items():
            study_results[model_name] = hpo_manager.save_study_results(
                study, f"study_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'all_models': optimization_results,
            'data_analysis': data_analysis,
            'recommended_models': recommended_models,
            'results_path': str(results_path),
            'study_results': study_results,
            'hpo_config': hpo_config
        }
        
    except Exception as e:
        logging.error(f"Error in HPO training: {e}")
        raise e

def compare_hpo_vs_default(hpo_results: Dict[str, Any], 
                          default_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare HPO results with default training results"""
    
    comparison = {
        'hpo_better': False,
        'improvement': 0.0,
        'summary': "No comparison available"
    }
    
    try:
        hpo_score = hpo_results.get('best_score', 0)
        
        # Try to extract best score from default results
        # This would depend on your existing training function's return format
        default_score = 0
        if isinstance(default_results, dict):
            # Assuming your training function returns something with performance metrics
            if 'best_score' in default_results:
                default_score = default_results['best_score']
            elif 'accuracy' in default_results:
                default_score = default_results['accuracy']
            elif 'performance' in default_results:
                default_score = default_results['performance']
        
        if hpo_score > 0 and default_score > 0:
            improvement = ((hpo_score - default_score) / default_score) * 100
            comparison = {
                'hpo_better': hpo_score > default_score,
                'improvement': improvement,
                'hpo_score': hpo_score,
                'default_score': default_score,
                'summary': f"HPO {'improved' if improvement > 0 else 'decreased'} performance by {abs(improvement):.2f}%"
            }
            
            print(f"\n📈 HPO vs Default Comparison:")
            print(f"   HPO score: {hpo_score:.4f}")
            print(f"   Default score: {default_score:.4f}")
            print(f"   {comparison['summary']}")
        
    except Exception as e:
        logging.warning(f"Error comparing results: {e}")
    
    return comparison

def create_training_integration_example():
    """Create an example of how to integrate HPO with existing training"""
    
    example_code = '''
# Example integration with your existing training function

# Before (your existing code):
def train_models(X, y, output_dir="pipeline_outputs"):
    # Your existing training logic
    return training_results

# After (with HPO integration):
from src.plugins.training_integration import enhance_training_with_hpo
from utils.simplified_config_manager import SimplifiedConfigManager

config_manager = SimplifiedConfigManager()

@enhance_training_with_hpo(train_models, config_manager, mode="custom")
def enhanced_train_models(X, y, output_dir="pipeline_outputs"):
    # Your existing training logic remains unchanged
    return training_results

# Usage:
results = enhanced_train_models(X, y)
# Results now include both HPO and original results for comparison
'''
    
    return example_code

# Example demonstration function
def demo_training_integration():
    """Demonstrate training integration"""
    
    print("🎯 Training Integration Demo")
    print("="*40)
    
    # This would be called from your main training pipeline
    example = create_training_integration_example()
    print("Example integration code:")
    print(example)
    
    print("\nIntegration benefits:")
    print("✅ Backward compatibility - existing code unchanged")
    print("✅ Auto HPO config generation")
    print("✅ Model recommendations based on data analysis") 
    print("✅ Comparison between HPO and default results")
    print("✅ Fallback to default if HPO fails")
    print("✅ Comprehensive logging and reporting")

if __name__ == "__main__":
    demo_training_integration()
