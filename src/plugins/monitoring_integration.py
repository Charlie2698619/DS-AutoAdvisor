#!/usr/bin/env python3
"""
📊 Monitoring Integration for Evaluation Stage
==============================================

Integrates Evidently monitoring into the evaluation stage for drift detection.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "utils"))

def enhance_evaluation_with_monitoring(evaluation_function, config_manager):
    """
    Decorator/wrapper to enhance evaluation with monitoring capabilities.
    """
    
    def enhanced_evaluation_wrapper(X_test, y_test, models_results, 
                                   X_train=None, y_train=None,
                                   output_dir: str = "pipeline_outputs", **kwargs):
        """Enhanced evaluation function with monitoring support"""
        
        # Import here to avoid circular imports
        from src.plugins.integration_helper import get_plugin_helper
        
        plugin_helper = get_plugin_helper(config_manager)
        
        results = {
            'evaluation_results': None,
            'monitoring_results': None,
            'drift_alerts': None
        }
        
        # Run original evaluation first
        print("\n📊 Running Model Evaluation...")
        results['evaluation_results'] = evaluation_function(
            X_test, y_test, models_results, output_dir=output_dir, **kwargs
        )
        
        # Add monitoring if enabled and training data is available
        if (plugin_helper.should_monitor_drift() and 
            X_train is not None and y_train is not None):
            
            print("\n📊 Running Drift Monitoring...")
            print("="*35)
            
            try:
                # Combine training and test data for monitoring
                train_data = X_train.copy()
                if y_train is not None:
                    target_col = y_train.name if hasattr(y_train, 'name') else 'target'
                    train_data[target_col] = y_train
                
                test_data = X_test.copy()
                if y_test is not None:
                    test_data[target_col] = y_test
                
                # Run monitoring
                monitoring_results = plugin_helper.run_drift_monitoring_if_enabled(
                    reference_data=train_data,
                    current_data=test_data,
                    target_column=target_col,
                    workflow_name=f"evaluation_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                if monitoring_results:
                    results['monitoring_results'] = monitoring_results
                    results['drift_alerts'] = monitoring_results.get('alerts', {})
                    
                    # Print monitoring summary
                    summary = monitoring_results.get('summary', {})
                    print(f"✅ Monitoring completed")
                    print(f"   Alert level: {results['drift_alerts'].get('alert_level', 'INFO')}")
                    print(f"   Total alerts: {len(results['drift_alerts'].get('alerts', []))}")
                    
                    if results['drift_alerts'].get('alerts'):
                        print("⚠️  Drift issues detected:")
                        for alert in results['drift_alerts']['alerts'][:3]:  # Show first 3
                            print(f"     - {alert.get('message', 'Unknown alert')}")
                
            except Exception as e:
                print(f"❌ Monitoring failed: {e}")
                logging.warning(f"Monitoring failed: {e}")
        
        elif plugin_helper.should_monitor_drift():
            print("⚠️  Monitoring enabled but no training data provided")
            print("   Provide X_train and y_train to enable drift monitoring")
        
        return results
    
    return enhanced_evaluation_wrapper

def create_monitoring_report(evaluation_results: Dict[str, Any], 
                           monitoring_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a comprehensive report combining evaluation and monitoring results"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_summary': {},
        'monitoring_summary': {},
        'recommendations': []
    }
    
    # Extract evaluation summary
    if evaluation_results:
        # This would depend on your evaluation function's return format
        report['evaluation_summary'] = {
            'models_evaluated': len(evaluation_results.get('models', {})),
            'best_model': evaluation_results.get('best_model', 'Unknown'),
            'performance_metrics': evaluation_results.get('metrics', {})
        }
    
    # Extract monitoring summary
    if monitoring_results:
        alerts = monitoring_results.get('alerts', {})
        summary = monitoring_results.get('summary', {})
        
        report['monitoring_summary'] = {
            'alert_level': alerts.get('alert_level', 'INFO'),
            'total_alerts': len(alerts.get('alerts', [])),
            'data_samples': summary.get('data_samples', {}),
            'drift_detected': len(alerts.get('alerts', [])) > 0
        }
        
        # Generate recommendations based on monitoring results
        if report['monitoring_summary']['drift_detected']:
            report['recommendations'].extend([
                "🔍 Investigate data drift detected in the monitoring analysis",
                "📊 Review the drift report for detailed analysis",
                "🔄 Consider retraining models if drift is significant",
                "📈 Monitor model performance in production closely"
            ])
        else:
            report['recommendations'].append(
                "✅ No significant drift detected - model appears stable"
            )
    
    return report

def save_comprehensive_report(report: Dict[str, Any], output_dir: str) -> Path:
    """Save comprehensive evaluation and monitoring report"""
    
    output_path = Path(output_dir) / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        import json
        json.dump(report, f, indent=2)
    
    print(f"📋 Comprehensive report saved: {output_path}")
    return output_path

def demo_monitoring_integration():
    """Demonstrate monitoring integration"""
    
    print("📊 Monitoring Integration Demo")
    print("="*40)
    
    example_code = '''
# Example integration with your existing evaluation function

# Before (your existing code):
def evaluate_models(X_test, y_test, models_results, output_dir="pipeline_outputs"):
    # Your existing evaluation logic
    return evaluation_results

# After (with monitoring integration):
from src.plugins.monitoring_integration import enhance_evaluation_with_monitoring
from utils.simplified_config_manager import SimplifiedConfigManager

config_manager = SimplifiedConfigManager()

@enhance_evaluation_with_monitoring(evaluate_models, config_manager)
def enhanced_evaluate_models(X_test, y_test, models_results, 
                           X_train=None, y_train=None, output_dir="pipeline_outputs"):
    # Your existing evaluation logic remains unchanged
    return evaluation_results

# Usage:
results = enhanced_evaluate_models(
    X_test, y_test, models_results, 
    X_train=X_train, y_train=y_train  # Add training data for monitoring
)
# Results now include both evaluation and monitoring results
'''
    
    print("Example integration code:")
    print(example_code)
    
    print("\nMonitoring integration benefits:")
    print("✅ Automatic drift detection between train and test data")
    print("✅ HTML reports for detailed analysis")
    print("✅ Alert system for significant drift")
    print("✅ Slack integration for notifications")
    print("✅ Comprehensive reporting combining evaluation and monitoring")
    print("✅ Non-intrusive - existing evaluation code unchanged")

if __name__ == "__main__":
    demo_monitoring_integration()
