#!/usr/bin/env python3
"""
⚡ Plugin CLI Manager
====================

Command-line interface for managing DS-AutoAdvisor plugins.
"""

import argparse
import sys
import os
import json
import yaml
from pathlib import Path
from datetime import datetime
import subprocess

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def check_dependencies():
    """Check plugin dependencies"""
    print("🔍 Checking Plugin Dependencies")
    print("="*35)
    
    deps_status = {
        'optuna': False,
        'evidently': False,
        'requests': False
    }
    
    for dep in deps_status.keys():
        try:
            __import__(dep)
            deps_status[dep] = True
            print(f"✅ {dep}: Installed")
        except ImportError:
            print(f"❌ {dep}: Not installed")
    
    # Installation suggestions
    missing = [dep for dep, installed in deps_status.items() if not installed]
    if missing:
        print(f"\n💡 To install missing dependencies:")
        print(f"   uv add {' '.join(missing)}")
    else:
        print(f"\n🎉 All dependencies are installed!")
    
    return deps_status

def install_dependencies(dependencies: list):
    """Install dependencies using uv"""
    print(f"📦 Installing dependencies: {', '.join(dependencies)}")
    
    try:
        cmd = ['uv', 'add'] + dependencies
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully!")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ UV not found. Please install UV first or use pip:")
        print(f"   pip install {' '.join(dependencies)}")
        return False

def show_plugin_status():
    """Show current plugin status"""
    print("🔌 Plugin Status")
    print("="*20)
    
    try:
        # Try to import the integration helper
        sys.path.append(str(project_root / "src"))
        from src.plugins.integration_helper import get_plugin_helper
        from utils.simplified_config_manager import SimplifiedConfigManager
        
        config_manager = SimplifiedConfigManager()
        plugin_helper = get_plugin_helper(config_manager)
        plugin_helper.print_plugin_status()
        
        # Show detailed status
        status = plugin_helper.get_plugin_status_summary()
        
        print(f"\n📊 Detailed Status:")
        for plugin_name, details in status['plugin_details'].items():
            print(f"\n{plugin_name}:")
            print(f"   Available: {details['available']}")
            print(f"   Enabled: {details['enabled']}")
            if details['config']:
                print(f"   Config keys: {list(details['config'].keys())}")
        
    except ImportError as e:
        print(f"❌ Cannot import plugin system: {e}")
        print("   Make sure dependencies are installed")

def generate_hpo_config(data_path: str, target_column: str = None, mode: str = "custom"):
    """Generate HPO configuration from data"""
    print(f"🔧 Generating HPO Config")
    print("="*25)
    
    try:
        import pandas as pd
        from src.plugins.optuna_hpo import HPOConfigGenerator
        from utils.simplified_config_manager import SimplifiedConfigManager
        
        # Load config to get delimiter
        config_manager = SimplifiedConfigManager()
        config = config_manager.get_global_config()
        delimiter = config.get('csv_delimiter', ',')
        
        # Load data with correct delimiter
        print(f"📊 Loading data from {data_path}...")
        df = pd.read_csv(data_path, sep=delimiter)
        
        # Prepare features and target
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            # Auto-detect target (last column)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            target_column = y.name if hasattr(y, 'name') else 'target'
        
        print(f"   Features: {X.shape[1]} columns")
        print(f"   Target: {target_column}")
        print(f"   Samples: {X.shape[0]} rows")
        
        # Generate config
        config_gen = HPOConfigGenerator()
        config, config_path = config_gen.create_complete_workflow_config(X, y, mode)
        
        print(f"✅ HPO config generated: {config_path}")
        
    except ImportError:
        print("❌ Optuna HPO plugin not available")
        print("   Install with: uv add optuna")
    except Exception as e:
        print(f"❌ Error generating config: {e}")

def run_monitoring_demo(reference_data_path: str, current_data_path: str = None,
                        target_column: str = None, drift_severity: str = "medium"):
    """Run monitoring demonstration"""
    print(f"📊 Running Monitoring Demo")
    print("="*26)
    
    try:
        import pandas as pd
        from src.plugins.evidently_monitor import EvidentiallyMonitor
        from utils.simplified_config_manager import SimplifiedConfigManager
        
        # Load config to get delimiter
        config_manager = SimplifiedConfigManager()
        config = config_manager.get_global_config()
        delimiter = config.get('csv_delimiter', ',')
        
        # Load reference data
        print(f"📊 Loading reference data from {reference_data_path}...")
        reference_df = pd.read_csv(reference_data_path, sep=delimiter)
        
        # Load or generate current data
        if current_data_path:
            print(f"📊 Loading current data from {current_data_path}...")
            current_df = pd.read_csv(current_data_path, sep=delimiter)
        else:
            print(f"🎭 Generating synthetic drift data (severity: {drift_severity})...")
            monitor = EvidentiallyMonitor()
            current_df = monitor.create_synthetic_drift_demo(
                reference_df, target_column, drift_severity
            )
        
        # Run monitoring
        print(f"🔍 Running drift detection...")
        monitor = EvidentiallyMonitor()
        
        results = monitor.run_monitoring_workflow(
            reference_df, current_df, target_column=target_column,
            workflow_name="cli_demo"
        )
        
        print(f"✅ Monitoring completed!")
        print(f"   Alert level: {results['alerts']['alert_level']}")
        print(f"   Total alerts: {len(results['alerts']['alerts'])}")
        print(f"   Report: {results.get('drift_report_path', 'N/A')}")
        
    except ImportError:
        print("❌ Evidently Monitor plugin not available")
        print("   Install with: uv add evidently")
    except Exception as e:
        print(f"❌ Error running monitoring: {e}")

def show_integration_examples():
    """Show integration examples"""
    print("🔗 Integration Examples")
    print("="*22)
    
    examples = {
        "Training Integration": """
# Enhance your existing training function with HPO
from src.plugins.training_integration import enhance_training_with_hpo
from utils.simplified_config_manager import SimplifiedConfigManager

config_manager = SimplifiedConfigManager()

@enhance_training_with_hpo(your_training_function, config_manager)
def enhanced_training(X, y, output_dir="pipeline_outputs"):
    return your_training_function(X, y, output_dir)

# Usage - automatically uses HPO if enabled in config
results = enhanced_training(X, y)
        """,
        
        "Monitoring Integration": """
# Enhance your existing evaluation function with monitoring
from src.plugins.monitoring_integration import enhance_evaluation_with_monitoring
from utils.simplified_config_manager import SimplifiedConfigManager

config_manager = SimplifiedConfigManager()

@enhance_evaluation_with_monitoring(your_evaluation_function, config_manager)
def enhanced_evaluation(X_test, y_test, models, X_train=None, y_train=None):
    return your_evaluation_function(X_test, y_test, models)

# Usage - automatically runs drift monitoring if enabled
results = enhanced_evaluation(X_test, y_test, models, X_train, y_train)
        """,
        
        "CLI Monitoring": """
# Command-line monitoring
python -m src.plugins.evidently_monitor.cli_monitor \\
    --reference-data data/train.csv \\
    --current-data data/production.csv \\
    --target-column target \\
    --config config/monitor_config.yaml

# Generate cron command
python -m src.plugins.evidently_monitor.cli_monitor \\
    --generate-cron \\
    --reference-data data/train.csv \\
    --current-data data/production.csv
        """
    }
    
    for title, example in examples.items():
        print(f"\n{title}:")
        print("-" * len(title))
        print(example)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='DS-AutoAdvisor Plugin Manager')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show plugin status')
    
    # Install command  
    install_parser = subparsers.add_parser('install', help='Install plugin dependencies')
    install_parser.add_argument('--deps', nargs='+', 
                               choices=['optuna', 'evidently', 'requests', 'all'],
                               default=['all'], help='Dependencies to install')
    
    # Check command
    subparsers.add_parser('check', help='Check plugin dependencies')
    
    # Generate HPO config
    hpo_parser = subparsers.add_parser('generate-hpo', help='Generate HPO configuration')
    hpo_parser.add_argument('--data', required=True, help='Path to data CSV file')
    hpo_parser.add_argument('--target', help='Target column name')
    hpo_parser.add_argument('--mode', choices=['fast', 'custom'], default='custom',
                           help='Configuration mode')
    
    # Monitoring demo
    monitor_parser = subparsers.add_parser('monitor-demo', help='Run monitoring demonstration')
    monitor_parser.add_argument('--reference-data', required=True, 
                               help='Path to reference data CSV')
    monitor_parser.add_argument('--current-data', 
                               help='Path to current data CSV (optional)')
    monitor_parser.add_argument('--target', help='Target column name')
    monitor_parser.add_argument('--drift-severity', choices=['low', 'medium', 'high'],
                               default='medium', help='Synthetic drift severity')
    
    # Examples command
    subparsers.add_parser('examples', help='Show integration examples')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'status':
            show_plugin_status()
        
        elif args.command == 'check':
            check_dependencies()
        
        elif args.command == 'install':
            deps_to_install = []
            if 'all' in args.deps:
                deps_to_install = ['optuna', 'evidently', 'requests']
            else:
                deps_to_install = args.deps
            install_dependencies(deps_to_install)
        
        elif args.command == 'generate-hpo':
            generate_hpo_config(args.data, args.target, args.mode)
        
        elif args.command == 'monitor-demo':
            run_monitoring_demo(args.reference_data, args.current_data, 
                              args.target, args.drift_severity)
        
        elif args.command == 'examples':
            show_integration_examples()
        
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
