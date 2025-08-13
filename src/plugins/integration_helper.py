#!/usr/bin/env python3
"""
🔗 Plugin Integration Helper
===========================

Helper functions to integrate plugins with the existing DS-AutoAdvisor pipeline.
"""

import sys
import os
import yaml
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "utils"))

from simplified_config_manager import SimplifiedConfigManager

class PluginIntegrationHelper:
    """Helper class for integrating plugins with the main pipeline"""
    
    def __init__(self, config_manager: SimplifiedConfigManager = None):
        """Initialize the plugin integration helper"""
        self.config_manager = config_manager or SimplifiedConfigManager()
        self.logger = self._setup_logging()
        
        # Check plugin availability
        self.available_plugins = self._check_plugin_availability()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for plugin integration"""
        logger = logging.getLogger('plugin_integration')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _detect_csv_delimiter(self, file_path: str) -> str:
        """Automatically detect CSV delimiter for a given file"""
        common_delimiters = [',', ';', '\t', '|']
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            
            # Count occurrences of each delimiter
            delimiter_counts = {}
            for delimiter in common_delimiters:
                delimiter_counts[delimiter] = first_line.count(delimiter)
            
            # Return the delimiter with the highest count (if > 0)
            best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            if delimiter_counts[best_delimiter] > 0:
                self.logger.info(f"🔍 Detected delimiter '{best_delimiter}' for {file_path}")
                return best_delimiter
            
            # Fallback to config setting
            global_config = self.config_manager.get_global_config()
            return global_config.get('csv_delimiter', ',')
            
        except Exception as e:
            self.logger.warning(f"Failed to detect delimiter for {file_path}: {e}")
            return self.config_manager.get_setting('global.csv_delimiter', ',')
    
    def load_data_with_auto_delimiter(self, file_path: str) -> pd.DataFrame:
        """Load CSV data with automatic delimiter detection"""
        delimiter = self._detect_csv_delimiter(file_path)
        
        try:
            df = pd.read_csv(file_path, sep=delimiter)
            self.logger.info(f"📊 Loaded data from {file_path} (shape: {df.shape}, delimiter: '{delimiter}')")
            return df
        except Exception as e:
            # Fallback to common delimiters
            for fallback_delimiter in [',', ';', '\t']:
                try:
                    df = pd.read_csv(file_path, sep=fallback_delimiter)
                    self.logger.info(f"📊 Loaded data with fallback delimiter '{fallback_delimiter}' (shape: {df.shape})")
                    return df
                except:
                    continue
            
            raise Exception(f"Could not load {file_path} with any common delimiter")
    
    def find_target_column(self, df: pd.DataFrame, target_hints: List[str] = None) -> Optional[str]:
        """Find the target column in dataframe using hints"""
        if target_hints is None:
            # Use config target column variants
            global_config = self.config_manager.get_global_config()
            target_hints = global_config.get('target_column_variants', ['Churn', 'Churn_binary_0'])
        
        for hint in target_hints:
            if hint in df.columns:
                self.logger.info(f"🎯 Found target column: {hint}")
                return hint
        
        # Look for any column containing 'churn' (case insensitive)
        churn_columns = [col for col in df.columns if 'churn' in col.lower()]
        if churn_columns:
            self.logger.info(f"🎯 Found churn-related column: {churn_columns[0]}")
            return churn_columns[0]
        
        self.logger.warning("❌ No target column found")
        return None
    
    def _check_plugin_availability(self) -> Dict[str, bool]:
        """Check which plugins are available"""
        available = {}
        
        # Check Optuna HPO
        try:
            import optuna
            from src.plugins.optuna_hpo import OptunaHPOManager
            available['optuna_hpo'] = True
            self.logger.info("✅ Optuna HPO plugin available")
        except ImportError as e:
            available['optuna_hpo'] = False
            self.logger.info(f"❌ Optuna HPO plugin not available: {e}")
        
        # Check Evidently Monitor - import without problematic components
        try:
            import evidently
            # Just check if evidently is importable, don't import the specific classes yet
            available['evidently_monitor'] = True
            self.logger.info("✅ Evidently Monitor plugin available")
        except ImportError as e:
            available['evidently_monitor'] = False
            self.logger.info(f"❌ Evidently Monitor plugin not available: {e}")
        
        return available
        plugins = {}
        
        # Check Optuna HPO plugin
        try:
            import optuna
            from src.plugins.optuna_hpo import OptunaHPOManager
            plugins['optuna_hpo'] = True
            self.logger.info("✅ Optuna HPO plugin available")
        except ImportError:
            plugins['optuna_hpo'] = False
            self.logger.info("❌ Optuna HPO plugin not available (install: uv add optuna)")
        
        # Check Evidently Monitor plugin
        try:
            import evidently
            from src.plugins.evidently_monitor import EvidentiallyMonitor
            plugins['evidently_monitor'] = True
            self.logger.info("✅ Evidently Monitor plugin available")
        except ImportError:
            plugins['evidently_monitor'] = False
            self.logger.info("❌ Evidently Monitor plugin not available (install: uv add evidently)")
        
        return plugins
    
    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled in configuration"""
        try:
            plugin_config = self.config_manager.config.get('infrastructure', {}).get('plugins', {})
            return (plugin_config.get(plugin_name, {}).get('enabled', False) and 
                    self.available_plugins.get(plugin_name, False))
        except Exception:
            return False
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin"""
        try:
            plugins_config = self.config_manager.config.get('infrastructure', {}).get('plugins', {})
            return plugins_config.get(plugin_name, {})
        except Exception:
            return {}
    
    def should_use_optuna_hpo(self, mode: str = "custom") -> bool:
        """Check if Optuna HPO should be used for training"""
        if not self.is_plugin_enabled('optuna_hpo'):
            return False
        
        # Check if tuning is enabled in the training config
        try:
            training_config = self.config_manager.get_mode_config(mode, "model_training")
            return training_config.get('enable_tuning', False)
        except Exception:
            return False
    
    def get_optuna_hpo_manager(self, output_base: str = "pipeline_outputs"):
        """Get Optuna HPO manager if available"""
        if not self.is_plugin_enabled('optuna_hpo'):
            self.logger.warning("Optuna HPO plugin not enabled or available")
            return None
        
        try:
            from src.plugins.optuna_hpo import OptunaHPOManager
            return OptunaHPOManager(self.config_manager, output_base)
        except ImportError:
            self.logger.error("Failed to import Optuna HPO manager")
            return None
    
    def get_evidently_monitor(self, output_base: str = "pipeline_outputs"):
        """Get Evidently monitor if available"""
        if not self.is_plugin_enabled('evidently_monitor'):
            self.logger.warning("Evidently Monitor plugin not enabled or available")
            return None
        
        try:
            from src.plugins.evidently_monitor import EvidentiallyMonitor
            return EvidentiallyMonitor(output_base)
        except ImportError:
            self.logger.error("Failed to import Evidently Monitor")
            return None
    
    def generate_hpo_config_if_needed(self, X, y, mode: str = "custom", 
                                      output_dir: str = None) -> Optional[Path]:
        """Generate HPO config if auto-generation is enabled
        
        NOTE: This is now deprecated as Optuna handles best params automatically.
        The HPO workflow reads configuration directly from unified_config_v3.yaml.
        """
        if not self.is_plugin_enabled('optuna_hpo'):
            return None
        
        plugin_config = self.get_plugin_config('optuna_hpo')
        
        # Check if auto-generation is explicitly enabled (default: False)
        if not plugin_config.get('auto_yaml_generation', False):
            self.logger.info("🔧 HPO auto-generation disabled - using direct YAML configuration")
            return None
        
        # If explicitly enabled, warn about redundancy but still generate
        self.logger.warning("⚠️ HPO auto-generation is enabled but redundant - Optuna handles best params automatically")
        
        try:
            from src.plugins.optuna_hpo import HPOConfigGenerator
            
            if output_dir is None:
                output_dir = "pipeline_outputs"
            
            config_gen = HPOConfigGenerator(output_dir)
            config, config_path = config_gen.create_complete_workflow_config(X, y, mode)
            
            self.logger.info(f"🔧 Auto-generated HPO config: {config_path}")
            return config_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate HPO config: {e}")
            return None
    
    def should_monitor_drift(self) -> bool:
        """Check if drift monitoring should be performed"""
        return self.is_plugin_enabled('evidently_monitor')
    
    def run_drift_monitoring_if_enabled(self, reference_data, current_data, 
                                        target_column: str = None,
                                        workflow_name: str = None) -> Optional[Dict[str, Any]]:
        """Run drift monitoring if enabled"""
        if not self.should_monitor_drift():
            return None
        
        try:
            monitor = self.get_evidently_monitor()
            if monitor is None:
                return None
            
            # Set thresholds from config
            plugin_config = self.get_plugin_config('evidently_monitor')
            drift_thresholds = plugin_config.get('settings', {}).get('drift_thresholds', {})
            if drift_thresholds:
                monitor.set_drift_thresholds(drift_thresholds)
            
            # Run monitoring workflow
            results = monitor.run_monitoring_workflow(
                reference_data, current_data, target_column, 
                workflow_name=workflow_name
            )
            
            self.logger.info(f"📊 Drift monitoring completed: {results['summary']}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to run drift monitoring: {e}")
            return None
    
    def get_plugin_status_summary(self) -> Dict[str, Any]:
        """Get summary of plugin status"""
        summary = {
            'plugins_available': len([p for p in self.available_plugins.values() if p]),
            'plugins_enabled': 0,
            'plugin_details': {}
        }
        
        for plugin_name in ['optuna_hpo', 'evidently_monitor']:
            is_available = self.available_plugins.get(plugin_name, False)
            is_enabled = self.is_plugin_enabled(plugin_name)
            
            if is_enabled:
                summary['plugins_enabled'] += 1
            
            summary['plugin_details'][plugin_name] = {
                'available': is_available,
                'enabled': is_enabled,
                'config': self.get_plugin_config(plugin_name) if is_enabled else {}
            }
        
        return summary
    
    def print_plugin_status(self):
        """Print plugin status for user"""
        status = self.get_plugin_status_summary()
        
        print(f"\n🔌 Plugin Status Summary:")
        print(f"   Available: {status['plugins_available']}/2")
        print(f"   Enabled: {status['plugins_enabled']}/2")
        
        for plugin_name, details in status['plugin_details'].items():
            icon = "✅" if details['enabled'] else "❌" if details['available'] else "📦"
            status_text = "Enabled" if details['enabled'] else "Available" if details['available'] else "Not Installed"
            print(f"   {icon} {plugin_name}: {status_text}")
        
        # Installation suggestions
        missing_plugins = [name for name, details in status['plugin_details'].items() 
                          if not details['available']]
        if missing_plugins:
            print(f"\n💡 To install missing plugins:")
            for plugin in missing_plugins:
                if plugin == 'optuna_hpo':
                    print(f"   uv add optuna")
                elif plugin == 'evidently_monitor':
                    print(f"   uv add evidently")


# Global instance for easy access
_plugin_helper = None

def get_plugin_helper(config_manager: SimplifiedConfigManager = None) -> PluginIntegrationHelper:
    """Get global plugin helper instance"""
    global _plugin_helper
    if _plugin_helper is None:
        _plugin_helper = PluginIntegrationHelper(config_manager)
    return _plugin_helper
