"""
Simplified Configuration Manager for DS-AutoAdvisor v3.0
Two-Mode System: FAST and CUSTOM
Complete YAML control with no hardcoded settings
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

class SimplifiedConfigManager:
    """Simplified configuration manager with fast/custom mode switching"""
    
    def __init__(self, config_path: str = "config/unified_config_v3.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.current_mode = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def get_mode_config(self, mode: str, stage: str) -> Dict[str, Any]:
        """
        Get configuration for a specific mode and stage
        
        Args:
            mode: "fast" or "custom"
            stage: "data_discovery", "data_cleaning", "ml_advisory", "model_training", "model_evaluation"
        
        Returns:
            Configuration dictionary for the specified mode and stage
        """
        if mode not in ["fast", "custom"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'fast' or 'custom'")
        
        self.current_mode = mode
        mode_key = f"{mode}_mode"
        
        if mode_key not in self.config:
            raise ValueError(f"Mode configuration not found: {mode_key}")
        
        if stage not in self.config[mode_key]:
            raise ValueError(f"Stage configuration not found: {stage} in {mode_key}")
        
        # Get the stage configuration
        stage_config = self.config[mode_key][stage].copy()
        
        # Add global configuration values
        stage_config.update(self.config.get("global", {}))
        
        return stage_config
    
    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration shared by all modes"""
        return self.config.get("global", {})
    
    def get_infrastructure_config(self) -> Dict[str, Any]:
        """Get infrastructure configuration"""
        return self.config.get("infrastructure", {})
    
    def get_business_features_config(self) -> Dict[str, Any]:
        """Get business features configuration"""
        return self.config.get("business_features", {})
    
    def get_workflow_config(self) -> Dict[str, Any]:
        """Get workflow configuration"""
        return self.config.get("workflow", {})
    
    def validate_mode_config(self, mode: str) -> bool:
        """Validate that all required stages are present in mode configuration"""
        required_stages = [
            "data_discovery",
            "data_cleaning", 
            "ml_advisory",
            "model_training",
            "model_evaluation"
        ]
        
        mode_key = f"{mode}_mode"
        if mode_key not in self.config:
            return False
        
        mode_config = self.config[mode_key]
        return all(stage in mode_config for stage in required_stages)
    
    def print_mode_summary(self, mode: str) -> None:
        """Print a summary of the mode configuration"""
        if not self.validate_mode_config(mode):
            print(f"❌ Invalid mode configuration: {mode}")
            return
        
        print(f"📋 {mode.upper()} MODE CONFIGURATION SUMMARY")
        print("=" * 50)
        
        mode_key = f"{mode}_mode"
        for stage, config in self.config[mode_key].items():
            print(f"\n🔧 {stage.upper().replace('_', ' ')}:")
            
            # Show key settings for each stage
            if stage == "data_discovery":
                profiling = config.get("profiling", {})
                print(f"  • Generate HTML: {profiling.get('generate_html', False)}")
                print(f"  • Advanced Stats: {profiling.get('enable_advanced_stats', False)}")
                
            elif stage == "data_cleaning":
                execution = config.get("execution", {})
                print(f"  • Validate Before: {execution.get('validate_before_cleaning', False)}")
                print(f"  • Generate Report: {execution.get('generate_cleaning_report', False)}")
                
            elif stage == "ml_advisory":
                assumption = config.get("assumption_testing", {})
                print(f"  • Enabled: {assumption.get('enabled', False)}")
                print(f"  • Verbose: {assumption.get('verbose', False)}")
                print(f"  • Generate Recommendations: {assumption.get('generate_recommendations', False)}")
                
            elif stage == "model_training":
                print(f"  • Max Models: {config.get('max_models', 0)}")
                print(f"  • Models to Use: {config.get('models_to_use', 'Auto-select')}")
                print(f"  • Enable Tuning: {config.get('enable_tuning', False)}")
                print(f"  • Verbose: {config.get('verbose', False)}")
                
            elif stage == "model_evaluation":
                print(f"  • Enable SHAP: {config.get('enable_shap', False)}")
                print(f"  • Enable Learning Curves: {config.get('enable_learning_curves', False)}")
                print(f"  • Models to Evaluate: {config.get('n_models_to_evaluate', 0)}")
                print(f"  • Verbose: {config.get('verbose', False)}")
        
        print(f"\n✅ {mode.upper()} mode validation: {'PASSED' if self.validate_mode_config(mode) else 'FAILED'}")

# Global instance for easy access
config_manager = SimplifiedConfigManager()

def get_config_for_stage(mode: str, stage: str) -> Dict[str, Any]:
    """
    Convenience function to get configuration for a specific mode and stage
    
    Usage:
        config = get_config_for_stage("custom", "model_training")
        trainer_config = TrainerConfig.from_yaml_config(config)
    """
    return config_manager.get_mode_config(mode, stage)

def validate_config_file(config_path: str = "config/unified_config_v3.yaml") -> bool:
    """Validate the configuration file structure"""
    try:
        manager = SimplifiedConfigManager(config_path)
        
        # Check both modes
        fast_valid = manager.validate_mode_config("fast")
        custom_valid = manager.validate_mode_config("custom")
        
        print(f"📋 CONFIGURATION VALIDATION REPORT")
        print("=" * 40)
        print(f"Config File: {config_path}")
        print(f"Fast Mode: {'✅ VALID' if fast_valid else '❌ INVALID'}")
        print(f"Custom Mode: {'✅ VALID' if custom_valid else '❌ INVALID'}")
        print(f"Overall: {'✅ VALID' if fast_valid and custom_valid else '❌ INVALID'}")
        
        return fast_valid and custom_valid
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode in ["fast", "custom"]:
            config_manager.print_mode_summary(mode)
        elif mode == "validate":
            validate_config_file()
        else:
            print("Usage: python simplified_config_manager.py [fast|custom|validate]")
    else:
        print("📋 DS-AutoAdvisor v3.0 Configuration Manager")
        print("=" * 45)
        print("Available modes: fast, custom")
        print("\nUsage examples:")
        print("  python simplified_config_manager.py fast")
        print("  python simplified_config_manager.py custom") 
        print("  python simplified_config_manager.py validate")
        print("\nPython usage:")
        print("  from simplified_config_manager import get_config_for_stage")
        print("  config = get_config_for_stage('custom', 'model_training')")
        print("  trainer_config = TrainerConfig.from_yaml_config(config)")
