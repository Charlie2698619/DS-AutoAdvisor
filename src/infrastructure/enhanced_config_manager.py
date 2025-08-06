"""
DS-AutoAdvisor v2.0 Core Infrastructure
======================================

Enhanced configuration management system with support for v2.0 features.
This module extends the v1.0 configuration system with:

- Environment-specific configurations
- Plugin configuration management  
- Feature toggles
- Configuration validation
- Dynamic configuration updates
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class DeploymentMode(Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"

@dataclass
class MetadataConfig:
    """Configuration for metadata management system"""
    enabled: bool = True
    store_type: str = "sqlite"
    connection_string: str = "sqlite:///metadata/pipeline_metadata.db"
    track_lineage: bool = True
    track_performance: bool = True
    retention_days: int = 90

@dataclass
class PluginConfig:
    """Configuration for plugin system"""
    enabled: bool = True
    plugin_dir: str = "plugins"
    auto_discovery: bool = True
    validation_required: bool = True
    allowed_plugins: List[str] = field(default_factory=list)

@dataclass
class DataQualityConfig:
    """Configuration for enhanced data quality features"""
    type_enforcement_enabled: bool = True
    lineage_tracking: bool = True
    outlier_detection_methods: List[str] = field(default_factory=lambda: ["isolation_forest", "z_score"])
    quality_scoring: bool = True

@dataclass
class AdvisorIntelligenceConfig:
    """Configuration for ML advisor intelligence features"""
    meta_learning_enabled: bool = True
    fairness_analysis: bool = True
    performance_prediction: bool = True
    schema_similarity_threshold: float = 0.8

@dataclass
class TrainingScalingConfig:
    """Configuration for training and scaling features"""
    cloud_runner_enabled: bool = False
    provider: str = "aws"
    advanced_optimization: bool = True
    resource_management: bool = True
    per_model_time_limit: int = 1800

@dataclass
class EvaluationReportingConfig:
    """Configuration for evaluation and reporting features"""
    unified_dashboard: bool = True
    automated_reports: bool = True
    production_monitoring: bool = True
    real_time_updates: bool = True

@dataclass
class ExtensibilityConfig:
    """Configuration for extensibility features"""
    plugins_enabled: bool = True
    versioning_enabled: bool = True
    custom_stages: bool = True
    api_extensions: bool = True

class EnhancedConfigManager:
    """
    Enhanced configuration manager for DS-AutoAdvisor v2.0
    
    Features:
    - Environment-specific configuration loading
    - Feature toggle management
    - Configuration validation
    - Plugin configuration
    - Runtime configuration updates
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: Optional[Environment] = None):
        """
        Initialize enhanced configuration manager
        
        Args:
            config_path: Path to configuration file
            environment: Target environment
        """
        self.environment = environment or Environment.DEVELOPMENT
        self.config_path = config_path or self._get_default_config_path()
        self.config = {}
        self.feature_flags = {}
        self.plugin_configs = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self._load_configuration()
        self._load_feature_flags()
        self._validate_configuration()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration path based on environment"""
        base_path = Path(__file__).parent.parent / "config"
        
        env_configs = {
            Environment.DEVELOPMENT: "unified_config_v2.yaml",
            Environment.STAGING: "unified_config_v2_staging.yaml", 
            Environment.PRODUCTION: "unified_config_v2_prod.yaml"
        }
        
        config_file = env_configs.get(self.environment, "unified_config_v2.yaml")
        config_path = base_path / config_file
        
        # Fallback to base config if environment-specific doesn't exist
        if not config_path.exists():
            config_path = base_path / "unified_config_v2.yaml"
            
        return str(config_path)
    
    def _load_configuration(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            self.logger.info(f"Configuration loaded from {self.config_path}")
            
            # Apply environment-specific overrides
            self._apply_environment_overrides()
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        if self.environment == Environment.PRODUCTION:
            # Production overrides
            self.config.setdefault('global', {})['environment'] = 'production'
            self.config.setdefault('infrastructure', {}).setdefault('logging', {})['level'] = 'WARNING'
            self.config.setdefault('training_scaling', {})['cloud_runner']['enabled'] = True
            
        elif self.environment == Environment.STAGING:
            # Staging overrides  
            self.config.setdefault('global', {})['environment'] = 'staging'
            self.config.setdefault('infrastructure', {}).setdefault('logging', {})['level'] = 'INFO'
            
        else:
            # Development overrides (default)
            self.config.setdefault('global', {})['environment'] = 'development'
            self.config.setdefault('infrastructure', {}).setdefault('logging', {})['level'] = 'DEBUG'
    
    def _load_feature_flags(self):
        """Load feature flags for gradual rollout"""
        feature_flags_path = Path(self.config_path).parent / "feature_flags.yaml"
        
        if feature_flags_path.exists():
            try:
                with open(feature_flags_path, 'r') as f:
                    self.feature_flags = yaml.safe_load(f) or {}
                self.logger.info("Feature flags loaded")
            except Exception as e:
                self.logger.warning(f"Failed to load feature flags: {e}")
                self.feature_flags = {}
        else:
            # Default feature flags
            self.feature_flags = {
                'data_quality_v2': True,
                'advisor_intelligence': True,
                'cloud_training': False,
                'unified_dashboard': True,
                'plugin_system': True
            }
    
    def _validate_configuration(self):
        """Validate configuration structure and values"""
        required_sections = ['global', 'infrastructure', 'data_quality', 'advisor_intelligence']
        
        for section in required_sections:
            if section not in self.config:
                self.logger.warning(f"Missing required configuration section: {section}")
                self.config[section] = {}
    
    def get_config(self, section: str = None, key: str = None, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            section: Configuration section name
            key: Configuration key name  
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        if section is None:
            return self.config
            
        section_config = self.config.get(section, {})
        
        if key is None:
            return section_config
            
        return section_config.get(key, default)
    
    def set_config(self, section: str, key: str, value: Any):
        """
        Set configuration value at runtime
        
        Args:
            section: Configuration section name
            key: Configuration key name
            value: Value to set
        """
        if section not in self.config:
            self.config[section] = {}
            
        self.config[section][key] = value
        self.logger.info(f"Configuration updated: {section}.{key} = {value}")
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature flag is enabled
        
        Args:
            feature_name: Name of the feature flag
            
        Returns:
            True if feature is enabled, False otherwise
        """
        return self.feature_flags.get(feature_name, False)
    
    def get_metadata_config(self) -> MetadataConfig:
        """Get metadata management configuration"""
        metadata_config = self.get_config('infrastructure', 'metadata', {})
        return MetadataConfig(**metadata_config)
    
    def get_plugin_config(self) -> PluginConfig:
        """Get plugin system configuration"""
        plugin_config = self.get_config('infrastructure', 'plugins', {})
        return PluginConfig(**plugin_config)
    
    def get_data_quality_config(self) -> DataQualityConfig:
        """Get data quality configuration"""
        dq_config = self.get_config('data_quality', default={})
        
        return DataQualityConfig(
            type_enforcement_enabled=dq_config.get('type_enforcement', {}).get('enabled', True),
            lineage_tracking=dq_config.get('lineage', {}).get('enabled', True),
            outlier_detection_methods=dq_config.get('outlier_detection', {}).get('methods', ["isolation_forest"]),
            quality_scoring=dq_config.get('profiling', {}).get('quality_scoring', True)
        )
    
    def get_advisor_intelligence_config(self) -> AdvisorIntelligenceConfig:
        """Get advisor intelligence configuration"""
        ai_config = self.get_config('advisor_intelligence', default={})
        
        return AdvisorIntelligenceConfig(
            meta_learning_enabled=ai_config.get('meta_learning', {}).get('enabled', True),
            fairness_analysis=ai_config.get('fairness', {}).get('enabled', True),
            performance_prediction=ai_config.get('model_selection', {}).get('performance_prediction', True),
            schema_similarity_threshold=ai_config.get('meta_learning', {}).get('schema_similarity_threshold', 0.8)
        )
    
    def get_training_scaling_config(self) -> TrainingScalingConfig:
        """Get training and scaling configuration"""
        ts_config = self.get_config('training_scaling', default={})
        
        return TrainingScalingConfig(
            cloud_runner_enabled=ts_config.get('cloud_runner', {}).get('enabled', False),
            provider=ts_config.get('cloud_runner', {}).get('provider', 'aws'),
            advanced_optimization=ts_config.get('hyperparameter_optimization', {}).get('framework') == 'optuna',
            resource_management=ts_config.get('resource_management', {}).get('per_model_time_limit', 1800) > 0,
            per_model_time_limit=ts_config.get('resource_management', {}).get('per_model_time_limit', 1800)
        )
    
    def get_evaluation_reporting_config(self) -> EvaluationReportingConfig:
        """Get evaluation and reporting configuration"""
        er_config = self.get_config('evaluation_reporting', default={})
        
        return EvaluationReportingConfig(
            unified_dashboard=er_config.get('dashboard', {}).get('enabled', True),
            automated_reports=er_config.get('automated_reports', {}).get('enabled', True),
            production_monitoring=er_config.get('production_monitoring', {}).get('drift_detection', True),
            real_time_updates=er_config.get('dashboard', {}).get('real_time_updates', True)
        )
    
    def get_extensibility_config(self) -> ExtensibilityConfig:
        """Get extensibility configuration"""
        ext_config = self.get_config('extensibility', default={})
        
        return ExtensibilityConfig(
            plugins_enabled=ext_config.get('plugins', {}).get('feature_selection', {}).get('enabled', True),
            versioning_enabled=ext_config.get('versioning', {}).get('models', {}).get('enabled', True),
            custom_stages=ext_config.get('extension_points', {}).get('custom_stages', True),
            api_extensions=ext_config.get('extension_points', {}).get('api_extensions', True)
        )
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to file
        
        Args:
            output_path: Path to save configuration file
        """
        output_path = output_path or self.config_path
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def reload_config(self):
        """Reload configuration from file"""
        self._load_configuration()
        self._load_feature_flags()
        self._validate_configuration()
        self.logger.info("Configuration reloaded")
    
    def get_stage_config(self, stage_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific pipeline stage
        
        Args:
            stage_name: Name of the pipeline stage
            
        Returns:
            Stage-specific configuration
        """
        # Map stage names to configuration sections
        stage_mapping = {
            'profiling': 'profiling',
            'data_cleaning': 'data_cleaning', 
            'ml_advisory': 'ml_advisory',
            'model_training': 'model_training',
            'model_evaluation': 'model_evaluation'
        }
        
        config_section = stage_mapping.get(stage_name, stage_name)
        return self.get_config(config_section, default={})
    
    def validate_plugin_config(self, plugin_name: str, plugin_config: Dict[str, Any]) -> bool:
        """
        Validate plugin-specific configuration
        
        Args:
            plugin_name: Name of the plugin
            plugin_config: Plugin configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        # This would be implemented with specific validation logic
        # For now, return True as placeholder
        return True


# Singleton instance for global access
_config_manager: Optional[EnhancedConfigManager] = None

def get_config_manager(config_path: Optional[str] = None, 
                      environment: Optional[Environment] = None) -> EnhancedConfigManager:
    """
    Get global configuration manager instance
    
    Args:
        config_path: Path to configuration file
        environment: Target environment
        
    Returns:
        Enhanced configuration manager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = EnhancedConfigManager(config_path, environment)
    
    return _config_manager

def reload_global_config():
    """Reload global configuration manager"""
    global _config_manager
    if _config_manager:
        _config_manager.reload_config()
