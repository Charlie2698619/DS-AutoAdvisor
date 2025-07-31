"""
Industrial-Grade Configuration Management
=======================================

Advanced configuration handling with validation, environment-specific configs,
and dynamic updates for production ML pipelines.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import hashlib

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    
class ConfigValidationError(Exception):
    """Configuration validation error"""
    pass

@dataclass
class ConfigMetadata:
    """Configuration metadata for tracking and auditing"""
    version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    checksum: Optional[str] = None
    validated: bool = False
    
class AdvancedConfigManager:
    """
    Industrial-grade configuration management with:
    - Environment-specific configs
    - Schema validation
    - Configuration versioning
    - Secret management
    - Hot reloading
    """
    
    def __init__(self, 
                 base_config_path: str,
                 environment: Environment = Environment.DEVELOPMENT,
                 secrets_backend: str = "env",  # env, vault, aws_secrets
                 enable_hot_reload: bool = False):
        
        self.base_config_path = Path(base_config_path)
        self.environment = environment
        self.secrets_backend = secrets_backend
        self.enable_hot_reload = enable_hot_reload
        self.logger = logging.getLogger(__name__)
        
        self._config_cache = {}
        self._config_metadata = {}
        self._file_watchers = {}
        
        # Load environment-specific config paths
        self.config_paths = self._discover_config_files()
        
    def _discover_config_files(self) -> List[Path]:
        """Discover configuration files based on environment"""
        config_dir = self.base_config_path.parent
        base_name = self.base_config_path.stem
        
        # Priority order: environment-specific -> base config
        possible_configs = [
            config_dir / f"{base_name}.{self.environment.value}.yaml",
            config_dir / f"{base_name}.{self.environment.value}.yml", 
            config_dir / f"{base_name}_override.yaml",
            self.base_config_path
        ]
        
        existing_configs = [p for p in possible_configs if p.exists()]
        
        if not existing_configs:
            raise FileNotFoundError(f"No configuration files found for {base_name}")
            
        self.logger.info(f"Config files discovered: {[str(p) for p in existing_configs]}")
        return existing_configs
    
    def load_config(self, validate: bool = True) -> Dict[str, Any]:
        """
        Load and merge configuration from multiple sources
        
        Args:
            validate: Whether to validate the configuration
            
        Returns:
            Merged configuration dictionary
        """
        config = {}
        
        # Load and merge configs in priority order
        for config_path in reversed(self.config_paths):  # Reverse for proper override
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config = self._deep_merge(config, file_config)
                        self.logger.debug(f"Loaded config from {config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_path}: {e}")
                if config_path == self.base_config_path:  # Base config is required
                    raise
        
        # Process secrets and environment variables
        config = self._process_secrets(config)
        config = self._process_environment_variables(config)
        
        # Add metadata
        metadata = ConfigMetadata(
            environment=self.environment,
            checksum=self._calculate_checksum(config),
            validated=False
        )
        
        if validate:
            self._validate_config(config)
            metadata.validated = True
        
        # Cache config and metadata
        config_key = "main"
        self._config_cache[config_key] = config
        self._config_metadata[config_key] = metadata
        
        self.logger.info(f"Configuration loaded successfully for {self.environment.value}")
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _process_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process secret references in configuration"""
        if self.secrets_backend == "env":
            return self._process_env_secrets(config)
        elif self.secrets_backend == "vault":
            return self._process_vault_secrets(config)
        elif self.secrets_backend == "aws_secrets":
            return self._process_aws_secrets(config)
        else:
            return config
    
    def _process_env_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process environment variable secrets"""
        def process_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                env_value = os.getenv(env_var)
                if env_value is None:
                    self.logger.warning(f"Environment variable {env_var} not found")
                    return value
                return env_value
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            else:
                return value
        
        return process_value(config)
    
    def _process_vault_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process HashiCorp Vault secrets (placeholder)"""
        # Implementation would integrate with Vault API
        self.logger.warning("Vault secrets backend not implemented")
        return config
    
    def _process_aws_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process AWS Secrets Manager secrets (placeholder)"""
        # Implementation would integrate with AWS Secrets Manager
        self.logger.warning("AWS Secrets backend not implemented")
        return config
    
    def _process_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process environment variable overrides"""
        # Allow environment variables to override config values
        # Format: DS_AUTOADVISOR_SECTION__SUBSECTION__KEY=value
        prefix = "DS_AUTOADVISOR_"
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                config_path = env_key[len(prefix):].lower().split("__")
                self._set_nested_value(config, config_path, env_value)
        
        return config
    
    def _set_nested_value(self, config: Dict, path: List[str], value: str):
        """Set nested configuration value from environment variable"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Try to parse value as JSON, fall back to string
        try:
            current[path[-1]] = json.loads(value)
        except json.JSONDecodeError:
            current[path[-1]] = value
    
    def _calculate_checksum(self, config: Dict[str, Any]) -> str:
        """Calculate configuration checksum for change detection"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration against schema"""
        required_sections = ['global', 'workflow']
        
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError(f"Required section '{section}' missing from configuration")
        
        # Validate global section
        global_config = config['global']
        required_global_keys = ['data_input_path', 'target_column', 'csv_delimiter']
        
        for key in required_global_keys:
            if key not in global_config:
                raise ConfigValidationError(f"Required global setting '{key}' missing")
        
        # Validate data paths exist
        data_path = Path(global_config['data_input_path'])
        if not data_path.exists():
            if self.environment == Environment.PRODUCTION:
                raise ConfigValidationError(f"Data file not found: {data_path}")
            else:
                self.logger.warning(f"Data file not found: {data_path} (non-prod environment)")
        
        # Environment-specific validations
        if self.environment == Environment.PRODUCTION:
            self._validate_production_config(config)
        
        self.logger.info("Configuration validation passed")
    
    def _validate_production_config(self, config: Dict[str, Any]):
        """Additional validations for production environment"""
        # Ensure human intervention is properly configured
        human_config = config.get('global', {}).get('human_intervention', {})
        
        if human_config.get('mode') == 'interactive':
            self.logger.warning("Interactive mode not recommended for production")
        
        # Ensure monitoring is enabled
        if 'monitoring' not in config:
            self.logger.warning("Monitoring configuration not found for production")
        
        # Ensure security settings
        if 'security' not in config:
            self.logger.warning("Security configuration not found for production")
    
    def get_config(self, section: Optional[str] = None) -> Union[Dict[str, Any], Any]:
        """Get configuration or specific section"""
        config = self._config_cache.get("main", {})
        
        if section:
            return config.get(section, {})
        return config
    
    def update_config(self, updates: Dict[str, Any], persist: bool = False):
        """Update configuration dynamically"""
        config = self._config_cache.get("main", {})
        config = self._deep_merge(config, updates)
        
        if persist:
            # Save to environment-specific override file
            override_path = self.base_config_path.parent / f"{self.base_config_path.stem}_override.yaml"
            with open(override_path, 'w') as f:
                yaml.dump(updates, f, default_flow_style=False)
            
        self._config_cache["main"] = config
        self.logger.info("Configuration updated successfully")
    
    def get_metadata(self) -> ConfigMetadata:
        """Get configuration metadata"""
        return self._config_metadata.get("main", ConfigMetadata())

# Environment-specific configuration examples
ENVIRONMENT_DEFAULTS = {
    Environment.DEVELOPMENT: {
        "global": {
            "human_intervention": {
                "mode": "interactive",
                "enabled": True
            }
        },
        "workflow": {
            "error_handling": {
                "stop_on_error": True
            },
            "logging": {
                "level": "DEBUG"
            }
        }
    },
    Environment.STAGING: {
        "global": {
            "human_intervention": {
                "mode": "semi_automated",
                "enabled": True
            }
        },
        "workflow": {
            "error_handling": {
                "stop_on_error": False,
                "skip_failed_stages": True
            },
            "logging": {
                "level": "INFO"
            }
        }
    },
    Environment.PRODUCTION: {
        "global": {
            "human_intervention": {
                "mode": "fully_automated",
                "enabled": False
            }
        },
        "workflow": {
            "error_handling": { 
                "stop_on_error": False,
                "skip_failed_stages": False,
                "retry_attempts": 3
            },
            "logging": {
                "level": "WARNING"
            }
        },
        "monitoring": {
            "enabled": True,
            "metrics_collection": True,
            "alerting": True
        }
    }
}

# Example usage
if __name__ == "__main__":
    # Initialize config manager for production
    config_manager = AdvancedConfigManager(
        base_config_path="config/unified_config.yaml",
        environment=Environment.PRODUCTION,
        secrets_backend="env"
    )
    
    # Load configuration
    config = config_manager.load_config(validate=True)
    
    # Get specific section
    global_config = config_manager.get_config("global")
    
    # Update configuration
    config_manager.update_config({
        "model_training": {
            "max_models": 5
        }
    })
    
    print(f"Config loaded for {config_manager.environment.value}")
    print(f"Metadata: {config_manager.get_metadata()}")
