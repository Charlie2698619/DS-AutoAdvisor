"""
DS-AutoAdvisor v2.0 Plugin Architecture
======================================

This module provides a flexible plugin system for extending DS-AutoAdvisor functionality.

Features:
- Dynamic plugin loading and registration
- Plugin interface definitions
- Plugin dependency management
- Configuration validation
- Lifecycle management
- Extension points for all pipeline stages
"""

import os
import sys
import importlib
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import yaml
import json

class PluginType(Enum):
    """Types of plugins supported by the system"""
    FEATURE_SELECTION = "feature_selection"
    SAMPLING = "sampling"
    DATA_CLEANING = "data_cleaning"
    MODEL_SELECTION = "model_selection"
    EVALUATION = "evaluation"
    VISUALIZATION = "visualization"
    DEEP_LEARNING = "deep_learning"
    DOMAIN_SPECIFIC = "domain_specific"
    CUSTOM_STAGE = "custom_stage"

class PluginStatus(Enum):
    """Plugin status states"""
    UNKNOWN = "unknown"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class PluginInfo:
    """Plugin information and metadata"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    python_version: str = ">=3.8"
    status: PluginStatus = PluginStatus.UNKNOWN
    error_message: Optional[str] = None
    config_schema: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PluginConfig:
    """Plugin-specific configuration"""
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 100  # Lower numbers = higher priority

class BasePlugin(ABC):
    """
    Base class for all DS-AutoAdvisor plugins
    
    All plugins must inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: PluginConfig = None):
        """
        Initialize plugin
        
        Args:
            config: Plugin-specific configuration
        """
        self.config = config or PluginConfig()
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self._is_initialized = False
    
    @property
    @abstractmethod
    def plugin_info(self) -> PluginInfo:
        """Return plugin information and metadata"""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute(self, data: Any, **kwargs) -> Any:
        """
        Execute the plugin's main functionality
        
        Args:
            data: Input data for processing
            **kwargs: Additional parameters
            
        Returns:
            Processed data or results
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        # Default implementation - override in subclasses for specific validation
        return True
    
    def cleanup(self):
        """Cleanup resources when plugin is unloaded"""
        pass
    
    def get_status(self) -> PluginStatus:
        """Get current plugin status"""
        if self._is_initialized:
            return PluginStatus.ACTIVE
        else:
            return PluginStatus.LOADED

class FeatureSelectionPlugin(BasePlugin):
    """Base class for feature selection plugins"""
    
    @abstractmethod
    def select_features(self, X, y, **kwargs) -> List[str]:
        """
        Select features from dataset
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional parameters
            
        Returns:
            List of selected feature names
        """
        pass

class SamplingPlugin(BasePlugin):
    """Base class for sampling plugins"""
    
    @abstractmethod
    def sample_data(self, X, y, **kwargs):
        """
        Apply sampling to dataset
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (X_sampled, y_sampled)
        """
        pass

class DataCleaningPlugin(BasePlugin):
    """Base class for data cleaning plugins"""
    
    @abstractmethod
    def clean_data(self, df, **kwargs):
        """
        Clean the input dataframe
        
        Args:
            df: Input dataframe
            **kwargs: Additional parameters
            
        Returns:
            Cleaned dataframe
        """
        pass

class ModelSelectionPlugin(BasePlugin):
    """Base class for model selection plugins"""
    
    @abstractmethod
    def recommend_models(self, X, y, **kwargs) -> List[Dict[str, Any]]:
        """
        Recommend models for the given dataset
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional parameters
            
        Returns:
            List of model recommendations
        """
        pass

class EvaluationPlugin(BasePlugin):
    """Base class for evaluation plugins"""
    
    @abstractmethod
    def evaluate_model(self, model, X_test, y_test, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a trained model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass

class VisualizationPlugin(BasePlugin):
    """Base class for visualization plugins"""
    
    @abstractmethod
    def create_visualization(self, data, **kwargs) -> str:
        """
        Create visualization from data
        
        Args:
            data: Data to visualize
            **kwargs: Additional parameters
            
        Returns:
            Path to generated visualization file
        """
        pass

class CustomStagePlugin(BasePlugin):
    """Base class for custom pipeline stage plugins"""
    
    @abstractmethod
    def execute_stage(self, pipeline_state, **kwargs) -> Dict[str, Any]:
        """
        Execute custom pipeline stage
        
        Args:
            pipeline_state: Current pipeline state
            **kwargs: Additional parameters
            
        Returns:
            Stage execution results
        """
        pass

class PluginManager:
    """
    Central plugin management system
    
    Features:
    - Plugin discovery and loading
    - Dependency resolution
    - Configuration management
    - Lifecycle management
    - Error handling and recovery
    """
    
    def __init__(self, plugin_dir: str = "plugins", auto_discovery: bool = True):
        """
        Initialize plugin manager
        
        Args:
            plugin_dir: Directory containing plugins
            auto_discovery: Whether to automatically discover plugins
        """
        self.plugin_dir = Path(plugin_dir)
        self.auto_discovery = auto_discovery
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_configs: Dict[str, PluginConfig] = {}
        self.plugin_infos: Dict[str, PluginInfo] = {}
        self.logger = logging.getLogger(__name__)
        
        # Create plugin directory if it doesn't exist
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        
        # Load plugin configurations
        self._load_plugin_configs()
        
        # Discover and load plugins if enabled
        if auto_discovery:
            self.discover_plugins()
            self.load_all_plugins()
    
    def _load_plugin_configs(self):
        """Load plugin configurations from file"""
        config_file = self.plugin_dir / "plugin_configs.yaml"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    configs = yaml.safe_load(f) or {}
                
                for plugin_name, config_data in configs.items():
                    self.plugin_configs[plugin_name] = PluginConfig(**config_data)
                    
                self.logger.info(f"Loaded configurations for {len(self.plugin_configs)} plugins")
                
            except Exception as e:
                self.logger.error(f"Failed to load plugin configurations: {e}")
        else:
            self.logger.info("No plugin configuration file found")
    
    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in the plugin directory
        
        Returns:
            List of discovered plugin names
        """
        discovered_plugins = []
        
        # Look for Python files in plugin directory
        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
                
            plugin_name = plugin_file.stem
            discovered_plugins.append(plugin_name)
        
        # Look for plugin packages (directories with __init__.py)
        for plugin_dir in self.plugin_dir.iterdir():
            if plugin_dir.is_dir() and (plugin_dir / "__init__.py").exists():
                discovered_plugins.append(plugin_dir.name)
        
        self.logger.info(f"Discovered {len(discovered_plugins)} plugins: {discovered_plugins}")
        return discovered_plugins
    
    def load_plugin(self, plugin_name: str) -> bool:
        """
        Load a specific plugin
        
        Args:
            plugin_name: Name of the plugin to load
            
        Returns:
            True if plugin loaded successfully, False otherwise
        """
        try:
            # Add plugin directory to Python path
            if str(self.plugin_dir) not in sys.path:
                sys.path.insert(0, str(self.plugin_dir))
            
            # Import the plugin module
            module = importlib.import_module(plugin_name)
            
            # Find plugin class (should inherit from BasePlugin)
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin and
                    not inspect.isabstract(obj)):
                    plugin_class = obj
                    break
            
            if plugin_class is None:
                self.logger.error(f"No valid plugin class found in {plugin_name}")
                return False
            
            # Get plugin configuration
            plugin_config = self.plugin_configs.get(plugin_name, PluginConfig())
            
            # Skip if plugin is disabled
            if not plugin_config.enabled:
                self.logger.info(f"Plugin {plugin_name} is disabled")
                return False
            
            # Create plugin instance
            plugin_instance = plugin_class(plugin_config)
            
            # Get plugin info
            plugin_info = plugin_instance.plugin_info
            plugin_info.status = PluginStatus.LOADING
            
            # Validate dependencies
            if not self._validate_dependencies(plugin_info.dependencies):
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = "Dependency validation failed"
                self.plugin_infos[plugin_name] = plugin_info
                return False
            
            # Initialize plugin
            if plugin_instance.initialize():
                self.plugins[plugin_name] = plugin_instance
                plugin_info.status = PluginStatus.ACTIVE
                self.plugin_infos[plugin_name] = plugin_info
                self.logger.info(f"Successfully loaded plugin: {plugin_name}")
                return True
            else:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = "Plugin initialization failed"
                self.plugin_infos[plugin_name] = plugin_info
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_name}: {e}")
            self.plugin_infos[plugin_name] = PluginInfo(
                name=plugin_name,
                version="unknown",
                description="Failed to load",
                author="unknown",
                plugin_type=PluginType.CUSTOM_STAGE,
                status=PluginStatus.ERROR,
                error_message=str(e)
            )
            return False
    
    def _validate_dependencies(self, dependencies: List[str]) -> bool:
        """
        Validate plugin dependencies
        
        Args:
            dependencies: List of dependency names
            
        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        for dependency in dependencies:
            try:
                importlib.import_module(dependency)
            except ImportError:
                self.logger.error(f"Missing dependency: {dependency}")
                return False
        
        return True
    
    def load_all_plugins(self) -> Dict[str, bool]:
        """
        Load all discovered plugins
        
        Returns:
            Dictionary mapping plugin names to load success status
        """
        discovered_plugins = self.discover_plugins()
        results = {}
        
        for plugin_name in discovered_plugins:
            results[plugin_name] = self.load_plugin(plugin_name)
        
        loaded_count = sum(results.values())
        self.logger.info(f"Loaded {loaded_count}/{len(discovered_plugins)} plugins")
        
        return results
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """
        Get a loaded plugin by name
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """
        Get all loaded plugins of a specific type
        
        Args:
            plugin_type: Type of plugins to retrieve
            
        Returns:
            List of plugin instances
        """
        matching_plugins = []
        
        for plugin_name, plugin in self.plugins.items():
            if plugin.plugin_info.plugin_type == plugin_type:
                matching_plugins.append(plugin)
        
        # Sort by priority (lower number = higher priority)
        matching_plugins.sort(key=lambda p: self.plugin_configs.get(p.plugin_info.name, PluginConfig()).priority)
        
        return matching_plugins
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if plugin unloaded successfully, False otherwise
        """
        if plugin_name in self.plugins:
            try:
                plugin = self.plugins[plugin_name]
                plugin.cleanup()
                del self.plugins[plugin_name]
                
                if plugin_name in self.plugin_infos:
                    self.plugin_infos[plugin_name].status = PluginStatus.DISABLED
                
                self.logger.info(f"Unloaded plugin: {plugin_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to unload plugin {plugin_name}: {e}")
                return False
        
        return False
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin
        
        Args:
            plugin_name: Name of the plugin to reload
            
        Returns:
            True if plugin reloaded successfully, False otherwise
        """
        # Unload if currently loaded
        if plugin_name in self.plugins:
            self.unload_plugin(plugin_name)
        
        # Reload module
        try:
            if plugin_name in sys.modules:
                importlib.reload(sys.modules[plugin_name])
        except Exception as e:
            self.logger.warning(f"Failed to reload module {plugin_name}: {e}")
        
        # Load plugin again
        return self.load_plugin(plugin_name)
    
    def get_plugin_status(self) -> Dict[str, PluginInfo]:
        """
        Get status of all plugins
        
        Returns:
            Dictionary mapping plugin names to their info
        """
        return self.plugin_infos.copy()
    
    def save_plugin_configs(self):
        """Save current plugin configurations to file"""
        config_file = self.plugin_dir / "plugin_configs.yaml"
        
        try:
            config_data = {}
            for plugin_name, config in self.plugin_configs.items():
                config_data[plugin_name] = {
                    'enabled': config.enabled,
                    'config': config.config,
                    'priority': config.priority
                }
            
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
                
            self.logger.info("Plugin configurations saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save plugin configurations: {e}")
    
    def execute_plugins(self, plugin_type: PluginType, data: Any, **kwargs) -> List[Any]:
        """
        Execute all plugins of a specific type
        
        Args:
            plugin_type: Type of plugins to execute
            data: Input data for plugins
            **kwargs: Additional parameters
            
        Returns:
            List of results from plugin executions
        """
        plugins = self.get_plugins_by_type(plugin_type)
        results = []
        
        for plugin in plugins:
            try:
                result = plugin.execute(data, **kwargs)
                results.append(result)
                self.logger.debug(f"Executed plugin {plugin.plugin_info.name}")
            except Exception as e:
                self.logger.error(f"Plugin {plugin.plugin_info.name} execution failed: {e}")
                results.append(None)
        
        return results


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None

def get_plugin_manager(plugin_dir: str = "plugins", auto_discovery: bool = True) -> PluginManager:
    """
    Get global plugin manager instance
    
    Args:
        plugin_dir: Directory containing plugins
        auto_discovery: Whether to automatically discover plugins
        
    Returns:
        Plugin manager instance
    """
    global _plugin_manager
    
    if _plugin_manager is None:
        _plugin_manager = PluginManager(plugin_dir, auto_discovery)
    
    return _plugin_manager
