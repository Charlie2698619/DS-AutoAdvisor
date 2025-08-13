"""
🔧 Optuna HPO Plugin for DS-AutoAdvisor
====================================

Hyperparameter optimization plugin that integrates seamlessly with the existing pipeline.

Features:
- Auto YAML config generation
- Training stage integration
- Model parsing and HPO recommendations
- Evaluation stage Optuna support
- Backward compatibility with default configs

Usage:
    from src.plugins.optuna_hpo import OptunaHPOManager
    
    hpo_manager = OptunaHPOManager(config)
    recommended_config = hpo_manager.generate_auto_config(models, data)
"""

from .hpo_manager import OptunaHPOManager
from .config_generator import HPOConfigGenerator
from .model_optimizer import ModelOptimizer

__all__ = ['OptunaHPOManager', 'HPOConfigGenerator', 'ModelOptimizer']
