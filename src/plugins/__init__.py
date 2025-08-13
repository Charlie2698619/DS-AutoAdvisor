"""
🔌 DS-AutoAdvisor Plugins
========================

Plugin system for extending DS-AutoAdvisor functionality.

Available Plugins:
- optuna_hpo: Hyperparameter optimization using Optuna
- evidently_monitor: Data drift monitoring using Evidently

Usage:
    from src.plugins.optuna_hpo import OptunaHPOManager
    from src.plugins.evidently_monitor import EvidentiallyMonitor
"""

# Plugin availability flags
try:
    from .optuna_hpo import OptunaHPOManager, HPOConfigGenerator, ModelOptimizer
    HAS_OPTUNA_HPO = True
except ImportError:
    HAS_OPTUNA_HPO = False

try:
    from .evidently_monitor import EvidentiallyMonitor, DriftDetector, ReportGenerator, CLIMonitor
    HAS_EVIDENTLY_MONITOR = True
except ImportError:
    HAS_EVIDENTLY_MONITOR = False

# Plugin registry
AVAILABLE_PLUGINS = {
    'optuna_hpo': HAS_OPTUNA_HPO,
    'evidently_monitor': HAS_EVIDENTLY_MONITOR
}

def get_available_plugins():
    """Get list of available plugins"""
    return [name for name, available in AVAILABLE_PLUGINS.items() if available]

def check_plugin_dependencies():
    """Check plugin dependencies and provide installation instructions"""
    missing_deps = []
    
    if not HAS_OPTUNA_HPO:
        missing_deps.append({
            'plugin': 'optuna_hpo',
            'install': 'pip install optuna',
            'description': 'Hyperparameter optimization'
        })
    
    if not HAS_EVIDENTLY_MONITOR:
        missing_deps.append({
            'plugin': 'evidently_monitor', 
            'install': 'pip install evidently',
            'description': 'Data drift monitoring'
        })
    
    return missing_deps

__all__ = [
    'AVAILABLE_PLUGINS', 'get_available_plugins', 'check_plugin_dependencies'
]

# Conditional exports
if HAS_OPTUNA_HPO:
    __all__.extend(['OptunaHPOManager', 'HPOConfigGenerator', 'ModelOptimizer'])

if HAS_EVIDENTLY_MONITOR:
    __all__.extend(['EvidentiallyMonitor', 'DriftDetector', 'ReportGenerator', 'CLIMonitor'])
