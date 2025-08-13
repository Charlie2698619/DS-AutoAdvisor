#!/usr/bin/env python3
"""
🎯 Optuna HPO Manager
====================

Main manager class for hyperparameter optimization integration.
Follows DS-AutoAdvisor patterns for consistency and maintainability.
"""

import sys
import os
import json
import yaml
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "utils"))
sys.path.append(str(project_root / "src"))

from simplified_config_manager import SimplifiedConfigManager

class OptunaHPOManager:
    """
    Main manager for Optuna hyperparameter optimization integration.
    
    Seamlessly integrates with existing DS-AutoAdvisor pipeline patterns.
    """
    
    def __init__(self, config_manager: SimplifiedConfigManager = None, 
                 output_base: str = "pipeline_outputs"):
        """Initialize HPO manager with config and output paths"""
        self.config_manager = config_manager or SimplifiedConfigManager()
        self.output_base = Path(output_base)
        
        # Create HPO output structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.hpo_outputs = {
            'base': self.output_base / f"hpo_outputs_{timestamp}",
            'studies': self.output_base / f"hpo_outputs_{timestamp}/studies",
            'configs': self.output_base / f"hpo_outputs_{timestamp}/configs",
            'recommendations': self.output_base / f"hpo_outputs_{timestamp}/recommendations",
            'logs': self.output_base / f"hpo_outputs_{timestamp}/logs"
        }
        
        # Create directories
        for path in self.hpo_outputs.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # HPO configuration cache
        self.hpo_config = None
        self.studies_cache = {}
        
        print("🎯 Optuna HPO Manager initialized")
        self.logger.info("OptunaHPOManager initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for HPO operations"""
        logger = logging.getLogger('optuna_hpo')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.hpo_outputs['logs'] / 'optuna_hpo.log'
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler if not already added
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def is_hpo_enabled(self, mode: str = "custom") -> bool:
        """Check if HPO is enabled in the current configuration"""
        try:
            if mode == "fast":
                # Fast mode typically disables HPO
                training_config = self.config_manager.get_mode_config("model_training", "fast_mode")
                return training_config.get('enable_tuning', False)
            else:
                # Custom mode - check custom_mode config
                training_config = self.config_manager.get_mode_config("model_training", "custom_mode")
                return training_config.get('enable_tuning', False)
        except Exception as e:
            self.logger.warning(f"Error checking HPO enablement: {e}")
            return False
    
    def get_hpo_config(self, mode: str = "custom") -> Dict[str, Any]:
        """Get HPO configuration for the specified mode"""
        if self.hpo_config is None:
            try:
                # Load base configuration
                if mode == "fast":
                    base_config = self.config_manager.get_mode_config("fast", "model_training")
                else:
                    base_config = self.config_manager.get_mode_config("custom", "model_training")
                
                # Get infrastructure plugin settings
                try:
                    infrastructure = self.config_manager.get_global_config().get('infrastructure', {})
                    plugin_settings = infrastructure.get('plugins', {}).get('optuna_hpo', {}).get('settings', {})
                except:
                    plugin_settings = {}
                
                # Get optuna-specific settings from model training config
                optuna_settings = base_config.get('optuna_hpo_settings', {})
                
                # Extract HPO-specific settings with YAML control
                self.hpo_config = {
                    'enabled': base_config.get('enable_tuning', False),
                    'method': base_config.get('tuning_method', 'optuna'),
                    'iterations': base_config.get('tuning_iterations', plugin_settings.get('default_iterations', 50)),
                    'cv_folds': base_config.get('tuning_cv_folds', plugin_settings.get('default_cv_folds', 3)),
                    'parallel_jobs': base_config.get('parallel_jobs', plugin_settings.get('parallel_jobs', 1)),
                    'max_time_minutes': optuna_settings.get('timeout_minutes', plugin_settings.get('timeout_minutes', 30)),
                    'random_state': self.config_manager.config.get('global', {}).get('random_state', 42),
                    'models_to_use': base_config.get('models_to_use', ['RandomForestClassifier', 'XGBClassifier', 'GradientBoostingClassifier']),
                    'max_models': base_config.get('max_models', 5),
                    
                    # Optuna-specific settings from YAML
                    'direction': optuna_settings.get('direction', plugin_settings.get('direction', 'maximize')),
                    'sampler': optuna_settings.get('sampler', plugin_settings.get('default_sampler', 'TPE')),
                    'pruner': optuna_settings.get('pruner', plugin_settings.get('default_pruner', 'MedianPruner')),
                    'n_startup_trials': optuna_settings.get('n_startup_trials', plugin_settings.get('n_startup_trials', 10)),
                    'n_warmup_steps': optuna_settings.get('n_warmup_steps', plugin_settings.get('n_warmup_steps', 10)),
                    'storage_type': plugin_settings.get('storage_type', 'sqlite'),
                    'study_name_prefix': plugin_settings.get('study_name_prefix', 'ds_autoadvisor'),
                    'study_name_suffix': optuna_settings.get('study_name_suffix', mode),
                    'study_name': f"{plugin_settings.get('study_name_prefix', 'ds_autoadvisor')}_{optuna_settings.get('study_name_suffix', mode)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
                
                self.logger.info(f"Loaded HPO config for {mode} mode: {self.hpo_config}")
                
            except Exception as e:
                self.logger.error(f"Error loading HPO config: {e}")
                # Fallback configuration
                self.hpo_config = {
                    'enabled': False,
                    'method': 'optuna',
                    'iterations': 20,
                    'cv_folds': 3,
                    'parallel_jobs': 1,
                    'max_time_minutes': 15,
                    'random_state': 42,
                    'models_to_use': ['RandomForestClassifier', 'XGBClassifier', 'GradientBoostingClassifier'],
                    'max_models': 3,
                    'direction': 'maximize',
                    'sampler': 'TPE',
                    'pruner': 'MedianPruner',
                    'n_startup_trials': 5,
                    'n_warmup_steps': 5,
                    'storage_type': 'sqlite',
                    'study_name_prefix': 'ds_autoadvisor',
                    'study_name_suffix': 'fallback',
                    'study_name': f"ds_autoadvisor_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
        
        return self.hpo_config
    
    def create_study(self, study_name: str = None, mode: str = "custom") -> optuna.Study:
        """Create an Optuna study with YAML-configured settings"""
        
        # Get configuration from YAML
        config = self.get_hpo_config(mode)
        
        if study_name is None:
            study_name = config.get('study_name', f"ds_autoadvisor_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        try:
            # Setup storage (SQLite for persistence)
            storage_path = self.hpo_outputs['studies'] / f"{study_name}.db"
            storage = f"sqlite:///{storage_path}"
            
            # Configure sampler from YAML
            sampler_name = config.get('sampler', 'TPE')
            random_state = config.get('random_state', 42)
            n_startup_trials = config.get('n_startup_trials', 10)
            
            if sampler_name.lower() == "tpe":
                sampler = optuna.samplers.TPESampler(seed=random_state, n_startup_trials=n_startup_trials)
            elif sampler_name.lower() == "random":
                sampler = optuna.samplers.RandomSampler(seed=random_state)
            elif sampler_name.lower() == "cmaes":
                sampler = optuna.samplers.CmaEsSampler(seed=random_state)
            else:
                sampler = optuna.samplers.TPESampler(seed=random_state, n_startup_trials=n_startup_trials)
            
            # Configure pruner from YAML
            pruner_name = config.get('pruner', 'MedianPruner')
            n_warmup_steps = config.get('n_warmup_steps', 10)
            
            if pruner_name.lower() == "medianpruner":
                pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)
            elif pruner_name.lower() == "hyperband":
                pruner = optuna.pruners.HyperbandPruner()
            elif pruner_name.lower() == "successivehalving":
                pruner = optuna.pruners.SuccessiveHalvingPruner()
            else:
                pruner = optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)
            
            # Get direction from YAML
            direction = config.get('direction', 'maximize')
            
            # Create study
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )
            
            # Cache the study
            self.studies_cache[study_name] = study
            
            self.logger.info(f"Created Optuna study: {study_name} (sampler: {sampler_name}, pruner: {pruner_name}, direction: {direction})")
            print(f"📊 Created Optuna study: {study_name}")
            print(f"   ⚙️ Sampler: {sampler_name}, Pruner: {pruner_name}")
            print(f"   📈 Direction: {direction}, Startup trials: {n_startup_trials}")
            
            return study
            
        except Exception as e:
            self.logger.error(f"Error creating study {study_name}: {e}")
            raise e
    
    def save_study_results(self, study: optuna.Study, output_name: str = None) -> Dict[str, Any]:
        """Save study results and generate recommendations"""
        
        if output_name is None:
            output_name = f"study_results_{study.study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Prepare results
            results = {
                'study_name': study.study_name,
                'direction': study.direction.name,
                'n_trials': len(study.trials),
                'best_trial': {
                    'number': study.best_trial.number,
                    'value': study.best_value,
                    'params': study.best_params,
                    'user_attrs': study.best_trial.user_attrs,
                    'datetime_start': study.best_trial.datetime_start.isoformat() if study.best_trial.datetime_start else None,
                    'datetime_complete': study.best_trial.datetime_complete.isoformat() if study.best_trial.datetime_complete else None
                },
                'all_trials': [],
                'parameter_importance': {},
                'optimization_history': []
            }
            
            # Add all trials
            for trial in study.trials:
                trial_data = {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name,
                    'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                    'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
                }
                results['all_trials'].append(trial_data)
            
            # Calculate parameter importance if enough trials
            if len(study.trials) >= 3:
                try:
                    importance = optuna.importance.get_param_importances(study)
                    results['parameter_importance'] = importance
                except Exception as e:
                    self.logger.warning(f"Could not calculate parameter importance: {e}")
            
            # Generate optimization history
            results['optimization_history'] = [
                {'trial': i, 'value': trial.value} 
                for i, trial in enumerate(study.trials) 
                if trial.value is not None
            ]
            
            # Save results to JSON
            results_file = self.hpo_outputs['recommendations'] / f"{output_name}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Generate YAML config with best parameters
            yaml_config = self._generate_yaml_config(study.best_params, study.best_value)
            yaml_file = self.hpo_outputs['configs'] / f"{output_name}_config.yaml"
            with open(yaml_file, 'w') as f:
                yaml.dump(yaml_config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Saved study results to {results_file}")
            self.logger.info(f"Generated YAML config: {yaml_file}")
            
            print(f"💾 Study results saved: {results_file}")
            print(f"📝 YAML config generated: {yaml_file}")
            print(f"🏆 Best trial value: {study.best_value:.4f}")
            print(f"🔧 Best parameters: {study.best_params}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error saving study results: {e}")
            raise e
    
    def _generate_yaml_config(self, best_params: Dict[str, Any], best_value: float) -> Dict[str, Any]:
        """Generate YAML configuration with optimized parameters"""
        
        # Get current config as base
        current_config = self.config_manager.config.copy()
        
        # Create optimized config section
        optimized_config = {
            'optuna_hpo_optimized': {
                'generated_at': datetime.now().isoformat(),
                'best_value': best_value,
                'optimization_source': 'optuna_hpo_plugin',
                'model_training': {
                    'enable_tuning': True,
                    'tuning_method': 'optuna_optimized',
                    'optimized_params': best_params
                }
            }
        }
        
        # Merge with existing config
        optimized_config.update(current_config)
        
        return optimized_config
    
    def load_study(self, study_name: str) -> Optional[optuna.Study]:
        """Load an existing study from storage"""
        try:
            if study_name in self.studies_cache:
                return self.studies_cache[study_name]
            
            storage_path = self.hpo_outputs['studies'] / f"{study_name}.db"
            if storage_path.exists():
                storage = f"sqlite:///{storage_path}"
                study = optuna.load_study(study_name=study_name, storage=storage)
                self.studies_cache[study_name] = study
                self.logger.info(f"Loaded existing study: {study_name}")
                return study
            else:
                self.logger.warning(f"Study storage not found: {storage_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading study {study_name}: {e}")
            return None
    
    def run_optimization_workflow(self, X: pd.DataFrame, y: pd.Series, 
                                  mode: str = "custom", study_name: str = None,
                                  models_to_optimize: List[str] = None,
                                  n_trials: int = None) -> Optional[Dict[str, Any]]:
        """
        Run complete HPO optimization workflow for integration with main pipeline.
        
        Args:
            X: Features dataframe
            y: Target series
            mode: Configuration mode ("fast" or "custom")
            study_name: Name for the study (optional, generated from YAML if None)
            models_to_optimize: List of model names to optimize (optional, from YAML if None)
            n_trials: Number of optimization trials (optional, from YAML if None)
            
        Returns:
            Dictionary with optimization results or None if failed
        """
        try:
            self.logger.info(f"🚀 Starting HPO optimization workflow in {mode} mode...")
            
            # Get configuration from YAML
            config = self.get_hpo_config(mode)
            
            # Use YAML settings with override capability
            if n_trials is None:
                n_trials = config.get('iterations', 20)
            
            if study_name is None:
                study_name = config.get('study_name', f"pipeline_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            if models_to_optimize is None:
                models_to_optimize = config.get('models_to_use', ['RandomForestClassifier', 'XGBClassifier', 'GradientBoostingClassifier'])
            
            # Respect max_models limit from config
            max_models = config.get('max_models', 5)
            if len(models_to_optimize) > max_models:
                models_to_optimize = models_to_optimize[:max_models]
                self.logger.info(f"🔧 Limited to {max_models} models as per YAML config")
            
            self.logger.info(f"🎯 Optimizing models: {models_to_optimize}")
            self.logger.info(f"🔄 Running {n_trials} trials per model")
            self.logger.info(f"⚙️ Using config: sampler={config.get('sampler')}, pruner={config.get('pruner')}")
            
            # Import required components
            from .model_optimizer import ModelOptimizer
            from .config_generator import HPOConfigGenerator
            
            # Get parameter spaces from config generator
            config_gen = HPOConfigGenerator(str(self.output_base))
            parameter_spaces = config_gen._define_parameter_spaces()
            
            # Create models config for optimization
            models_config = {}
            for model_name in models_to_optimize:
                if model_name in parameter_spaces:
                    models_config[model_name] = parameter_spaces[model_name]
                    self.logger.info(f"✅ Parameter space loaded for {model_name}")
                else:
                    self.logger.warning(f"⚠️ No parameter space for {model_name}, skipping")
            
            if not models_config:
                self.logger.error("No valid parameter spaces found for any models")
                return None
            
            # Create study with YAML configuration
            study = self.create_study(study_name=study_name, mode=mode)
            if not study:
                self.logger.error("Failed to create optimization study")
                return None
            
            # Initialize optimizer
            optimizer = ModelOptimizer(output_dir=str(self.output_base))
            
            # Run optimization using optimize_multiple_models
            self.logger.info(f"🚀 Starting batch optimization...")
            study_results = optimizer.optimize_multiple_models(
                models_config, X, y, n_trials=n_trials, cv_folds=5
            )
            
            # Process results
            best_overall_score = 0
            best_overall_model = None
            best_overall_params = None
            optimization_results = []
            
            for model_name, study in study_results.items():
                if study and study.best_trial:
                    best_score = study.best_trial.value
                    best_params = study.best_trial.params
                    
                    optimization_results.append({
                        'model_name': model_name,
                        'best_score': best_score,
                        'best_params': best_params,
                        'n_trials': len(study.trials)
                    })
                    
                    # Track overall best
                    if best_score > best_overall_score:
                        best_overall_score = best_score
                        best_overall_model = model_name
                        best_overall_params = best_params
                    
                    self.logger.info(f"✅ {model_name}: {best_score:.4f}")
                else:
                    self.logger.warning(f"❌ No results for {model_name}")
            
            # Save study results
            study_summary = self.save_study_results(study, f"workflow_{study_name}")
            
            # Prepare workflow results
            workflow_results = {
                'study_name': study_name,
                'best_model': best_overall_model,
                'best_score': best_overall_score,
                'best_params': best_overall_params,
                'total_trials': sum(len(study.trials) for study in study_results.values() if study),
                'models_optimized': len(optimization_results),
                'optimization_results': optimization_results,
                'study_results': study_summary,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"🏆 Optimization complete!")
            self.logger.info(f"   Best model: {best_overall_model}")
            self.logger.info(f"   Best score: {best_overall_score:.4f}")
            self.logger.info(f"   Total trials: {len(optimization_results) * n_trials}")
            
            return workflow_results
            
        except ImportError as e:
            self.logger.error(f"Missing HPO components: {e}")
            return None
        except Exception as e:
            self.logger.error(f"HPO workflow failed: {e}")
            return None
    
    def get_recommendations_summary(self) -> Dict[str, Any]:
        """Get a summary of all HPO recommendations"""
        try:
            recommendations_dir = self.hpo_outputs['recommendations']
            summary = {
                'total_studies': 0,
                'best_overall': None,
                'studies': []
            }
            
            # Process all recommendation files
            for rec_file in recommendations_dir.glob("*.json"):
                try:
                    with open(rec_file, 'r') as f:
                        data = json.load(f)
                    
                    study_summary = {
                        'file': rec_file.name,
                        'study_name': data.get('study_name'),
                        'best_value': data.get('best_trial', {}).get('value'),
                        'n_trials': data.get('n_trials'),
                        'best_params': data.get('best_trial', {}).get('params')
                    }
                    
                    summary['studies'].append(study_summary)
                    summary['total_studies'] += 1
                    
                    # Track best overall
                    if (summary['best_overall'] is None or 
                        (study_summary['best_value'] and 
                         study_summary['best_value'] > summary['best_overall'].get('best_value', 0))):
                        summary['best_overall'] = study_summary
                        
                except Exception as e:
                    self.logger.warning(f"Error processing recommendation file {rec_file}: {e}")
            
            self.logger.info(f"Generated recommendations summary: {summary['total_studies']} studies")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations summary: {e}")
            return {'total_studies': 0, 'best_overall': None, 'studies': []}
