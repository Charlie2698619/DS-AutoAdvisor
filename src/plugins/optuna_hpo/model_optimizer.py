#!/usr/bin/env python3
"""
🎯 Model Optimizer
==================

Handles the actual hyperparameter optimization using Optuna.
Integrates with DS-AutoAdvisor's training and evaluation pipeline.
"""

import sys
import os
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from datetime import datetime
import logging
import joblib
import json

# Machine learning imports
try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.pipeline import Pipeline
    
    # Model imports
    from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                                  GradientBoostingClassifier, GradientBoostingRegressor,
                                  ExtraTreesClassifier, ExtraTreesRegressor)
    from sklearn.linear_model import (LogisticRegression, LinearRegression, 
                                      Ridge, Lasso, ElasticNet)
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    
    # XGBoost and LightGBM (optional)
    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False
    
    try:
        import lightgbm as lgb
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False
    
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False

class ModelOptimizer:
    """
    Handles hyperparameter optimization for machine learning models using Optuna.
    
    Integrates seamlessly with DS-AutoAdvisor's existing training pipeline.
    """
    
    def __init__(self, random_state: int = 42, output_dir: str = "pipeline_outputs"):
        """Initialize the model optimizer"""
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.logger = self._setup_logging()
        
        # Model registry
        self.model_registry = self._create_model_registry()
        
        # Optimization results cache
        self.optimization_results = {}
        
        # Current optimization context
        self.current_X = None
        self.current_y = None
        self.current_problem_type = None
        self.current_cv_folds = 5
        
        print("🎯 Model Optimizer initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for optimization"""
        logger = logging.getLogger('model_optimizer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_model_registry(self) -> Dict[str, Any]:
        """Create registry of available models"""
        registry = {}
        
        if not HAS_ML_LIBS:
            self.logger.warning("Scikit-learn not available - limited functionality")
            return registry
        
        # Classification models
        registry.update({
            'RandomForestClassifier': RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'LogisticRegression': LogisticRegression,
            'SVC': SVC,
            'KNeighborsClassifier': KNeighborsClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'ExtraTreesClassifier': ExtraTreesClassifier,
            'GaussianNB': GaussianNB,
            'MLPClassifier': MLPClassifier
        })
        
        # Regression models
        registry.update({
            'RandomForestRegressor': RandomForestRegressor,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'LinearRegression': LinearRegression,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'ElasticNet': ElasticNet,
            'SVR': SVR,
            'KNeighborsRegressor': KNeighborsRegressor,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'ExtraTreesRegressor': ExtraTreesRegressor,
            'MLPRegressor': MLPRegressor
        })
        
        # Add XGBoost if available
        if HAS_XGB:
            registry.update({
                'XGBClassifier': xgb.XGBClassifier,
                'XGBRegressor': xgb.XGBRegressor
            })
        
        # Add LightGBM if available
        if HAS_LGB:
            registry.update({
                'LGBMClassifier': lgb.LGBMClassifier,
                'LGBMRegressor': lgb.LGBMRegressor
            })
        
        self.logger.info(f"Model registry created with {len(registry)} models")
        return registry
    
    def _detect_problem_type(self, y: pd.Series) -> str:
        """Detect if the problem is classification or regression"""
        if y.dtype == 'object' or len(y.unique()) < 20:
            return 'classification'
        else:
            return 'regression'
    
    def _get_scoring_metric(self, problem_type: str) -> str:
        """Get appropriate scoring metric for the problem type"""
        if problem_type == 'classification':
            return 'f1_macro'  # Good for imbalanced datasets
        else:
            return 'neg_mean_squared_error'
    
    def _create_objective_function(self, model_name: str, parameter_space: Dict[str, Any]) -> Callable:
        """Create Optuna objective function for a specific model"""
        
        def objective(trial):
            try:
                # Suggest parameters based on parameter space
                params = {}
                for param_name, param_config in parameter_space.items():
                    if param_config['type'] == 'int':
                        if 'step' in param_config:
                            params[param_name] = trial.suggest_int(
                                param_name, param_config['low'], param_config['high'], 
                                step=param_config['step']
                            )
                        else:
                            params[param_name] = trial.suggest_int(
                                param_name, param_config['low'], param_config['high']
                            )
                    elif param_config['type'] == 'float':
                        log = param_config.get('log', False)
                        params[param_name] = trial.suggest_float(
                            param_name, param_config['low'], param_config['high'], log=log
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config['choices']
                        )
                
                # Create model with suggested parameters
                model_class = self.model_registry[model_name]
                
                # Handle special cases for different models
                if model_name in ['SVC', 'SVR'] and 'kernel' in params:
                    # Remove degree parameter if kernel is not 'poly'
                    if params['kernel'] != 'poly' and 'degree' in params:
                        del params['degree']
                
                # Add random state if the model supports it
                if hasattr(model_class(), 'random_state'):
                    params['random_state'] = self.random_state
                
                # Create model instance
                model = model_class(**params)
                
                # Set up cross-validation
                if self.current_problem_type == 'classification':
                    cv = StratifiedKFold(n_splits=self.current_cv_folds, shuffle=True, 
                                         random_state=self.random_state)
                else:
                    cv = KFold(n_splits=self.current_cv_folds, shuffle=True, 
                               random_state=self.random_state)
                
                # Perform cross-validation
                scoring = self._get_scoring_metric(self.current_problem_type)
                scores = cross_val_score(model, self.current_X, self.current_y, 
                                         cv=cv, scoring=scoring, n_jobs=1)
                
                # Return mean score (Optuna maximizes, so negate if needed)
                mean_score = scores.mean()
                if scoring.startswith('neg_'):
                    mean_score = -mean_score
                
                # Store additional information
                trial.set_user_attr('std_score', scores.std())
                trial.set_user_attr('individual_scores', scores.tolist())
                trial.set_user_attr('model_name', model_name)
                
                return mean_score
                
            except Exception as e:
                self.logger.warning(f"Trial failed for {model_name}: {e}")
                # Return a very low score for failed trials
                return 0.0 if self.current_problem_type == 'classification' else float('inf')
        
        return objective
    
    def optimize_model(self, model_name: str, parameter_space: Dict[str, Any],
                       X: pd.DataFrame, y: pd.Series, n_trials: int = 50,
                       cv_folds: int = 5, study_name: str = None) -> optuna.Study:
        """Optimize hyperparameters for a specific model"""
        
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        # Set optimization context
        self.current_X = X
        self.current_y = y
        self.current_problem_type = self._detect_problem_type(y)
        self.current_cv_folds = cv_folds
        
        # Create study name
        if study_name is None:
            study_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Create objective function
        objective = self._create_objective_function(model_name, parameter_space)
        
        # Optimize
        print(f"🔧 Optimizing {model_name} with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Store results
        self.optimization_results[model_name] = {
            'study': study,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
        
        print(f"✅ Optimization complete for {model_name}")
        print(f"   🏆 Best score: {study.best_value:.4f}")
        print(f"   🔧 Best params: {study.best_params}")
        
        self.logger.info(f"Completed optimization for {model_name}: {study.best_value:.4f}")
        
        return study
    
    def optimize_multiple_models(self, models_config: Dict[str, Dict[str, Any]],
                                 X: pd.DataFrame, y: pd.Series, n_trials: int = 50,
                                 cv_folds: int = 5) -> Dict[str, optuna.Study]:
        """Optimize multiple models and return results"""
        
        results = {}
        
        print(f"🚀 Starting optimization for {len(models_config)} models...")
        
        for i, (model_name, parameter_space) in enumerate(models_config.items(), 1):
            print(f"\n📊 [{i}/{len(models_config)}] Optimizing {model_name}")
            
            try:
                study = self.optimize_model(
                    model_name, parameter_space, X, y, n_trials, cv_folds
                )
                results[model_name] = study
                
            except Exception as e:
                self.logger.error(f"Failed to optimize {model_name}: {e}")
                print(f"❌ Failed to optimize {model_name}: {e}")
        
        print(f"\n🎉 Optimization complete for {len(results)} models")
        
        return results
    
    def get_best_model_across_all(self) -> Tuple[str, Dict[str, Any], float]:
        """Get the best model across all optimizations"""
        
        if not self.optimization_results:
            raise ValueError("No optimization results available")
        
        best_model = None
        best_params = None
        best_score = -float('inf')
        
        for model_name, result in self.optimization_results.items():
            if result['best_value'] > best_score:
                best_score = result['best_value']
                best_model = model_name
                best_params = result['best_params']
        
        return best_model, best_params, best_score
    
    def create_optimized_model(self, model_name: str, params: Dict[str, Any]):
        """Create a model instance with optimized parameters"""
        
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        model_class = self.model_registry[model_name]
        
        # Add random state if supported
        if hasattr(model_class(), 'random_state'):
            params['random_state'] = self.random_state
        
        return model_class(**params)
    
    def save_optimization_results(self, output_name: str = None) -> Path:
        """Save all optimization results to JSON"""
        
        if output_name is None:
            output_name = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        results_json = {}
        for model_name, result in self.optimization_results.items():
            study = result['study']
            results_json[model_name] = {
                'best_params': result['best_params'],
                'best_value': result['best_value'],
                'n_trials': result['n_trials'],
                'study_name': study.study_name,
                'all_trials': [
                    {
                        'number': trial.number,
                        'value': trial.value,
                        'params': trial.params,
                        'state': trial.state.name
                    }
                    for trial in study.trials
                ]
            }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"💾 Optimization results saved: {output_path}")
        self.logger.info(f"Optimization results saved to {output_path}")
        
        return output_path
    
    def generate_optimization_summary(self) -> Dict[str, Any]:
        """Generate a summary of all optimization results"""
        
        if not self.optimization_results:
            return {'message': 'No optimization results available'}
        
        # Find best overall model
        best_model, best_params, best_score = self.get_best_model_across_all()
        
        # Create summary
        summary = {
            'total_models_optimized': len(self.optimization_results),
            'best_overall': {
                'model': best_model,
                'score': best_score,
                'params': best_params
            },
            'all_models': {}
        }
        
        # Add details for each model
        for model_name, result in self.optimization_results.items():
            summary['all_models'][model_name] = {
                'best_score': result['best_value'],
                'n_trials': result['n_trials'],
                'best_params': result['best_params']
            }
        
        return summary
