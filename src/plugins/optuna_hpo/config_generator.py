#!/usr/bin/env python3
"""
🔧 HPO Config Generator
======================

Automatically generates YAML configurations based on model analysis and HPO results.
Integrates with DS-AutoAdvisor's configuration management system.
"""

import sys
import os
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging

# Scikit-learn imports for model analysis
try:
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
    import xgboost as xgb
    import lightgbm as lgb
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False

class HPOConfigGenerator:
    """
    Generates optimized YAML configurations based on data characteristics and model analysis.
    
    Follows DS-AutoAdvisor patterns and integrates with existing config management.
    """
    
    def __init__(self, output_dir: str = "pipeline_outputs"):
        """Initialize config generator"""
        self.output_dir = Path(output_dir)
        self.logger = self._setup_logging()
        
        # Model parameter spaces - defines search spaces for different algorithms
        self.parameter_spaces = self._define_parameter_spaces()
        
        # Data characteristics cache
        self.data_analysis = {}
        
        print("🔧 HPO Config Generator initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for config generation"""
        logger = logging.getLogger('hpo_config_generator')
        logger.setLevel(logging.INFO)
        
        # Avoid adding multiple handlers
        if not logger.handlers:
            # Create console handler
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _define_parameter_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Define hyperparameter search spaces for different algorithms"""
        
        spaces = {
            # Random Forest
            'RandomForestClassifier': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500, 'step': 50},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]}
            },
            'RandomForestRegressor': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500, 'step': 50},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]}
            },
            
            # Gradient Boosting
            'GradientBoostingClassifier': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300, 'step': 50},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
            },
            'GradientBoostingRegressor': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300, 'step': 50},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
            },
            
            # XGBoost
            'XGBClassifier': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300, 'step': 50},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 0, 'high': 1},
                'reg_lambda': {'type': 'float', 'low': 0, 'high': 1}
            },
            'XGBRegressor': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300, 'step': 50},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 0, 'high': 1},
                'reg_lambda': {'type': 'float', 'low': 0, 'high': 1}
            },
            
            # Support Vector Machines
            'SVC': {
                'C': {'type': 'float', 'low': 0.1, 'high': 100, 'log': True},
                'kernel': {'type': 'categorical', 'choices': ['rbf', 'poly', 'sigmoid']},
                'gamma': {'type': 'categorical', 'choices': ['scale', 'auto']},
                'degree': {'type': 'int', 'low': 2, 'high': 5}  # Only for poly kernel
            },
            'SVR': {
                'C': {'type': 'float', 'low': 0.1, 'high': 100, 'log': True},
                'kernel': {'type': 'categorical', 'choices': ['rbf', 'poly', 'sigmoid']},
                'gamma': {'type': 'categorical', 'choices': ['scale', 'auto']},
                'epsilon': {'type': 'float', 'low': 0.01, 'high': 1.0},
                'degree': {'type': 'int', 'low': 2, 'high': 5}
            },
            
            # Logistic Regression
            'LogisticRegression': {
                'C': {'type': 'float', 'low': 0.01, 'high': 100, 'log': True},
                'penalty': {'type': 'categorical', 'choices': ['l1', 'l2', 'elasticnet']},
                'solver': {'type': 'categorical', 'choices': ['liblinear', 'saga', 'lbfgs']},
                'max_iter': {'type': 'int', 'low': 100, 'high': 1000, 'step': 100}
            },
            
            # Neural Networks
            'MLPClassifier': {
                'hidden_layer_sizes': {'type': 'categorical', 'choices': [(50,), (100,), (50, 50), (100, 50)]},
                'learning_rate_init': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
                'alpha': {'type': 'float', 'low': 0.0001, 'high': 0.1, 'log': True},
                'max_iter': {'type': 'int', 'low': 200, 'high': 1000, 'step': 100}
            },
            'MLPRegressor': {
                'hidden_layer_sizes': {'type': 'categorical', 'choices': [(50,), (100,), (50, 50), (100, 50)]},
                'learning_rate_init': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
                'alpha': {'type': 'float', 'low': 0.0001, 'high': 0.1, 'log': True},
                'max_iter': {'type': 'int', 'low': 200, 'high': 1000, 'step': 100}
            }
        }
        
        return spaces
    
    def analyze_data_characteristics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze data characteristics to inform HPO configuration"""
        
        try:
            n_samples, n_features = X.shape
            
            analysis = {
                'n_samples': n_samples,
                'n_features': n_features,
                'dataset_size': 'small' if n_samples < 1000 else 'medium' if n_samples < 10000 else 'large',
                'feature_complexity': 'low' if n_features < 10 else 'medium' if n_features < 50 else 'high',
                'problem_type': 'classification' if y.dtype == 'object' or len(y.unique()) < 20 else 'regression',
                'class_balance': None,
                'missing_values': X.isnull().sum().sum(),
                'categorical_features': len(X.select_dtypes(include=['object', 'category']).columns),
                'numerical_features': len(X.select_dtypes(include=[np.number]).columns)
            }
            
            # Analyze class balance for classification
            if analysis['problem_type'] == 'classification':
                class_counts = y.value_counts()
                analysis['n_classes'] = len(class_counts)
                analysis['class_balance'] = {
                    'balanced': (class_counts.min() / class_counts.max()) > 0.7,
                    'ratio': class_counts.min() / class_counts.max(),
                    'minority_class_size': class_counts.min()
                }
            
            # Cache analysis
            self.data_analysis = analysis
            
            self.logger.info(f"Data analysis complete: {n_samples} samples, {n_features} features")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in data analysis: {e}")
            return {}
    
    def recommend_models_for_data(self, data_analysis: Dict[str, Any] = None) -> List[str]:
        """Recommend models based on data characteristics"""
        
        if data_analysis is None:
            data_analysis = self.data_analysis
        
        if not data_analysis:
            self.logger.warning("No data analysis available, using default model recommendations")
            return ['RandomForestClassifier', 'LogisticRegression', 'GradientBoostingClassifier']
        
        recommendations = []
        problem_type = data_analysis.get('problem_type', 'classification')
        dataset_size = data_analysis.get('dataset_size', 'medium')
        feature_complexity = data_analysis.get('feature_complexity', 'medium')
        
        if problem_type == 'classification':
            # Always include Random Forest - robust and interpretable
            recommendations.append('RandomForestClassifier')
            
            # Add models based on dataset characteristics
            if dataset_size in ['medium', 'large']:
                recommendations.extend(['GradientBoostingClassifier', 'XGBClassifier'])
            
            if feature_complexity == 'low':
                recommendations.append('LogisticRegression')
            
            if dataset_size == 'large' and feature_complexity == 'high':
                recommendations.append('MLPClassifier')
            
            # For small datasets, add simpler models
            if dataset_size == 'small':
                recommendations.extend(['LogisticRegression', 'SVC'])
                
        else:  # regression
            recommendations.append('RandomForestRegressor')
            
            if dataset_size in ['medium', 'large']:
                recommendations.extend(['GradientBoostingRegressor', 'XGBRegressor'])
            
            if feature_complexity == 'low':
                recommendations.append('LinearRegression')
            
            if dataset_size == 'large' and feature_complexity == 'high':
                recommendations.append('MLPRegressor')
            
            if dataset_size == 'small':
                recommendations.extend(['LinearRegression', 'SVR'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for model in recommendations:
            if model not in seen:
                seen.add(model)
                unique_recommendations.append(model)
        
        self.logger.info(f"Model recommendations: {unique_recommendations}")
        
        return unique_recommendations
    
    def generate_hpo_config_yaml(self, models: List[str] = None, 
                                 data_analysis: Dict[str, Any] = None,
                                 mode: str = "custom") -> Dict[str, Any]:
        """Generate complete HPO configuration YAML"""
        
        if models is None:
            models = self.recommend_models_for_data(data_analysis)
        
        if data_analysis is None:
            data_analysis = self.data_analysis
        
        # Base configuration structure following DS-AutoAdvisor patterns
        hpo_config = {
            'optuna_hpo_auto_generated': {
                'generated_at': datetime.now().isoformat(),
                'data_analysis': data_analysis,
                'recommended_models': models,
                'mode': mode
            },
            
            # Extend the existing config structure
            f'{mode}_mode': {
                'model_training': {
                    # Enable HPO
                    'enable_tuning': True,
                    'tuning_method': 'optuna',
                    'tuning_iterations': self._get_iterations_for_dataset(data_analysis),
                    'tuning_cv_folds': 5 if data_analysis.get('dataset_size') != 'small' else 3,
                    'parallel_jobs': 1,  # Conservative for stability
                    'max_training_time_minutes': self._get_time_limit(data_analysis),
                    
                    # Model selection
                    'models_to_use': models,
                    'max_models': min(len(models), 5),  # Limit for efficiency
                    
                    # HPO-specific settings
                    'optuna_settings': {
                        'sampler': 'TPE',
                        'pruner': 'MedianPruner',
                        'direction': 'maximize',
                        'study_name_prefix': 'ds_autoadvisor',
                        'storage_type': 'sqlite',
                        'n_startup_trials': 10,
                        'n_warmup_steps': 10
                    },
                    
                    # Parameter spaces for each model
                    'model_parameter_spaces': {}
                }
            }
        }
        
        # Add parameter spaces for recommended models
        for model in models:
            if model in self.parameter_spaces:
                hpo_config[f'{mode}_mode']['model_training']['model_parameter_spaces'][model] = \
                    self.parameter_spaces[model]
        
        self.logger.info(f"Generated HPO config for {len(models)} models")
        
        return hpo_config
    
    def _get_iterations_for_dataset(self, data_analysis: Dict[str, Any]) -> int:
        """Determine optimal number of iterations based on dataset characteristics"""
        
        if not data_analysis:
            return 50
        
        dataset_size = data_analysis.get('dataset_size', 'medium')
        feature_complexity = data_analysis.get('feature_complexity', 'medium')
        
        # More iterations for complex datasets, fewer for simple ones
        if dataset_size == 'large' and feature_complexity == 'high':
            return 100
        elif dataset_size == 'small' or feature_complexity == 'low':
            return 30
        else:
            return 50
    
    def _get_time_limit(self, data_analysis: Dict[str, Any]) -> int:
        """Determine time limit based on dataset characteristics"""
        
        if not data_analysis:
            return 30
        
        dataset_size = data_analysis.get('dataset_size', 'medium')
        
        if dataset_size == 'large':
            return 60
        elif dataset_size == 'small':
            return 15
        else:
            return 30
    
    def save_config_yaml(self, config: Dict[str, Any], filename: str = None) -> Path:
        """Save generated configuration to YAML file"""
        
        if filename is None:
            filename = f"hpo_auto_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        
        output_path = self.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
            
            self.logger.info(f"HPO config saved to: {output_path}")
            print(f"📝 HPO config saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            raise e
    
    def create_complete_workflow_config(self, X: pd.DataFrame, y: pd.Series, 
                                        mode: str = "custom") -> Tuple[Dict[str, Any], Path]:
        """Complete workflow: analyze data, recommend models, generate config"""
        
        print("🔍 Analyzing data characteristics...")
        data_analysis = self.analyze_data_characteristics(X, y)
        
        print("🤖 Recommending models...")
        models = self.recommend_models_for_data(data_analysis)
        
        print("⚙️ Generating HPO configuration...")
        config = self.generate_hpo_config_yaml(models, data_analysis, mode)
        
        print("💾 Saving configuration...")
        config_path = self.save_config_yaml(config)
        
        print(f"✅ Complete HPO configuration ready!")
        print(f"   📊 Data: {data_analysis['n_samples']} samples, {data_analysis['n_features']} features")
        print(f"   🤖 Models: {', '.join(models)}")
        print(f"   📝 Config: {config_path}")
        
        return config, config_path
