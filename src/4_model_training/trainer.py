import pandas as pd
import numpy as np
import sys
import json
import warnings
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold, KFold
)
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score, 
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, 
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Advanced models (if available)
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

import joblib

# Import business features
try:
    sys.path.append('/mnt/c/Users/tony3/Desktop/tidytuesday/ds-autoadvisor/plugins')
    from feature_selection.business_feature_selector import BusinessFeatureSelector, FeatureSelectionResult
    from business_metrics.kpi_tracker import BusinessKPITracker, calculate_business_value
    HAS_BUSINESS_FEATURES = True
except ImportError:
    HAS_BUSINESS_FEATURES = False
    print("⚠️ Business features not available - continuing with standard training")
    
    # Create dummy classes for type hints
    class FeatureSelectionResult:
        def __init__(self):
            self.selected_features = []
            self.feature_scores = {}
            self.selection_method = "fallback"
    
    class BusinessFeatureSelector:
        def __init__(self, *args, **kwargs):
            pass
    
    class BusinessKPITracker:
        def __init__(self, *args, **kwargs):
            pass
    
    def calculate_business_value(*args, **kwargs):
        return 0.0

warnings.filterwarnings('ignore')

@dataclass
class TrainerConfig:
    """Type-safe configuration for model training - ALL values loaded from YAML"""
    # Data splitting
    test_size: float
    validation_size: float
    random_state: int
    
    # Model selection
    max_models: int
    include_ensemble: bool
    include_advanced: bool
    models_to_use: Optional[List[str]]
    
    # Hyperparameter tuning
    enable_tuning: bool
    tuning_method: str
    tuning_iterations: int
    tuning_cv_folds: int
    
    # Validation
    cv_folds: int
    scoring_strategy: str
    
    # Performance
    parallel_jobs: int
    max_training_time_minutes: Optional[int]
    memory_limit_gb: Optional[float]
    
    # Encoding and preprocessing
    encoding_strategy: str
    scaling_strategy: str
    
    # Business features
    enable_business_features: bool
    feature_selection_enabled: bool
    business_kpi_tracking: bool
    business_rules_file: str
    business_kpis_file: str
    
    # Output
    save_models: bool
    model_dir: str
    save_predictions: bool
    verbose: bool
    
    @classmethod
    def from_yaml_config(cls, config_dict: Dict[str, Any]) -> 'TrainerConfig':
        """Create TrainerConfig from YAML configuration dictionary"""
        return cls(
            # Data splitting
            test_size=config_dict.get('test_size', 0.2),
            validation_size=config_dict.get('validation_size', 0.1),
            random_state=config_dict.get('random_state', 42),
            
            # Model selection
            max_models=config_dict.get('max_models', 2),
            include_ensemble=config_dict.get('include_ensemble', True),
            include_advanced=config_dict.get('include_advanced', True),
            models_to_use=config_dict.get('models_to_use', None),
            
            # Hyperparameter tuning
            enable_tuning=config_dict.get('enable_tuning', True),
            tuning_method=config_dict.get('tuning_method', 'random'),
            tuning_iterations=config_dict.get('tuning_iterations', 50),
            tuning_cv_folds=config_dict.get('tuning_cv_folds', 6),
            
            # Validation
            cv_folds=config_dict.get('cv_folds', 6),
            scoring_strategy=config_dict.get('scoring_strategy', 'comprehensive'),
            
            # Performance
            parallel_jobs=config_dict.get('parallel_jobs', -1),
            max_training_time_minutes=config_dict.get('max_training_time_minutes', 30),
            memory_limit_gb=config_dict.get('memory_limit_gb', None),
            
            # Encoding and preprocessing
            encoding_strategy=config_dict.get('encoding_strategy', 'none'),
            scaling_strategy=config_dict.get('scaling_strategy', 'none'),
            
            # Business features
            enable_business_features=config_dict.get('enable_business_features', True),
            feature_selection_enabled=config_dict.get('feature_selection_enabled', True),
            business_kpi_tracking=config_dict.get('business_kpi_tracking', True),
            business_rules_file=config_dict.get('business_rules_file', 'config/business_rules.yaml'),
            business_kpis_file=config_dict.get('business_kpis_file', 'config/business_kpis.yaml'),
            
            # Output
            save_models=config_dict.get('save_models', True),
            model_dir=config_dict.get('model_dir', 'models'),
            save_predictions=config_dict.get('save_predictions', True),
            verbose=config_dict.get('verbose', True)
        )

@dataclass
class ModelResult:
    """Individual model training result"""
    name: str
    model: Any
    pipeline: Pipeline
    scores: Dict[str, float]
    cv_scores: Dict[str, float]
    feature_importance: Optional[Dict[str, float]]
    training_time: float
    best_params: Optional[Dict[str, Any]]
    model_path: Optional[str]
    
    # Business features
    selected_features: Optional[List[str]] = None
    feature_selection_result: Optional[FeatureSelectionResult] = None
    business_metrics: Optional[Dict[str, float]] = None
    business_value: Optional[Dict[str, float]] = None

class EnhancedModelTrainer:
    """Enhanced trainer with comprehensive model selection and business features"""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.results = []
        self.best_model = None
        self.feature_selector = None
        self.kpi_tracker = None
        self.feature_selection_results = {}
        
        self._setup_model_registry()
        self._setup_business_features()
        
    def _setup_business_features(self):
        """Initialize business feature components"""
        if not self.config.enable_business_features or not HAS_BUSINESS_FEATURES:
            if self.config.verbose:
                print("📊 Business features disabled or not available")
            return
            
        try:
            # Initialize feature selector
            if self.config.feature_selection_enabled:
                from infrastructure.plugin_system import PluginConfig
                plugin_config = PluginConfig(
                    enabled=True,
                    config={
                        'business_rules_file': self.config.business_rules_file,
                        'human_approval_required': True,
                        'feature_consistency_strategy': 'intersection',
                        'min_feature_count': 5,
                        'max_feature_count': 50
                    }
                )
                self.feature_selector = BusinessFeatureSelector(plugin_config)
                self.feature_selector.initialize()
                
            # Initialize KPI tracker
            if self.config.business_kpi_tracking:
                self.kpi_tracker = BusinessKPITracker(self.config.business_kpis_file)
                
            if self.config.verbose:
                print("✅ Business features initialized successfully")
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ Failed to initialize business features: {e}")
            self.feature_selector = None
            self.kpi_tracker = None
        
    def _setup_model_registry(self):
        """Initialize comprehensive model registry with hyperparameters"""
        
        # Regression models with hyperparameter grids
        self.regression_models = {
            "LinearRegression": {
                "model": LinearRegression,
                "params": {},
                "tune": False
            },
            "Ridge": {
                "model": Ridge,
                "params": {
                    "alpha": [0.1, 1.0, 10.0, 100.0]
                },
                "tune": True
            },
            "Lasso": {
                "model": Lasso,
                "params": {
                    "alpha": [0.01, 0.1, 1.0, 10.0]
                },
                "tune": True
            },
            "ElasticNet": {
                "model": ElasticNet,
                "params": {
                    "alpha": [0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.5, 0.9]
                },
                "tune": True
            },
            "RandomForestRegressor": {
                "model": RandomForestRegressor,
                "params": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },
                "tune": True
            },
            "GradientBoostingRegressor": {
                "model": GradientBoostingRegressor,
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 6]
                },
                "tune": True
            },
            "ExtraTreesRegressor": {
                "model": ExtraTreesRegressor,
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None]
                },
                "tune": True
            },
            "SVR": {
                "model": SVR,
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "gamma": ["scale", "auto"],
                    "kernel": ["rbf", "linear"]
                },
                "tune": True
            },
            "KNeighborsRegressor": {
                "model": KNeighborsRegressor,
                "params": {
                    "n_neighbors": [3, 5, 7, 10],
                    "weights": ["uniform", "distance"]
                },
                "tune": True
            }
        }
        
        # Classification models with hyperparameter grids
        self.classification_models = {
            "LogisticRegression": {
                "model": LogisticRegression,
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"]
                },
                "tune": True
            },
            "RandomForestClassifier": {
                "model": RandomForestClassifier,
                "params": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5],
                    "class_weight": ["balanced", None]
                },
                "tune": True
            },
            "GradientBoostingClassifier": {
                "model": GradientBoostingClassifier,
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 6]
                },
                "tune": True
            },
            "ExtraTreesClassifier": {
                "model": ExtraTreesClassifier,
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "class_weight": ["balanced", None]
                },
                "tune": True
            },
            "LinearSVC": {
                "model": LinearSVC,
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "dual": [False]
                },
                "tune": True
            },
            "SVC": {
                "model": SVC,
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "gamma": ["scale", "auto"],
                    "kernel": ["rbf", "linear"]
                },
                "tune": True
            },
            "GaussianNB": {
                "model": GaussianNB,
                "params": {},
                "tune": False
            },
            "KNeighborsClassifier": {
                "model": KNeighborsClassifier,
                "params": {
                    "n_neighbors": [3, 5, 7, 10],
                    "weights": ["uniform", "distance"]
                },
                "tune": True
            }
        }
        
        # Add advanced models if available
        if HAS_XGBOOST and self.config.include_advanced:
            self.regression_models["XGBRegressor"] = {
                "model": XGBRegressor,
                "params": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 6, 10],
                    "subsample": [0.8, 1.0]
                },
                "tune": True
            }
            
            self.classification_models["XGBClassifier"] = {
                "model": XGBClassifier,
                "params": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 6, 10],
                    "scale_pos_weight": [1, 3, 5]
                },
                "tune": True
            }
        
        if HAS_LIGHTGBM and self.config.include_advanced:
            self.regression_models["LGBMRegressor"] = {
                "model": LGBMRegressor,
                "params": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 6, 10],
                    "num_leaves": [31, 50, 100]
                },
                "tune": True
            }
            
            self.classification_models["LGBMClassifier"] = {
                "model": LGBMClassifier,
                "params": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 6, 10],
                    "class_weight": ["balanced", None]
                },
                "tune": True
            }
    
    def apply_business_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                       target_type: str, models_to_train: List[str]) -> Tuple[pd.DataFrame, FeatureSelectionResult]:
        """Apply business feature selection if enabled"""
        
        if not self.feature_selector:
            if self.config.verbose:
                print("📊 Business feature selection not available - using all features")
            return X, None
            
        try:
            if self.config.verbose:
                print("🔍 Applying business feature selection...")
                
            # Apply feature selection
            feature_result = self.feature_selector.execute(
                (X, y),
                target_type=target_type,
                models=models_to_train
            )
            
            # Filter features
            selected_features = feature_result.selected_features
            X_selected = X[selected_features]
            
            if self.config.verbose:
                print(f"✅ Feature selection complete:")
                print(f"   Original features: {len(X.columns)}")
                print(f"   Selected features: {len(selected_features)}")
                print(f"   Reduction: {(1 - len(selected_features)/len(X.columns)):.1%}")
                
            return X_selected, feature_result
            
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ Feature selection failed: {e}")
            return X, None
    
    def calculate_business_metrics(self, model_result: ModelResult, y_true, y_pred, y_prob=None) -> Dict[str, float]:
        """Calculate business metrics for a model"""
        
        if not self.kpi_tracker:
            return {}
            
        try:
            # Calculate business value from predictions
            business_value = calculate_business_value(y_true, y_pred, y_prob)
            
            # Prepare ML results for KPI calculation
            ml_results = {
                'model_accuracy': model_result.scores.get('test_accuracy', 0),
                'model_precision': model_result.scores.get('test_precision', 0),
                'model_recall': model_result.scores.get('test_recall', 0),
                'true_positives': business_value.get('true_positives', 0),
                'false_positives': business_value.get('false_positives', 0),
                'false_negatives': business_value.get('false_negatives', 0),
                'true_negatives': business_value.get('true_negatives', 0),
            }
            
            # Calculate KPIs
            kpi_results = self.kpi_tracker.calculate_kpis(ml_results)
            
            # Extract KPI values
            business_metrics = {
                name: result.value for name, result in kpi_results.items()
            }
            
            return business_metrics
            
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ Business metrics calculation failed: {e}")
            return {}
    
    def infer_target_type(self, y: pd.Series) -> str:
        """Enhanced target type inference"""
        n_unique = y.nunique()
        dtype = y.dtype
        
        if dtype.name in ["object", "category", "bool"]:
            return "classification"
        if np.issubdtype(dtype, np.integer) and n_unique <= 20:
            return "classification"
        return "regression"
    
    def _get_scaler(self):
        """Get scaler based on configuration"""
        if self.config.scaling_strategy == "standard":
            return StandardScaler()
        elif self.config.scaling_strategy == "minmax":
            return MinMaxScaler()
        elif self.config.scaling_strategy == "robust":
            return RobustScaler()
        elif self.config.scaling_strategy == "none":
            return None
        else:
            if self.config.verbose:
                print(f"⚠️ Unknown scaling strategy '{self.config.scaling_strategy}', using StandardScaler")
            return StandardScaler()
    
    def _get_actual_preprocessing_method(self):
        """Get accurate description of preprocessing method used"""
        if self.config.encoding_strategy == "none" and self.config.scaling_strategy == "none":
            return "Pre-cleaned data (no preprocessing needed)"
        elif self.config.encoding_strategy in ["onehot", "ordinal"]:
            return "ColumnTransformer (encoding + scaling)"
        elif self.config.encoding_strategy == "label":
            return "In-place LabelEncoder + Scaler"
        elif self.config.encoding_strategy == "none" and self.config.scaling_strategy != "none":
            return f"Scaling only ({self.config.scaling_strategy.title()}Scaler)"
        else:
            return f"Custom preprocessing ({self.config.encoding_strategy} + {self.config.scaling_strategy})"
    
    def _get_pipeline_structure_description(self):
        """Get accurate description of pipeline structure"""
        if self.config.encoding_strategy == "none" and self.config.scaling_strategy == "none":
            return "data -> model (no preprocessing)"
        elif self.config.encoding_strategy in ["onehot", "ordinal"]:
            return "data -> ColumnTransformer -> model"
        elif self.config.encoding_strategy == "none" and self.config.scaling_strategy != "none":
            return "data -> scaler -> model"
        elif self.config.encoding_strategy == "label" and self.config.scaling_strategy != "none":
            return "data -> label_encoder -> scaler -> model"
        elif self.config.encoding_strategy == "label" and self.config.scaling_strategy == "none":
            return "data -> label_encoder -> model"
        else:
            return f"data -> {self.config.encoding_strategy}+{self.config.scaling_strategy} -> model"
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target: str
    ) -> Tuple[pd.DataFrame, pd.Series, str]:
        """Prepare data for training - handles both raw and pre-cleaned data"""
        X = df.drop(columns=[target])
        y = df[target]
        
        # Infer target type
        target_type = self.infer_target_type(y)
        
        # Handle categorical target (if not already encoded)
        if target_type == "classification" and y.dtype in ["object", "category"]:
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)
        
        # Check if data is already preprocessed (all numeric = likely cleaned)
        is_preprocessed = all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number)))
        
        if is_preprocessed and self.config.encoding_strategy == "none" and self.config.scaling_strategy == "none":
            # Data is already cleaned - no preprocessing needed
            if self.config.verbose:
                print("✅ Data appears to be pre-cleaned (all numeric) - skipping preprocessing")
            return X, y, target_type, None
        
        # Original preprocessing logic for raw data
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.to_list()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.to_list()        
        
        # Get scaler based on configuration
        scaler = self._get_scaler()
        
        preprocessor = None
        
        if self.config.encoding_strategy == "label":
            # Label Encode categoricals
            for col in categorical_cols:
                if self.config.verbose:
                    print(f"🔄 Label encoding column: {col}")
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            preprocessor = None
        elif self.config.encoding_strategy == "onehot":
            # Defer to ColumnTransformer during pipeline creation
            if self.config.verbose:
                print(f"🔧 OneHot encoding deferred to pipeline for columns: {categorical_cols}")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', scaler, numeric_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ]
            )
        elif self.config.encoding_strategy == "ordinal":
            # Ordinal encoding for categoricals
            if self.config.verbose:
                print(f"🔧 Ordinal encoding deferred to pipeline for columns: {categorical_cols}")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', scaler, numeric_cols),
                    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
                ]
            )
        elif self.config.encoding_strategy == "none":
            # No encoding - assume data is already processed
            if scaler and self.config.scaling_strategy != "none":
                preprocessor = scaler
            else:
                preprocessor = None
        else:
            raise ValueError(f"Unknown encoding strategy: {self.config.encoding_strategy}")

        return X, y, target_type, preprocessor
     
    
    def create_pipeline(self, model_class, target_type: str, preprocessor=None) -> Pipeline:
        """Create preprocessing pipeline"""
        if preprocessor is not None:
            steps = [
                ('preprocessor', preprocessor),
                ('model', model_class())
            ]
        else:
            # Fall back to configurable scaler
            scaler = self._get_scaler()
            if scaler is not None:
                steps = [
                    ('scaler', scaler),
                    ('model', model_class())
                ]
            else:
                steps = [
                    ('model', model_class())
                ]
        return Pipeline(steps)
    
    def tune_hyperparameters(
        self,
        pipeline: Pipeline,
        param_grid: Dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_type: str
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """Hyperparameter tuning with GridSearch or RandomizedSearch"""
        
        if not param_grid or not self.config.enable_tuning:
            return pipeline, {}
        
        # Prepend 'model__' to parameter names for pipeline
        pipeline_params = {f"model__{k}": v for k, v in param_grid.items()}
        
        # Choose scoring metric
        scoring = "neg_root_mean_squared_error" if target_type == "regression" else "accuracy"
        
        # Choose CV strategy
        cv = KFold(n_splits=self.config.tuning_cv_folds, shuffle=True, random_state=self.config.random_state)
        if target_type == "classification":
            cv = StratifiedKFold(n_splits=self.config.tuning_cv_folds, shuffle=True, random_state=self.config.random_state)
        
        # Choose search method
        if self.config.tuning_method == "grid":
            search = GridSearchCV(
                pipeline, 
                pipeline_params, 
                cv=cv, 
                scoring=scoring,
                n_jobs=self.config.parallel_jobs
            )
        else:  # random search
            search = RandomizedSearchCV(
                pipeline,
                pipeline_params,
                n_iter=min(self.config.tuning_iterations, 50),
                cv=cv,
                scoring=scoring,
                random_state=self.config.random_state,
                n_jobs=self.config.parallel_jobs
            )
        
        # Fit with timeout if specified
        try:
            search.fit(X_train, y_train)
            return search.best_estimator_, search.best_params_
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ Hyperparameter tuning failed: {e}")
            return pipeline, {}
    
    def evaluate_model(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        target_type: str = "regression"
    ) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        
        scores = {}
        
        # Test set evaluation
        y_pred = pipeline.predict(X_test)
        
        if target_type == "regression":
            scores["test_rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
            scores["test_mae"] = mean_absolute_error(y_test, y_pred)
            scores["test_r2"] = r2_score(y_test, y_pred)
            
            # Validation set if available
            if X_val is not None and y_val is not None:
                y_val_pred = pipeline.predict(X_val)
                scores["val_rmse"] = np.sqrt(mean_squared_error(y_val, y_val_pred))
                scores["val_r2"] = r2_score(y_val, y_val_pred)
                
        else:  # classification
            scores["test_accuracy"] = accuracy_score(y_test, y_pred)
            scores["test_f1"] = f1_score(y_test, y_pred, average="weighted")
            
            # ROC AUC if possible
            if hasattr(pipeline.named_steps['model'], 'predict_proba'):
                try:
                    y_proba = pipeline.predict_proba(X_test)
                    if len(np.unique(y_test)) == 2:
                        scores["test_roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        scores["test_roc_auc"] = roc_auc_score(y_test, y_proba, multi_class="ovr")
                except:
                    pass
            
            # Validation set if available
            if X_val is not None and y_val is not None:
                y_val_pred = pipeline.predict(X_val)
                scores["val_accuracy"] = accuracy_score(y_val, y_val_pred)
                scores["val_f1"] = f1_score(y_val, y_val_pred, average="weighted")
        
        return scores
    
    def cross_validate_model(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        target_type: str
    ) -> Dict[str, float]:
        """Cross-validation evaluation"""
        
        cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        if target_type == "classification":
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        
        cv_scores = {}
        
        if target_type == "regression":
            rmse_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error")
            r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
            
            cv_scores["cv_rmse_mean"] = -rmse_scores.mean()
            cv_scores["cv_rmse_std"] = rmse_scores.std()
            cv_scores["cv_r2_mean"] = r2_scores.mean()
            cv_scores["cv_r2_std"] = r2_scores.std()
        else:
            acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
            f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_weighted")
            
            cv_scores["cv_accuracy_mean"] = acc_scores.mean()
            cv_scores["cv_accuracy_std"] = acc_scores.std()
            cv_scores["cv_f1_mean"] = f1_scores.mean()
            cv_scores["cv_f1_std"] = f1_scores.std()
        
        return cv_scores
    
    def extract_feature_importance(
        self,
        pipeline: Pipeline,
        feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """Extract feature importance/coefficients"""
        
        model = pipeline.named_steps['model']
        
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                coef = coef[0]  # Take first class for multi-class
            return dict(zip(feature_names, coef))
        else:
            return None
    
    def train_single_model(
        self,
        model_name: str,
        model_config: Dict,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        target_type: str,
        preprocessor=None
    ) -> ModelResult:
        """Train a single model with comprehensive evaluation"""
        
        start_time = time.time()
        
        try:
            # Create pipeline
            pipeline = self.create_pipeline(model_config["model"], target_type, preprocessor)
            
            # Hyperparameter tuning
            best_params = {}
            if model_config["tune"]:
                pipeline, best_params = self.tune_hyperparameters(
                    pipeline, model_config["params"], X_train, y_train, target_type
                )
            
            # Fit the model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            scores = self.evaluate_model(
                pipeline, X_train, X_test, y_train, y_test, X_val, y_val, target_type
            )
            
            # Cross-validation (if comprehensive scoring)
            cv_scores = {}
            if self.config.scoring_strategy == "comprehensive":
                # Use smaller dataset for CV to save time
                X_cv = pd.concat([X_train, X_test])
                y_cv = pd.concat([y_train, y_test])
                cv_scores = self.cross_validate_model(pipeline, X_cv, y_cv, target_type)
            
            # Feature importance
            feature_importance = self.extract_feature_importance(pipeline, X_train.columns.tolist())
            
            # Save model
            model_path = None
            if self.config.save_models:
                model_dir = Path(self.config.model_dir)
                model_dir.mkdir(exist_ok=True)
                model_path = model_dir / f"{model_name}_{int(time.time())}.pkl"
                joblib.dump(pipeline, model_path)
            
            training_time = time.time() - start_time
            
            return ModelResult(
                name=model_name,
                model=pipeline.named_steps['model'],
                pipeline=pipeline,
                scores=scores,
                cv_scores=cv_scores,
                feature_importance=feature_importance,
                training_time=training_time,
                best_params=best_params,
                model_path=str(model_path) if model_path else None
            )
            
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Failed to train {model_name}: {e}")
            return None
    
    def train_all_models(
        self,
        df: pd.DataFrame,
        target: str
    ) -> List[ModelResult]:
        """Train multiple models and return results with business features integration"""
        
        if self.config.verbose:
            print("🚀 Starting comprehensive model training...")
        
        # Prepare data
        X, y, target_type, preprocessor = self.prepare_data(df, target)
        
        if self.config.verbose:
            print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"🎯 Target type: {target_type}")
            print(f"🔧 Encoding strategy: {self.config.encoding_strategy}")
            print(f"📏 Scaling strategy: {self.config.scaling_strategy}")
        
        # Select model registry first to know which models we'll train
        if target_type == "regression":
            models_to_try = self.regression_models
        else:
            models_to_try = self.classification_models
        
        # Check for custom model selection (models_to_use) - prioritize this in custom mode
        if hasattr(self.config, 'models_to_use') and self.config.models_to_use:
            if self.config.verbose:
                print(f"🎯 Custom model selection: {self.config.models_to_use}")
            
            # Filter models to only include those specified in models_to_use
            custom_models = {}
            for model_name in self.config.models_to_use:
                if model_name in models_to_try:
                    custom_models[model_name] = models_to_try[model_name]
                    if self.config.verbose:
                        print(f"   ✅ {model_name} - Found in registry")
                else:
                    if self.config.verbose:
                        print(f"   ❌ {model_name} - Not found in registry")
            
            if custom_models:
                models_to_try = custom_models
                if self.config.verbose:
                    print(f"🤖 Will train {len(models_to_try)} custom-selected models")
            else:
                if self.config.verbose:
                    print(f"⚠️ No valid models found in models_to_use list, falling back to max_models limit")
                
        # If no custom model selection, apply original max_models logic
        elif self.config.max_models < len(models_to_try):
            # Priority: keep advanced models and ensemble methods
            priority_models = ["XGBRegressor", "XGBClassifier", "RandomForestRegressor", 
                             "RandomForestClassifier", "GradientBoostingRegressor", 
                             "GradientBoostingClassifier"]
            
            kept_models = {}
            for name in priority_models:
                if name in models_to_try:
                    kept_models[name] = models_to_try[name]
            
            # Fill remaining slots
            remaining = self.config.max_models - len(kept_models)
            for name, config in models_to_try.items():
                if name not in kept_models and remaining > 0:
                    kept_models[name] = config
                    remaining -= 1
            
            models_to_try = kept_models
        
        # Apply business feature selection BEFORE data splitting
        model_names = list(models_to_try.keys())
        X_selected, feature_selection_result = self.apply_business_feature_selection(
            X, y, target_type, model_names
        )
        
        # Store feature selection result for all models
        if feature_selection_result:
            for model_name in model_names:
                self.feature_selection_results[model_name] = feature_selection_result
        
        # Split data using selected features
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_selected, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if target_type == "classification" else None
        )
        
        # Further split for validation set
        if self.config.validation_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=self.config.validation_size,
                random_state=self.config.random_state,
                stratify=y_temp if target_type == "classification" else None
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
        
        if self.config.verbose:
            print(f"🤖 Training {len(models_to_try)} models: {list(models_to_try.keys())}")
            if feature_selection_result:
                print(f"🔍 Using {len(feature_selection_result.selected_features)} selected features")
        
        # Train models
        results = []
        for model_name, model_config in models_to_try.items():
            if self.config.verbose:
                print(f"  🔄 Training {model_name}...")
            
            result = self.train_single_model(
                model_name, model_config,
                X_train, X_test, y_train, y_test,
                X_val, y_val, target_type, preprocessor
            )
            
            if result:
                # Add business feature information
                if feature_selection_result:
                    result.selected_features = feature_selection_result.selected_features
                    result.feature_selection_result = feature_selection_result
                
                # Calculate business metrics if enabled
                if self.config.business_kpi_tracking:
                    y_pred = result.pipeline.predict(X_test)
                    y_prob = None
                    if hasattr(result.pipeline.named_steps['model'], 'predict_proba'):
                        try:
                            y_prob = result.pipeline.predict_proba(X_test)
                        except:
                            pass
                    
                    business_metrics = self.calculate_business_metrics(result, y_test, y_pred, y_prob)
                    result.business_metrics = business_metrics
                    
                    # Calculate business value
                    if y_prob is not None and target_type == "classification":
                        business_value = calculate_business_value(y_test, y_pred, y_prob)
                        result.business_value = business_value
                
                results.append(result)
                
                if self.config.verbose:
                    primary_score = result.scores.get("test_r2" if target_type == "regression" else "test_accuracy", 0)
                    business_score = result.business_metrics.get("churn_prevention_value", 0) if result.business_metrics else 0
                    print(f"    ✅ {model_name}: {primary_score:.4f} ({result.training_time:.2f}s)")
                    if business_score > 0:
                        print(f"       💰 Business value: ${business_score:,.0f}")
        
        self.results = results
        return results
    
    def get_best_models(
        self, 
        results: List[ModelResult], 
        target_type: str,
        top_k: int = 5
    ) -> List[ModelResult]:
        """Get top k best models based on performance"""
        
        if not results:
            return []
        
        # Choose primary metric for ranking
        if target_type == "regression":
            primary_metric = "test_r2"  # Higher is better
            reverse = True
        else:
            primary_metric = "test_accuracy"  # Higher is better  
            reverse = True
        
        # Sort by primary metric
        valid_results = [r for r in results if primary_metric in r.scores]
        sorted_results = sorted(
            valid_results, 
            key=lambda x: x.scores[primary_metric], 
            reverse=reverse
        )
        
        return sorted_results[:top_k]
    
    def create_ensemble(
        self,
        best_models: List[ModelResult],
        target_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Optional[ModelResult]:
        """Create ensemble from best models"""
        
        if not self.config.include_ensemble or len(best_models) < 2:
            return None
        
        try:
            # Take top 3 models for ensemble
            top_models = best_models[:3]
            
            estimators = [(f"model_{i}", result.pipeline) for i, result in enumerate(top_models)]
            
            if target_type == "regression":
                ensemble = VotingRegressor(estimators=estimators)
            else:
                # Only use models that support predict_proba
                prob_estimators = []
                for name, pipeline in estimators:
                    if hasattr(pipeline.named_steps['model'], 'predict_proba'):
                        prob_estimators.append((name, pipeline))
                
                if len(prob_estimators) >= 2:
                    ensemble = VotingClassifier(estimators=prob_estimators, voting='soft')
                else:
                    ensemble = VotingClassifier(estimators=estimators, voting='hard')
            
            start_time = time.time()
            ensemble.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            return ModelResult(
                name="Ensemble",
                model=ensemble,
                pipeline=ensemble,  # Ensemble is already a pipeline-like object
                scores={},  # Will be filled by evaluate_model
                cv_scores={},
                feature_importance=None,
                training_time=training_time,
                best_params={},
                model_path=None
            )
            
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Ensemble creation failed: {e}")
            return None
    
    def generate_report(
        self,
        results: List[ModelResult],
        target_type: str,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive training report with business metrics"""
        
        if not results:
            return {"error": "No successful model training results"}
        
        best_models = self.get_best_models(results, target_type)
        
        # Calculate business summary
        business_summary = {}
        if self.config.business_kpi_tracking and any(r.business_metrics for r in results):
            total_business_value = sum(
                r.business_value.get('total_business_value', 0) 
                for r in results if r.business_value
            ) / len([r for r in results if r.business_value])
            
            business_summary = {
                "average_business_value": total_business_value,
                "models_with_business_metrics": len([r for r in results if r.business_metrics]),
                "best_business_model": max(
                    [r for r in results if r.business_value], 
                    key=lambda x: x.business_value.get('total_business_value', 0)
                ).name if any(r.business_value for r in results) else None
            }
        
        # Feature selection summary
        feature_selection_summary = {}
        if any(r.feature_selection_result for r in results):
            fs_result = next(r.feature_selection_result for r in results if r.feature_selection_result)
            feature_selection_summary = {
                "feature_selection_enabled": True,
                "original_features": fs_result.metadata.get('original_features', 0),
                "selected_features": len(fs_result.selected_features),
                "reduction_percentage": (1 - len(fs_result.selected_features) / fs_result.metadata.get('original_features', 1)) * 100,
                "selection_method": fs_result.selection_method,
                "business_rules_applied": fs_result.business_rules_applied,
                "human_approved": fs_result.human_approved
            }
        
        report = {
            "summary": {
                "total_models_trained": len(results),
                "target_type": target_type,
                "best_model": best_models[0].name if best_models else None,
                "total_training_time": sum(r.training_time for r in results),
                "encoding_strategy": self.config.encoding_strategy,
                "preprocessing_method": self._get_actual_preprocessing_method(),
                **business_summary
            },
            "preprocessing_details": {
                "encoding_strategy": self.config.encoding_strategy,
                "scaling_strategy": self.config.scaling_strategy,
                "scaling_method": self.config.scaling_strategy.title() + "Scaler" if self.config.scaling_strategy != "none" else "No scaling",
                "pipeline_structure": self._get_pipeline_structure_description()
            },
            "feature_selection": feature_selection_summary,
            "business_features": {
                "enabled": self.config.enable_business_features,
                "feature_selection_enabled": self.config.feature_selection_enabled,
                "kpi_tracking_enabled": self.config.business_kpi_tracking,
                "business_rules_file": self.config.business_rules_file,
                "business_kpis_file": self.config.business_kpis_file
            },
            "model_rankings": [],
            "detailed_results": {},
            "config_used": self.config.__dict__
        }
        
        # Model rankings with business metrics
        for i, result in enumerate(best_models, 1):
            primary_score = result.scores.get("test_r2" if target_type == "regression" else "test_accuracy", 0)
            business_value = result.business_value.get('total_business_value', 0) if result.business_value else 0
            
            ranking_entry = {
                "rank": i,
                "model": result.name,
                "primary_score": round(primary_score, 4),
                "training_time": round(result.training_time, 2)
            }
            
            if business_value > 0:
                ranking_entry["business_value"] = round(business_value, 2)
                
            report["model_rankings"].append(ranking_entry)
        
        # Detailed results for each model
        for result in results:
            detailed_result = {
                "scores": result.scores,
                "cv_scores": result.cv_scores,
                "best_params": result.best_params,
                "training_time": result.training_time,
                "model_path": result.model_path,
                "feature_importance": result.feature_importance
            }
            
            # Add business information
            if result.selected_features:
                detailed_result["selected_features"] = result.selected_features
                detailed_result["feature_count"] = len(result.selected_features)
                
            if result.business_metrics:
                detailed_result["business_metrics"] = result.business_metrics
                
            if result.business_value:
                detailed_result["business_value"] = result.business_value
                
            report["detailed_results"][result.name] = detailed_result
        
        # Business correlation analysis
        if self.kpi_tracker and len(results) > 1:
            try:
                ml_metrics = {
                    result.name: result.scores.get("test_accuracy" if target_type == "classification" else "test_r2", 0)
                    for result in results
                }
                business_metrics = {
                    result.name: result.business_value.get('total_business_value', 0) if result.business_value else 0
                    for result in results
                }
                
                correlations = self.kpi_tracker.calculate_business_ml_correlation(ml_metrics, business_metrics)
                report["business_ml_correlation"] = correlations
                
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️ Failed to calculate business-ML correlation: {e}")
        
        # Generate business report if KPI tracker available
        if self.kpi_tracker and any(r.business_metrics for r in results):
            try:
                # Use best model's metrics for business report
                best_model = best_models[0] if best_models else results[0]
                business_report = self.kpi_tracker.generate_business_report(
                    {k: type('KPIResult', (), {'value': v, 'target_value': None, 'achievement_rate': None, 
                                               'status': 'good', 'impact_score': v, 'metadata': {}})() 
                     for k, v in best_model.business_metrics.items()}
                )
                report["business_analysis"] = business_report
                
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️ Failed to generate business report: {e}")
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report

# Convenience function for backward compatibility
def train_model(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    target_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
    metric: Optional[str] = None,
    use_cv: bool = False,
    n_splits: int = 5,
    model_out: Optional[str] = None
) -> Dict[str, Any]:
    """Legacy function - train single model"""
    
    config = TrainerConfig(
        test_size=test_size,
        random_state=random_state,
        max_models=1,
        enable_tuning=False,
        save_models=model_out is not None,
        verbose=True
    )
    
    trainer = EnhancedModelTrainer(config)
    
    # Prepare data
    X, y, inferred_type, preprocessor = trainer.prepare_data(df, target)
    target_type = target_type or inferred_type
    
    # Get model config
    if target_type == "regression":
        model_config = trainer.regression_models.get(model_name)
    else:
        model_config = trainer.classification_models.get(model_name)
    
    if not model_config:
        raise ValueError(f"Model '{model_name}' not found")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if target_type == "classification" else None
    )
    
    # Train single model
    result = trainer.train_single_model(
        model_name, model_config,
        X_train, X_test, y_train, y_test,
        None, None, target_type, preprocessor
    )
    
    if not result:
        raise RuntimeError(f"Training failed for {model_name}")
    
    # Legacy format
    primary_score = result.scores.get("test_r2" if target_type == "regression" else "test_accuracy", 0)
    
    return {
        "model_name": model_name,
        "model_path": result.model_path,
        "metrics": result.scores,
        "score_metric": "r2" if target_type == "regression" else "accuracy",
        "score": primary_score,
        "feature_importances": result.feature_importance,
    }

# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Enhanced Trainer Usage:")
        print("  Single model: python trainer.py <csv> <target> <model_name>")
        print("  All models:   python trainer.py <csv> <target> --all")
        print("  Config file:  python trainer.py <csv> <target> --config <config.json>")
        sys.exit(1)
    
    df = pd.read_csv(sys.argv[1])
    target = sys.argv[2]
    
    if len(sys.argv) > 3 and sys.argv[3] == "--all":
        # Train all models
        config = TrainerConfig(verbose=True, save_models=True)
        trainer = EnhancedModelTrainer(config)
        
        results = trainer.train_all_models(df, target)
        report = trainer.generate_report(results, trainer.infer_target_type(df[target]))
        
        print(f"\n🏆 TRAINING COMPLETE!")
        print(f"📊 Trained {len(results)} models")
        print(f"🥇 Best model: {report['summary']['best_model']}")
        print(f"⏱️ Total time: {report['summary']['total_training_time']:.2f}s")
        
        print(f"\n🏅 TOP 5 MODELS:")
        for ranking in report['model_rankings'][:5]:
            print(f"  {ranking['rank']}. {ranking['model']}: {ranking['primary_score']:.4f}")
        
        # Save report
        with open("training_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n💾 Detailed report saved to: training_report.json")
        
    else:
        # Single model (legacy)
        model_name = sys.argv[3]
        result = train_model(df, target, model_name, None)
        print(f"\n✅ {model_name} training complete")
        print(f"Score: {result['score']:.4f}")