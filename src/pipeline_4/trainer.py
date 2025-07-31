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
warnings.filterwarnings('ignore')

@dataclass
class TrainerConfig:
    """Type-safe configuration for model training"""
    # Data splitting
    test_size: float = 0.2
    validation_size: float = 0.1  # From training set
    random_state: int = 42
    
    # Model selection
    max_models: int = 10  # Limit number of models to try
    include_ensemble: bool = True
    include_advanced: bool = True  # XGBoost, LightGBM, etc.
    
    # Hyperparameter tuning
    enable_tuning: bool = True
    tuning_method: str = "random"  # "grid", "random", "none"
    tuning_iterations: int = 50
    tuning_cv_folds: int = 3
    
    # Validation
    cv_folds: int = 5
    scoring_strategy: str = "comprehensive"  # "fast", "comprehensive"
    
    # Performance
    parallel_jobs: int = -1
    max_training_time_minutes: Optional[int] = 30
    memory_limit_gb: Optional[float] = None
    
    # Encoding and preprocessing
    encoding_strategy: str = "onehot"  # "onehot", "label", "ordinal"
    scaling_strategy: str = "standard"  # "standard", "minmax", "robust", "none"
    
    # Output
    save_models: bool = True
    model_dir: str = "models"
    save_predictions: bool = True
    verbose: bool = True

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

class EnhancedModelTrainer:
    """Enhanced trainer with comprehensive model selection"""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.results = []
        self.best_model = None
        self._setup_model_registry()
        
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
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target: str
    ) -> Tuple[pd.DataFrame, pd.Series, str]:
        """Prepare data for training"""
        X = df.drop(columns=[target])
        y = df[target]
        
        # Infer target type
        target_type = self.infer_target_type(y)
        
        # Handle categorical target
        if target_type == "classification" and y.dtype in ["object", "category"]:
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)
        
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
        """Train multiple models and return results"""
        
        if self.config.verbose:
            print("🚀 Starting comprehensive model training...")
        
        # Prepare data
        X, y, target_type, preprocessor = self.prepare_data(df, target)
        
        if self.config.verbose:
            print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"🎯 Target type: {target_type}")
            print(f"🔧 Encoding strategy: {self.config.encoding_strategy}")
            print(f"📏 Scaling strategy: {self.config.scaling_strategy}")
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
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
        
        # Select model registry
        if target_type == "regression":
            models_to_try = self.regression_models
        else:
            models_to_try = self.classification_models
        
        # Limit number of models if specified
        if self.config.max_models < len(models_to_try):
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
        
        if self.config.verbose:
            print(f"🤖 Training {len(models_to_try)} models: {list(models_to_try.keys())}")
        
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
                results.append(result)
                if self.config.verbose:
                    primary_score = result.scores.get("test_r2" if target_type == "regression" else "test_accuracy", 0)
                    print(f"    ✅ {model_name}: {primary_score:.4f} ({result.training_time:.2f}s)")
        
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
        """Generate comprehensive training report"""
        
        if not results:
            return {"error": "No successful model training results"}
        
        best_models = self.get_best_models(results, target_type)
        
        report = {
            "summary": {
                "total_models_trained": len(results),
                "target_type": target_type,
                "best_model": best_models[0].name if best_models else None,
                "total_training_time": sum(r.training_time for r in results),
                "encoding_strategy": self.config.encoding_strategy,
                "preprocessing_method": "ColumnTransformer" if self.config.encoding_strategy in ["onehot", "ordinal"] else "In-place LabelEncoder"
            },
            "preprocessing_details": {
                "encoding_strategy": self.config.encoding_strategy,
                "scaling_strategy": self.config.scaling_strategy,
                "scaling_method": self.config.scaling_strategy.title() + "Scaler" if self.config.scaling_strategy != "none" else "No scaling",
                "pipeline_structure": "preprocessor -> model" if self.config.encoding_strategy in ["onehot", "ordinal"] else "scaler -> model"
            },
            "model_rankings": [],
            "detailed_results": {},
            "config_used": self.config.__dict__
        }
        
        # Model rankings
        for i, result in enumerate(best_models, 1):
            primary_score = result.scores.get("test_r2" if target_type == "regression" else "test_accuracy", 0)
            report["model_rankings"].append({
                "rank": i,
                "model": result.name,
                "primary_score": round(primary_score, 4),
                "training_time": round(result.training_time, 2)
            })
        
        # Detailed results for each model
        for result in results:
            report["detailed_results"][result.name] = {
                "scores": result.scores,
                "cv_scores": result.cv_scores,
                "best_params": result.best_params,
                "training_time": result.training_time,
                "model_path": result.model_path,
                "feature_importance": result.feature_importance
            }
        
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