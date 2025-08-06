"""
Advanced ML Preprocessing Steps (POST-CLEANING)
===============================================

These steps happen AFTER basic data cleaning and are more ML-specific.
They assume you have clean, properly formatted data.

Pipeline Order:
1. Basic Cleaning (duplicates, missing values, outliers) ← Your original cleaner
2. Advanced ML Preprocessing (this file) ← Feature engineering, selection, etc.
3. Model Training

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from sklearn.feature_selection import (
    SelectKBest, chi2, f_regression, mutual_info_regression, 
    mutual_info_classif, RFE
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from category_encoders import TargetEncoder, BinaryEncoder, HashingEncoder
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import warnings

warnings.filterwarnings("ignore")

class MLPreprocessingStep(ABC):
    """Base class for ML preprocessing steps (post-cleaning)"""
    
    def __init__(self, name: str):
        self.name = name
        self.fitted_objects = {}  # Store fitted transformers for transform phase
        
    @abstractmethod
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                     config: dict = None, log: dict = None) -> pd.DataFrame:
        """Fit and transform training data"""
        pass
        
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test/validation data using fitted objects"""
        pass
        
    def log_action(self, log: Dict, message: str, step_type: str = "action"):
        if log:
            log[f"{step_type}s"].append(f"[{self.name}] {message}")

class FeatureSelectionStep(MLPreprocessingStep):
    """Feature Selection using various methods"""
    
    def __init__(self, method: str = "mutual_info", k: int = 10):
        super().__init__("FeatureSelection")
        self.method = method
        self.k = k
        self.selector = None
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                     config: dict = None, log: dict = None) -> pd.DataFrame:
        if y is None:
            self.log_action(log, "Skipping feature selection - no target provided", "warning")
            return X
            
        original_features = X.shape[1]
        
        # Choose selection method
        if self.method == "chi2":
            # For categorical features and targets
            self.selector = SelectKBest(chi2, k=self.k)
        elif self.method == "f_regression":
            # For continuous targets
            self.selector = SelectKBest(f_regression, k=self.k)
        elif self.method == "mutual_info":
            # For both categorical and continuous
            if y.dtype == 'object' or y.nunique() < 20:
                self.selector = SelectKBest(mutual_info_classif, k=self.k)
            else:
                self.selector = SelectKBest(mutual_info_regression, k=self.k)
        elif self.method == "rfe":
            # Recursive Feature Elimination (requires estimator)
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=10, random_state=42)
            self.selector = RFE(estimator, n_features_to_select=self.k)
        
        # Fit and transform
        X_selected = pd.DataFrame(
            self.selector.fit_transform(X, y),
            columns=X.columns[self.selector.get_support()],
            index=X.index
        )
        
        selected_features = X_selected.shape[1]
        self.log_action(log, f"Selected {selected_features} features from {original_features} using {self.method}")
        
        return X_selected
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selector is None:
            return X
        return pd.DataFrame(
            self.selector.transform(X),
            columns=X.columns[self.selector.get_support()],
            index=X.index
        )

class AdvancedEncodingStep(MLPreprocessingStep):
    """Advanced categorical encoding methods"""
    
    def __init__(self, method: str = "target", max_cardinality: int = 50):
        super().__init__("AdvancedEncoding")
        self.method = method
        self.max_cardinality = max_cardinality
        self.encoders = {}
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                     config: dict = None, log: dict = None) -> pd.DataFrame:
        
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if not cat_cols:
            return X
            
        X_encoded = X.copy()
        
        for col in cat_cols:
            n_unique = X[col].nunique()
            
            if n_unique > self.max_cardinality:
                self.log_action(log, f"Skipping {col} - too many categories ({n_unique})", "warning")
                continue
                
            if self.method == "target" and y is not None:
                self.encoders[col] = TargetEncoder()
                X_encoded[col] = self.encoders[col].fit_transform(X[col], y)
                
            elif self.method == "binary":
                self.encoders[col] = BinaryEncoder(cols=[col])
                encoded_df = self.encoders[col].fit_transform(X[[col]])
                X_encoded = X_encoded.drop(columns=[col])
                X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
                
            elif self.method == "hash":
                self.encoders[col] = HashingEncoder(cols=[col], n_components=min(8, n_unique))
                encoded_df = self.encoders[col].fit_transform(X[[col]])
                X_encoded = X_encoded.drop(columns=[col])
                X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
        
        encoded_cols = len(self.encoders)
        if encoded_cols > 0:
            self.log_action(log, f"Applied {self.method} encoding to {encoded_cols} columns")
            
        return X_encoded
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = X.copy()
        
        for col, encoder in self.encoders.items():
            if col in X_encoded.columns:
                if isinstance(encoder, (BinaryEncoder, HashingEncoder)):
                    encoded_df = encoder.transform(X_encoded[[col]])
                    X_encoded = X_encoded.drop(columns=[col])
                    X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
                else:
                    X_encoded[col] = encoder.transform(X_encoded[col])
        
        return X_encoded

class TimeSeriesFeaturesStep(MLPreprocessingStep):
    """Time series feature engineering"""
    
    def __init__(self, date_cols: List[str] = None, lag_periods: List[int] = [1, 7, 30]):
        super().__init__("TimeSeriesFeatures")
        self.date_cols = date_cols or []
        self.lag_periods = lag_periods
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                     config: dict = None, log: dict = None) -> pd.DataFrame:
        X_ts = X.copy()
        features_added = 0
        
        # Auto-detect date columns if not specified
        if not self.date_cols:
            self.date_cols = X.select_dtypes(include=['datetime64']).columns.tolist()
        
        for col in self.date_cols:
            if col not in X.columns:
                continue
                
            # Cyclical encoding for time features
            if X[col].dtype == 'datetime64[ns]':
                # Month cyclical (1-12 -> 0-1)
                X_ts[f'{col}_month_sin'] = np.sin(2 * np.pi * X[col].dt.month / 12)
                X_ts[f'{col}_month_cos'] = np.cos(2 * np.pi * X[col].dt.month / 12)
                
                # Day of week cyclical (0-6 -> 0-1)
                X_ts[f'{col}_dow_sin'] = np.sin(2 * np.pi * X[col].dt.dayofweek / 7)
                X_ts[f'{col}_dow_cos'] = np.cos(2 * np.pi * X[col].dt.dayofweek / 7)
                
                # Hour cyclical (if time info available)
                if X[col].dt.hour.nunique() > 1:
                    X_ts[f'{col}_hour_sin'] = np.sin(2 * np.pi * X[col].dt.hour / 24)
                    X_ts[f'{col}_hour_cos'] = np.cos(2 * np.pi * X[col].dt.hour / 24)
                    features_added += 2
                
                features_added += 4
        
        # Lag features for numeric columns (if data is sorted by time)
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols[:5]:  # Limit to first 5 numeric columns
            for lag in self.lag_periods:
                X_ts[f'{col}_lag_{lag}'] = X[col].shift(lag)
                features_added += 1
        
        if features_added > 0:
            self.log_action(log, f"Added {features_added} time series features")
            
        return X_ts
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Same logic as fit_transform for time series features
        return self.fit_transform(X)

class ImbalancedDataStep(MLPreprocessingStep):
    """Handle imbalanced datasets"""
    
    def __init__(self, method: str = "smote", sampling_strategy: str = "auto"):
        super().__init__("ImbalancedData")
        self.method = method
        self.sampling_strategy = sampling_strategy
        self.sampler = None
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                     config: dict = None, log: dict = None) -> Tuple[pd.DataFrame, pd.Series]:
        if y is None:
            self.log_action(log, "Skipping imbalance handling - no target provided", "warning")
            return X, y
            
        # Check if imbalanced (>70% in majority class)
        class_counts = y.value_counts(normalize=True)
        if class_counts.max() < 0.7:
            self.log_action(log, "Dataset appears balanced - skipping resampling", "info")
            return X, y
        
        original_size = len(X)
        
        # Choose sampling method
        if self.method == "smote":
            self.sampler = SMOTE(sampling_strategy=self.sampling_strategy, random_state=42)
        elif self.method == "adasyn":
            self.sampler = ADASYN(sampling_strategy=self.sampling_strategy, random_state=42)
        elif self.method == "borderline_smote":
            self.sampler = BorderlineSMOTE(sampling_strategy=self.sampling_strategy, random_state=42)
        elif self.method == "undersample":
            self.sampler = RandomUnderSampler(sampling_strategy=self.sampling_strategy, random_state=42)
        elif self.method == "tomek":
            self.sampler = TomekLinks(sampling_strategy=self.sampling_strategy)
        
        try:
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled = pd.Series(y_resampled, name=y.name if hasattr(y, 'name') else 'target')
            
            new_size = len(X_resampled)
            self.log_action(log, f"Resampled data: {original_size} → {new_size} samples using {self.method}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            self.log_action(log, f"Resampling failed: {str(e)}", "error")
            return X, y
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Imbalanced data handling is only applied to training data
        return X

class AdvancedMLPreprocessor:
    """Orchestrator for advanced ML preprocessing steps"""
    
    def __init__(self, steps: List[MLPreprocessingStep]):
        self.steps = steps
        self.log = {
            "actions": [],
            "warnings": [],
            "errors": [],
            "info": []
        }
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply all preprocessing steps to training data"""
        X_processed = X.copy()
        y_processed = y.copy() if y is not None else None
        
        for step in self.steps:
            try:
                if isinstance(step, ImbalancedDataStep):
                    X_processed, y_processed = step.fit_transform(X_processed, y_processed, log=self.log)
                else:
                    X_processed = step.fit_transform(X_processed, y_processed, log=self.log)
            except Exception as e:
                self.log["errors"].append(f"{step.name} failed: {str(e)}")
                print(f"⚠️ {step.name} failed: {e}")
                continue
        
        return X_processed, y_processed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations to test/validation data"""
        X_processed = X.copy()
        
        for step in self.steps:
            try:
                if not isinstance(step, ImbalancedDataStep):  # Don't apply resampling to test data
                    X_processed = step.transform(X_processed)
            except Exception as e:
                self.log["errors"].append(f"{step.name} transform failed: {str(e)}")
                continue
        
        return X_processed

# Example usage
if __name__ == "__main__":
    # This would run AFTER your basic data cleaning
    
    # Example: Load cleaned data
    # df_clean = pd.read_csv("../../data/bank_cleaned.csv")  # From your cleaner
    # X = df_clean.drop('target', axis=1)
    # y = df_clean['target']
    
    # Create advanced preprocessing pipeline
    steps = [
        FeatureSelectionStep(method="mutual_info", k=15),
        AdvancedEncodingStep(method="target"),
        TimeSeriesFeaturesStep(lag_periods=[1, 7]),
        ImbalancedDataStep(method="smote")
    ]
    
    preprocessor = AdvancedMLPreprocessor(steps)
    
    # Fit on training data
    # X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Transform test data
    # X_test_processed = preprocessor.transform(X_test)
    
    print("✅ Advanced ML preprocessing pipeline created")
