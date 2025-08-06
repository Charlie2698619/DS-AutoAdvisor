"""
Example Feature Selection Plugin for DS-AutoAdvisor v2.0
========================================================

This plugin demonstrates how to create a feature selection extension 
using the DS-AutoAdvisor plugin architecture.
"""

from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

from src.infrastructure.plugin_system import FeatureSelectionPlugin, PluginInfo, PluginType, PluginStatus, PluginConfig

class AdvancedFeatureSelector(FeatureSelectionPlugin):
    """
    Advanced feature selection plugin with multiple methods
    """
    
    @property
    def plugin_info(self) -> PluginInfo:
        """Return plugin information"""
        return PluginInfo(
            name="advanced_feature_selector",
            version="1.0.0",
            description="Advanced feature selection with multiple methods and ensemble voting",
            author="DS-AutoAdvisor Team",
            plugin_type=PluginType.FEATURE_SELECTION,
            dependencies=["sklearn", "pandas", "numpy"],
            config_schema={
                "methods": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["mutual_info", "rfe", "importance"]
                },
                "k_features": {
                    "type": "integer", 
                    "default": 10,
                    "description": "Number of features to select"
                },
                "ensemble_voting": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use ensemble voting across methods"
                }
            }
        )
    
    def initialize(self) -> bool:
        """Initialize the plugin"""
        try:
            self.available_methods = {
                'mutual_info': self._mutual_info_selection,
                'rfe': self._rfe_selection,
                'importance': self._importance_selection,
                'univariate': self._univariate_selection
            }
            
            self._is_initialized = True
            self.logger.info("Advanced Feature Selector plugin initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin: {e}")
            return False
    
    def execute(self, data: Any, **kwargs) -> Any:
        """Execute feature selection"""
        X, y = data
        return self.select_features(X, y, **kwargs)
    
    def select_features(self, X, y, **kwargs) -> List[str]:
        """
        Select features using ensemble of methods
        
        Args:
            X: Feature matrix (DataFrame)
            y: Target variable (Series)
            **kwargs: Additional parameters
            
        Returns:
            List of selected feature names
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        methods = self.config.config.get('methods', ['mutual_info', 'rfe'])
        k_features = self.config.config.get('k_features', 10)
        ensemble_voting = self.config.config.get('ensemble_voting', True)
        
        # Ensure k_features doesn't exceed available features
        k_features = min(k_features, len(X.columns))
        
        feature_scores = {}
        method_results = {}
        
        # Apply each method
        for method in methods:
            if method in self.available_methods:
                try:
                    selected_features = self.available_methods[method](X, y, k_features)
                    method_results[method] = selected_features
                    
                    # Assign scores based on selection order (earlier = higher score)
                    for i, feature in enumerate(selected_features):
                        score = len(selected_features) - i
                        if feature not in feature_scores:
                            feature_scores[feature] = 0
                        feature_scores[feature] += score
                        
                except Exception as e:
                    self.logger.warning(f"Method {method} failed: {e}")
                    continue
        
        if not feature_scores:
            self.logger.warning("All feature selection methods failed, returning all features")
            return list(X.columns[:k_features])
        
        # Select top features based on ensemble scores
        if ensemble_voting:
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feature for feature, score in sorted_features[:k_features]]
        else:
            # Use first successful method
            selected_features = list(method_results.values())[0][:k_features]
        
        self.logger.info(f"Selected {len(selected_features)} features using methods: {methods}")
        return selected_features
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """Mutual information based feature selection"""
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        
        # Determine if classification or regression
        if self._is_classification_task(y):
            scores = mutual_info_classif(X, y, random_state=42)
        else:
            scores = mutual_info_regression(X, y, random_state=42)
        
        # Get top k features
        feature_scores = list(zip(X.columns, scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [feature for feature, score in feature_scores[:k]]
    
    def _rfe_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """Recursive Feature Elimination"""
        # Choose estimator based on task type
        if self._is_classification_task(y):
            estimator = LogisticRegression(random_state=42, max_iter=1000)
        else:
            estimator = LinearRegression()
        
        rfe = RFE(estimator, n_features_to_select=k)
        rfe.fit(X, y)
        
        selected_features = X.columns[rfe.support_].tolist()
        return selected_features
    
    def _importance_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """Feature importance based selection using Random Forest"""
        # Choose Random Forest based on task type
        if self._is_classification_task(y):
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        feature_importance = list(zip(X.columns, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return [feature for feature, importance in feature_importance[:k]]
    
    def _univariate_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """Univariate statistical test based selection"""
        # Choose score function based on task type
        if self._is_classification_task(y):
            score_func = f_classif
        else:
            score_func = f_regression
        
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        return selected_features
    
    def _is_classification_task(self, y: pd.Series) -> bool:
        """Determine if this is a classification task"""
        # Simple heuristic: if target has few unique values, it's classification
        return y.nunique() < 20 and y.dtype in ['object', 'category', 'bool']
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration"""
        try:
            methods = config.get('methods', [])
            if not isinstance(methods, list):
                return False
            
            for method in methods:
                if method not in self.available_methods:
                    self.logger.warning(f"Unknown method: {method}")
                    return False
            
            k_features = config.get('k_features', 10)
            if not isinstance(k_features, int) or k_features <= 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Config validation failed: {e}")
            return False
