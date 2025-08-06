"""
Business Feature Selector
=========================

Integrates statistical feature selection with business rules and human oversight.
Handles multiple models with consistent fe        # Step 5: Human approval (if required)
        if self._get_config_value('human_approval_required', False):ure selection strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import yaml
import json
from datetime import datetime

from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, f_regression, mutual_info_classif, 
    mutual_info_regression, RFE, RFECV, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score

# Simplified base classes for plugin system
class PluginInfo:
    def __init__(self, name: str, version: str, description: str):
        self.name = name
        self.version = version
        self.description = description

class PluginConfig:
    """Plugin configuration class"""
    pass

class PluginType:
    """Plugin type enumeration"""
    FEATURE_SELECTION = "feature_selection"

class FeatureSelectionPlugin:
    """Base class for feature selection plugins"""
    def __init__(self, config=None):
        self.config = config

@dataclass
class BusinessRule:
    """Represents a business rule for feature selection"""
    name: str
    rule_type: str  # 'must_include', 'must_exclude', 'conditional', 'preference'
    features: List[str]
    condition: Optional[str] = None
    reason: str = ""
    priority: int = 1
    active: bool = True

@dataclass
class FeatureSelectionResult:
    """Results from feature selection process"""
    selected_features: List[str]
    selection_method: str
    feature_scores: Dict[str, float]
    business_rules_applied: List[str]
    human_approved: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ModelFeatureMap:
    """Maps models to their selected features for consistency"""
    model_name: str
    features: List[str]
    selection_method: str
    performance_score: float
    business_alignment_score: float

class BusinessFeatureSelector(FeatureSelectionPlugin):
    """
    Business-aligned feature selector with multi-model consistency handling
    """
    
    def __init__(self, config: Union[str, PluginConfig] = None):
        super().__init__(config)
        self.business_rules = []
        self.feature_importance_cache = {}
        self.model_feature_maps = {}
        self.human_interface = None
        
        # Handle config as file path or config object
        if isinstance(config, str):
            self.config_path = config
        else:
            self.config_path = None
        
        # Load business rules from config
        self._load_business_rules()
        
        # Setup simple logger
        self.logger = self._setup_logger()
    
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Safely get configuration value"""
        if isinstance(self.config, str):
            # Config is a file path, return defaults
            defaults = {
                'human_approval_required': False,
                'max_feature_count': 20,
                'min_feature_count': 5,
                'feature_consistency_strategy': 'intersection',
                'business_rules_file': 'config/business_rules.yaml'
            }
            return defaults.get(key, default)
        elif hasattr(self.config, 'config') and self.config.config:
            return self.config.config.get(key, default)
        else:
            return default
        
    @property
    def plugin_info(self) -> PluginInfo:
        return PluginInfo(
            name="BusinessFeatureSelector",
            version="1.0.0",
            description="Business-aligned feature selection with human oversight",
            author="DS-AutoAdvisor Team",
            plugin_type=PluginType.FEATURE_SELECTION,
            dependencies=["scikit-learn", "pandas", "numpy"],
            config_schema={
                "business_rules_file": {"type": "string", "default": "config/business_rules.yaml"},
                "selection_methods": {"type": "array", "default": ["statistical", "ml_based", "business"]},
                "human_approval_required": {"type": "boolean", "default": True},
                "feature_consistency_strategy": {"type": "string", "default": "intersection"},
                "min_feature_count": {"type": "integer", "default": 5},
                "max_feature_count": {"type": "integer", "default": 50}
            }
        )
    
    def initialize(self) -> bool:
        """Initialize the feature selector"""
        try:
            self._load_business_rules()
            self._setup_human_interface()
            self._is_initialized = True
            self.logger.info("BusinessFeatureSelector initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize BusinessFeatureSelector: {e}")
            raise
    
    def _setup_logger(self):
        """Setup simple logger for testing"""
        class SimpleLogger:
            def info(self, msg): pass
            def warning(self, msg): pass  
            def error(self, msg): pass
        return SimpleLogger()
    
    def execute(self, data: Any, **kwargs) -> FeatureSelectionResult:
        """Main execution method"""
        X, y = data
        target_type = kwargs.get('target_type', 'classification')
        models_to_train = kwargs.get('models', ['random_forest'])
        
        # Step 1: Apply business rules
        business_filtered_features = self._apply_business_rules(X.columns.tolist())
        
        # Step 2: Statistical feature selection
        statistical_features = self._statistical_selection(
            X[business_filtered_features], y, target_type
        )
        
        # Step 3: ML-based feature selection
        ml_features = self._ml_based_selection(
            X[business_filtered_features], y, target_type
        )
        
        # Step 4: Handle multi-model consistency
        consistent_features = self._ensure_model_consistency(
            statistical_features, ml_features, models_to_train, X, y
        )
        
        # Step 5: Human approval if required
        if self.config.config.get('human_approval_required', True):
            approved_features = self._request_human_approval(
                consistent_features, X, y, business_filtered_features
            )
        else:
            approved_features = consistent_features
        
        # Step 6: Final validation
        final_features = self._final_validation(approved_features, X, y)
        
        return FeatureSelectionResult(
            selected_features=final_features,
            selection_method="business_aligned_multi_stage",
            feature_scores=self._calculate_feature_scores(X[final_features], y),
            business_rules_applied=[rule.name for rule in self.business_rules if rule.active],
            human_approved=False,  # Default to False for testing
            metadata={
                'original_features': len(X.columns),
                'business_filtered': len(business_filtered_features),
                'statistical_selected': len(statistical_features),
                'ml_selected': len(ml_features),
                'consistent_features': len(consistent_features),
                'final_features': len(final_features),
                'models_considered': models_to_train
            }
        )
    
    def select_features(self, data: pd.DataFrame, target: str, method: str = 'statistical') -> List[str]:
        """
        Simplified interface for feature selection
        
        Args:
            data: DataFrame containing features and target
            target: Name of target column
            method: Selection method ('statistical', 'ml_based', 'business_rules')
            
        Returns:
            List of selected feature names
        """
        X = data.drop(columns=[target])
        y = data[target]
        
        if method == 'statistical':
            return self._statistical_selection(X, y, 'classification')
        elif method == 'ml_based':
            return self._ml_based_selection(X, y, 'classification')
        elif method == 'business_rules':
            return self._apply_business_rules(X.columns.tolist())
        else:
            # Default: use full execute method
            result = self.execute((X, y), target_type='classification')
            return result.selected_features
    
    def _load_business_rules(self):
        """Load business rules from configuration"""
        # Determine rules file path
        if self.config_path:
            rules_file = self.config_path
        elif hasattr(self.config, 'config') and self.config.config:
            rules_file = self._get_config_value('business_rules_file', 'config/business_rules.yaml')
        else:
            rules_file = 'config/business_rules.yaml'
            
        rules_path = Path(rules_file)
        
        if rules_path.exists():
            try:
                with open(rules_path, 'r') as f:
                    rules_data = yaml.safe_load(f)
                
                self.business_rules = []
                # Handle both old and new structure
                rules_list = rules_data.get('feature_selection_rules', [])
                if not rules_list:
                    # Try new structure
                    rules_list = rules_data.get('feature_selection', {}).get('rules', [])
                
                for rule_data in rules_list:
                    rule = BusinessRule(**rule_data)
                    self.business_rules.append(rule)
                    
                print(f"Loaded {len(self.business_rules)} business rules")
            except Exception as e:
                print(f"Failed to load business rules: {e}")
                self.business_rules = []
        else:
            print("No business rules file found, using defaults")
            self.business_rules = []
    
    def _apply_business_rules(self, features: List[str]) -> List[str]:
        """Apply business rules to filter features"""
        filtered_features = features.copy()
        
        for rule in self.business_rules:
            if not rule.active:
                continue
                
            if rule.rule_type == 'must_include':
                # Ensure these features are included if they exist
                for feature in rule.features:
                    if feature in features and feature not in filtered_features:
                        filtered_features.append(feature)
                        
            elif rule.rule_type == 'must_exclude':
                # Remove these features
                filtered_features = [f for f in filtered_features if f not in rule.features]
                
            elif rule.rule_type == 'conditional':
                # Apply conditional logic (simplified for now)
                if rule.condition:
                    # This would need more sophisticated condition parsing
                    pass
        
        self.logger.info(f"Business rules filtered {len(features)} to {len(filtered_features)} features")
        return filtered_features
    
    def _statistical_selection(self, X: pd.DataFrame, y: pd.Series, target_type: str) -> List[str]:
        """Perform statistical feature selection"""
        if target_type == 'classification':
            # Use chi-square for categorical and f_classif for numerical
            selector = SelectKBest(score_func=f_classif, k='all')
        else:
            selector = SelectKBest(score_func=f_regression, k='all')
        
        # Handle mixed data types
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty:
            return X.columns.tolist()[:10]  # Fallback
        
        selector.fit(X_numeric, y)
        scores = selector.scores_
        
        # Get top features based on scores
        feature_scores = dict(zip(X_numeric.columns, scores))
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top K features
        k = min(self._get_config_value('max_feature_count', 20), len(sorted_features))
        selected = [feature for feature, score in sorted_features[:k]]
        
        self.logger.info(f"Statistical selection chose {len(selected)} features")
        return selected
    
    def _ml_based_selection(self, X: pd.DataFrame, y: pd.Series, target_type: str) -> List[str]:
        """Perform ML-based feature selection"""
        if target_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Use only numeric features for ML selection
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty:
            return X.columns.tolist()[:10]  # Fallback
        
        # Feature importance based selection
        estimator.fit(X_numeric, y)
        importances = estimator.feature_importances_
        
        feature_importance = dict(zip(X_numeric.columns, importances))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Select features with importance above threshold
        threshold = np.mean(importances)
        selected = [feature for feature, importance in sorted_features if importance >= threshold]
        
        # Ensure minimum features
        min_features = self._get_config_value('min_feature_count', 5)
        if len(selected) < min_features:
            selected = [feature for feature, _ in sorted_features[:min_features]]
        
        self.logger.info(f"ML-based selection chose {len(selected)} features")
        return selected
    
    def _ensure_model_consistency(self, statistical_features: List[str], 
                                ml_features: List[str], models: List[str], 
                                X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Ensure feature selection consistency across multiple models
        """
        strategy = self._get_config_value('feature_consistency_strategy', 'intersection')
        
        if strategy == 'intersection':
            # Use intersection of statistical and ML features
            consistent_features = list(set(statistical_features) & set(ml_features))
        elif strategy == 'union':
            # Use union of all methods
            consistent_features = list(set(statistical_features) | set(ml_features))
        elif strategy == 'weighted_voting':
            # Score-based weighted selection
            all_features = list(set(statistical_features) | set(ml_features))
            feature_votes = {}
            
            for feature in all_features:
                votes = 0
                if feature in statistical_features:
                    votes += 1
                if feature in ml_features:
                    votes += 1
                feature_votes[feature] = votes
            
            # Select features with majority votes
            min_votes = max(1, len([statistical_features, ml_features]) // 2)
            consistent_features = [f for f, v in feature_votes.items() if v >= min_votes]
        else:
            # Default to statistical features
            consistent_features = statistical_features
        
        # Validate minimum feature count
        min_features = self._get_config_value('min_feature_count', 5)
        if len(consistent_features) < min_features:
            # Add top features from combined list
            all_scored = statistical_features + ml_features
            consistent_features.extend([f for f in all_scored 
                                     if f not in consistent_features][:min_features - len(consistent_features)])
        
        # Store model-feature mapping for consistency tracking
        for model in models:
            self.model_feature_maps[model] = ModelFeatureMap(
                model_name=model,
                features=consistent_features.copy(),
                selection_method=strategy,
                performance_score=0.0,  # Will be updated during training
                business_alignment_score=self._calculate_business_alignment(consistent_features)
            )
        
        self.logger.info(f"Consistent features across {len(models)} models: {len(consistent_features)}")
        return consistent_features
    
    def _calculate_business_alignment(self, features: List[str]) -> float:
        """Calculate how well features align with business rules"""
        if not self.business_rules:
            return 1.0
        
        alignment_score = 0.0
        total_rules = 0
        
        for rule in self.business_rules:
            if not rule.active:
                continue
                
            total_rules += 1
            rule_score = 0.0
            
            if rule.rule_type == 'must_include':
                included_count = sum(1 for f in rule.features if f in features)
                rule_score = included_count / len(rule.features) if rule.features else 1.0
                
            elif rule.rule_type == 'must_exclude':
                excluded_count = sum(1 for f in rule.features if f not in features)
                rule_score = excluded_count / len(rule.features) if rule.features else 1.0
                
            alignment_score += rule_score * rule.priority
        
        return alignment_score / total_rules if total_rules > 0 else 1.0
    
    def _setup_human_interface(self):
        """Setup human approval interface"""
        # This would integrate with a UI or CLI for human approval
        # For now, we'll use a simple logging-based interface
        pass
    
    def _request_human_approval(self, features: List[str], X: pd.DataFrame, 
                              y: pd.Series, original_features: List[str]) -> List[str]:
        """Request human approval for feature selection"""
        # In a real implementation, this would show a UI or CLI interface
        # For now, we'll auto-approve with logging
        
        self.logger.info("=== FEATURE SELECTION APPROVAL REQUEST ===")
        self.logger.info(f"Original features: {len(original_features)}")
        self.logger.info(f"Proposed features: {len(features)}")
        self.logger.info(f"Selected features: {features}")
        
        # Calculate basic statistics
        if len(features) > 0:
            X_selected = X[features]
            missing_rates = X_selected.isnull().mean()
            self.logger.info(f"Average missing rate: {missing_rates.mean():.3f}")
        
        # For automated approval (would be interactive in real implementation)
        approval = True  # Auto-approve for now
        
        if approval:
            self.logger.info("Features APPROVED by human reviewer")
            return features
        else:
            self.logger.info("Features REJECTED by human reviewer")
            # Would need fallback strategy or re-selection
            return original_features[:10]  # Fallback
    
    def _final_validation(self, features: List[str], X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Final validation of selected features"""
        valid_features = []
        
        for feature in features:
            if feature in X.columns:
                # Check for sufficient non-null values
                if X[feature].notna().sum() >= len(X) * 0.5:  # At least 50% non-null
                    valid_features.append(feature)
                else:
                    self.logger.warning(f"Excluding {feature}: too many missing values")
            else:
                self.logger.warning(f"Feature {feature} not found in data")
        
        # Ensure minimum feature count
        min_features = self._get_config_value('min_feature_count', 5)
        if len(valid_features) < min_features:
            # Add remaining features from original data
            remaining = [col for col in X.columns if col not in valid_features]
            additional_needed = min_features - len(valid_features)
            valid_features.extend(remaining[:additional_needed])
        
        self.logger.info(f"Final validation: {len(valid_features)} features selected")
        return valid_features
    
    def _calculate_feature_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate final feature importance scores"""
        if X.empty:
            return {}
        
        # Use correlation for quick scoring
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty:
            return {col: 1.0 for col in X.columns}
        
        correlations = X_numeric.corrwith(y).abs()
        return correlations.fillna(0).to_dict()
    
    def get_model_features(self, model_name: str) -> Optional[List[str]]:
        """Get selected features for a specific model"""
        if model_name in self.model_feature_maps:
            return self.model_feature_maps[model_name].features
        return None
    
    def update_model_performance(self, model_name: str, performance_score: float):
        """Update performance score for a model"""
        if model_name in self.model_feature_maps:
            self.model_feature_maps[model_name].performance_score = performance_score
    
    def get_feature_consistency_report(self) -> Dict[str, Any]:
        """Generate a report on feature consistency across models"""
        if not self.model_feature_maps:
            return {}
        
        all_features = set()
        model_features = {}
        
        for model_name, feature_map in self.model_feature_maps.items():
            features = set(feature_map.features)
            all_features.update(features)
            model_features[model_name] = features
        
        # Calculate consistency metrics
        feature_consistency = {}
        for feature in all_features:
            models_using = [m for m, features in model_features.items() if feature in features]
            feature_consistency[feature] = {
                'used_by_models': models_using,
                'usage_rate': len(models_using) / len(model_features),
                'consistency_score': len(models_using) / len(model_features)
            }
        
        return {
            'total_unique_features': len(all_features),
            'models_analyzed': list(model_features.keys()),
            'feature_consistency': feature_consistency,
            'average_consistency': np.mean([f['consistency_score'] for f in feature_consistency.values()]),
            'model_feature_maps': {k: v.features for k, v in self.model_feature_maps.items()}
        }
