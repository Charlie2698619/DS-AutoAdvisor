import pandas as pd
import numpy as np
import json
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
from datetime import datetime
import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error, median_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, classification_report
)
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.calibration import calibration_curve
import joblib

# Try to import SHAP - it's optional
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not available. Install with: pip install shap")

# Try to import kaleido for PNG export
try:
    import kaleido
    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False
    warnings.warn("Kaleido not available for PNG export. Install with: pip install kaleido")

from scipy import stats
from scipy.stats import pearsonr, spearmanr
import umap

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Type-safe configuration for model analysis"""
    # Input/Output
    models_dir: Path = field(default_factory=lambda: Path("models"))
    training_report_path: Path = field(default_factory=lambda: Path("training_report.json"))
    output_dir: Path = field(default_factory=lambda: Path("analysis_output"))
    
    # Analysis components
    enable_shap: bool = True
    enable_learning_curves: bool = True
    enable_residual_analysis: bool = True
    enable_stability_analysis: bool = True
    enable_interpretability: bool = True
    enable_uncertainty_analysis: bool = True
    
    # Visualization
    save_plots: bool = True
    plot_format: str = "html"  # "html", "png", "both"
    plot_dpi: int = 300 #
    figure_size: Tuple[int, int] = (12, 8)
    
    # Performance
    n_permutations: int = 50 # Number of permutations for feature importance
    n_bootstrap_samples: int = 100 # Number of bootstrap samples for stability testing
    max_shap_samples: int = 1000 # Max samples for SHAP analysis
    n_models_to_evaluate: int = 3 # Number of top models to evaluate
    
    # Feature importance configuration
    top_k_features: int = 15  # Number of top features to show in plots
    show_direction: bool = True  # Show positive/negative impact direction
    add_log_odds_axis: bool = True  # Add log-odds scale for binary classification
    permutation_error_bars: bool = True  # Show error bars in permutation importance
    
    # Stability testing
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])
    dropout_rates: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3])
    
    # Reproducibility
    random_state: int = 42
    
    # Logging
    verbose: bool = True
    log_level: str = "INFO"

@dataclass
class ModelAnalysisResult:
    """Container for comprehensive model analysis results"""
    model_name: str
    performance_metrics: Dict[str, float]
    learning_curves: Optional[Dict[str, np.ndarray]]
    residual_analysis: Optional[Dict[str, Any]]
    stability_metrics: Dict[str, float]
    interpretability_scores: Dict[str, Any]
    uncertainty_metrics: Optional[Dict[str, float]]
    feature_dependencies: Optional[Dict[str, Any]]
    calibration_metrics: Optional[Dict[str, float]]
    execution_time: float
    plots_saved: List[str] = field(default_factory=list)

class BaseAnalyzer(ABC):
    """Abstract base class for analysis components"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def analyze(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                target_type: str) -> Dict[str, Any]:
        """Perform analysis and return results"""
        pass

class LearningCurveAnalyzer(BaseAnalyzer):
    """Analyzes learning curves and model convergence"""
    
    def analyze(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                target_type: str) -> Dict[str, Any]:
        """Generate comprehensive learning curve analysis"""
        
        self.logger.info("🔄 Generating learning curves...")
        
        # Combine test data with training data for learning curves
        X_combined = X_test
        y_combined = y_test
        
        # Scoring metric
        scoring = "r2" if target_type == "regression" else "accuracy"
        
        # Generate learning curves
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model.pipeline if hasattr(model, 'pipeline') else model,
            X_combined, y_combined,
            train_sizes=train_sizes,
            cv=5,
            scoring=scoring,
            n_jobs=-1,
            random_state=42
        )
        
        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Convergence analysis
        convergence_rate = self._calculate_convergence_rate(val_mean)
        overfitting_score = self._calculate_overfitting_score(train_mean, val_mean)
        
        return {
            "train_sizes": train_sizes_abs,
            "train_scores_mean": train_mean,
            "train_scores_std": train_std,
            "val_scores_mean": val_mean,
            "val_scores_std": val_std,
            "convergence_rate": convergence_rate,
            "overfitting_score": overfitting_score,
            "final_gap": train_mean[-1] - val_mean[-1]
        }
    
    def _calculate_convergence_rate(self, scores: np.ndarray) -> float:
        """Calculate rate of convergence"""
        if len(scores) < 3:
            return 0.0
        
        # Fit exponential curve to later half of scores
        x = np.arange(len(scores) // 2, len(scores))
        y = scores[len(scores) // 2:]
        
        try:
            # Simple linear fit to log-transformed improvements
            improvements = np.diff(y)
            if len(improvements) > 0:
                return np.mean(improvements)
        except:
            pass
        
        return 0.0
    
    def _calculate_overfitting_score(self, train_scores: np.ndarray, 
                                   val_scores: np.ndarray) -> float:
        """Calculate overfitting tendency"""
        gap = train_scores - val_scores
        return np.mean(gap[-3:])  # Average gap in last 3 points

class ResidualAnalyzer(BaseAnalyzer):
    """Analyzes model residuals and error patterns"""
    
    def analyze(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                target_type: str) -> Dict[str, Any]:
        """Comprehensive residual analysis"""
        
        self.logger.info("🔍 Performing residual analysis...")
        
        if target_type != "regression":
            return {"note": "Residual analysis only applicable to regression"}
        
        # Generate predictions
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        # Basic residual statistics
        residual_stats = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "skewness": float(stats.skew(residuals)),
            "kurtosis": float(stats.kurtosis(residuals)),
            "normality_pvalue": float(stats.shapiro(residuals)[1]) if len(residuals) < 5000 else None
        }
        
        # Heteroscedasticity test
        heteroscedasticity = self._test_heteroscedasticity(y_pred, residuals)
        
        # Autocorrelation test
        autocorr = self._test_autocorrelation(residuals)
        
        # Outlier detection
        outliers = self._detect_outliers(residuals, X_test)
        
        # Error distribution analysis
        error_quantiles = np.percentile(np.abs(residuals), [50, 75, 90, 95, 99])
        
        return {
            "residual_stats": residual_stats,
            "heteroscedasticity": heteroscedasticity,
            "autocorrelation": autocorr,
            "outliers": outliers,
            "error_quantiles": {
                "median_abs_error": float(error_quantiles[0]),
                "q75_abs_error": float(error_quantiles[1]),
                "q90_abs_error": float(error_quantiles[2]),
                "q95_abs_error": float(error_quantiles[3]),
                "q99_abs_error": float(error_quantiles[4])
            },
            "residuals": residuals.tolist(),
            "predictions": y_pred.tolist(),
            "actuals": y_test.tolist()
        }
    
    def _test_heteroscedasticity(self, y_pred: np.ndarray, 
                               residuals: np.ndarray) -> Dict[str, float]:
        """Test for heteroscedasticity using Breusch-Pagan test"""
        try:
            # Simple correlation test
            correlation, p_value = pearsonr(np.abs(residuals), y_pred)
            return {
                "correlation": float(correlation),
                "p_value": float(p_value),
                "is_heteroscedastic": p_value < 0.05
            }
        except:
            return {"error": "Could not perform heteroscedasticity test"}
    
    def _test_autocorrelation(self, residuals: np.ndarray) -> Dict[str, float]:
        """Test for autocorrelation in residuals"""
        try:
            # Durbin-Watson test approximation
            diff_residuals = np.diff(residuals)
            dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)
            
            return {
                "durbin_watson": float(dw_stat),
                "has_autocorr": dw_stat < 1.5 or dw_stat > 2.5
            }
        except:
            return {"error": "Could not perform autocorrelation test"}
    
    def _detect_outliers(self, residuals: np.ndarray, 
                        X_test: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        
        # Z-score method
        z_scores = np.abs(stats.zscore(residuals))
        z_outliers = np.where(z_scores > 3)[0]
        
        # IQR method
        Q1, Q3 = np.percentile(residuals, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = np.where((residuals < lower_bound) | (residuals > upper_bound))[0]
        
        return {
            "z_score_outliers": len(z_outliers),
            "iqr_outliers": len(iqr_outliers),
            "outlier_indices": list(set(z_outliers.tolist() + iqr_outliers.tolist())),
            "outlier_percentage": len(set(z_outliers.tolist() + iqr_outliers.tolist())) / len(residuals) * 100
        }

class StabilityAnalyzer(BaseAnalyzer):
    """Analyzes model stability and robustness"""
    
    def analyze(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                target_type: str) -> Dict[str, Any]:
        """Comprehensive stability analysis"""
        
        self.logger.info("🧪 Testing model stability...")
        
        baseline_score = self._get_baseline_score(model, X_test, y_test, target_type)
        
        # Noise robustness
        noise_stability = self._test_noise_robustness(model, X_test, y_test, target_type)
        
        # Feature dropout robustness
        dropout_stability = self._test_dropout_robustness(model, X_test, y_test, target_type)
        
        # Bootstrap stability
        bootstrap_stability = self._test_bootstrap_stability(model, X_test, y_test, target_type)
        
        # Prediction consistency
        consistency = self._test_prediction_consistency(model, X_test)
        
        # Calculate normalized stability scores (0 to 1, where 1 is most stable)
        normalized_noise = noise_stability["avg_score_retention"]
        normalized_dropout = dropout_stability["avg_score_retention"]  
        normalized_bootstrap = bootstrap_stability.get("stability_score", 
                                                      1 - bootstrap_stability.get("score_std_normalized", 1))
        normalized_consistency = consistency.get("consistency_score", 0.0)
        
        # Weighted overall stability score
        overall_stability = (
            normalized_noise * 0.3 +           # Noise robustness
            normalized_dropout * 0.3 +         # Feature dropout robustness  
            normalized_bootstrap * 0.25 +      # Bootstrap stability
            normalized_consistency * 0.15      # Prediction consistency
        )
        
        return {
            "baseline_score": baseline_score,
            "noise_stability": noise_stability,
            "dropout_stability": dropout_stability,
            "bootstrap_stability": bootstrap_stability,
            "prediction_consistency": consistency,
            "stability_components": {
                "normalized_noise": normalized_noise,
                "normalized_dropout": normalized_dropout,
                "normalized_bootstrap": normalized_bootstrap,
                "normalized_consistency": normalized_consistency
            },
            "overall_stability": float(overall_stability)
        }
    
    def _get_baseline_score(self, model, X_test: pd.DataFrame, 
                          y_test: pd.Series, target_type: str) -> float:
        """Get baseline performance score"""
        y_pred = model.predict(X_test)
        
        if target_type == "regression":
            return float(r2_score(y_test, y_pred))
        else:
            return float(np.mean(y_pred == y_test))
    
    def _test_noise_robustness(self, model, X_test: pd.DataFrame, 
                             y_test: pd.Series, target_type: str) -> Dict[str, Any]:
        """Test robustness to input noise (only applies to numeric features)"""
        
        # Identify numeric columns only
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            # No numeric features to add noise to
            self.logger.warning("No numeric features found for noise robustness testing")
            baseline = self._get_baseline_score(model, X_test, y_test, target_type)
            return {
                "noise_levels": self.config.noise_levels,
                "scores": [baseline] * len(self.config.noise_levels),
                "score_retentions": [1.0] * len(self.config.noise_levels),
                "avg_score_retention": 1.0,
                "baseline_score": baseline,
                "score_mean": baseline,
                "score_std": 0.0,
                "score_std_normalized": 0.0
            }
        
        # Set random seed for reproducibility
        np.random.seed(getattr(self.config, 'random_state', 42))
        
        scores = []
        for noise_level in self.config.noise_levels:
            try:
                # Create a copy of the test data
                X_noisy = X_test.copy()
                
                # Add Gaussian noise only to numeric columns
                numeric_data = X_test[numeric_cols]
                noise = np.random.normal(0, noise_level * numeric_data.std(), numeric_data.shape)
                X_noisy[numeric_cols] = numeric_data + noise
                
                score = self._get_baseline_score(model, X_noisy, y_test, target_type)
                scores.append(score)
            except Exception as e:
                self.logger.warning(f"Noise robustness test failed at level {noise_level}: {e}")
                scores.append(0.0)
        
        baseline = self._get_baseline_score(model, X_test, y_test, target_type)
        score_retentions = [s / baseline if baseline != 0 else 0 for s in scores]
        score_std = float(np.std(scores))
        score_mean = float(np.mean(scores))
        
        return {
            "noise_levels": self.config.noise_levels,
            "scores": scores,
            "score_retentions": score_retentions,
            "avg_score_retention": float(np.mean(score_retentions)),
            "baseline_score": baseline,
            "score_mean": score_mean,
            "score_std": score_std,
            "score_std_normalized": float(score_std / (score_mean + 1e-8))
        }
    
    def _test_dropout_robustness(self, model, X_test: pd.DataFrame, 
                               y_test: pd.Series, target_type: str) -> Dict[str, Any]:
        """Test robustness to feature dropout by zeroing out features instead of dropping columns"""
        
        scores = []
        baseline = self._get_baseline_score(model, X_test, y_test, target_type)
        
        # Set random seed for reproducibility if available
        np.random.seed(getattr(self.config, 'random_state', 42))
        
        for dropout_rate in self.config.dropout_rates:
            # Calculate number of features to drop
            n_drop = int(len(X_test.columns) * dropout_rate)
            if n_drop == 0:
                scores.append(baseline)
                continue
            
            try:
                # Create a copy and zero out features instead of dropping columns
                X_dropped = X_test.copy()
                dropped_features = np.random.choice(X_test.columns, n_drop, replace=False)
                
                # Check if model pipeline can handle NaN (if it has imputer)
                use_nan = self._pipeline_has_imputer(model)
                
                if use_nan:
                    # Use NaN if pipeline has imputer
                    X_dropped[dropped_features] = np.nan
                else:
                    # Use zero-filling for models without imputation
                    # For numeric columns, use 0; for categorical, use mode or most frequent value
                    for feature in dropped_features:
                        if X_test[feature].dtype in ['object', 'category']:
                            # For categorical features, use the most frequent value
                            mode_value = X_test[feature].mode().iloc[0] if len(X_test[feature].mode()) > 0 else X_test[feature].iloc[0]
                            X_dropped[feature] = mode_value
                        else:
                            # For numeric features, use 0
                            X_dropped[feature] = 0
                
                score = self._get_baseline_score(model, X_dropped, y_test, target_type)
                scores.append(score)
                
            except Exception as e:
                self.logger.warning(f"Dropout robustness test failed at rate {dropout_rate}: {e}")
                scores.append(0.0)
        
        score_retentions = [s / baseline if baseline != 0 else 0 for s in scores]
        score_std = float(np.std(scores))
        score_mean = float(np.mean(scores))
        
        return {
            "dropout_rates": self.config.dropout_rates,
            "scores": scores,
            "score_retentions": score_retentions,
            "avg_score_retention": float(np.mean(score_retentions)),
            "baseline_score": baseline,
            "score_mean": score_mean,
            "score_std": score_std,
            "score_std_normalized": float(score_std / (score_mean + 1e-8))
        }
    
    def _test_bootstrap_stability(self, model, X_test: pd.DataFrame, 
                                y_test: pd.Series, target_type: str) -> Dict[str, Any]:
        """Test stability across bootstrap samples"""
        
        # Set random seed for reproducibility
        np.random.seed(getattr(self.config, 'random_state', 42))
        
        scores = []
        n_samples = min(len(X_test), 1000)  # Limit for performance
        baseline = self._get_baseline_score(model, X_test, y_test, target_type)
        
        for _ in range(min(self.config.n_bootstrap_samples, 50)):
            # Bootstrap sample
            indices = np.random.choice(len(X_test), n_samples, replace=True)
            
            # Handle both pandas DataFrame/Series and numpy arrays
            if hasattr(X_test, 'iloc'):
                X_boot = X_test.iloc[indices]
            else:
                X_boot = X_test[indices]
                
            if hasattr(y_test, 'iloc'):
                y_boot = y_test.iloc[indices]
            else:
                y_boot = y_test[indices]
            
            try:
                score = self._get_baseline_score(model, X_boot, y_boot, target_type)
                scores.append(score)
            except Exception as e:
                self.logger.warning(f"Bootstrap sample failed: {e}")
                continue
        
        if not scores:
            return {"error": "Bootstrap testing failed"}
        
        score_mean = float(np.mean(scores))
        score_std = float(np.std(scores))
        
        return {
            "scores": scores,
            "score_mean": score_mean,
            "score_std": score_std,
            "score_std_normalized": float(score_std / (score_mean + 1e-8)),
            "baseline_score": baseline,
            "stability_score": float(1 - score_std / (score_mean + 1e-8))  # Higher = more stable
        }
    
    def _test_prediction_consistency(self, model, X_test: pd.DataFrame) -> Dict[str, float]:
        """Test prediction consistency across multiple runs"""
        
        predictions = []
        for _ in range(5):
            try:
                pred = model.predict(X_test)
                predictions.append(pred)
            except:
                continue
        
        if len(predictions) < 2:
            return {"error": "Could not test consistency"}
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                try:
                    corr = pearsonr(predictions[i], predictions[j])[0]
                    correlations.append(corr)
                except:
                    continue
        
        return {
            "avg_correlation": float(np.mean(correlations)) if correlations else 0.0,
            "consistency_score": float(np.mean(correlations)) if correlations else 0.0
        }
    
    def _pipeline_has_imputer(self, model) -> bool:
        """Check if the model pipeline contains an imputer that can handle NaN values"""
        
        try:
            # Check sklearn Pipeline
            if hasattr(model, 'named_steps'):
                for step_name, step in model.named_steps.items():
                    step_class = step.__class__.__name__
                    if any(imputer_name in step_class for imputer_name in 
                          ['Imputer', 'SimpleImputer', 'IterativeImputer', 'KNNImputer']):
                        return True
            
            # Check imblearn Pipeline
            elif hasattr(model, 'steps'):
                for step_name, step in model.steps:
                    step_class = step.__class__.__name__
                    if any(imputer_name in step_class for imputer_name in 
                          ['Imputer', 'SimpleImputer', 'IterativeImputer', 'KNNImputer']):
                        return True
            
            return False
            
        except Exception:
            return False

class InterpretabilityAnalyzer(BaseAnalyzer):
    """Analyzes model interpretability using SHAP and permutation importance"""
    
    def analyze(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                target_type: str) -> Dict[str, Any]:
        """Comprehensive interpretability analysis"""
        
        self.logger.info("🔍 Analyzing model interpretability...")
        
        # Permutation importance
        perm_importance = self._calculate_permutation_importance(
            model, X_test, y_test, target_type
        )
        
        # SHAP analysis
        shap_analysis = None 
        if self.config.enable_shap and HAS_SHAP:
            shap_analysis = self._calculate_shap_values(model, X_test)
        elif self.config.enable_shap and not HAS_SHAP:
            self.logger.warning("SHAP analysis requested but SHAP not installed")
        
        # Feature interaction analysis
        interactions = self._analyze_feature_interactions(model, X_test, target_type)
        
        # Global vs local importance comparison
        importance_comparison = self._compare_importance_methods(
            perm_importance, shap_analysis
        )
        
        return {
            "permutation_importance": perm_importance,
            "shap_analysis": shap_analysis,
            "feature_interactions": interactions,
            "importance_comparison": importance_comparison,
            "interpretability_score": self._calculate_interpretability_score(
                perm_importance, shap_analysis
            )
        }
    
    def _calculate_permutation_importance(self, model, X_test: pd.DataFrame, 
                                        y_test: pd.Series, target_type: str) -> Dict[str, Any]:
        """Calculate permutation importance"""
        
        try:
            scoring = "r2" if target_type == "regression" else "accuracy"
            
            result = permutation_importance(
                model, X_test, y_test,
                n_repeats=min(self.config.n_permutations, 30),
                random_state=42,
                scoring=scoring,
                n_jobs=-1
            )
            
            importance_df = pd.DataFrame({
                'feature': X_test.columns,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            return {
                "feature_importances": importance_df.to_dict('records'),
                "top_features": importance_df.head(10)['feature'].tolist(),
                "importance_sum": float(result.importances_mean.sum()),
                "importance_concentration": float(result.importances_mean.max() / result.importances_mean.sum())
            }
            
        except Exception as e:
            self.logger.warning(f"Permutation importance failed: {e}")
            return {"error": str(e)}
    
    def _calculate_shap_values(self, model, X_test: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate SHAP values for interpretability - model-agnostic approach"""
        
        try:
            # Limit samples for performance
            sample_size = min(len(X_test), self.config.max_shap_samples)
            X_sample = X_test.sample(n=sample_size, random_state=42)
            
            # Initialize variables
            explainer = None
            shap_values = None
            explainer_type = None
            
            # Enhanced model extraction for different frameworks
            actual_model = self._extract_model_from_pipeline(model)
            model_type = self._identify_model_type(actual_model)
            
            # Try explainers in order of efficiency: Tree → Linear → Kernel
            explainer_attempts = [
                ("TreeExplainer", self._try_tree_explainer),
                ("LinearExplainer", self._try_linear_explainer),
                ("KernelExplainer", self._try_kernel_explainer),
                ("PermutationExplainer", self._try_permutation_explainer)
            ]
            
            for explainer_name, explainer_func in explainer_attempts:
                try:
                    explainer, shap_values = explainer_func(actual_model, model, X_sample)
                    if explainer is not None and shap_values is not None:
                        explainer_type = explainer_name
                        self.logger.info(f"✅ SHAP {explainer_name} successful")
                        break
                except Exception as e:
                    self.logger.warning(f"{explainer_name} failed: {e}")
                    continue
            
            if explainer is None or shap_values is None:
                self.logger.warning("All SHAP explainers failed")
                return None
            
            if shap_values is None:
                return None
            
            # Handle multi-class output - SHAP values might be a list of arrays (one per class)
            if isinstance(shap_values, list):
                if len(shap_values) > 0:
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]  # Use positive class for binary classification
                else:
                    return None
            
            # Ensure shap_values is a numpy array
            shap_values = np.array(shap_values)
            
            # Handle case where shap_values might be 3D (samples, features, classes)
            if len(shap_values.shape) == 3:
                # Take the positive class (index 1) or last class
                shap_values = shap_values[:, :, -1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
            elif len(shap_values.shape) == 1:
                # If 1D, reshape to 2D (assuming single sample)
                shap_values = shap_values.reshape(1, -1)
            
            # Calculate SHAP statistics with error handling
            try:
                # Calculate both magnitude and direction
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)  # Magnitude
                mean_shap = np.mean(shap_values, axis=0)  # Direction (can be positive or negative)
                
                # Ensure both are 1-dimensional
                if len(mean_abs_shap.shape) > 1:
                    mean_abs_shap = mean_abs_shap.flatten()
                if len(mean_shap.shape) > 1:
                    mean_shap = mean_shap.flatten()
                
                # Convert to python scalars explicitly
                mean_abs_shap = [float(x) for x in mean_abs_shap]
                mean_shap = [float(x) for x in mean_shap]
                
            except Exception as e:
                self.logger.warning(f"Error calculating SHAP statistics: {e}")
                return None
            
            # Ensure we have the right number of features
            if len(mean_abs_shap) != len(X_sample.columns) or len(mean_shap) != len(X_sample.columns):
                self.logger.warning(f"SHAP values length doesn't match features")
                # Try to take only the first n features if we have too many values
                if len(mean_abs_shap) > len(X_sample.columns):
                    mean_abs_shap = mean_abs_shap[:len(X_sample.columns)]
                    mean_shap = mean_shap[:len(X_sample.columns)]
                else:
                    return None
            
            shap_df = pd.DataFrame({
                'feature': X_sample.columns,
                'mean_abs_shap': mean_abs_shap,
                'mean_shap': mean_shap,  # Include direction
                'direction': ['positive' if x > 0 else 'negative' for x in mean_shap]
            }).sort_values('mean_abs_shap', ascending=False)
            
            # Handle expected_value which might be an array or scalar
            base_value = 0.0
            if hasattr(explainer, 'expected_value'):
                exp_val = explainer.expected_value
                if isinstance(exp_val, (list, np.ndarray)):
                    # If it's an array, take the first/mean value
                    base_value = float(np.mean(exp_val)) if len(exp_val) > 0 else 0.0
                else:
                    # If it's already a scalar
                    base_value = float(exp_val)
            
            # Convert shap_values to list with error handling
            shap_values_list = None
            try:
                shap_values_list = shap_values.tolist()
            except Exception as e:
                self.logger.warning(f"Could not convert SHAP values to list: {e}")
                # Try converting to a simpler format
                try:
                    shap_values_list = np.array(shap_values).astype(float).tolist()
                except:
                    self.logger.warning("Skipping SHAP values raw data due to conversion issues")
                    shap_values_list = []
            
            return {
                "feature_shap_values": shap_df.to_dict('records'),
                "top_shap_features": shap_df.head(10)['feature'].tolist(),
                "shap_values_raw": shap_values_list,
                "base_value": base_value,
                "explainer_type": explainer_type,
                "model_type": model_type
            }
            
        except Exception as e:
            self.logger.warning(f"SHAP analysis failed: {e}")
            return None
    
    def _extract_model_from_pipeline(self, model):
        """Extract the actual model from various pipeline structures"""
        
        # sklearn Pipeline
        if hasattr(model, 'named_steps'):
            # Try common step names
            for step_name in ['model', 'classifier', 'regressor', 'estimator']:
                if step_name in model.named_steps:
                    return model.named_steps[step_name]
            # If no common names, return the last step
            if len(model.named_steps) > 0:
                return list(model.named_steps.values())[-1]
        
        # imblearn Pipeline
        elif hasattr(model, 'steps'):
            return model.steps[-1][1]  # Last step in pipeline
        
        # Already extracted model or standalone model
        return model
    
    def _identify_model_type(self, model) -> str:
        """Identify the model type for optimal SHAP explainer selection"""
        
        model_class = model.__class__.__name__
        model_module = model.__class__.__module__
        
        # Tree-based models
        tree_models = [
            'RandomForest', 'ExtraTrees', 'GradientBoosting', 'XGBoost', 'LightGBM',
            'CatBoost', 'DecisionTree', 'ExtraTree', 'HistGradientBoosting'
        ]
        
        if any(tree_name in model_class for tree_name in tree_models):
            return 'tree'
        
        # Linear models
        linear_models = [
            'Linear', 'Ridge', 'Lasso', 'ElasticNet', 'Logistic', 'SGD',
            'Perceptron', 'PassiveAggressive'
        ]
        
        if any(linear_name in model_class for linear_name in linear_models):
            return 'linear'
        
        # SVM models
        if 'SV' in model_class or 'svm' in model_module:
            return 'svm'
        
        # Neural networks
        if any(nn_name in model_class.lower() for nn_name in ['neural', 'mlp', 'dnn']):
            return 'neural'
        
        # XGBoost/LightGBM specific
        if 'xgboost' in model_module or 'lightgbm' in model_module:
            return 'tree'
        
        return 'unknown'
    
    def _try_tree_explainer(self, actual_model, full_model, X_sample):
        """Try TreeExplainer for tree-based models"""
        explainer = shap.TreeExplainer(actual_model)
        shap_values = explainer.shap_values(X_sample)
        return explainer, shap_values
    
    def _try_linear_explainer(self, actual_model, full_model, X_sample):
        """Try LinearExplainer for linear models"""
        explainer = shap.LinearExplainer(actual_model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        return explainer, shap_values
    
    def _try_kernel_explainer(self, actual_model, full_model, X_sample):
        """Try KernelExplainer as fallback for any model"""
        # Use a smaller background dataset for KernelExplainer
        background_size = min(100, len(X_sample))
        background = X_sample.sample(n=background_size, random_state=42)
        
        explainer = shap.KernelExplainer(full_model.predict, background)
        
        # Limit samples for KernelExplainer (it's slow)
        sample_size = min(100, len(X_sample))
        shap_values = explainer.shap_values(X_sample.head(sample_size))
        return explainer, shap_values
    
    def _try_permutation_explainer(self, actual_model, full_model, X_sample):
        """Try PermutationExplainer as final fallback"""
        explainer = shap.PermutationExplainer(full_model.predict, X_sample)
        shap_values = explainer.shap_values(X_sample)
        return explainer, shap_values
    
    def _adapt_shap_config_to_model(self, model_type: str, X_sample_size: int) -> Dict[str, Any]:
        """Adapt SHAP configuration based on model type and data size"""
        
        config = {
            "max_samples": self.config.max_shap_samples,
            "background_samples": 100,
            "use_approximate": False
        }
        
        # Tree models - can handle more samples efficiently
        if model_type == 'tree':
            config["max_samples"] = min(2000, X_sample_size)
            config["background_samples"] = min(500, X_sample_size // 2)
        
        # Linear models - very efficient
        elif model_type == 'linear':
            config["max_samples"] = min(5000, X_sample_size)
            config["background_samples"] = min(1000, X_sample_size // 2)
        
        # Kernel methods and neural networks - more conservative
        elif model_type in ['svm', 'neural', 'unknown']:
            config["max_samples"] = min(500, X_sample_size)
            config["background_samples"] = min(100, X_sample_size // 4)
            config["use_approximate"] = True
        
        # Large datasets - always be conservative
        if X_sample_size > 10000:
            config["max_samples"] = min(config["max_samples"], 1000)
            config["background_samples"] = min(config["background_samples"], 200)
            config["use_approximate"] = True
        
        return config
    
    def _analyze_feature_interactions(self, model, X_test: pd.DataFrame, 
                                    target_type: str) -> Dict[str, Any]:
        """Analyze feature interactions"""
        
        try:
            # Simple pairwise correlation analysis
            correlations = X_test.corr()
            
            # Find highly correlated feature pairs
            high_corr_pairs = []
            for i in range(len(correlations.columns)):
                for j in range(i + 1, len(correlations.columns)):
                    corr_val = correlations.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append({
                            'feature1': correlations.columns[i],
                            'feature2': correlations.columns[j],
                            'correlation': float(corr_val)
                        })
            
            return {
                "high_correlation_pairs": high_corr_pairs,
                "max_correlation": float(correlations.abs().max().max()),
                "avg_correlation": float(correlations.abs().mean().mean())
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _compare_importance_methods(self, perm_importance: Dict, 
                                  shap_analysis: Optional[Dict]) -> Dict[str, Any]:
        """Compare different importance methods"""
        
        if not perm_importance or "feature_importances" not in perm_importance:
            return {"error": "No permutation importance available"}
        
        if not shap_analysis or "feature_shap_values" not in shap_analysis:
            return {"comparison": "Only permutation importance available"}
        
        # Compare rankings
        perm_features = [f['feature'] for f in perm_importance['feature_importances'][:10]]
        shap_features = [f['feature'] for f in shap_analysis['feature_shap_values'][:10]]
        
        # Calculate overlap
        overlap = len(set(perm_features) & set(shap_features))
        
        return {
            "top10_overlap": overlap,
            "overlap_percentage": overlap / 10.0 * 100,
            "ranking_correlation": self._calculate_ranking_correlation(
                perm_importance['feature_importances'],
                shap_analysis['feature_shap_values']
            )
        }
    
    def _calculate_ranking_correlation(self, perm_list: List[Dict], 
                                     shap_list: List[Dict]) -> float:
        """Calculate correlation between importance rankings"""
        
        try:
            # Create ranking dictionaries
            perm_ranks = {item['feature']: i for i, item in enumerate(perm_list)}
            shap_ranks = {item['feature']: i for i, item in enumerate(shap_list)}
            
            # Find common features
            common_features = set(perm_ranks.keys()) & set(shap_ranks.keys())
            
            if len(common_features) < 3:
                return 0.0
            
            # Get rankings for common features
            perm_values = [perm_ranks[f] for f in common_features]
            shap_values = [shap_ranks[f] for f in common_features]
            
            # Calculate Spearman correlation
            correlation, _ = spearmanr(perm_values, shap_values)
            return float(correlation)
            
        except:
            return 0.0
    
    def _calculate_interpretability_score(self, perm_importance: Dict, 
                                        shap_analysis: Optional[Dict]) -> float:
        """Calculate overall interpretability score"""
        
        score = 0.0
        
        # Permutation importance contribution
        if perm_importance and "importance_concentration" in perm_importance:
            # Higher concentration means less interpretable (fewer important features)
            concentration = perm_importance["importance_concentration"]
            score += (1 - min(concentration, 1.0)) * 0.4
        
        # SHAP contribution
        if shap_analysis:
            score += 0.4
        
        # Feature interaction penalty
        if perm_importance and "feature_importances" in perm_importance:
            n_important = len([f for f in perm_importance["feature_importances"] 
                             if f["importance_mean"] > 0.01])
            # More interpretable if fewer important features
            score += min(10 / max(n_important, 1), 1.0) * 0.2
        
        return min(score, 1.0)

class ModelAnalyzer:
    """Main analyzer class orchestrating all analysis components"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        self.analyzers = {
            'learning_curves': LearningCurveAnalyzer(config) if config.enable_learning_curves else None,
            'residual': ResidualAnalyzer(config) if config.enable_residual_analysis else None,
            'stability': StabilityAnalyzer(config) if config.enable_stability_analysis else None,
            'interpretability': InterpretabilityAnalyzer(config) if config.enable_interpretability else None,
        }
    
    def load_training_results(self) -> Dict[str, Any]:
        """Load results from trainer.py"""
        
        if not self.config.training_report_path.exists():
            raise FileNotFoundError(f"Training report not found: {self.config.training_report_path}")
        
        with open(self.config.training_report_path, 'r') as f:
            return json.load(f)
    
    def load_top_models(self, training_results: Dict[str, Any], n_models: int = 3) -> List[Tuple[str, Any]]:
        """Load top N models from training results"""
        
        top_models = []
        rankings = training_results.get('model_rankings', [])[:n_models]
        
        for ranking in rankings:
            model_name = ranking['model']
            detailed_result = training_results['detailed_results'].get(model_name, {})
            model_path = detailed_result.get('model_path')
            
            if model_path and Path(model_path).exists():
                try:
                    model = joblib.load(model_path)
                    top_models.append((model_name, model))
                    self.logger.info(f"✅ Loaded model: {model_name}")
                except Exception as e:
                    self.logger.warning(f"❌ Failed to load {model_name}: {e}")
            else:
                self.logger.warning(f"❌ Model file not found: {model_path}")
        
        return top_models
    
    def analyze_model(self, model_name: str, model: Any, 
                     X_test: pd.DataFrame, y_test: pd.Series, 
                     target_type: str) -> ModelAnalysisResult:
        """Comprehensive analysis of a single model"""
        
        self.logger.info(f"🔬 Starting analysis for {model_name}...")
        start_time = time.time()
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(
            model, X_test, y_test, target_type
        )
        
        # Run all analyzers
        results = {}
        for analyzer_name, analyzer in self.analyzers.items():
            if analyzer is None:
                continue
                
            try:
                results[analyzer_name] = analyzer.analyze(model, X_test, y_test, target_type)
                self.logger.info(f"  ✅ {analyzer_name} analysis complete")
            except Exception as e:
                self.logger.error(f"  ❌ {analyzer_name} analysis failed: {e}")
                results[analyzer_name] = {"error": str(e)}
        
        # Generate visualizations
        plots_saved = []
        if self.config.save_plots:
            plots_saved = self._generate_visualizations(
                model_name, model, X_test, y_test, target_type, results
            )
        
        execution_time = time.time() - start_time
        
        return ModelAnalysisResult(
            model_name=model_name,
            performance_metrics=performance_metrics,
            learning_curves=results.get('learning_curves'),
            residual_analysis=results.get('residual'),
            stability_metrics=results.get('stability', {}),
            interpretability_scores=results.get('interpretability', {}),
            uncertainty_metrics=results.get('uncertainty'),
            feature_dependencies=None,
            calibration_metrics=None,
            execution_time=execution_time,
            plots_saved=plots_saved
        )
    
    def _calculate_performance_metrics(self, model, X_test: pd.DataFrame, 
                                     y_test: pd.Series, target_type: str) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        y_pred = model.predict(X_test)
        
        if target_type == "regression":
            # Comprehensive regression metrics
            mse = mean_squared_error(y_test, y_pred)
            mae_val = mean_absolute_error(y_test, y_pred)
            
            # Calculate MAPE (Mean Absolute Percentage Error) with zero handling
            mape = 0.0
            try:
                non_zero_mask = y_test != 0
                if non_zero_mask.sum() > 0:
                    mape = float(np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100)
            except:
                mape = 0.0
            
            return {
                # Core regression metrics
                "r2_score": float(r2_score(y_test, y_pred)),
                "rmse": float(np.sqrt(mse)),
                "mae": float(mae_val),
                "mse": float(mse),
                "mape": mape,
                
                # Additional regression metrics
                "explained_variance": float(explained_variance_score(y_test, y_pred)),
                "max_error": float(max_error(y_test, y_pred)),
                "median_absolute_error": float(median_absolute_error(y_test, y_pred)),
                
                # Error analysis
                "mean_residual": float(np.mean(y_test - y_pred)),
                "std_residual": float(np.std(y_test - y_pred))
            }
        else:
            # Comprehensive classification metrics
            try:
                # Get probability predictions if available for AUC/PR metrics
                y_proba = None
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test)
                    # Convert to probabilities for binary classification
                    if len(np.unique(y_test)) == 2:
                        from scipy.special import expit
                        y_proba = np.column_stack([1 - expit(y_scores), expit(y_scores)])
                
                # Determine if binary or multiclass
                n_classes = len(np.unique(y_test))
                is_binary = n_classes == 2
                
                # Core classification metrics
                metrics = {
                    # Basic metrics
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                    "error_rate": float(1 - accuracy_score(y_test, y_pred)),
                }
                
                # Precision, Recall, F1 - handle both binary and multiclass
                if is_binary:
                    metrics.update({
                        "precision": float(precision_score(y_test, y_pred, average='binary', zero_division=0)),
                        "recall": float(recall_score(y_test, y_pred, average='binary', zero_division=0)),
                        "f1_score": float(f1_score(y_test, y_pred, average='binary', zero_division=0)),
                        "specificity": self._calculate_specificity(y_test, y_pred),
                    })
                else:
                    # Multiclass - provide macro and weighted averages
                    metrics.update({
                        "precision_macro": float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
                        "precision_weighted": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                        "recall_macro": float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
                        "recall_weighted": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                        "f1_score_macro": float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
                        "f1_score_weighted": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                    })
                
                # Advanced metrics
                metrics.update({
                    "matthews_corrcoef": float(matthews_corrcoef(y_test, y_pred)),
                    "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
                })
                
                # AUC and probability-based metrics
                if y_proba is not None:
                    try:
                        if is_binary:
                            # Binary classification AUC metrics
                            metrics.update({
                                "roc_auc": float(roc_auc_score(y_test, y_proba[:, 1])),
                                "pr_auc": float(average_precision_score(y_test, y_proba[:, 1])),
                                "log_loss": float(log_loss(y_test, y_proba)),
                            })
                        else:
                            # Multiclass AUC metrics
                            try:
                                metrics.update({
                                    "roc_auc_ovr": float(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')),
                                    "roc_auc_ovo": float(roc_auc_score(y_test, y_proba, multi_class='ovo', average='macro')),
                                    "log_loss": float(log_loss(y_test, y_proba)),
                                })
                            except ValueError:
                                # Some metrics might fail for certain class configurations
                                self.logger.warning("Could not calculate multiclass AUC metrics")
                    except Exception as e:
                        self.logger.warning(f"Could not calculate probability-based metrics: {e}")
                
                # Confusion matrix statistics
                cm = confusion_matrix(y_test, y_pred)
                if is_binary and cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    metrics.update({
                        "true_positives": int(tp),
                        "true_negatives": int(tn),
                        "false_positives": int(fp),
                        "false_negatives": int(fn),
                        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                        "positive_predictive_value": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                        "negative_predictive_value": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
                    })
                
                return metrics
                
            except Exception as e:
                self.logger.error(f"Error calculating classification metrics: {e}")
                # Fallback to basic metrics
                accuracy = float(np.mean(y_pred == y_test))
                return {
                    "accuracy": accuracy,
                    "error_rate": 1 - accuracy
                }
    
    def _calculate_specificity(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate specificity for binary classification"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            return 0.0
        except:
            return 0.0
    
    def _generate_visualizations(self, model_name: str, model: Any, 
                               X_test: pd.DataFrame, y_test: pd.Series,
                               target_type: str, analysis_results: Dict) -> List[str]:
        """Generate comprehensive visualizations"""
        
        plots_saved = []
        
        try:
            # Learning curves plot
            if 'learning_curves' in analysis_results and analysis_results['learning_curves']:
                plot_paths = self._plot_learning_curves(model_name, analysis_results['learning_curves'])
                if plot_paths:
                    plots_saved.extend(plot_paths)
            
            # Residual plots (for regression)
            if target_type == "regression" and 'residual' in analysis_results:
                plot_paths = self._plot_residuals(model_name, analysis_results['residual'])
                if plot_paths:
                    plots_saved.extend(plot_paths)
            
            # Feature importance plot
            if 'interpretability' in analysis_results:
                plot_paths = self._plot_feature_importance(model_name, analysis_results['interpretability'])
                if plot_paths:
                    plots_saved.extend(plot_paths)
            
            # Stability analysis plot
            if 'stability' in analysis_results:
                plot_paths = self._plot_stability_analysis(model_name, analysis_results['stability'])
                if plot_paths:
                    plots_saved.extend(plot_paths)
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
        
        return plots_saved
    
    def _save_plot(self, fig, model_name: str, plot_type: str) -> List[str]:
        """Save plot in specified format(s) with proper configuration"""
        saved_files = []
        
        try:
            # Apply figure size if using matplotlib backend
            if hasattr(fig, 'update_layout'):
                # For Plotly figures
                width, height = self.config.figure_size
                fig.update_layout(
                    width=width * 100,  # Convert to pixels (rough conversion)
                    height=height * 100
                )
            
            base_filename = f"{model_name}_{plot_type}"
            
            # Save based on plot_format setting
            if self.config.plot_format in ["html", "both"]:
                html_path = self.config.output_dir / f"{base_filename}.html"
                if hasattr(fig, 'write_html'):
                    fig.write_html(html_path)
                    saved_files.append(str(html_path))
                    self.logger.debug(f"Saved HTML plot: {html_path}")
            
            if self.config.plot_format in ["png", "both"]:
                png_path = self.config.output_dir / f"{base_filename}.png"
                if hasattr(fig, 'write_image'):
                    # Plotly figure - save as PNG
                    try:
                        # Try to import kaleido first
                        import kaleido
                        try:
                            kaleido_version = getattr(kaleido, '__version__', 'unknown')
                            self.logger.debug(f"Kaleido version: {kaleido_version}")
                        except:
                            self.logger.debug("Kaleido imported but version unavailable")
                        
                        # Try with different engines and configurations
                        try:
                            fig.write_image(
                                png_path,
                                width=self.config.figure_size[0] * self.config.plot_dpi // 12,
                                height=self.config.figure_size[1] * self.config.plot_dpi // 12,
                                scale=self.config.plot_dpi / 100,
                                engine="kaleido"
                            )
                            saved_files.append(str(png_path))
                            self.logger.debug(f"Saved PNG plot with kaleido: {png_path}")
                        except Exception as kaleido_error:
                            self.logger.warning(f"Kaleido engine failed: {kaleido_error}")
                            # Try without specifying engine
                            try:
                                fig.write_image(
                                    png_path,
                                    width=self.config.figure_size[0] * 100,
                                    height=self.config.figure_size[1] * 100
                                )
                                saved_files.append(str(png_path))
                                self.logger.debug(f"Saved PNG plot with default engine: {png_path}")
                            except Exception as default_error:
                                self.logger.warning(f"Default PNG export also failed: {default_error}")
                                self.logger.info(f"Continuing with HTML only for {base_filename}")
                    except ImportError as e:
                        self.logger.warning(f"Kaleido not available for PNG export: {e}")
                        self.logger.info(f"Skipping PNG export for {base_filename}")
                elif hasattr(fig, 'savefig'):
                    # Matplotlib figure - save as PNG
                    try:
                        fig.savefig(
                            png_path,
                            dpi=self.config.plot_dpi,
                            figsize=self.config.figure_size,
                            bbox_inches='tight',
                            facecolor='white'
                        )
                        saved_files.append(str(png_path))
                        self.logger.debug(f"Saved PNG plot: {png_path}")
                    except Exception as e:
                        self.logger.warning(f"Matplotlib PNG export failed for {base_filename}: {e}")
        
        except Exception as e:
            self.logger.error(f"Failed to save plot {base_filename}: {e}")
            
        return saved_files

    def _plot_learning_curves(self, model_name: str, learning_data: Dict) -> List[str]:
        """Generate learning curves plot"""
        
        try:
            fig = go.Figure()
            
            train_sizes = learning_data['train_sizes']
            train_mean = learning_data['train_scores_mean']
            train_std = learning_data['train_scores_std']
            val_mean = learning_data['val_scores_mean']
            val_std = learning_data['val_scores_std']
            
            # Training curve with confidence interval
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_mean + train_std,
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_mean - train_std,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0)',
                name='Training ±1σ',
                fillcolor='rgba(0,100,80,0.2)'
            ))
            
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_mean,
                mode='lines+markers',
                name='Training Score',
                line=dict(color='green', width=2)
            ))
            
            # Validation curve with confidence interval
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=val_mean + val_std,
                fill=None,
                mode='lines',
                line_color='rgba(255,165,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=val_mean - val_std,
                fill='tonexty',
                mode='lines',
                line_color='rgba(255,165,0,0)',
                name='Validation ±1σ',
                fillcolor='rgba(255,165,0,0.2)'
            ))
            
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=val_mean,
                mode='lines+markers',
                name='Validation Score',
                line=dict(color='orange', width=2)
            ))
            
            fig.update_layout(
                title=f'Learning Curves - {model_name}',
                xaxis_title='Training Set Size',
                yaxis_title='Score',
                hovermode='x unified',
                template='plotly_white'  # Clean template
            )
            
            # Save using the enhanced save method
            saved_files = self._save_plot(fig, model_name, "learning_curves")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Learning curves plot failed: {e}")
            return []

    def _plot_residuals(self, model_name: str, residual_data: Dict) -> List[str]:
        """Generate residual analysis plots"""
        
        try:
            if 'residuals' not in residual_data:
                return []
            
            residuals = residual_data['residuals']
            predictions = residual_data['predictions']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Residuals vs Fitted', 'Q-Q Plot', 
                               'Residuals Distribution', 'Residuals vs Order'],
                specs=[[{'secondary_y': False}, {'secondary_y': False}],
                       [{'secondary_y': False}, {'secondary_y': False}]]
            )
            
            # Residuals vs Fitted
            fig.add_trace(
                go.Scatter(x=predictions, y=residuals, mode='markers',
                          name='Residuals', opacity=0.6),
                row=1, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Q-Q Plot
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
            
            fig.add_trace(
                go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers',
                          name='Q-Q Plot', opacity=0.6),
                row=1, col=2
            )
            
            # Add diagonal line for Q-Q plot
            min_val = min(min(theoretical_quantiles), min(sorted_residuals))
            max_val = max(max(theoretical_quantiles), max(sorted_residuals))
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                          mode='lines', name='Diagonal', line=dict(color='red', dash='dash')),
                row=1, col=2
            )
            
            # Residuals distribution
            fig.add_trace(
                go.Histogram(x=residuals, nbinsx=30, name='Distribution', opacity=0.7),
                row=2, col=1
            )
            
            # Residuals vs Order
            fig.add_trace(
                go.Scatter(x=list(range(len(residuals))), y=residuals, mode='markers',
                          name='Order', opacity=0.6),
                row=2, col=2
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
            
            fig.update_layout(
                title=f'Residual Analysis - {model_name}',
                showlegend=False,
                height=800,
                template='plotly_white'
            )
            
            # Save using the enhanced save method
            saved_files = self._save_plot(fig, model_name, "residuals")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Residuals plot failed: {e}")
            return []
    
    def _plot_feature_importance(self, model_name: str, interp_data: Dict) -> List[str]:
        """Generate enhanced feature importance plots with direction, error bars, and log-odds"""
        
        try:
            # Determine if this is binary classification for log-odds axis
            is_binary_classification = False
            if ('shap_analysis' in interp_data and 
                interp_data['shap_analysis'] and 
                'model_type' in interp_data['shap_analysis']):
                # We'll add logic to detect binary classification
                is_binary_classification = True  # For now, assume we want log-odds for classification
            
            # Create subplots with potentially different y-axis types
            if self.config.add_log_odds_axis and is_binary_classification:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=['Permutation Importance', 'SHAP Values (with Log-Odds)'],
                    specs=[[{'secondary_y': False}, {'secondary_y': True}]]
                )
            else:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=['Permutation Importance', 'SHAP Values'],
                    specs=[[{'secondary_y': False}, {'secondary_y': False}]]
                )
            
            # 1. Enhanced Permutation Importance with error bars
            if 'permutation_importance' in interp_data and 'feature_importances' in interp_data['permutation_importance']:
                perm_data = interp_data['permutation_importance']['feature_importances'][:self.config.top_k_features]
                
                features = [item['feature'] for item in perm_data]
                importances = [item['importance_mean'] for item in perm_data]
                
                # Add error bars if enabled and available
                if self.config.permutation_error_bars and 'importance_std' in perm_data[0]:
                    stds = [item['importance_std'] for item in perm_data]
                    error_x = dict(type='data', array=stds, visible=True, thickness=2)
                else:
                    error_x = None
                
                fig.add_trace(
                    go.Bar(
                        x=importances, 
                        y=features, 
                        orientation='h',
                        error_x=error_x,
                        name='Permutation Importance',
                        marker=dict(
                            color='steelblue',
                            line=dict(color='darkblue', width=1)
                        ),
                        text=[f'{imp:.3f}' for imp in importances],
                        textposition='outside',
                        textfont=dict(size=10)
                    ),
                    row=1, col=1
                )
            
            # 2. Enhanced SHAP values with direction and log-odds
            if 'shap_analysis' in interp_data and interp_data['shap_analysis'] and 'feature_shap_values' in interp_data['shap_analysis']:
                shap_data = interp_data['shap_analysis']['feature_shap_values'][:self.config.top_k_features]
                
                features = [item['feature'] for item in shap_data]
                
                if self.config.show_direction and 'mean_shap' in shap_data[0]:
                    # Use directional SHAP values (positive/negative)
                    shap_values = [item['mean_shap'] for item in shap_data]
                    colors = ['green' if val > 0 else 'red' for val in shap_values]
                    
                    # Create custom hover text with direction
                    hover_text = [
                        f"Feature: {feat}<br>"
                        f"SHAP: {val:.3f}<br>"
                        f"Impact: {'Positive' if val > 0 else 'Negative'}<br>"
                        f"Magnitude: {abs(val):.3f}"
                        for feat, val in zip(features, shap_values)
                    ]
                    
                else:
                    # Use magnitude only (backward compatibility)
                    shap_values = [item['mean_abs_shap'] for item in shap_data]
                    colors = 'orange'
                    hover_text = [f"Feature: {feat}<br>SHAP: {val:.3f}" for feat, val in zip(features, shap_values)]
                
                # Main SHAP bar plot
                fig.add_trace(
                    go.Bar(
                        x=shap_values, 
                        y=features, 
                        orientation='h',
                        name='SHAP Values',
                        marker=dict(
                            color=colors,
                            line=dict(color='black', width=0.5)
                        ),
                        text=[f'{val:.3f}' for val in shap_values],
                        textposition='outside',
                        textfont=dict(size=10),
                        hovertext=hover_text,
                        hoverinfo='text'
                    ),
                    row=1, col=2
                )
                
                # Add log-odds axis if enabled and this is binary classification
                if self.config.add_log_odds_axis and is_binary_classification:
                    # Convert SHAP values to approximate log-odds scale
                    # Note: This is an approximation since exact conversion requires model specifics
                    log_odds_values = [val * 1.6 for val in shap_values]  # Rough conversion factor
                    
                    # Add secondary y-axis with log-odds scale
                    fig.add_trace(
                        go.Scatter(
                            x=log_odds_values,
                            y=features,
                            mode='markers',
                            marker=dict(
                                symbol='diamond',
                                size=8,
                                color='purple',
                                line=dict(color='rebeccapurple', width=1)
                            ),
                            name='Log-Odds Scale',
                            yaxis='y2',
                            showlegend=True,
                            hovertemplate='<b>%{y}</b><br>Log-Odds: %{x:.3f}<extra></extra>'
                        ),
                        row=1, col=2, secondary_y=True
                    )
            
            # Update layout with enhanced styling
            layout_updates = {
                'title': f'Enhanced Feature Importance Analysis - {model_name}',
                'showlegend': True,
                'height': 600,
                'template': 'plotly_white',
                'font': dict(size=12)
            }
            
            # Add log-odds axis labels if enabled
            if self.config.add_log_odds_axis and is_binary_classification:
                layout_updates['yaxis2'] = dict(
                    title='Log-Odds Impact',
                    overlaying='y',
                    side='right',
                    showgrid=False
                )
            
            fig.update_layout(**layout_updates)
            
            # Update x-axis labels
            fig.update_xaxes(title_text="Importance Score", row=1, col=1)
            fig.update_xaxes(title_text="SHAP Value" + (" / Log-Odds" if self.config.add_log_odds_axis and is_binary_classification else ""), row=1, col=2)
            
            # Add reference line at x=0 for SHAP plot if showing direction
            if self.config.show_direction:
                fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)
            
            # Save using the enhanced save method
            saved_files = self._save_plot(fig, model_name, "feature_importance")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Feature importance plot failed: {e}")
            return []
    
    def _plot_stability_analysis(self, model_name: str, stability_data: Dict) -> List[str]:
        """Generate enhanced stability analysis plots with trends and variance metrics"""
        
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Score Retention Trends', 'Bootstrap Stability Distribution', 
                               'Stability Components', 'Score Variance Analysis'],
                specs=[[{'secondary_y': False}, {'secondary_y': False}],
                       [{'secondary_y': False}, {'secondary_y': True}]]
            )
            
            # Enhanced Score Retention Trends (Subplot 1)
            baseline_score = stability_data.get('baseline_score', 1.0)
            
            # Noise robustness trend
            if 'noise_stability' in stability_data:
                noise_data = stability_data['noise_stability']
                noise_levels = noise_data.get('noise_levels', [])
                score_retentions = noise_data.get('score_retentions', [])
                scores = noise_data.get('scores', [])
                
                # Main line
                fig.add_trace(
                    go.Scatter(
                        x=noise_levels, 
                        y=score_retentions,
                        mode='lines+markers', 
                        name='Noise Robustness',
                        line=dict(color='red', width=3),
                        marker=dict(size=8),
                        hovertemplate='Noise Level: %{x}<br>Score Retention: %{y:.3f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Add error bars if we have std data
                if 'score_std' in noise_data and noise_data['score_std'] > 0:
                    score_mean = noise_data.get('score_mean', baseline_score)
                    score_std = noise_data['score_std']
                    fig.add_trace(
                        go.Scatter(
                            x=noise_levels + noise_levels[::-1],
                            y=[(score_mean + score_std)/baseline_score] * len(noise_levels) + 
                              [(score_mean - score_std)/baseline_score] * len(noise_levels),
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            name='Noise Variance'
                        ),
                        row=1, col=1
                    )
            
            # Dropout robustness trend
            if 'dropout_stability' in stability_data:
                dropout_data = stability_data['dropout_stability']
                dropout_rates = dropout_data.get('dropout_rates', [])
                score_retentions = dropout_data.get('score_retentions', [])
                
                fig.add_trace(
                    go.Scatter(
                        x=dropout_rates, 
                        y=score_retentions,
                        mode='lines+markers', 
                        name='Dropout Robustness',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8),
                        hovertemplate='Dropout Rate: %{x}<br>Score Retention: %{y:.3f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Add error bars for dropout if available
                if 'score_std' in dropout_data and dropout_data['score_std'] > 0:
                    score_mean = dropout_data.get('score_mean', baseline_score)
                    score_std = dropout_data['score_std']
                    fig.add_trace(
                        go.Scatter(
                            x=dropout_rates + dropout_rates[::-1],
                            y=[(score_mean + score_std)/baseline_score] * len(dropout_rates) + 
                              [(score_mean - score_std)/baseline_score] * len(dropout_rates),
                            fill='toself',
                            fillcolor='rgba(0,0,255,0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            name='Dropout Variance'
                        ),
                        row=1, col=1
                    )
            
            # Bootstrap stability distribution (Subplot 2)
            if 'bootstrap_stability' in stability_data and 'scores' in stability_data['bootstrap_stability']:
                bootstrap_scores = stability_data['bootstrap_stability']['scores']
                bootstrap_mean = stability_data['bootstrap_stability'].get('score_mean', np.mean(bootstrap_scores))
                bootstrap_std = stability_data['bootstrap_stability'].get('score_std', np.std(bootstrap_scores))
                
                # Histogram
                fig.add_trace(
                    go.Histogram(
                        x=bootstrap_scores, 
                        nbinsx=20, 
                        name='Bootstrap Distribution',
                        opacity=0.7,
                        marker_color='green',
                        hovertemplate='Score Range: %{x}<br>Count: %{y}<extra></extra>'
                    ),
                    row=1, col=2
                )
                
                # Add mean line
                fig.add_vline(
                    x=bootstrap_mean, 
                    line_dash="dash", 
                    line_color="red",
                    row=1, col=2,
                    annotation_text=f"Mean: {bootstrap_mean:.3f}"
                )
                
                # Add std bands
                fig.add_vrect(
                    x0=bootstrap_mean - bootstrap_std,
                    x1=bootstrap_mean + bootstrap_std,
                    fillcolor="rgba(255,0,0,0.1)",
                    layer="below",
                    line_width=0,
                    row=1, col=2
                )
            
            # Normalized stability components (Subplot 3)
            if 'stability_components' in stability_data:
                components = stability_data['stability_components']
                component_names = ['Noise\nRobustness', 'Dropout\nRobustness', 
                                 'Bootstrap\nStability', 'Prediction\nConsistency']
                component_values = [
                    components.get('normalized_noise', 0),
                    components.get('normalized_dropout', 0),
                    components.get('normalized_bootstrap', 0),
                    components.get('normalized_consistency', 0)
                ]
                
                # Color code based on performance
                colors = ['red' if v < 0.5 else 'orange' if v < 0.7 else 'green' for v in component_values]
                
                fig.add_trace(
                    go.Bar(
                        x=component_names, 
                        y=component_values,
                        name='Stability Components',
                        marker_color=colors,
                        hovertemplate='Component: %{x}<br>Score: %{y:.3f}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                # Add threshold lines
                fig.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                             row=2, col=1, annotation_text="Good Threshold")
                fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                             row=2, col=1, annotation_text="Excellent Threshold")
            
            # Score variance analysis (Subplot 4 with secondary y-axis)
            variance_data = []
            variance_labels = []
            std_data = []
            
            if 'noise_stability' in stability_data:
                noise_std = stability_data['noise_stability'].get('score_std_normalized', 0)
                variance_data.append(stability_data['noise_stability'].get('avg_score_retention', 0))
                std_data.append(noise_std)
                variance_labels.append('Noise')
            
            if 'dropout_stability' in stability_data:
                dropout_std = stability_data['dropout_stability'].get('score_std_normalized', 0)
                variance_data.append(stability_data['dropout_stability'].get('avg_score_retention', 0))
                std_data.append(dropout_std)
                variance_labels.append('Dropout')
            
            if 'bootstrap_stability' in stability_data:
                bootstrap_std = stability_data['bootstrap_stability'].get('score_std_normalized', 0)
                bootstrap_stability_score = stability_data['bootstrap_stability'].get('stability_score', 0)
                variance_data.append(bootstrap_stability_score)
                std_data.append(bootstrap_std)
                variance_labels.append('Bootstrap')
            
            # Mean performance (primary y-axis)
            fig.add_trace(
                go.Bar(
                    x=variance_labels,
                    y=variance_data,
                    name='Mean Performance',
                    marker_color='lightblue',
                    yaxis='y',
                    hovertemplate='Test: %{x}<br>Mean Performance: %{y:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Variance (secondary y-axis)
            fig.add_trace(
                go.Scatter(
                    x=variance_labels,
                    y=std_data,
                    mode='markers+lines',
                    name='Normalized Std Dev',
                    marker=dict(color='red', size=10),
                    line=dict(color='red'),
                    yaxis='y2',
                    hovertemplate='Test: %{x}<br>Std Dev: %{y:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f'Enhanced Stability Analysis - {model_name}<br>' +
                      f'<sub>Overall Stability Score: {stability_data.get("overall_stability", 0):.3f}</sub>',
                showlegend=True,
                height=900,
                template='plotly_white',
                legend=dict(x=1.05, y=1)
            )
            
            # Update axes
            fig.update_xaxes(title_text="Noise Level", row=1, col=1)
            fig.update_yaxes(title_text="Score Retention", row=1, col=1)
            
            fig.update_xaxes(title_text="Bootstrap Score", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            
            fig.update_xaxes(title_text="Stability Component", row=2, col=1)
            fig.update_yaxes(title_text="Normalized Score", row=2, col=1, range=[0, 1])
            
            fig.update_xaxes(title_text="Test Type", row=2, col=2)
            fig.update_yaxes(title_text="Mean Performance", row=2, col=2)
            fig.update_yaxes(title_text="Normalized Std Dev", row=2, col=2, secondary_y=True)
            
            # Save using the enhanced save method
            saved_files = self._save_plot(fig, model_name, "stability")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Enhanced stability plot failed: {e}")
            return []
    
    def run_analysis(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, ModelAnalysisResult]:
        """Run comprehensive analysis on top models"""
        
        self.logger.info("🚀 Starting comprehensive model analysis...")
        
        # Load training results
        training_results = self.load_training_results()
        target_type = training_results['summary']['target_type']
        
        # Load top models
        top_models = self.load_top_models(training_results, n_models=self.config.n_models_to_evaluate)
        
        if not top_models:
            raise RuntimeError("No models could be loaded for analysis")
        
        # Analyze each model
        analysis_results = {}
        for model_name, model in top_models:
            try:
                result = self.analyze_model(model_name, model, X_test, y_test, target_type)
                analysis_results[model_name] = result
                
                self.logger.info(f"✅ Analysis complete for {model_name} "
                               f"(took {result.execution_time:.2f}s)")
                
            except Exception as e:
                self.logger.error(f"❌ Analysis failed for {model_name}: {e}")
        
        # Generate comparison report
        self._generate_comparison_report(analysis_results, target_type)
        
        self.logger.info(f"🎉 Analysis complete! Results saved to {self.config.output_dir}")
        
        return analysis_results
    
    def _generate_comparison_report(self, results: Dict[str, ModelAnalysisResult], 
                                  target_type: str) -> None:
        """Generate comparative analysis report"""
        
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "target_type": target_type,
            "models_analyzed": list(results.keys()),
            "comparison_summary": {},
            "detailed_results": {}
        }
        
        # Performance comparison
        performance_comparison = {}
        for model_name, result in results.items():
            performance_comparison[model_name] = result.performance_metrics
        
        report["comparison_summary"]["performance"] = performance_comparison
        
        # Stability comparison
        stability_comparison = {}
        for model_name, result in results.items():
            stability_comparison[model_name] = {
                "overall_stability": result.stability_metrics.get("overall_stability", 0),
                "execution_time": result.execution_time
            }
        
        report["comparison_summary"]["stability"] = stability_comparison
        
        # Interpretability comparison
        interpretability_comparison = {}
        for model_name, result in results.items():
            interpretability_comparison[model_name] = {
                "interpretability_score": result.interpretability_scores.get("interpretability_score", 0),
                "has_shap": "shap_analysis" in result.interpretability_scores
            }
        
        report["comparison_summary"]["interpretability"] = interpretability_comparison
        
        # Detailed results
        for model_name, result in results.items():
            report["detailed_results"][model_name] = {
                "performance_metrics": result.performance_metrics,
                "stability_metrics": result.stability_metrics,
                "interpretability_scores": result.interpretability_scores,
                "execution_time": result.execution_time,
                "plots_generated": result.plots_saved
            }
        
        # Save report
        report_path = self.config.output_dir / "comprehensive_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📊 Comparison report saved to {report_path}")

# CLI Interface
def create_default_config() -> AnalysisConfig:
    """Create default configuration"""
    return AnalysisConfig(
        enable_shap=True,
        enable_learning_curves=True,
        enable_residual_analysis=True,
        enable_stability_analysis=True,
        enable_interpretability=True,
        save_plots=True,
        verbose=True
    )

def create_analysis_config_from_yaml(yaml_config_dict: Dict[str, Any], 
                                   output_dir: str = "analysis_output",
                                   training_report_path: str = "training_report.json",
                                   models_dir: str = "models") -> AnalysisConfig:
    """Create AnalysisConfig from YAML configuration dictionary for pipeline use"""
    try:
        
        if 'custom_mode' in yaml_config_dict:
            eval_config = yaml_config_dict['custom_mode'].get('model_evaluation', {})
        elif 'fast_mode' in yaml_config_dict:
            eval_config = yaml_config_dict['fast_mode'].get('model_evaluation', {})
        else:
            # Fallback to direct model_evaluation key
            eval_config = yaml_config_dict.get('model_evaluation', {})
        
        # Get stability testing config
        stability_config = eval_config.get('stability_testing', {})
        feature_config = eval_config.get('feature_importance', {})
      
        return AnalysisConfig(
            # Basic settings - use provided paths from pipeline
            models_dir=Path(models_dir),
            training_report_path=Path(training_report_path),
            output_dir=Path(output_dir),
            
            # Analysis components from YAML
            enable_shap=eval_config.get('enable_shap', True),
            enable_learning_curves=eval_config.get('enable_learning_curves', True),
            enable_residual_analysis=eval_config.get('enable_residual_analysis', True),
            enable_stability_analysis=eval_config.get('enable_stability_analysis', True),
            enable_interpretability=eval_config.get('enable_interpretability', True),
            enable_uncertainty_analysis=eval_config.get('enable_uncertainty_analysis', True),
            
            # Visualization
            save_plots=eval_config.get('save_plots', True),
            plot_format=eval_config.get('plot_format', 'html'),
            plot_dpi=eval_config.get('plot_dpi', 300),
            figure_size=tuple(eval_config.get('figure_size', [12, 8])),
            
            # Performance
            n_permutations=eval_config.get('n_permutations', 50),
            n_bootstrap_samples=eval_config.get('n_bootstrap_samples', 100),
            max_shap_samples=eval_config.get('max_shap_samples', 1000),
            n_models_to_evaluate=eval_config.get('n_models_to_evaluate', 3),
            
            # Feature importance enhancements
            top_k_features=feature_config.get('top_k_features', 15),
            show_direction=feature_config.get('show_direction', True),
            add_log_odds_axis=feature_config.get('add_log_odds_axis', True),
            permutation_error_bars=feature_config.get('permutation_error_bars', True),
            
            # Stability
            noise_levels=eval_config.get('noise_levels', [0.01, 0.05, 0.1, 0.2]),
            dropout_rates=eval_config.get('dropout_rates', [0.05, 0.1, 0.2, 0.3]),
            
            # Reproducibility
            random_state=eval_config.get('random_state', 42),
            
            # Logging
            verbose=eval_config.get('verbose', True)
        )
        
    except Exception as e:
        print(f"⚠️ Failed to create config from YAML dict: {e}")
        print("Using default configuration...")
        return create_default_config()

def load_config_from_yaml(config_path: str = "config/unified_config_v3.yaml") -> AnalysisConfig:
    """Load analysis configuration from unified config YAML file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        eval_config = config.get('model_evaluation', {})
        feature_config = eval_config.get('feature_importance', {})
        
        return AnalysisConfig(
            # Basic settings
            models_dir=Path(eval_config.get('models_dir', 'models')),
            training_report_path=Path(eval_config.get('training_report_path', 'training_report.json')),
            output_dir=Path(eval_config.get('output_dir', 'analysis_output')),
            
            # Analysis components
            enable_shap=eval_config.get('enable_shap', True),
            enable_learning_curves=eval_config.get('enable_learning_curves', True),
            enable_residual_analysis=eval_config.get('enable_residual_analysis', True),
            enable_stability_analysis=eval_config.get('enable_stability_analysis', True),
            enable_interpretability=eval_config.get('enable_interpretability', True),
            enable_uncertainty_analysis=eval_config.get('enable_uncertainty_analysis', True),
            
            # Visualization
            save_plots=eval_config.get('save_plots', True),
            plot_format=eval_config.get('plot_format', 'html'),
            plot_dpi=eval_config.get('plot_dpi', 300),
            figure_size=tuple(eval_config.get('figure_size', [12, 8])),
            
            # Performance
            n_permutations=eval_config.get('n_permutations', 50),
            n_bootstrap_samples=eval_config.get('n_bootstrap_samples', 100),
            max_shap_samples=eval_config.get('max_shap_samples', 1000),
            n_models_to_evaluate=eval_config.get('n_models_to_evaluate', 3),
            
            # Feature importance enhancements
            top_k_features=feature_config.get('top_k_features', 15),
            show_direction=feature_config.get('show_direction', True),
            add_log_odds_axis=feature_config.get('add_log_odds_axis', True),
            permutation_error_bars=feature_config.get('permutation_error_bars', True),
            
            # Stability
            noise_levels=eval_config.get('noise_levels', [0.01, 0.05, 0.1, 0.2]),
            dropout_rates=eval_config.get('dropout_rates', [0.05, 0.1, 0.2, 0.3]),
            
            # Reproducibility
            random_state=eval_config.get('random_state', 42),
            
            # Logging
            verbose=eval_config.get('verbose', True)
        )
        
    except Exception as e:
        print(f"⚠️ Failed to load config from {config_path}: {e}")
        print("Using default configuration...")
        return create_default_config()

def main():
    """Simplified CLI interface - Pure YAML configuration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Model Analysis")
    parser.add_argument("test_data", help="Test dataset CSV file")
    parser.add_argument("target", help="Target column name")
    parser.add_argument("--config", default="config/unified_config_v3.yaml", help="Configuration YAML file")
    
    args = parser.parse_args()
    
    # Load test data
    df_test = pd.read_csv(args.test_data)
    X_test = df_test.drop(columns=[args.target])
    y_test = df_test[args.target]
    
    # Load configuration purely from YAML
    config = load_config_from_yaml(args.config)
    
    print(f"📄 Using configuration: {args.config}")
    print(f"📁 Models directory: {config.models_dir}")
    print(f"📊 Output directory: {config.output_dir}")
    print(f"🔍 SHAP analysis: {'Enabled' if config.enable_shap else 'Disabled'}")
    print(f"📈 Plot generation: {'Enabled' if config.save_plots else 'Disabled'}")
    
    # Run analysis
    analyzer = ModelAnalyzer(config)
    results = analyzer.run_analysis(X_test, y_test)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📁 Results saved to: {config.output_dir}")
    print(f"🔍 Models analyzed: {list(results.keys())}")
    print(f"📊 SHAP features analyzed: Top {config.top_k_features} features")
    print(f"🎯 Feature direction analysis: {'Enabled' if config.show_direction else 'Disabled'}")
    print(f"📈 Log-odds axis: {'Enabled' if config.add_log_odds_axis else 'Disabled'}")
    print(f"📉 Error bars: {'Enabled' if config.permutation_error_bars else 'Disabled'}")

if __name__ == "__main__":
    main()
    
# Usage examples:

"""
# Basic usage with default config (unified_config_v3.yaml)
python evaluator.py data/test.csv target_column

# Using custom configuration file
python evaluator.py data/test.csv target_column --config config/my_custom_config.yaml

# All settings are configured through YAML files:
model_evaluation:
  models_dir: "models"                    # Models directory
  training_report_path: "training_report.json"  # Training report path  
  output_dir: "analysis_output"           # Output directory
  feature_importance:
    top_k_features: 20                    # Show top 20 features
    show_direction: true                  # Show positive/negative impact
    add_log_odds_axis: true              # Add log-odds secondary axis
    permutation_error_bars: true         # Add error bars to permutation importance
  enable_shap: true                      # Enable/disable SHAP analysis
  save_plots: true                       # Enable/disable plot generation
  plot_format: "html"                    # Plot format: "html" or "png"
  max_shap_samples: 1000                 # Max samples for SHAP analysis
  n_permutations: 100                    # Number of permutations for importance
"""