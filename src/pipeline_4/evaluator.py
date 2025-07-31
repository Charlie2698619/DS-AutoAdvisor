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
    explained_variance_score, max_error, median_absolute_error
)
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.calibration import calibration_curve
import joblib
import shap
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
    plot_dpi: int = 300
    figure_size: Tuple[int, int] = (12, 8)
    
    # Performance
    n_permutations: int = 50 # Number of permutations for feature importance
    n_bootstrap_samples: int = 100 # Number of bootstrap samples for stability testing
    max_shap_samples: int = 1000 # Max samples for SHAP analysis
    
    # Stability testing
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])
    dropout_rates: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3])
    
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
        
        return {
            "baseline_score": baseline_score,
            "noise_stability": noise_stability,
            "dropout_stability": dropout_stability,
            "bootstrap_stability": bootstrap_stability,
            "prediction_consistency": consistency,
            "overall_stability": np.mean([
                noise_stability["avg_score_retention"],
                dropout_stability["avg_score_retention"],
                bootstrap_stability["score_std_normalized"]
            ])
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
            return {
                "noise_levels": self.config.noise_levels,
                "scores": [self._get_baseline_score(model, X_test, y_test, target_type)] * len(self.config.noise_levels),
                "score_retentions": [1.0] * len(self.config.noise_levels),
                "avg_score_retention": 1.0
            }
        
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
        
        return {
            "noise_levels": self.config.noise_levels,
            "scores": scores,
            "score_retentions": score_retentions,
            "avg_score_retention": float(np.mean(score_retentions))
        }
    
    def _test_dropout_robustness(self, model, X_test: pd.DataFrame, 
                               y_test: pd.Series, target_type: str) -> Dict[str, Any]:
        """Test robustness to feature dropout"""
        
        scores = []
        baseline = self._get_baseline_score(model, X_test, y_test, target_type)
        
        for dropout_rate in self.config.dropout_rates:
            # Randomly drop features
            n_drop = int(len(X_test.columns) * dropout_rate)
            if n_drop == 0:
                scores.append(baseline)
                continue
                
            dropped_features = np.random.choice(X_test.columns, n_drop, replace=False)
            X_dropped = X_test.drop(columns=dropped_features)
            
            try:
                score = self._get_baseline_score(model, X_dropped, y_test, target_type)
                scores.append(score)
            except:
                scores.append(0.0)
        
        score_retentions = [s / baseline if baseline != 0 else 0 for s in scores]
        
        return {
            "dropout_rates": self.config.dropout_rates,
            "scores": scores,
            "score_retentions": score_retentions,
            "avg_score_retention": float(np.mean(score_retentions))
        }
    
    def _test_bootstrap_stability(self, model, X_test: pd.DataFrame, 
                                y_test: pd.Series, target_type: str) -> Dict[str, Any]:
        """Test stability across bootstrap samples"""
        
        scores = []
        n_samples = min(len(X_test), 1000)  # Limit for performance
        
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
        
        return {
            "scores": scores,
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_std_normalized": float(np.std(scores) / (np.mean(scores) + 1e-8))
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
        if self.config.enable_shap:
            shap_analysis = self._calculate_shap_values(model, X_test)
        
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
        """Calculate SHAP values for interpretability"""
        
        try:
            # Limit samples for performance
            sample_size = min(len(X_test), self.config.max_shap_samples)
            X_sample = X_test.sample(n=sample_size, random_state=42)
            
            # Try different explainers based on model type
            explainer = None
            
            # Try TreeExplainer first (for tree-based models)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            except:
                # Fallback to KernelExplainer
                try:
                    explainer = shap.KernelExplainer(
                        model.predict, 
                        X_sample.sample(n=min(100, len(X_sample)), random_state=42)
                    )
                    shap_values = explainer.shap_values(X_sample.head(min(100, len(X_sample))))
                except:
                    return None
            
            if shap_values is None:
                return None
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Use first class
            
            # Calculate SHAP statistics
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            shap_df = pd.DataFrame({
                'feature': X_sample.columns,
                'mean_abs_shap': mean_abs_shap
            }).sort_values('mean_abs_shap', ascending=False)
            
            return {
                "feature_shap_values": shap_df.to_dict('records'),
                "top_shap_features": shap_df.head(10)['feature'].tolist(),
                "shap_values_raw": shap_values.tolist(),
                "base_value": float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0
            }
            
        except Exception as e:
            self.logger.warning(f"SHAP analysis failed: {e}")
            return None
    
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
    
    def load_top_models(self, training_results: Dict[str, Any], n_models: int = 2) -> List[Tuple[str, Any]]:
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
            return {
                "r2_score": float(r2_score(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "explained_variance": float(explained_variance_score(y_test, y_pred)),
                "max_error": float(max_error(y_test, y_pred)),
                "median_absolute_error": float(median_absolute_error(y_test, y_pred))
            }
        else:
            accuracy = float(np.mean(y_pred == y_test))
            return {
                "accuracy": accuracy,
                "error_rate": 1 - accuracy
            }
    
    def _generate_visualizations(self, model_name: str, model: Any, 
                               X_test: pd.DataFrame, y_test: pd.Series,
                               target_type: str, analysis_results: Dict) -> List[str]:
        """Generate comprehensive visualizations"""
        
        plots_saved = []
        
        try:
            # Learning curves plot
            if 'learning_curves' in analysis_results and analysis_results['learning_curves']:
                plot_path = self._plot_learning_curves(model_name, analysis_results['learning_curves'])
                if plot_path:
                    plots_saved.append(plot_path)
            
            # Residual plots (for regression)
            if target_type == "regression" and 'residual' in analysis_results:
                plot_path = self._plot_residuals(model_name, analysis_results['residual'])
                if plot_path:
                    plots_saved.append(plot_path)
            
            # Feature importance plot
            if 'interpretability' in analysis_results:
                plot_path = self._plot_feature_importance(model_name, analysis_results['interpretability'])
                if plot_path:
                    plots_saved.append(plot_path)
            
            # Stability analysis plot
            if 'stability' in analysis_results:
                plot_path = self._plot_stability_analysis(model_name, analysis_results['stability'])
                if plot_path:
                    plots_saved.append(plot_path)
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
        
        return plots_saved
    
    def _plot_learning_curves(self, model_name: str, learning_data: Dict) -> Optional[str]:
        """Generate learning curves plot"""
        
        try:
            fig = go.Figure()
            
            train_sizes = learning_data['train_sizes']
            train_mean = learning_data['train_scores_mean']
            train_std = learning_data['train_scores_std']
            val_mean = learning_data['val_scores_mean']
            val_std = learning_data['val_scores_std']
            
            # Training curve
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
            
            # Validation curve
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
                hovermode='x unified'
            )
            
            plot_path = self.config.output_dir / f"{model_name}_learning_curves.html"
            fig.write_html(plot_path)
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Learning curves plot failed: {e}")
            return None
    
    def _plot_residuals(self, model_name: str, residual_data: Dict) -> Optional[str]:
        """Generate residual analysis plots"""
        
        try:
            if 'residuals' not in residual_data:
                return None
            
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
                height=800
            )
            
            plot_path = self.config.output_dir / f"{model_name}_residuals.html"
            fig.write_html(plot_path)
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Residuals plot failed: {e}")
            return None
    
    def _plot_feature_importance(self, model_name: str, interp_data: Dict) -> Optional[str]:
        """Generate feature importance plots"""
        
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Permutation Importance', 'SHAP Values'],
                specs=[[{'secondary_y': False}, {'secondary_y': False}]]
            )
            
            # Permutation importance
            if 'permutation_importance' in interp_data and 'feature_importances' in interp_data['permutation_importance']:
                perm_data = interp_data['permutation_importance']['feature_importances'][:15]  # Top 15
                
                features = [item['feature'] for item in perm_data]
                importances = [item['importance_mean'] for item in perm_data]
                stds = [item['importance_std'] for item in perm_data]
                
                fig.add_trace(
                    go.Bar(x=importances, y=features, orientation='h',
                          error_x=dict(type='data', array=stds),
                          name='Permutation Importance'),
                    row=1, col=1
                )
            
            # SHAP values
            if 'shap_analysis' in interp_data and interp_data['shap_analysis'] and 'feature_shap_values' in interp_data['shap_analysis']:
                shap_data = interp_data['shap_analysis']['feature_shap_values'][:15]  # Top 15
                
                features = [item['feature'] for item in shap_data]
                shap_values = [item['mean_abs_shap'] for item in shap_data]
                
                fig.add_trace(
                    go.Bar(x=shap_values, y=features, orientation='h',
                          name='SHAP Values'),
                    row=1, col=2
                )
            
            fig.update_layout(
                title=f'Feature Importance - {model_name}',
                showlegend=False,
                height=600
            )
            
            plot_path = self.config.output_dir / f"{model_name}_feature_importance.html"
            fig.write_html(plot_path)
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Feature importance plot failed: {e}")
            return None
    
    def _plot_stability_analysis(self, model_name: str, stability_data: Dict) -> Optional[str]:
        """Generate stability analysis plots"""
        
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Noise Robustness', 'Dropout Robustness', 
                               'Bootstrap Stability', 'Overall Stability'],
                specs=[[{'secondary_y': False}, {'secondary_y': False}],
                       [{'secondary_y': False}, {'secondary_y': False}]]
            )
            
            # Noise robustness
            if 'noise_stability' in stability_data:
                noise_data = stability_data['noise_stability']
                fig.add_trace(
                    go.Scatter(x=noise_data['noise_levels'], y=noise_data['score_retentions'],
                              mode='lines+markers', name='Noise Robustness'),
                    row=1, col=1
                )
            
            # Dropout robustness
            if 'dropout_stability' in stability_data:
                dropout_data = stability_data['dropout_stability']
                fig.add_trace(
                    go.Scatter(x=dropout_data['dropout_rates'], y=dropout_data['score_retentions'],
                              mode='lines+markers', name='Dropout Robustness'),
                    row=1, col=2
                )
            
            # Bootstrap stability
            if 'bootstrap_stability' in stability_data and 'scores' in stability_data['bootstrap_stability']:
                bootstrap_scores = stability_data['bootstrap_stability']['scores']
                fig.add_trace(
                    go.Histogram(x=bootstrap_scores, nbinsx=20, name='Bootstrap Scores'),
                    row=2, col=1
                )
            
            # Overall stability metrics
            stability_metrics = {
                'Noise Robustness': stability_data.get('noise_stability', {}).get('avg_score_retention', 0),
                'Dropout Robustness': stability_data.get('dropout_stability', {}).get('avg_score_retention', 0),
                'Bootstrap Stability': 1 - stability_data.get('bootstrap_stability', {}).get('score_std_normalized', 1),
                'Prediction Consistency': stability_data.get('prediction_consistency', {}).get('consistency_score', 0)
            }
            
            fig.add_trace(
                go.Bar(x=list(stability_metrics.keys()), y=list(stability_metrics.values()),
                      name='Stability Metrics'),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'Stability Analysis - {model_name}',
                showlegend=False,
                height=800
            )
            
            plot_path = self.config.output_dir / f"{model_name}_stability.html"
            fig.write_html(plot_path)
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Stability plot failed: {e}")
            return None
    
    def run_analysis(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, ModelAnalysisResult]:
        """Run comprehensive analysis on top models"""
        
        self.logger.info("🚀 Starting comprehensive model analysis...")
        
        # Load training results
        training_results = self.load_training_results()
        target_type = training_results['summary']['target_type']
        
        # Load top models
        top_models = self.load_top_models(training_results, n_models=2)
        
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

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Model Analysis")
    parser.add_argument("test_data", help="Test dataset CSV file")
    parser.add_argument("target", help="Target column name")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--report-path", default="training_report.json", help="Training report path")
    parser.add_argument("--output-dir", default="analysis_output", help="Output directory")
    parser.add_argument("--disable-shap", action="store_true", help="Disable SHAP analysis")
    parser.add_argument("--disable-plots", action="store_true", help="Disable plot generation")
    
    args = parser.parse_args()
    
    # Load test data
    df_test = pd.read_csv(args.test_data)
    X_test = df_test.drop(columns=[args.target])
    y_test = df_test[args.target]
    
    # Create configuration
    config = AnalysisConfig(
        models_dir=Path(args.models_dir),
        training_report_path=Path(args.report_path),
        output_dir=Path(args.output_dir),
        enable_shap=not args.disable_shap,
        save_plots=not args.disable_plots,
        verbose=True
    )
    
    # Run analysis
    analyzer = ModelAnalyzer(config)
    results = analyzer.run_analysis(X_test, y_test)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📁 Results saved to: {config.output_dir}")
    print(f"🔍 Models analyzed: {list(results.keys())}")

if __name__ == "__main__":
    main()
    
# Usage examples


# Advanced analysis with custom config
# Basic analysis
#python model_analyzer.py test_data.csv target_column

# Advanced analysis with custom config
#python model_analyzer.py test_data.csv target_column --models-dir ./models --output-dir ./analysis --disable-shap