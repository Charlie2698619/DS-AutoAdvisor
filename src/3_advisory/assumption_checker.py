import pandas as pd
import numpy as np
from scipy.stats import shapiro, anderson, levene, jarque_bera, normaltest
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import linear_harvey_collier
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AssumptionConfig:
    """Type-safe configuration for assumption testing - ALL values loaded from YAML"""
    # Normality testing
    normality_alpha: float
    normality_max_sample: int
    normality_method: str
    
    # Homoscedasticity  
    homo_alpha: float
    homo_method: str
    
    # Multicollinearity
    vif_threshold: float
    correlation_threshold: float
    
    # Class balance
    imbalance_threshold: float
    min_class_size: int
    
    # New assumptions
    linearity_alpha: float
    independence_alpha: float
    
    # Scalability
    chunk_size: Optional[int]
    enable_sampling: bool
    max_features_vif: int
    
    # Output options
    verbose: bool
    generate_recommendations: bool
    
    @classmethod
    def from_yaml_config(cls, config_dict: Dict[str, Any]) -> 'AssumptionConfig':
        """Create AssumptionConfig from YAML configuration dictionary"""
        return cls(
            # Normality testing
            normality_alpha=config_dict.get('normality_alpha', 0.05),
            normality_max_sample=config_dict.get('normality_max_sample', 5000),
            normality_method=config_dict.get('normality_method', 'shapiro'),
            
            # Homoscedasticity  
            homo_alpha=config_dict.get('homo_alpha', 0.05),
            homo_method=config_dict.get('homo_method', 'breusch_pagan'),
            
            # Multicollinearity
            vif_threshold=config_dict.get('vif_threshold', 10.0),
            correlation_threshold=config_dict.get('correlation_threshold', 0.95),
            
            # Class balance
            imbalance_threshold=config_dict.get('imbalance_threshold', 0.9),
            min_class_size=config_dict.get('min_class_size', 30),
            
            # New assumptions
            linearity_alpha=config_dict.get('linearity_alpha', 0.05),
            independence_alpha=config_dict.get('independence_alpha', 0.05),
            
            # Scalability
            chunk_size=config_dict.get('chunk_size', None),
            enable_sampling=config_dict.get('enable_sampling', True),
            max_features_vif=config_dict.get('max_features_vif', 50),
            
            # Output options
            verbose=config_dict.get('verbose', True),
            generate_recommendations=config_dict.get('generate_recommendations', True)
        ) 

class EnhancedAssumptionChecker:
    """Enhanced assumption checker with better scalability and features"""
    
    def __init__(self, config: AssumptionConfig):
        self.config = config
        self.results = {}
        self.recommendations = []
        
    def check_normality_enhanced(
        self, 
        df: pd.DataFrame, 
        num_cols: List[str]
    ) -> Dict[str, Any]:
        """Enhanced normality testing with multiple methods"""
        result = {
            "passed": True, 
            "failed_columns": [], 
            "tested_columns": [], 
            "column_results": {}, 
            "skipped": [],
            "method_used": self.config.normality_method
        }
        
        for col in num_cols:
            sample = df[col].dropna()
            if len(sample) < 3 or sample.nunique() < 3:
                result["skipped"].append({
                    "col": col, 
                    "reason": "Too few non-NA or unique values"
                })
                continue
                
            try:
                # Apply sampling for large datasets
                if len(sample) > self.config.normality_max_sample:
                    sample = sample.sample(self.config.normality_max_sample, random_state=42)
                
                # Choose normality test method
                if self.config.normality_method == "shapiro":
                    stat, p = shapiro(sample)
                elif self.config.normality_method == "anderson":
                    result_ad = anderson(sample, dist='norm')
                    stat, p = result_ad.statistic, 0.05 if result_ad.statistic > result_ad.critical_values[2] else 0.1
                elif self.config.normality_method == "jarque_bera":
                    stat, p = jarque_bera(sample)[:2]
                else:
                    stat, p = normaltest(sample)
                
                passed = p > self.config.normality_alpha
                result["column_results"][col] = {
                    "statistic": float(stat), 
                    "p_value": float(p), 
                    "n": len(sample), 
                    "passed": passed,
                    "skewness": float(sample.skew()),
                    "kurtosis": float(sample.kurtosis())
                }
                result["tested_columns"].append(col)
                if not passed:
                    result["failed_columns"].append(col)
                    
            except Exception as e:
                result["skipped"].append({
                    "col": col, 
                    "reason": f"Test failed: {str(e)}"
                })
        
        result["passed"] = len(result["failed_columns"]) == 0
        return result
    
    def check_homoscedasticity_enhanced(
        self,
        df: pd.DataFrame,
        target: str,
        features: List[str]
    ) -> Dict[str, Any]:
        """Enhanced homoscedasticity testing with multiple methods"""
        result = {
            "passed": True, 
            "p_value": None, 
            "statistic": None, 
            "error": None, 
            "n": None,
            "method_used": self.config.homo_method
        }
        
        try:
            data = df.dropna(subset=[target] + features)
            
            # Apply sampling for large datasets
            if self.config.chunk_size and len(data) > self.config.chunk_size:
                data = data.sample(n=self.config.chunk_size, random_state=42)
                result["note"] = f"Test performed on sample of {len(data)} rows"
            
            X = add_constant(data[features])
            y = data[target]
            result["n"] = len(data)
            
            model = OLS(y, X).fit()
            
            # Choose homoscedasticity test method
            if self.config.homo_method == "breusch_pagan":
                stat, pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
            elif self.config.homo_method == "white":
                stat, pval, _, _ = het_white(model.resid, model.model.exog)
            else:
                stat, pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
            
            result["statistic"] = float(stat)
            result["p_value"] = float(pval)
            result["passed"] = pval > self.config.homo_alpha
            
        except Exception as e:
            result["error"] = str(e)
            result["passed"] = False
            
        return result
    
    def check_multicollinearity_scalable(
        self,
        df: pd.DataFrame,
        features: List[str]
    ) -> Dict[str, Any]:
        """Enhanced multicollinearity checking with rich metadata for manual decision-making"""
        result = {
            "passed": True, 
            "high_vif_features": [], 
            "high_correlation_pairs": [],
            "vif_analysis": {
                "all_features": [],
                "risk_categories": {
                    "low_risk": [],      # VIF < 5
                    "moderate_risk": [], # 5 <= VIF < 10  
                    "high_risk": [],     # VIF >= 10
                    "extreme_risk": []   # VIF >= 20
                }
            },
            "correlation_analysis": {
                "matrix": {},
                "risk_pairs": {
                    "moderate": [],      # 0.7 <= |corr| < 0.9
                    "high": [],          # 0.9 <= |corr| < 0.95
                    "extreme": []        # |corr| >= 0.95
                },
                "feature_risk_scores": {}  # Aggregated risk per feature
            },
            "recommendations": {
                "safe_to_keep": [],
                "review_required": [],
                "strong_candidates_for_removal": [],
                "feature_relationships": {}
            },
            "metadata": {
                "n_features": len(features),
                "n_samples_used": 0,
                "computation_notes": []
            },
            "error": None
        }
        
        try:
            X = df[features].dropna()
            result["metadata"]["n_samples_used"] = len(X)
            
            if X.shape[1] < 2:
                result["passed"] = True
                result["metadata"]["computation_notes"].append("Less than 2 features - no multicollinearity possible")
                return result
            
            # PHASE 1: Comprehensive Correlation Analysis
            corr_matrix = X.corr()
            result["correlation_analysis"]["matrix"] = corr_matrix.to_dict()
            
            # Analyze correlation pairs and categorize by risk
            feature_corr_counts = {feat: {"moderate": 0, "high": 0, "extreme": 0} for feat in features}
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_val = abs(corr_matrix.iloc[i, j])
                    
                    pair_info = {
                        "feature1": feat1,
                        "feature2": feat2, 
                        "correlation": float(corr_val),
                        "interpretation": self._interpret_correlation(corr_val)
                    }
                    
                    # Categorize correlation risk
                    if corr_val >= 0.95:
                        result["correlation_analysis"]["risk_pairs"]["extreme"].append(pair_info)
                        feature_corr_counts[feat1]["extreme"] += 1
                        feature_corr_counts[feat2]["extreme"] += 1
                    elif corr_val >= 0.9:
                        result["correlation_analysis"]["risk_pairs"]["high"].append(pair_info)
                        feature_corr_counts[feat1]["high"] += 1
                        feature_corr_counts[feat2]["high"] += 1
                    elif corr_val >= 0.7:
                        result["correlation_analysis"]["risk_pairs"]["moderate"].append(pair_info)
                        feature_corr_counts[feat1]["moderate"] += 1
                        feature_corr_counts[feat2]["moderate"] += 1
            
            # Calculate feature-level correlation risk scores
            for feat in features:
                counts = feature_corr_counts[feat]
                risk_score = (counts["extreme"] * 3 + counts["high"] * 2 + counts["moderate"] * 1)
                result["correlation_analysis"]["feature_risk_scores"][feat] = {
                    "risk_score": risk_score,
                    "extreme_correlations": counts["extreme"],
                    "high_correlations": counts["high"], 
                    "moderate_correlations": counts["moderate"],
                    "risk_level": self._categorize_correlation_risk(risk_score)
                }
            
            # PHASE 2: Comprehensive VIF Analysis
            if X.shape[1] <= self.config.max_features_vif:
                # Apply sampling for large datasets while preserving relationships
                if self.config.chunk_size and len(X) > self.config.chunk_size:
                    X_sample = X.sample(n=self.config.chunk_size, random_state=42)
                    result["metadata"]["computation_notes"].append(f"VIF calculated on sample of {len(X_sample)} rows")
                else:
                    X_sample = X
                
                # Calculate VIF for ALL features
                for i, feat in enumerate(X_sample.columns):
                    try:
                        vif_value = variance_inflation_factor(X_sample.values, i)
                        
                        vif_info = {
                            "feature": feat,
                            "vif": float(vif_value) if not np.isinf(vif_value) else "infinite",
                            "risk_level": self._categorize_vif_risk(vif_value),
                            "interpretation": self._interpret_vif(vif_value),
                            "r_squared_with_others": (vif_value - 1) / vif_value if vif_value > 1 else 0
                        }
                        
                        result["vif_analysis"]["all_features"].append(vif_info)
                        
                        # Categorize by risk level
                        if vif_value >= 20:
                            result["vif_analysis"]["risk_categories"]["extreme_risk"].append(feat)
                        elif vif_value >= self.config.vif_threshold:  # Default 10
                            result["vif_analysis"]["risk_categories"]["high_risk"].append(feat)
                            result["high_vif_features"].append(feat)
                        elif vif_value >= 5:
                            result["vif_analysis"]["risk_categories"]["moderate_risk"].append(feat)
                        else:
                            result["vif_analysis"]["risk_categories"]["low_risk"].append(feat)
                            
                    except Exception as e:
                        vif_info = {
                            "feature": feat,
                            "vif": None,
                            "error": str(e),
                            "risk_level": "unknown",
                            "interpretation": "VIF calculation failed"
                        }
                        result["vif_analysis"]["all_features"].append(vif_info)
                        result["metadata"]["computation_notes"].append(f"VIF failed for {feat}: {str(e)}")
            else:
                result["metadata"]["computation_notes"].append(
                    f"VIF skipped: too many features ({X.shape[1]} > {self.config.max_features_vif}). "
                    "Consider feature selection or increase max_features_vif."
                )
            
            # PHASE 3: Generate Smart Recommendations
            self._generate_multicollinearity_recommendations(result)
            
            # Update legacy fields for backward compatibility
            result["high_correlation"] = result["correlation_analysis"]["risk_pairs"]["extreme"]
            result["vif_table"] = result["vif_analysis"]["all_features"]
            result["correlation_matrix"] = result["correlation_analysis"]["matrix"]
            
            # Overall pass/fail based on extreme cases only
            extreme_vif = len(result["vif_analysis"]["risk_categories"]["extreme_risk"])
            extreme_corr = len(result["correlation_analysis"]["risk_pairs"]["extreme"])
            result["passed"] = extreme_vif == 0 and extreme_corr == 0
            
        except Exception as e:
            result["error"] = str(e)
            result["passed"] = False
            
        return result
    
    def _interpret_correlation(self, corr_val: float) -> str:
        """Interpret correlation strength"""
        if corr_val >= 0.95:
            return "Nearly perfect - strong redundancy"
        elif corr_val >= 0.9:
            return "Very strong - likely redundant"
        elif corr_val >= 0.7:
            return "Strong - monitor for multicollinearity"
        elif corr_val >= 0.5:
            return "Moderate - acceptable"
        else:
            return "Weak - no concern"
    
    def _categorize_correlation_risk(self, risk_score: int) -> str:
        """Categorize overall correlation risk for a feature"""
        if risk_score >= 3:
            return "extreme"
        elif risk_score >= 2:
            return "high" 
        elif risk_score >= 1:
            return "moderate"
        else:
            return "low"
    
    def _categorize_vif_risk(self, vif_value: float) -> str:
        """Categorize VIF risk level"""
        if np.isinf(vif_value) or vif_value >= 20:
            return "extreme"
        elif vif_value >= 10:
            return "high"
        elif vif_value >= 5:
            return "moderate"
        else:
            return "low"
    
    def _interpret_vif(self, vif_value: float) -> str:
        """Interpret VIF value"""
        if np.isinf(vif_value):
            return "Perfect multicollinearity - feature is linear combination of others"
        elif vif_value >= 20:
            return "Extreme multicollinearity - serious concern"
        elif vif_value >= 10:
            return "High multicollinearity - consider removal"
        elif vif_value >= 5:
            return "Moderate multicollinearity - monitor"
        else:
            return "Low multicollinearity - acceptable"
    
    def _generate_multicollinearity_recommendations(self, result: Dict) -> None:
        """Generate intelligent recommendations for feature management"""
        recs = result["recommendations"]
        
        # Features that are safe to keep
        safe_features = result["vif_analysis"]["risk_categories"]["low_risk"]
        recs["safe_to_keep"] = safe_features
        
        # Features requiring review (moderate risk)
        moderate_vif = result["vif_analysis"]["risk_categories"]["moderate_risk"]
        moderate_corr_features = [
            feat for feat, info in result["correlation_analysis"]["feature_risk_scores"].items()
            if info["risk_level"] == "moderate"
        ]
        recs["review_required"] = list(set(moderate_vif + moderate_corr_features))
        
        # Strong candidates for removal (high/extreme risk)
        high_risk_vif = (result["vif_analysis"]["risk_categories"]["high_risk"] + 
                        result["vif_analysis"]["risk_categories"]["extreme_risk"])
        high_risk_corr = [
            feat for feat, info in result["correlation_analysis"]["feature_risk_scores"].items()
            if info["risk_level"] in ["high", "extreme"]
        ]
        recs["strong_candidates_for_removal"] = list(set(high_risk_vif + high_risk_corr))
        
        # Detailed feature relationships for manual decision-making
        for feat in result["correlation_analysis"]["feature_risk_scores"].keys():
            feat_info = {
                "vif_risk": "unknown",
                "correlation_risk": result["correlation_analysis"]["feature_risk_scores"][feat]["risk_level"],
                "related_features": [],
                "decision_guidance": ""
            }
            
            # Add VIF info if available
            vif_data = [v for v in result["vif_analysis"]["all_features"] if v["feature"] == feat]
            if vif_data:
                feat_info["vif_risk"] = vif_data[0].get("risk_level", "unknown")
                feat_info["vif_value"] = vif_data[0].get("vif", "unknown")
            
            # Find related features
            all_pairs = (result["correlation_analysis"]["risk_pairs"]["moderate"] +
                        result["correlation_analysis"]["risk_pairs"]["high"] + 
                        result["correlation_analysis"]["risk_pairs"]["extreme"])
            
            related = []
            for pair in all_pairs:
                if pair["feature1"] == feat:
                    related.append({"feature": pair["feature2"], "correlation": pair["correlation"]})
                elif pair["feature2"] == feat:
                    related.append({"feature": pair["feature1"], "correlation": pair["correlation"]})
            
            feat_info["related_features"] = sorted(related, key=lambda x: x["correlation"], reverse=True)
            
            # Generate decision guidance
            if feat_info["vif_risk"] == "extreme" or feat_info["correlation_risk"] == "extreme":
                feat_info["decision_guidance"] = "Strong candidate for removal - very high redundancy"
            elif feat_info["vif_risk"] == "high" or feat_info["correlation_risk"] == "high":
                feat_info["decision_guidance"] = "Consider removal or feature engineering"
            elif feat_info["vif_risk"] == "moderate" or feat_info["correlation_risk"] == "moderate":
                feat_info["decision_guidance"] = "Monitor - may need attention in some models"
            else:
                feat_info["decision_guidance"] = "Safe to keep - low multicollinearity risk"
            
            recs["feature_relationships"][feat] = feat_info
    
    def check_linearity(
        self,
        df: pd.DataFrame,
        target: str,
        features: List[str]
    ) -> Dict[str, Any]:
        """Test linearity assumption using Harvey-Collier test"""
        result = {
            "passed": True,
            "p_value": None,
            "statistic": None,
            "error": None,
            "n": None
        }
        
        try:
            data = df.dropna(subset=[target] + features)
            
            # Apply sampling for large datasets
            if self.config.chunk_size and len(data) > self.config.chunk_size:
                data = data.sample(n=self.config.chunk_size, random_state=42)
                result["note"] = f"Test performed on sample of {len(data)} rows"
            
            X = add_constant(data[features])
            y = data[target]
            result["n"] = len(data)
            
            # Harvey-Collier test for linearity
            model = OLS(y, X).fit()
            stat, pval = linear_harvey_collier(model)
            
            result["statistic"] = float(stat)
            result["p_value"] = float(pval)
            result["passed"] = pval > self.config.linearity_alpha
            
        except Exception as e:
            result["error"] = str(e)
            result["passed"] = False
            
        return result
    
    def check_independence(
        self,
        df: pd.DataFrame,
        target: str,
        features: List[str],
        time_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """Test independence assumption using Durbin-Watson test"""
        result = {
            "passed": True,
            "durbin_watson": None,
            "interpretation": None,
            "error": None,
            "n": None
        }
        
        try:
            data = df.dropna(subset=[target] + features)
            
            # Sort by time column if provided
            if time_col and time_col in data.columns:
                data = data.sort_values(time_col)
            
            X = add_constant(data[features])
            y = data[target]
            result["n"] = len(data)
            
            model = OLS(y, X).fit()
            dw_stat = durbin_watson(model.resid)
            
            result["durbin_watson"] = float(dw_stat)
            
            # Interpretation
            if dw_stat < 1.5:
                result["interpretation"] = "positive_autocorrelation"
                result["passed"] = False
            elif dw_stat > 2.5:
                result["interpretation"] = "negative_autocorrelation"  
                result["passed"] = False
            else:
                result["interpretation"] = "no_autocorrelation"
                result["passed"] = True
                
        except Exception as e:
            result["error"] = str(e)
            result["passed"] = False
            
        return result
    
    def generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations based on test results"""
        recommendations = []
        
        # Normality violations
        if results.get("normality") and not results["normality"]["passed"]:
            failed_cols = results["normality"]["failed_columns"]
            recommendations.append(
                f"🔄 Non-normal features detected: {failed_cols}. "
                "Consider log, sqrt, or Box-Cox transformations."
            )
        
        # Multicollinearity violations
        if results.get("multicollinearity"):
            mc_result = results["multicollinearity"]
            
            # Use new enhanced structure
            if mc_result.get("high_vif_features"):
                recommendations.append(
                    f"⚠️ High VIF features detected: {mc_result['high_vif_features']}. "
                    "These features have VIF ≥ 10, indicating high multicollinearity risk."
                )
            
            # Enhanced recommendations with risk categorization
            if mc_result.get("recommendations"):
                recs = mc_result["recommendations"]
                
                if recs.get("strong_candidates_for_removal"):
                    recommendations.append(
                        f"🔴 Strong removal candidates: {recs['strong_candidates_for_removal']}. "
                        "These features show extreme multicollinearity (VIF ≥ 20 or correlation ≥ 0.95)."
                    )
                
                if recs.get("review_required"):
                    recommendations.append(
                        f"🟡 Features requiring review: {recs['review_required']}. "
                        "Moderate multicollinearity detected - consider for specific models."
                    )
                
                if recs.get("safe_to_keep"):
                    recommendations.append(
                        f"🟢 Low-risk features (safe to keep): {len(recs['safe_to_keep'])} features "
                        "show minimal multicollinearity concerns."
                    )
            
            # Provide specific guidance based on correlation analysis
            corr_analysis = mc_result.get("correlation_analysis", {})
            if corr_analysis.get("risk_pairs", {}).get("extreme"):
                extreme_pairs = corr_analysis["risk_pairs"]["extreme"]
                recommendations.append(
                    f"🔗 Extreme correlation pairs found: {len(extreme_pairs)} pairs with |correlation| ≥ 0.95. "
                    "Consider feature selection, PCA, or domain-specific feature engineering."
                )
        
        # Homoscedasticity violations
        if results.get("homoscedasticity") and not results["homoscedasticity"]["passed"]:
            recommendations.append(
                "📊 Heteroscedasticity detected. Consider robust standard errors "
                "or weighted least squares regression."
            )
        
        # Linearity violations
        if results.get("linearity") and not results["linearity"]["passed"]:
            recommendations.append(
                "📈 Non-linear relationships detected. Consider polynomial features "
                "or non-linear models (Random Forest, XGBoost)."
            )
        
        # Independence violations
        if results.get("independence") and not results["independence"]["passed"]:
            recommendations.append(
                "🔄 Autocorrelation detected. Consider adding lagged features "
                "or using time series models."
            )
        
        # Class imbalance
        if results.get("class_balance") and results["class_balance"].get("imbalance"):
            major_class = results["class_balance"]["major_class"]
            prop = results["class_balance"]["major_class_prop"]
            recommendations.append(
                f"⚖️ Class imbalance: '{major_class}' dominates ({prop:.1%}). "
                "Consider SMOTE, undersampling, or stratified sampling."
            )
        
        return recommendations
    
    def run_all_checks(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        time_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive assumption checks with enhanced features"""
        
        if self.config.verbose:
            print("🔍 Running comprehensive assumption checks...")
        
        report = {
            "normality": None,
            "homoscedasticity": None,
            "multicollinearity": None,
            "linearity": None,
            "independence": None,
            "class_balance": None,
            "recommendations": [],
            "meta": {},
            "warnings": [],
            "errors": [],
            "config_used": self.config.__dict__
        }
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        
        # 1. Normality check (all numerics)
        if numeric_cols:
            if self.config.verbose:
                print("  📊 Testing normality...")
            ncheck = self.check_normality_enhanced(df, numeric_cols)
            report["normality"] = ncheck
            if ncheck.get("skipped"):
                report["warnings"].append(f"Normality skipped: {[x['col'] for x in ncheck['skipped']]}")
        
        # 2. Regression assumptions (if target is numeric)
        if target and target in numeric_cols:
            features = [col for col in numeric_cols if col != target]
            
            if len(features) >= 1:
                if self.config.verbose:
                    print("  🏠 Testing homoscedasticity...")
                hcheck = self.check_homoscedasticity_enhanced(df, target, features)
                report["homoscedasticity"] = hcheck
                if hcheck.get("error"):
                    report["errors"].append(f"Homoscedasticity error: {hcheck['error']}")
                
                if self.config.verbose:
                    print("  📈 Testing linearity...")
                lcheck = self.check_linearity(df, target, features)
                report["linearity"] = lcheck
                if lcheck.get("error"):
                    report["errors"].append(f"Linearity error: {lcheck['error']}")
                
                if self.config.verbose:
                    print("  🔄 Testing independence...")
                icheck = self.check_independence(df, target, features, time_col)
                report["independence"] = icheck
                if icheck.get("error"):
                    report["errors"].append(f"Independence error: {icheck['error']}")
            
            # 3. Multicollinearity (at least 2 features)
            if len(features) >= 2:
                if self.config.verbose:
                    print("  🔗 Testing multicollinearity...")
                mcheck = self.check_multicollinearity_scalable(df, features)
                report["multicollinearity"] = mcheck
                if mcheck.get("error"):
                    report["errors"].append(f"Multicollinearity error: {mcheck['error']}")
        
        # 4. Class balance for classification (categorical target)
        if target and target in df.columns and df[target].nunique() <= 20:
            if self.config.verbose:
                print("  ⚖️ Checking class balance...")
            cl_balance = self.check_class_balance(df, target)
            report["class_balance"] = cl_balance
        
        # 5. Generate recommendations
        if self.config.generate_recommendations:
            if self.config.verbose:
                print("  💡 Generating recommendations...")
            report["recommendations"] = self.generate_recommendations(report)
        
        # Meta information
        report["meta"] = {
            "columns": df.columns.tolist(),
            "n_rows": len(df),
            "n_num_cols": len(numeric_cols),
            "n_cat_cols": len(cat_cols),
            "target": target,
            "time_col": time_col
        }
        
        if self.config.verbose:
            print("✅ Assumption checking complete!")
        
        return report
    
    def check_class_balance(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Enhanced class balance checking"""
        result = {
            "imbalance": False, 
            "distribution": {}, 
            "major_class": None, 
            "major_class_prop": None,
            "minor_class": None,
            "minor_class_prop": None,
            "imbalance_ratio": None
        }
        
        try:
            value_counts = df[target].value_counts(normalize=True)
            result["distribution"] = value_counts.to_dict()
            result["major_class"] = value_counts.idxmax()
            result["major_class_prop"] = value_counts.max()
            result["minor_class"] = value_counts.idxmin()
            result["minor_class_prop"] = value_counts.min()
            result["imbalance_ratio"] = result["major_class_prop"] / result["minor_class_prop"]
            
            if value_counts.max() > self.config.imbalance_threshold:
                result["imbalance"] = True
                
        except Exception as e:
            result["error"] = str(e)
            
        return result

# Convenience function for backward compatibility
def run_enhanced_checks(
    df: pd.DataFrame,
    target: Optional[str] = None,
    time_col: Optional[str] = None,
    config: Optional[AssumptionConfig] = None
) -> Dict[str, Any]:
    """Convenience function to run enhanced assumption checks"""
    if config is None:
        config = AssumptionConfig()
    
    checker = EnhancedAssumptionChecker(config)
    return checker.run_all_checks(df, target, time_col)

# Example usage and testing
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python assumption_checker_enhanced.py <input_csv> [<target_col>] [<time_col>]")
        sys.exit(1)
    
    # Load data
    df = pd.read_csv(sys.argv[1])
    target = sys.argv[2] if len(sys.argv) >= 3 else None
    time_col = sys.argv[3] if len(sys.argv) >= 4 else None
    
    # Create config
    config = AssumptionConfig(
        verbose=True,
        generate_recommendations=True,
        normality_method="shapiro",
        homo_method="breusch_pagan"
    )
    
    # Run checks
    report = run_enhanced_checks(df, target, time_col, config)
    
    # Output results
    print("\n" + "="*50)
    print("📋 ASSUMPTION CHECK RESULTS")
    print("="*50)
    
    if report["recommendations"]:
        print("\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  {rec}")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Rows: {report['meta']['n_rows']:,}")
    print(f"  Numeric columns: {report['meta']['n_num_cols']}")
    print(f"  Categorical columns: {report['meta']['n_cat_cols']}")
    
    # Save detailed results
    with open("assumption_check_results.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n💾 Detailed results saved to: assumption_check_results.json")