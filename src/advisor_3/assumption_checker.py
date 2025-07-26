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
    """Type-safe configuration for assumption testing"""
    # Normality testing
    normality_alpha: float = 0.05
    normality_max_sample: int = 5000
    normality_method: str = "shapiro"  # "shapiro", "anderson", "jarque_bera"
    
    # Homoscedasticity  
    homo_alpha: float = 0.05
    homo_method: str = "breusch_pagan"  # "breusch_pagan", "white"
    
    # Multicollinearity
    vif_threshold: float = 10.0 
    correlation_threshold: float = 0.95
    
    # Class balance
    imbalance_threshold: float = 0.9
    min_class_size: int = 30
    
    # New assumptions
    linearity_alpha: float = 0.05
    independence_alpha: float = 0.05
    
    # Scalability
    chunk_size: Optional[int] = None
    enable_sampling: bool = True
    max_features_vif: int = 50  # Limit VIF calculation for performance
    
    # Output options
    verbose: bool = True
    generate_recommendations: bool = True 

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
        """Scalable multicollinearity checking with performance optimizations"""
        result = {
            "passed": True, 
            "high_vif": [], 
            "high_correlation": [],
            "vif_table": [], 
            "correlation_matrix": {},
            "error": None
        }
        
        try:
            X = df[features].dropna()
            
            if X.shape[1] < 2:
                result["passed"] = True
                return result
            
            # Step 1: Quick correlation check first (faster)
            corr_matrix = X.corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.config.correlation_threshold:
                        high_corr_pairs.append({
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": float(corr_matrix.iloc[i, j])
                        })
            
            result["high_correlation"] = high_corr_pairs
            result["correlation_matrix"] = corr_matrix.to_dict()
            
            # Step 2: VIF calculation (limit features for performance)
            if X.shape[1] <= self.config.max_features_vif:
                # Apply sampling for large datasets
                if self.config.chunk_size and len(X) > self.config.chunk_size:
                    X = X.sample(n=self.config.chunk_size, random_state=42)
                    result["vif_note"] = f"VIF calculated on sample of {len(X)} rows"
                
                vif_data = []
                for i, col in enumerate(X.columns):
                    try:
                        vif_value = variance_inflation_factor(X.values, i)
                        vif_data.append({"feature": col, "VIF": float(vif_value)})
                        if vif_value > self.config.vif_threshold:
                            result["high_vif"].append(col)
                    except Exception as e:
                        vif_data.append({"feature": col, "VIF": None, "error": str(e)})
                
                result["vif_table"] = vif_data
            else:
                result["vif_note"] = f"VIF skipped: too many features ({X.shape[1]} > {self.config.max_features_vif})"
            
            result["passed"] = len(result["high_vif"]) == 0 and len(high_corr_pairs) == 0
            
        except Exception as e:
            result["error"] = str(e)
            result["passed"] = False
            
        return result
    
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
            if mc_result.get("high_vif"):
                recommendations.append(
                    f"⚠️ High VIF features: {mc_result['high_vif']}. "
                    "Consider removing or combining correlated features."
                )
            if mc_result.get("high_correlation"):
                recommendations.append(
                    f"🔗 High correlation pairs detected. "
                    "Consider dimensionality reduction (PCA) or feature selection."
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