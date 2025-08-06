# ds_autoadvisor/advisor/model_recommender.py

from typing import List, Dict, Optional, Any
import numpy as np

def infer_target_type(df, target: str, class_threshold: int = 20) -> str:
    """Classify target column as regression or classification (with robust dtype and unique-count heuristics)."""
    n_unique = df[target].nunique()
    dtype = df[target].dtype
    if dtype.name in ["object", "category", "bool"]:
        return "classification"
    if np.issubdtype(dtype, np.integer) and n_unique <= class_threshold:
        return "classification"
    return "regression"

def recommend_model(
    assumptions: Dict[str, dict],
    target_type: str
) -> Dict[str, Any]:
    """
    Suggest models based on failed assumptions and target type.
    Returns recommended model(s), reason, violated assumptions, resolved target type.
    """

    # Build violated list including explicit multicollinearity if present
    violated = [
        key for key in ["normality", "linearity", "homoscedasticity", "independence"]
        if assumptions.get(key) and not assumptions[key].get("passed")
    ]
    # Add multicollinearity check, if not present then assume it's passed
    if assumptions.get("multicollinearity") and not assumptions["multicollinearity"].get("passed"):
        violated.append("multicollinearity")

    result = {
        "recommended": [],
        "reason": "",
        "violated": violated,
        "target_type": target_type,
    }

    if target_type == "regression":
        if not violated:
            result.update({
                "recommended": ["LinearRegression", "Ridge", "Lasso"],
                "reason": "All linear regression assumptions passed."
            })
        elif "multicollinearity" in violated:
            result.update({
                "recommended": ["Ridge", "Lasso"],
                "reason": "Multicollinearity detected — use regularized linear models."
            })
        elif "linearity" in violated or "normality" in violated:
            result.update({
                "recommended": ["RandomForestRegressor", "GradientBoostingRegressor", "HistGradientBoostingRegressor"],
                "reason": "Linearity and/or normality assumption failed — use tree-based models."
            })
        else:
            result.update({
                "recommended": ["XGBoostRegressor", "LightGBMRegressor", "SVR"],
                "reason": "Fallback to robust or non-parametric models."
            })

    elif target_type == "classification":
        imbalance_flag = assumptions.get("class_balance", {}).get("imbalance", False)

        if not violated and not imbalance_flag:
            result.update({
                "recommended": ["LogisticRegression", "LinearSVC"],
                "reason": "Assumptions passed and class balance acceptable."
            })
        elif imbalance_flag:
            result.update({
                "recommended": ["XGBoostClassifier", "BalancedRandomForestClassifier", "CatBoostClassifier"],
                "reason": "Class imbalance detected; recommend models robust to imbalance."
            })
        else:
            result.update({
                "recommended": ["RandomForestClassifier", "GradientBoostingClassifier", "ExtraTreesClassifier"],
                "reason": "Fallback to robust classifiers (assumptions violated but classes balanced)."
            })
    else:
        result.update({
            "recommended": ["AutoML tools like AutoGluon, PyCaret"],
            "reason": "Target type not clearly identified from data/metadata."
        })

    return result

# --- CLI/Script usage for workflow automation ---

if __name__ == "__main__":
    import pandas as pd
    import json
    import sys

    # Usage: python model_recommender.py <cleaned_csv> <assumptions_json>
    if len(sys.argv) < 3:
        print("Usage: python model_recommender.py <cleaned_csv> <assumption_check_results.json>")
        sys.exit(1)

    # Load data & assumptions report
    df = pd.read_csv(sys.argv[1])
    with open(sys.argv[2], "r") as f:
        assumptions = json.load(f)

    target = assumptions.get("meta", {}).get("target")
    if not target:
        raise ValueError("Target column missing in assumptions report metadata.")
    # Robust target type inference
    target_type = infer_target_type(df, target)
    rec = recommend_model(assumptions, target_type)

    print("\n🔍 Model Recommendation:")
    print(json.dumps(rec, indent=2))
