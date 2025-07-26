import pandas as pd
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.advisor.model_recommender import infer_target_type, recommend_model

def test_with_bank_data():
    print("🏦 Testing Model Recommender with Bank Dataset")
    print("="*50)
    
    # Load bank data
    try:
        df = pd.read_csv("data/bank_cleaned.csv")
        print("✅ Using cleaned bank data")
    except FileNotFoundError:
        df = pd.read_csv("data/bank.csv", delimiter=';')
        print("⚠️ Using original bank data")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Find target column (usually 'y' in bank dataset)
    target = 'y' if 'y' in df.columns else df.columns[-1]
    print(f"Target column: {target}")
    print(f"Target values: {df[target].value_counts().to_dict()}")
    
    # Infer target type
    target_type = infer_target_type(df, target)
    print(f"Inferred target type: {target_type}")
    
    # Create mock assumptions (you can replace with real assumption results)
    print(f"\n🧪 Testing different assumption scenarios...")
    
    # Scenario 1: All assumptions passed
    assumptions_good = {
        "normality": {"passed": True},
        "linearity": {"passed": True},
        "homoscedasticity": {"passed": True},
        "independence": {"passed": True},
        "multicollinearity": {"passed": True},
        "class_balance": {"imbalance": False}
    }
    
    result1 = recommend_model(assumptions_good, target_type)
    print(f"\n✅ Best case scenario:")
    print(f"   Recommended: {result1['recommended']}")
    print(f"   Reason: {result1['reason']}")
    
    # Scenario 2: Typical real-world issues
    assumptions_realistic = {
        "normality": {"passed": False},
        "linearity": {"passed": False}, 
        "homoscedasticity": {"passed": True},
        "independence": {"passed": True},
        "multicollinearity": {"passed": True},
        "class_balance": {"imbalance": True}  # Bank data often imbalanced
    }
    
    result2 = recommend_model(assumptions_realistic, target_type)
    print(f"\n⚠️ Realistic scenario:")
    print(f"   Recommended: {result2['recommended']}")
    print(f"   Reason: {result2['reason']}")
    print(f"   Violations: {result2['violated']}")
    
    # Scenario 3: Multicollinearity problem
    assumptions_multicol = {
        "normality": {"passed": True},
        "linearity": {"passed": True},
        "multicollinearity": {"passed": False},
        "class_balance": {"imbalance": False}
    }
    
    result3 = recommend_model(assumptions_multicol, target_type)
    print(f"\n🔗 Multicollinearity scenario:")
    print(f"   Recommended: {result3['recommended']}")
    print(f"   Reason: {result3['reason']}")

if __name__ == "__main__":
    test_with_bank_data()