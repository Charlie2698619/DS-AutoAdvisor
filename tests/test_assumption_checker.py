import sys
import os
# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.advisor.assumption_checker import EnhancedAssumptionChecker, AssumptionConfig
import pandas as pd



# Load your cleaned bank data (from previous data cleaner step)
print("🏦 Loading bank dataset...")
df = pd.read_csv("data/bank_cleaned.csv")  # Use cleaned data from previous step

# Create configuration
config = AssumptionConfig(
    verbose=True,
    generate_recommendations=True,
    normality_method="shapiro", 
    homo_method="breusch_pagan",
    vif_threshold=5.0,  # Stricter than default
    imbalance_threshold=0.8,  # Detect moderate imbalance
    chunk_size=10000  # For large datasets
)

# Initialize checker
checker = EnhancedAssumptionChecker(config)

# Run comprehensive checks (assuming 'y' is your target variable)
# Adjust target column name based on your actual data
target_col = "y"  # Change this to your actual target column name

print(f"\n🎯 Running assumption checks with target: '{target_col}'")
results = checker.run_all_checks(df, target=target_col)

# Display key results
print("\n" + "="*60)
print("📋 ASSUMPTION CHECK SUMMARY")
print("="*60)

# Show recommendations
if results["recommendations"]:
    print("\n💡 ACTIONABLE RECOMMENDATIONS:")
    for i, rec in enumerate(results["recommendations"], 1):
        print(f"  {i}. {rec}")
else:
    print("\n✅ No major assumption violations detected!")

# Show test results summary
print(f"\n📊 TEST RESULTS:")
tests = ["normality", "homoscedasticity", "multicollinearity", "linearity", "independence", "class_balance"]
for test in tests:
    if results.get(test) is not None:
        passed = results[test].get("passed", "N/A")
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test.title()}: {status}")

# Show data summary
meta = results["meta"]
print(f"\n📈 DATA SUMMARY:")
print(f"  Dataset shape: {meta['n_rows']:,} rows × {meta['n_num_cols']} numeric + {meta['n_cat_cols']} categorical columns")
print(f"  Target variable: {meta['target']}")

# Save detailed results
import json
with open("docs/assumption_check_detailed.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n💾 Detailed results saved to: docs/assumption_check_detailed.json")
print("\n🎯 Next steps: Review recommendations and apply suggested transformations!")