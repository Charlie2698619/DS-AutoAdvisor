"""
Fixed test for evaluator.py - properly handles preprocessing consistency
"""
import pandas as pd
import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline_4.evaluator import AnalysisConfig, ModelAnalyzer

def load_bank_data():
    """Load bank data for testing"""
    data_path = Path(__file__).parent.parent / "data" / "bank.csv"
    if data_path.exists():
        return pd.read_csv(data_path, delimiter=';')
    else:
        raise FileNotFoundError(f"Bank data not found at {data_path}")

def prepare_test_data_matching_training():
    """Prepare test data using RAW data (models have built-in preprocessing)"""
    print("🔧 Preparing RAW test data (models contain preprocessing pipeline)...")
    
    # Load training report to get preprocessing details
    report_path = Path("comprehensive_training_report.json")
    with open(report_path, 'r') as f:
        training_report = json.load(f)
    
    preprocessing_details = training_report['preprocessing_details']
    encoding_strategy = preprocessing_details['encoding_strategy']
    scaling_strategy = preprocessing_details['scaling_strategy']
    
    print(f"   - Training used: {encoding_strategy} encoding + {scaling_strategy} scaling")
    print("   - Models expect RAW data (they have built-in preprocessing)")
    
    # Load raw data
    df = load_bank_data()
    target = 'y'
    
    X = df.drop(columns=[target])
    y = df[target]
    
    # Handle categorical target (same as trainer)
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Identify column types for display
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.to_list()
    numeric_cols = X.select_dtypes(include=['number']).columns.to_list()
    
    print(f"   - Categorical columns: {len(categorical_cols)}")
    print(f"   - Numeric columns: {len(numeric_cols)}")
    
    # Split data (same as trainer), but keep RAW format
    # Models are Pipeline objects with preprocessing built-in
    _, X_test, _, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    print(f"✅ RAW test data prepared: {X_test.shape}")
    print(f"   - Features: {list(X_test.columns)[:5]}... (showing first 5)")
    print("   - Data format: RAW (models will apply preprocessing internally)")
    
    return X_test, y_test

def test_evaluator_with_proper_preprocessing():
    """Test evaluator with proper preprocessing consistency"""
    print("🧪 Testing Evaluator with Proper Preprocessing")
    print("="*60)
    
    # Check if comprehensive report exists
    report_path = Path("comprehensive_training_report.json")
    if not report_path.exists():
        print(f"❌ Training report not found: {report_path}")
        print("Please run: python tests/test_trainer.py --comprehensive")
        return False
    
    print(f"✅ Found training report: {report_path}")
    
    # Prepare test data with matching preprocessing
    try:
        X_test, y_test = prepare_test_data_matching_training()
    except Exception as e:
        print(f"❌ Failed to prepare test data: {e}")
        return False
    
    # Setup evaluator config
    config = AnalysisConfig(
        training_report_path=report_path,
        output_dir=Path("evaluator_test_output"),
        enable_shap=False,  # Disable for faster testing
        enable_learning_curves=True,
        enable_residual_analysis=False,  # Disable for faster testing
        enable_stability_analysis=True,
        enable_interpretability=True,
        verbose=True
    )
    
    print(f"📁 Output directory: {config.output_dir}")
    
    # Initialize analyzer
    try:
        analyzer = ModelAnalyzer(config)
        print("✅ ModelAnalyzer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize ModelAnalyzer: {e}")
        return False
    
    # Load training results and models
    try:
        training_results = analyzer.load_training_results()
        top_models = analyzer.load_top_models(training_results, n_models=2)
        
        if not top_models:
            print("❌ No models could be loaded")
            return False
        
        print(f"🤖 Loaded {len(top_models)} models for analysis")
        
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return False
    
    # Run analysis on the first model
    try:
        model_name, model = top_models[0]
        target_type = training_results['summary']['target_type']
        
        print(f"🔬 Running analysis on: {model_name}")
        print(f"🎯 Target type: {target_type}")
        
        # This should now work without feature name mismatch
        result = analyzer.analyze_model(model_name, model, X_test, y_test, target_type)
        
        print(f"✅ Analysis completed successfully!")
        print(f"   - Execution time: {result.execution_time:.2f}s")
        print(f"   - Performance metrics: {len(result.performance_metrics)}")
        print(f"   - Stability metrics: {len(result.stability_metrics)}")
        print(f"   - Interpretability scores: {len(result.interpretability_scores)}")
        print(f"   - Plots saved: {len(result.plots_saved)}")
        
        # Show some results
        if result.performance_metrics:
            print(f"\n📊 Performance Metrics:")
            for metric, value in result.performance_metrics.items():
                print(f"   - {metric}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_evaluator_pipeline():
    """Test the complete evaluator pipeline"""
    print("\n🧪 Testing Full Evaluator Pipeline")
    print("="*60)
    
    # Prepare test data
    try:
        X_test, y_test = prepare_test_data_matching_training()
    except Exception as e:
        print(f"❌ Failed to prepare test data: {e}")
        return False
    
    # Setup evaluator
    report_path = Path("comprehensive_training_report.json")
    config = AnalysisConfig(
        training_report_path=report_path,
        output_dir=Path("evaluator_test_output"),
        enable_shap=False,
        enable_learning_curves=True,
        enable_residual_analysis=True,
        enable_stability_analysis=True,
        enable_interpretability=True,
        verbose=True
    )
    
    analyzer = ModelAnalyzer(config)
    
    # Run full analysis
    try:
        print("🚀 Running full analysis pipeline...")
        results = analyzer.run_analysis(X_test, y_test)
        
        print(f"✅ Full analysis completed!")
        print(f"📊 Models analyzed: {list(results.keys())}")
        
        # Check output files
        output_dir = Path("evaluator_test_output")
        if output_dir.exists():
            files = list(output_dir.glob("*"))
            print(f"\n📁 Output files created: {len(files)}")
            for file in files[:10]:  # Show first 10 files
                print(f"   📄 {file.name}")
            
            # Check for the main report
            report_file = output_dir / "comprehensive_analysis_report.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    report = json.load(f)
                print(f"\n📋 Analysis Report Summary:")
                print(f"   - Models analyzed: {len(report.get('models_analyzed', []))}")
                print(f"   - Analysis timestamp: {report.get('analysis_timestamp', 'N/A')}")
                print(f"   - Target type: {report.get('target_type', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Fixed Evaluator Integration Test")
    print("="*70)
    
    # Test 1: Single model analysis with proper preprocessing
    success1 = test_evaluator_with_proper_preprocessing()
    
    if success1:
        print("\n" + "="*70)
        # Test 2: Full pipeline
        success2 = test_full_evaluator_pipeline()
        
        if success1 and success2:
            print("\n🎉 All evaluator tests passed!")
            print("✅ Evaluator successfully ingested trainer report data")
            print("✅ Preprocessing consistency maintained")
            print("✅ Analysis pipeline working correctly")
            return 0
        else:
            print("\n❌ Some tests failed")
            return 1
    else:
        print("\n❌ Basic analysis test failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
