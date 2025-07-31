"""
Comprehensive test of Enhanced Trainer with new encoding/scaling features
"""
import pandas as pd
import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline_4.trainer import TrainerConfig, EnhancedModelTrainer

def load_bank_data():
    """Load bank data"""
    data_path = Path(__file__).parent.parent / "data" / "bank.csv"
    if data_path.exists():
        return pd.read_csv(data_path, delimiter=';')
    else:
        raise FileNotFoundError(f"Bank data not found at {data_path}")

def test_encoding_strategies():
    """Test different encoding strategies"""
    print("🧪 Testing Different Encoding Strategies")
    print("="*50)
    
    df = load_bank_data()
    target = 'y'
    
    strategies = ["label", "onehot", "ordinal"]
    
    for strategy in strategies:
        print(f"\n🔧 Testing {strategy} encoding...")
        
        config = TrainerConfig(
            max_models=2,  # Just test 2 models
            encoding_strategy=strategy,
            scaling_strategy="standard",
            enable_tuning=False,  # Faster testing
            save_models=False,
            verbose=True
        )
        
        trainer = EnhancedModelTrainer(config)
        results = trainer.train_all_models(df, target)
        
        if results and any(r for r in results if r):
            valid_results = [r for r in results if r]
            print(f"✅ {strategy} encoding: {len(valid_results)} models trained successfully")
            
            # Generate mini report
            target_type = trainer.infer_target_type(df[target])
            report = trainer.generate_report(valid_results, target_type)
            
            # Check if preprocessing details are in report
            if "preprocessing_details" in report:
                print(f"📊 Encoding in report: {report['preprocessing_details']['encoding_strategy']}")
                print(f"📏 Scaling in report: {report['preprocessing_details']['scaling_strategy']}")
            else:
                print("⚠️ Missing preprocessing details in report")
        else:
            print(f"❌ {strategy} encoding failed")

def test_scaling_strategies():
    """Test different scaling strategies"""
    print("\n🧪 Testing Different Scaling Strategies")
    print("="*50)
    
    df = load_bank_data()
    target = 'y'
    
    strategies = ["standard", "minmax", "robust", "none"]
    
    for strategy in strategies:
        print(f"\n📏 Testing {strategy} scaling...")
        
        config = TrainerConfig(
            max_models=1,  # Just test 1 model
            encoding_strategy="label",  # Keep encoding simple
            scaling_strategy=strategy,
            enable_tuning=False,
            save_models=False,
            verbose=True
        )
        
        trainer = EnhancedModelTrainer(config)
        results = trainer.train_all_models(df, target)
        
        if results and results[0]:
            print(f"✅ {strategy} scaling: Model trained successfully")
        else:
            print(f"❌ {strategy} scaling failed")
    
def test_comprehensive_report():
    """Test comprehensive reporting with all features"""
    print("\n🧪 Testing Comprehensive Report Generation")
    print("="*60)
    
    df = load_bank_data()
    target = 'y'
    
    print(f"📊 Dataset: {df.shape}")
    print(f"🎯 Target: {target} - {df[target].value_counts().to_dict()}")
    
    # Comprehensive config to test all features
    config = TrainerConfig(
        max_models=5,               # Reasonable number for testing
        include_advanced=True,      # Include XGBoost, LightGBM
        include_ensemble=True,      # Test ensemble creation
        enable_tuning=True,        # Enable hyperparameter tuning
        tuning_method="random",    # Faster than grid
        tuning_iterations=10,      # Small number for testing
        encoding_strategy="onehot", # Test complex encoding
        scaling_strategy="standard", # Test scaling
        scoring_strategy="comprehensive", # Full cross-validation
        save_models=True,          # Test model saving
        verbose=True               # Show all output
    )
    
    print(f"⚙️ Config Summary:")
    print(f"   - Models: {config.max_models}")
    print(f"   - Encoding: {config.encoding_strategy}")
    print(f"   - Scaling: {config.scaling_strategy}")
    print(f"   - Tuning: {config.enable_tuning}")
    print(f"   - Advanced models: {config.include_advanced}")
    
    # Train models
    trainer = EnhancedModelTrainer(config)
    results = trainer.train_all_models(df, target)
    
    # Generate comprehensive report
    if results:
        print(f"\n📋 Generating comprehensive report...")
        target_type = trainer.infer_target_type(df[target])
        report = trainer.generate_report(results, target_type, "comprehensive_training_report.json")
        
        # Validate report structure
        print(f"\n🔍 Report Validation:")
        required_sections = ["summary", "preprocessing_details", "model_rankings", "detailed_results", "config_used"]
        
        for section in required_sections:
            if section in report:
                print(f"   ✅ {section}: Present")
            else:
                print(f"   ❌ {section}: Missing")
        
        # Check preprocessing details specifically
        if "preprocessing_details" in report:
            prep_details = report["preprocessing_details"]
            print(f"\n📊 Preprocessing Details:")
            print(f"   - Encoding Strategy: {prep_details.get('encoding_strategy', 'N/A')}")
            print(f"   - Scaling Strategy: {prep_details.get('scaling_strategy', 'N/A')}")
            print(f"   - Scaling Method: {prep_details.get('scaling_method', 'N/A')}")
            print(f"   - Pipeline Structure: {prep_details.get('pipeline_structure', 'N/A')}")
        
        # Show model rankings
        if "model_rankings" in report and report["model_rankings"]:
            print(f"\n🏅 Top 3 Models:")
            for i, model in enumerate(report["model_rankings"][:3]):
                print(f"   {model['rank']}. {model['model']}: {model['primary_score']:.4f} ({model['training_time']:.2f}s)")
        
        print(f"\n💾 Comprehensive report saved to: comprehensive_training_report.json")
        
        # Show summary stats
        valid_results = [r for r in results if r]
        failed_results = [r for r in results if not r]
        
        print(f"\n🎉 Training Summary:")
        print(f"   ✅ Successful models: {len(valid_results)}")
        print(f"   ❌ Failed models: {len(failed_results)}")
        print(f"   📊 Total training time: {sum(r.training_time for r in valid_results):.2f}s")
        
        if valid_results:
            best = max(valid_results, key=lambda x: list(x.scores.values())[0] if x.scores else 0)
            print(f"   🏆 Best model: {best.name}")
            if best.scores:
                best_metric, best_value = list(best.scores.items())[0]
                print(f"   📈 Best score: {best_metric} = {best_value:.4f}")
        
        return True
    else:
        print("❌ No models were trained successfully!")
        return False

def test_all_models():
    """Test training all available models (legacy test)"""
    print("🧪 Testing Enhanced Trainer - ALL MODELS")
    print("="*60)
    
    # Load data
    df = load_bank_data()
    target = 'y'
    
    print(f"📊 Dataset: {df.shape}")
    print(f"🎯 Target: {target} - {df[target].value_counts().to_dict()}")
    
    # Simple config for ALL models
    config = TrainerConfig(
        max_models=100,          # Large number = train all available
        include_advanced=True,   # Include XGBoost, LightGBM
        enable_tuning=True,      # Enable hyperparameter tuning
        save_models=True,        # Save trained models
        verbose=True             # Show progress
    )
    
    print(f"⚙️ Config: max_models={config.max_models}, tuning={config.enable_tuning}")
    
    # Train all models
    trainer = EnhancedModelTrainer(config)
    results = trainer.train_all_models(df, target)
    
    # Generate report
    if results:
        print(f"\n📋 Generating comprehensive report...")
        target_type = trainer.infer_target_type(df[target])
        report = trainer.generate_report(results, target_type, "training_report.json")
        
        print(f"💾 Report saved to: training_report.json")
    
    # Simple results summary
    if results:
        print(f"\n🎉 Training completed!")
        print(f"✅ Models trained: {len([r for r in results if r])}")
        print(f"❌ Models failed: {len([r for r in results if not r])}")
        
        # Show best model
        valid_results = [r for r in results if r]
        if valid_results:
            best = max(valid_results, key=lambda x: list(x.scores.values())[0] if x.scores else 0)
            print(f"🏆 Best model: {best.name}")
            print(f"📊 Best score: {list(best.scores.items())[0] if best.scores else 'N/A'}")
    else:
        print("❌ Training failed!")

def test_single_model():
    """Quick single model test"""
    print("🧪 Testing Single Model")
    print("="*40)
    
    df = load_bank_data()
    target = 'y'
    
    # Minimal config
    config = TrainerConfig(
        max_models=1,
        enable_tuning=False,
        save_models=False,
        verbose=True
    )
    
    trainer = EnhancedModelTrainer(config)
    results = trainer.train_all_models(df, target)
    
    if results and results[0]:
        result = results[0]
        print(f"✅ {result.name} - Score: {list(result.scores.items())[0] if result.scores else 'N/A'}")
    else:
        print("❌ Single model test failed!")

def main():
    """Main function with comprehensive testing options"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Enhanced Trainer")
    parser.add_argument("--all", action="store_true", help="Train all available models")
    parser.add_argument("--quick", action="store_true", help="Quick single model test")
    parser.add_argument("--comprehensive", action="store_true", help="Test comprehensive features and report generation")
    parser.add_argument("--encoding", action="store_true", help="Test all encoding strategies")
    parser.add_argument("--scaling", action="store_true", help="Test all scaling strategies")
    parser.add_argument("--full-test", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    try:
        if args.full_test:
            print("🚀 Running Full Test Suite")
            print("="*60)
            test_encoding_strategies()
            test_scaling_strategies()
            test_comprehensive_report()
            test_single_model()
            print("\n🎉 All tests completed!")
            
        elif args.comprehensive:
            success = test_comprehensive_report()
            if success:
                print("\n✅ Comprehensive test passed!")
            else:
                print("\n❌ Comprehensive test failed!")
                
        elif args.encoding:
            test_encoding_strategies()
            
        elif args.scaling:
            test_scaling_strategies()
            
        elif args.quick:
            test_single_model()
            
        elif args.all:
            test_all_models()
            
        else:
            # Default: run comprehensive test
            print("🔧 Running comprehensive test (default)")
            print("Use --help to see all options")
            print()
            test_comprehensive_report()
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)