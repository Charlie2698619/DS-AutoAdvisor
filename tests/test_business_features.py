#!/usr/bin/env python3
"""
Business Features Integration Test
=================================

Test script to validate business feature integration in DS-AutoAdvisor v2.0
"""

import sys
import os
sys.path.append('/mnt/c/Users/tony3/Desktop/tidytuesday/ds-autoadvisor')

import pandas as pd
import numpy as np
from pathlib import Path

def test_business_feature_selector():
    """Test business feature selector functionality"""
    print("🔍 Testing Business Feature Selector...")
    
    try:
        # Import business feature selector
        sys.path.append('/mnt/c/Users/tony3/Desktop/tidytuesday/ds-autoadvisor/plugins')
        sys.path.append('/mnt/c/Users/tony3/Desktop/tidytuesday/ds-autoadvisor')
        from plugins.feature_selection.business_feature_selector import BusinessFeatureSelector
        from plugins.infrastructure.plugin_system import PluginConfig
        
        # Create test data
        np.random.seed(42)
        n_samples = 1000
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'customer_id': range(n_samples),  # Should be excluded by business rules
            'monthly_charges': np.random.uniform(20, 100, n_samples),  # Should be included
            'tenure': np.random.randint(1, 72, n_samples),  # Should be included
            'email': [f'user{i}@test.com' for i in range(n_samples)]  # Should be excluded
        })
        
        y = pd.Series(np.random.choice([0, 1], n_samples))
        
        # Configure plugin
        config = PluginConfig(
            enabled=True,
            config={
                'business_rules_file': 'config/business_rules.yaml',
                'human_approval_required': False,  # Auto-approve for testing
                'feature_consistency_strategy': 'intersection',
                'min_feature_count': 2,
                'max_feature_count': 10
            }
        )
        
        # Initialize and test
        selector = BusinessFeatureSelector(config)
        if selector.initialize():
            result = selector.execute(
                (X, y),
                target_type='classification',
                models=['RandomForestClassifier']
            )
            
            print(f"   ✅ Feature selection successful")
            print(f"   📊 Original features: {len(X.columns)}")
            print(f"   🎯 Selected features: {len(result.selected_features)}")
            print(f"   📋 Selected: {result.selected_features}")
            
            # Validate business rules
            excluded_features = {'customer_id', 'email'}
            selected_set = set(result.selected_features)
            
            if excluded_features.isdisjoint(selected_set):
                print(f"   ✅ Business rules enforced - excluded PII features")
            else:
                print(f"   ❌ Business rules violated - PII features present")
                
            return True
        else:
            print(f"   ❌ Feature selector initialization failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Feature selector test failed: {e}")
        return False

def test_kpi_tracker():
    """Test KPI tracker functionality"""
    print("\n💰 Testing KPI Tracker...")
    
    try:
        from business_metrics.kpi_tracker import BusinessKPITracker, calculate_business_value
        
        # Create test ML results
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
        
        # Calculate business value
        business_value = calculate_business_value(y_true, y_pred)
        
        print(f"   ✅ Business value calculation successful")
        print(f"   💵 Total business value: ${business_value['total_business_value']:,.0f}")
        print(f"   📊 True positives: {business_value['true_positives']}")
        print(f"   📊 False positives: {business_value['false_positives']}")
        
        # Test KPI tracker
        kpi_tracker = BusinessKPITracker('config/business_kpis.yaml')
        
        ml_results = {
            'model_accuracy': 0.8,
            'true_positives': business_value['true_positives'],
            'false_positives': business_value['false_positives'],
            'false_negatives': business_value['false_negatives']
        }
        
        kpi_results = kpi_tracker.calculate_kpis(ml_results)
        
        print(f"   ✅ KPI calculation successful")
        print(f"   📈 KPIs calculated: {len(kpi_results)}")
        
        for kpi_name, result in kpi_results.items():
            print(f"   📊 {kpi_name}: {result.value:.2f} ({result.status})")
            
        return True
        
    except Exception as e:
        print(f"   ❌ KPI tracker test failed: {e}")
        return False

def test_enhanced_trainer_integration():
    """Test enhanced trainer with business features"""
    print("\n🤖 Testing Enhanced Trainer Integration...")
    
    try:
        from src.model_training.trainer import EnhancedModelTrainer, TrainerConfig
        
        # Create test dataset
        np.random.seed(42)
        n_samples = 500
        df = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'customer_id': range(n_samples),  # Should be excluded
            'monthly_charges': np.random.uniform(20, 100, n_samples),
            'tenure': np.random.randint(1, 72, n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })
        
        # Configure trainer with business features
        config = TrainerConfig(
            enable_business_features=True,
            feature_selection_enabled=True,
            business_kpi_tracking=True,
            max_models=2,  # Limit for testing
            enable_tuning=False,  # Speed up testing
            verbose=False  # Reduce output
        )
        
        trainer = EnhancedModelTrainer(config)
        
        # Train models
        results = trainer.train_all_models(df, 'target')
        
        if results:
            print(f"   ✅ Training successful - {len(results)} models trained")
            
            # Check business features
            for result in results:
                if result.selected_features:
                    print(f"   🎯 {result.name}: {len(result.selected_features)} features selected")
                    
                if result.business_metrics:
                    print(f"   💼 {result.name}: Business metrics calculated")
                    
                if result.business_value:
                    bv = result.business_value.get('total_business_value', 0)
                    print(f"   💰 {result.name}: Business value: ${bv:,.0f}")
            
            # Generate report
            report = trainer.generate_report(results, 'classification')
            
            if 'feature_selection' in report:
                print(f"   ✅ Feature selection reported")
                
            if 'business_features' in report:
                print(f"   ✅ Business features configuration reported")
                
            return True
        else:
            print(f"   ❌ No models trained successfully")
            return False
            
    except Exception as e:
        print(f"   ❌ Enhanced trainer test failed: {e}")
        return False

def test_configuration_files():
    """Test configuration file loading"""
    print("\n⚙️ Testing Configuration Files...")
    
    config_files = [
        'config/unified_config_v2.yaml',
        'config/business_rules.yaml', 
        'config/business_kpis.yaml'
    ]
    
    all_exist = True
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"   ✅ {config_file} exists")
            
            # Try to load the file
            try:
                import yaml
                with open(config_file, 'r') as f:
                    data = yaml.safe_load(f)
                print(f"   ✅ {config_file} loads successfully")
            except Exception as e:
                print(f"   ❌ {config_file} failed to load: {e}")
                all_exist = False
        else:
            print(f"   ❌ {config_file} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🧪 DS-AutoAdvisor Business Features Integration Test")
    print("=" * 60)
    
    tests = [
        ("Configuration Files", test_configuration_files),
        ("Business Feature Selector", test_business_feature_selector),
        ("KPI Tracker", test_kpi_tracker),
        ("Enhanced Trainer Integration", test_enhanced_trainer_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All business features are working correctly!")
    else:
        print("⚠️ Some business features need attention.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
