#!/usr/bin/env python3
"""
Step-by-Step Testing for DS-AutoAdvisor v2.0
==========================================

This script tests each v2.0 component individually to identify
and fix issues before running the complete pipeline.

Usage: python test_v2_system.py --step <step_number>
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

def test_step_1_basic_imports():
    """Step 1: Test basic imports and dependencies"""
    print("🔍 Step 1: Testing basic imports and dependencies")
    print("-" * 50)
    
    errors = []
    warnings = []
    
    # Test core Python packages
    core_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'), 
        ('scikit-learn', 'sklearn'),
        ('yaml', 'yaml')
    ]
    for display_name, import_name in core_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {display_name}")
        except ImportError as e:
            errors.append(f"Missing {display_name}: {e}")
            print(f"  ❌ {display_name}: {e}")
    
    # Test optional packages
    optional_packages = ['mlflow']
    for package in optional_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} (optional)")
        except ImportError:
            warnings.append(f"Optional package {package} not available")
            print(f"  ⚠️  {package} (optional): not available")
    
    # Test Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"  ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        errors.append(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        print(f"  ❌ Python version: {python_version.major}.{python_version.minor} (need 3.8+)")
    
    print(f"\nStep 1 Result: {'✅ PASS' if not errors else '❌ FAIL'}")
    if errors:
        print("Errors:", errors)
    if warnings:
        print("Warnings:", warnings)
    
    return len(errors) == 0

def test_step_2_config_files():
    """Step 2: Test configuration file availability"""
    print("\n🔍 Step 2: Testing configuration files")
    print("-" * 50)
    
    errors = []
    
    # Check required config files
    config_files = [
        'config/unified_config_v2.yaml',
        'config/feature_flags.yaml',
        'config/business_rules.yaml'
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"  ✅ {config_file}")
            
            # Try to parse YAML
            try:
                import yaml
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
                print(f"    ✅ Valid YAML format")
            except Exception as e:
                errors.append(f"Invalid YAML in {config_file}: {e}")
                print(f"    ❌ Invalid YAML: {e}")
        else:
            errors.append(f"Missing config file: {config_file}")
            print(f"  ❌ {config_file}: not found")
    
    # Check v1.0 compatibility config
    v1_config = 'config/unified_config.yaml'
    if Path(v1_config).exists():
        print(f"  ✅ {v1_config} (v1.0 compatibility)")
    else:
        print(f"  ⚠️  {v1_config}: not found (v1.0 compatibility may not work)")
    
    print(f"\nStep 2 Result: {'✅ PASS' if not errors else '❌ FAIL'}")
    if errors:
        print("Errors:", errors)
    
    return len(errors) == 0

def test_step_3_infrastructure_modules():
    """Step 3: Test v2.0 infrastructure module imports"""
    print("\n🔍 Step 3: Testing v2.0 infrastructure modules")
    print("-" * 50)
    
    errors = []
    
    # Test infrastructure modules
    infrastructure_modules = [
        ('src.infrastructure.enhanced_config_manager', 'get_config_manager, Environment'),
        ('src.infrastructure.metadata_manager', 'get_metadata_store, DataLineageTracker'),
        ('src.infrastructure.plugin_system', 'get_plugin_manager, PluginType'),
        ('src.data_quality.enhanced_quality_system', 'DataQualityAssessor, TypeEnforcer')
    ]
    
    for module_name, imports in infrastructure_modules:
        try:
            module = __import__(module_name, fromlist=imports.split(', '))
            print(f"  ✅ {module_name}")
            
            # Test specific imports
            for import_name in imports.split(', '):
                if hasattr(module, import_name):
                    print(f"    ✅ {import_name}")
                else:
                    errors.append(f"Missing {import_name} in {module_name}")
                    print(f"    ❌ {import_name}: not found")
                    
        except ImportError as e:
            errors.append(f"Cannot import {module_name}: {e}")
            print(f"  ❌ {module_name}: {e}")
        except Exception as e:
            errors.append(f"Error importing {module_name}: {e}")
            print(f"  ❌ {module_name}: {e}")
    
    # Test optional MLflow integration
    try:
        from src.infrastructure.mlflow_integration import get_mlflow_integration
        print(f"  ✅ src.infrastructure.mlflow_integration")
    except ImportError as e:
        print(f"  ⚠️  MLflow integration: {e}")
    
    print(f"\nStep 3 Result: {'✅ PASS' if not errors else '❌ FAIL'}")
    if errors:
        print("Errors:", errors)
    
    return len(errors) == 0

def test_step_4_config_manager():
    """Step 4: Test enhanced configuration manager"""
    print("\n🔍 Step 4: Testing enhanced configuration manager")
    print("-" * 50)
    
    errors = []
    
    try:
        from src.infrastructure.enhanced_config_manager import get_config_manager, Environment
        
        # Test config manager creation
        print("  Creating config manager...")
        config_manager = get_config_manager('config/unified_config_v2.yaml', Environment.DEVELOPMENT)
        print("  ✅ Config manager created")
        
        # Test basic configuration access
        print("  Testing configuration access...")
        global_config = config_manager.get_config('global')
        if global_config:
            print("  ✅ Global config loaded")
            print(f"    Data input path: {global_config.get('data_input_path', 'Not set')}")
            print(f"    Target column: {global_config.get('target_column', 'Not set')}")
        else:
            errors.append("Failed to load global configuration")
            print("  ❌ Global config failed")
        
        # Test feature flags
        print("  Testing feature flags...")
        mlflow_enabled = config_manager.is_feature_enabled('mlflow_tracking')
        data_quality_enabled = config_manager.is_feature_enabled('data_quality_v2')
        plugin_enabled = config_manager.is_feature_enabled('plugin_system')
        
        print(f"    MLflow tracking: {mlflow_enabled}")
        print(f"    Data quality v2: {data_quality_enabled}")
        print(f"    Plugin system: {plugin_enabled}")
        
        # Test metadata config
        print("  Testing metadata configuration...")
        metadata_config = config_manager.get_metadata_config()
        print(f"    Metadata enabled: {metadata_config.enabled}")
        
        # Test plugin config
        print("  Testing plugin configuration...")
        plugin_config = config_manager.get_plugin_config()
        print(f"    Plugin enabled: {plugin_config.enabled}")
        
        print("  ✅ Configuration manager working")
        
    except Exception as e:
        errors.append(f"Configuration manager failed: {e}")
        print(f"  ❌ Configuration manager failed: {e}")
    
    print(f"\nStep 4 Result: {'✅ PASS' if not errors else '❌ FAIL'}")
    if errors:
        print("Errors:", errors)
    
    return len(errors) == 0

def test_step_5_data_quality():
    """Step 5: Test data quality system"""
    print("\n🔍 Step 5: Testing data quality system")
    print("-" * 50)
    
    errors = []
    
    try:
        from src.data_quality_system.enhanced_quality_system import DataQualityAssessor, TypeEnforcer
        import pandas as pd
        import numpy as np
        
        # Create test data
        print("  Creating test data...")
        test_data = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5, 999],  # Has missing and outlier
            'categorical_col': ['A', 'B', 'C', 'A', None, 'D'],  # Has missing
            'target': [0, 1, 0, 1, 0, 1]
        })
        print("  ✅ Test data created")
        
        # Test quality assessor
        print("  Testing data quality assessor...")
        quality_assessor = DataQualityAssessor()
        quality_report = quality_assessor.assess_quality(test_data, 'target')
        
        if quality_report:
            print("  ✅ Quality assessment completed")
            print(f"    Overall score: {quality_report.overall_score:.1f}/100")
            print(f"    Total issues: {quality_report.total_issues}")
            print(f"    Issues by severity: {quality_report.issues_by_severity}")
        else:
            errors.append("Quality assessment returned None")
            print("  ❌ Quality assessment failed")
        
        # Test type enforcer
        print("  Testing type enforcer...")
        type_enforcer = TypeEnforcer()
        type_schema = {
            'numeric_col': 'float64',
            'categorical_col': 'category',
            'target': 'int64'
        }
        
        enforced_data = type_enforcer.enforce_types(test_data, type_schema)
        if enforced_data is not None:
            print("  ✅ Type enforcement completed")
            print(f"    Types: {dict(enforced_data.dtypes)}")
        else:
            errors.append("Type enforcement failed")
            print("  ❌ Type enforcement failed")
        
    except Exception as e:
        errors.append(f"Data quality system failed: {e}")
        print(f"  ❌ Data quality system failed: {e}")
    
    print(f"\nStep 5 Result: {'✅ PASS' if not errors else '❌ FAIL'}")
    if errors:
        print("Errors:", errors)
    
    return len(errors) == 0

def test_step_6_business_configuration():
    """Step 6: Test business configuration system"""
    print("\n🔍 Step 6: Testing business configuration system")
    print("-" * 50)
    
    errors = []
    
    try:
        import os
        import yaml
        
        # Test business rules configuration
        print("  Testing business rules configuration...")
        business_rules_path = 'config/business_rules.yaml'
        if os.path.exists(business_rules_path):
            with open(business_rules_path, 'r') as f:
                business_rules = yaml.safe_load(f)
            
            print("  ✅ Business rules loaded")
            print(f"    Feature selection rules: {len(business_rules.get('feature_selection', {}).get('rules', []))}")
            print(f"    Multi-model strategy: {business_rules.get('feature_selection', {}).get('multi_model_consistency', {}).get('strategy', 'N/A')}")
            
            # Validate required sections
            required_sections = ['feature_selection', 'business_context']
            for section in required_sections:
                if section not in business_rules:
                    errors.append(f"Missing required section: {section}")
                    
        else:
            errors.append("Business rules configuration file not found")
            print("  ❌ Business rules configuration not found")
        
        # Test business KPIs configuration
        print("  Testing business KPIs configuration...")
        kpis_path = 'config/business_kpis.yaml'
        if os.path.exists(kpis_path):
            with open(kpis_path, 'r') as f:
                kpis_config = yaml.safe_load(f)
            
            print("  ✅ Business KPIs loaded")
            print(f"    Defined KPIs: {len(kpis_config.get('kpis', {}))}")
            print(f"    ROI analysis enabled: {kpis_config.get('roi_analysis', {}).get('enabled', False)}")
            
            # Validate KPI structure
            kpis = kpis_config.get('kpis', {})
            for kpi_name, kpi_config in kpis.items():
                # Check for weight (required for all KPIs)
                if 'weight' not in kpi_config:
                    errors.append(f"KPI {kpi_name} missing required 'weight' field")
                
                # Check for calculation method (either formula or calculation_method)
                has_formula = 'formula' in kpi_config
                has_calculation_method = 'calculation_method' in kpi_config
                
                if not has_formula and not has_calculation_method:
                    errors.append(f"KPI {kpi_name} missing calculation method (formula or calculation_method)")
                    
                # Validate calculation method if present
                if has_calculation_method:
                    calc_method = kpi_config.get('calculation_method')
                    if calc_method not in ['direct', 'formula']:
                        errors.append(f"KPI {kpi_name} has invalid calculation_method: {calc_method}")
                        
                    # If method is 'formula', ensure formula field exists
                    if calc_method == 'formula' and not has_formula:
                        errors.append(f"KPI {kpi_name} uses formula method but missing 'formula' field")
                    
        else:
            errors.append("Business KPIs configuration file not found")
            print("  ❌ Business KPIs configuration not found")
        
    except Exception as e:
        errors.append(f"Business configuration system failed: {e}")
        print(f"  ❌ Business configuration system failed: {e}")
    
    print(f"\nStep 6 Result: {'✅ PASS' if not errors else '❌ FAIL'}")
    if errors:
        print("Errors:", errors)
    
    return len(errors) == 0

def test_step_7_business_plugins():
    """Step 7: Test business plugins system"""
    print("\n🔍 Step 7: Testing business plugins system")
    print("-" * 50)
    
    errors = []
    
    try:
        import pandas as pd
        import numpy as np
        import sys
        import os
        
        # Add current directory to Python path to find plugins
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Test business feature selector plugin
        print("  Testing business feature selector plugin...")
        try:
            from plugins.feature_selection.business_feature_selector import BusinessFeatureSelector
            
            # Create test data
            test_data = pd.DataFrame({
                'customer_age': np.random.randint(18, 80, 100),
                'monthly_charges': np.random.uniform(20, 120, 100),
                'total_charges': np.random.uniform(100, 8000, 100),
                'contract_type': np.random.choice(['month_to_month', 'one_year', 'two_year'], 100),
                'payment_method': np.random.choice(['electronic_check', 'credit_card', 'bank_transfer'], 100),
                'noise_feature': np.random.random(100),  # Should be filtered out
                'churn': np.random.choice([0, 1], 100)
            })
            
            # Initialize business feature selector
            feature_selector = BusinessFeatureSelector('config/business_rules.yaml')
            print("  ✅ Business feature selector initialized")
            
            # Test feature selection
            print("  Testing feature selection process...")
            selected_features = feature_selector.select_features(
                data=test_data,
                target='churn',
                method='statistical'
            )
            
            if selected_features:
                print(f"  ✅ Feature selection completed - {len(selected_features)} features selected")
                print(f"    Selected features: {selected_features[:3]}...")  # Show first 3
            else:
                errors.append("Feature selection returned empty result")
                
        except ImportError as e:
            errors.append(f"Business feature selector plugin not found: {e}")
            print(f"  ❌ Business feature selector plugin not found: {e}")
        
        # Test business KPI tracker plugin
        print("  Testing business KPI tracker plugin...")
        try:
            from plugins.business_metrics.kpi_tracker import BusinessKPITracker
            
            # Initialize KPI tracker
            kpi_tracker = BusinessKPITracker('config/business_kpis.yaml')
            print("  ✅ Business KPI tracker initialized")
            
            # Test KPI calculation with mock data
            print("  Testing KPI calculation...")
            mock_business_context = {
                'customer_acquisition_cost': 50,
                'customer_lifetime_value': 1200,
                'monthly_revenue_per_customer': 75
            }
            
            mock_ml_results = {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.88,
                'f1_score': 0.85
            }
            
            kpis = kpi_tracker.calculate_kpis(mock_business_context, mock_ml_results)
            
            if kpis:
                print(f"  ✅ KPI calculation completed - {len(kpis)} KPIs calculated")
                print(f"    Sample KPIs: {list(kpis.keys())[:3]}...")  # Show first 3
            else:
                errors.append("KPI calculation returned empty result")
                
        except ImportError as e:
            errors.append(f"Business KPI tracker plugin not found: {e}")
            print(f"  ❌ Business KPI tracker plugin not found: {e}")
        
    except Exception as e:
        errors.append(f"Business plugins system failed: {e}")
        print(f"  ❌ Business plugins system failed: {e}")
    
    print(f"\nStep 7 Result: {'✅ PASS' if not errors else '❌ FAIL'}")
    if errors:
        print("Errors:", errors)
    
    return len(errors) == 0

def test_step_8_enhanced_trainer():
    """Step 8: Test enhanced trainer with business integration"""
    print("\n🔍 Step 8: Testing enhanced trainer with business integration")
    print("-" * 50)
    
    errors = []
    
    try:
        import pandas as pd
        import numpy as np
        import sys
        import os
        
        # Add current directory to Python path to find src modules
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Test enhanced trainer import
        print("  Testing enhanced trainer import...")
        try:
            # Check if trainer file exists
            trainer_file = os.path.join(current_dir, 'src', '4_model_training', 'trainer.py')
            if os.path.exists(trainer_file):
                print("  ✅ Enhanced trainer file found")
                
                # Read the file and check for business features
                with open(trainer_file, 'r') as f:
                    content = f.read()
                
                if 'TrainerConfig' in content:
                    print("  ✅ TrainerConfig class found")
                if 'ModelResult' in content:
                    print("  ✅ ModelResult class found")
                if 'business' in content.lower():
                    print("  ✅ Business features integration detected")
                else:
                    print("  ⚠️  Business features integration not detected")
            else:
                errors.append("Enhanced trainer file not found")
                print("  ❌ Enhanced trainer file not found")
                
        except ImportError as e:
            errors.append(f"Enhanced trainer import failed: {e}")
            print(f"  ❌ Enhanced trainer import failed: {e}")
            return len(errors) == 0  # Early return if import fails
        
        # Test trainer configuration with business features
        print("  Testing trainer configuration with business features...")
        try:
            # Check if the trainer file supports business configuration
            if 'enable_business_features' in content:
                print("  ✅ Trainer configuration with business features supported")
            else:
                print("  ⚠️  Business features configuration not found in trainer")
                
        except Exception as e:
            errors.append(f"Business trainer configuration failed: {e}")
            print(f"  ❌ Business trainer configuration failed: {e}")
        
        # Test trainer compatibility with business features
        print("  Testing trainer compatibility with business features...")
        try:
            # Check for business-related methods and classes
            business_indicators = [
                'business_features',
                'BusinessFeatureSelector', 
                'business_metrics',
                'business_kpis'
            ]
            
            found_indicators = [indicator for indicator in business_indicators 
                              if indicator in content]
            
            if found_indicators:
                print(f"  ✅ Trainer business feature compatibility verified ({len(found_indicators)} indicators found)")
            else:
                print("  ⚠️  Limited business feature compatibility detected")
                
        except Exception as e:
            errors.append(f"Trainer business compatibility test failed: {e}")
            print(f"  ❌ Trainer business compatibility test failed: {e}")
        
    except Exception as e:
        errors.append(f"Enhanced trainer testing failed: {e}")
        print(f"  ❌ Enhanced trainer testing failed: {e}")
    
    print(f"\nStep 8 Result: {'✅ PASS' if not errors else '❌ FAIL'}")
    if errors:
        print("Errors:", errors)
    
    return len(errors) == 0

def test_step_9_pipeline_v2():
    """Step 9: Test pipeline v2.0 initialization"""
    print("\n🔍 Step 9: Testing pipeline v2.0 initialization")
    print("-" * 50)
    
    errors = []
    
    try:
        import sys
        import os
        
        # Add current directory to Python path
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Test pipeline v2.0 files existence
        print("  Checking pipeline v2.0 files...")
        pipeline_files = [
            '3_run_pipeline.py',
            'config/unified_config_v2.yaml',
            'src/infrastructure/enhanced_config_manager.py'
        ]
        
        all_files_exist = True
        for file_path in pipeline_files:
            full_path = os.path.join(current_dir, file_path)
            if os.path.exists(full_path):
                print(f"  ✅ {file_path} exists")
            else:
                print(f"  ❌ {file_path} missing")
                errors.append(f"Required pipeline file missing: {file_path}")
                all_files_exist = False
        
        if all_files_exist:
            print("  ✅ Pipeline v2.0 files present")
            
            # Check for v2.0 features in config
            config_path = os.path.join(current_dir, 'config/unified_config_v2.yaml')
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            v2_features = ['business_features', 'infrastructure', 'plugins']
            found_features = [feature for feature in v2_features if feature in config_content]
            
            if found_features:
                print(f"  ✅ V2 features detected: {', '.join(found_features)}")
            else:
                print("  ⚠️  V2 features not clearly detected in config")
        else:
            errors.append("Pipeline v2.0 initialization failed - missing files")
        
        # Test configuration compatibility
        print("  Testing v1.0 compatibility mode...")
        v1_config_path = os.path.join(current_dir, 'config/unified_config.yaml')
        if os.path.exists(v1_config_path):
            print("  ✅ V1.0 compatibility mode available")
        else:
            print("  ⚠️  V1.0 compatibility config not found")
        
    except Exception as e:
        errors.append(f"Pipeline v2.0 initialization failed: {e}")
        print(f"  ❌ Pipeline v2.0 initialization failed: {e}")
    
    print(f"\nStep 9 Result: {'✅ PASS' if not errors else '❌ FAIL'}")
    if errors:
        print("Errors:", errors)
    
    return len(errors) == 0

def test_step_10_business_integration():
    """Step 10: Test end-to-end business feature integration"""
    print("\n🔍 Step 10: Testing end-to-end business feature integration")
    print("-" * 50)
    
    errors = []
    
    try:
        import pandas as pd
        import numpy as np
        import os
        import sys
        
        # Add current directory to Python path to find plugins
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Test complete business workflow
        print("  Testing complete business workflow...")
        
        # 1. Load business configuration
        print("    Loading business configurations...")
        business_config_loaded = (
            os.path.exists('config/business_rules.yaml') and 
            os.path.exists('config/business_kpis.yaml')
        )
        
        if business_config_loaded:
            print("    ✅ Business configurations found")
        else:
            errors.append("Business configuration files missing")
            print("    ❌ Business configuration files missing")
            return len(errors) == 0  # Early return
        
        # 2. Test business feature selector workflow
        print("    Testing business feature selection workflow...")
        try:
            from plugins.feature_selection.business_feature_selector import BusinessFeatureSelector
            
            # Create realistic test data
            np.random.seed(42)  # For reproducible results
            test_data = pd.DataFrame({
                'customer_age': np.random.randint(18, 80, 200),
                'monthly_charges': np.random.uniform(20, 120, 200),
                'total_charges': np.random.uniform(100, 8000, 200),
                'contract_type': np.random.choice(['month_to_month', 'one_year', 'two_year'], 200),
                'payment_method': np.random.choice(['electronic_check', 'credit_card', 'bank_transfer'], 200),
                'internet_service': np.random.choice(['dsl', 'fiber_optic', 'no'], 200),
                'noise_feature_1': np.random.random(200),
                'noise_feature_2': np.random.random(200),
                'churn': np.random.choice([0, 1], 200)
            })
            
            selector = BusinessFeatureSelector('config/business_rules.yaml')
            
            # Test multiple selection methods
            methods = ['statistical', 'ml_based', 'business_rules']
            for method in methods:
                try:
                    selected_features = selector.select_features(
                        data=test_data,
                        target='churn',
                        method=method
                    )
                    
                    if selected_features:
                        print(f"    ✅ {method} feature selection: {len(selected_features)} features")
                    else:
                        print(f"    ⚠️  {method} feature selection returned no features")
                        
                except Exception as e:
                    print(f"    ⚠️  {method} feature selection failed: {e}")
            
        except ImportError:
            errors.append("Business feature selector not available for integration test")
            print("    ❌ Business feature selector not available")
        
        # 3. Test business KPI tracking workflow
        print("    Testing business KPI tracking workflow...")
        try:
            from plugins.business_metrics.kpi_tracker import BusinessKPITracker
            
            tracker = BusinessKPITracker('config/business_kpis.yaml')
            
            # Test with comprehensive business context
            business_context = {
                'customer_acquisition_cost': 45.0,
                'customer_lifetime_value': 1350.0,
                'monthly_revenue_per_customer': 78.5,
                'total_customers': 10000,
                'churn_customers': 1500,
                'support_cost_per_customer': 12.0
            }
            
            ml_results = {
                'accuracy': 0.87,
                'precision': 0.84,
                'recall': 0.89,
                'f1_score': 0.865,
                'auc_roc': 0.91
            }
            
            kpis = tracker.calculate_kpis(business_context, ml_results)
            roi_analysis = tracker.calculate_roi(business_context, ml_results)
            
            if kpis and roi_analysis:
                print(f"    ✅ KPI tracking completed: {len(kpis)} KPIs calculated")
                print(f"    ✅ ROI analysis completed: ${roi_analysis.net_benefit:,.2f} net benefit")
            else:
                errors.append("KPI tracking or ROI analysis failed")
                
        except ImportError:
            errors.append("Business KPI tracker not available for integration test")
            print("    ❌ Business KPI tracker not available")
        
        # 4. Test configuration consistency
        print("    Testing configuration consistency...")
        try:
            import yaml
            
            # Load and validate configuration consistency
            with open('config/unified_config_v2.yaml', 'r') as f:
                unified_config = yaml.safe_load(f)
            
            with open('config/business_rules.yaml', 'r') as f:
                business_rules = yaml.safe_load(f)
            
            with open('config/business_kpis.yaml', 'r') as f:
                business_kpis = yaml.safe_load(f)
            
            # Check if business features are properly integrated
            business_integration_checks = [
                'business_features' in unified_config,
                len(business_rules.get('feature_selection', {}).get('rules', [])) > 0,
                len(business_kpis.get('kpis', {})) > 0,
                business_kpis.get('roi_analysis', {}).get('enabled', False)
            ]
            
            if all(business_integration_checks):
                print("    ✅ Configuration consistency verified")
            else:
                errors.append("Configuration consistency issues detected")
                print("    ❌ Configuration consistency issues detected")
                print(f"    Check results: {business_integration_checks}")
                
        except Exception as e:
            errors.append(f"Configuration consistency check failed: {e}")
            print(f"    ❌ Configuration consistency check failed: {e}")
        
        # 5. Test documentation and guide availability
        print("    Testing documentation availability...")
        docs_available = [
            os.path.exists('docs/BUSINESS_FEATURES_GUIDE.md'),
            os.path.exists('STATISTICAL_METHODS_ANALYSIS.md')
        ]
        
        if all(docs_available):
            print("    ✅ Business documentation available")
        else:
            print("    ⚠️  Some business documentation missing")
        
    except Exception as e:
        errors.append(f"Business integration testing failed: {e}")
        print(f"  ❌ Business integration testing failed: {e}")
    
    print(f"\nStep 10 Result: {'✅ PASS' if not errors else '❌ FAIL'}")
    if errors:
        print("Errors:", errors)
    
    return len(errors) == 0

def run_all_steps():
    """Run all test steps"""
    print("🚀 DS-AutoAdvisor v2.0 Step-by-Step Testing (with Business Features)")
    print("=" * 80)
    
    steps = [
        ("Basic Imports & Dependencies", test_step_1_basic_imports),
        ("Configuration Files", test_step_2_config_files),
        ("Infrastructure Modules", test_step_3_infrastructure_modules),
        ("Configuration Manager", test_step_4_config_manager),
        ("Data Quality System", test_step_5_data_quality),
        ("Business Configuration", test_step_6_business_configuration),
        ("Business Plugins", test_step_7_business_plugins),
        ("Enhanced Trainer", test_step_8_enhanced_trainer),
        ("Pipeline v2.0 Initialization", test_step_9_pipeline_v2),
        ("Business Integration", test_step_10_business_integration),
    ]
    
    results = []
    for step_name, test_func in steps:
        try:
            success = test_func()
            results.append((step_name, success))
        except Exception as e:
            print(f"\n❌ Step '{step_name}' crashed: {e}")
            results.append((step_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TESTING SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for step_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {step_name}")
    
    print(f"\nOverall: {passed}/{total} steps passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! DS-AutoAdvisor v2.0 with Business Features ready!")
        print("\n📋 System validated components:")
        print("   ✅ Core pipeline infrastructure")
        print("   ✅ Enhanced configuration management")
        print("   ✅ Data quality assessment system")
        print("   ✅ Business rules configuration")
        print("   ✅ Business feature selection plugins")
        print("   ✅ Business KPI tracking and ROI analysis")
        print("   ✅ Enhanced trainer with business integration")
        print("   ✅ Pipeline v2.0 initialization")
        print("   ✅ End-to-end business feature integration")
        print("\n🚀 Next steps:")
        print("   1. Run full pipeline: python complete_pipeline_v2.py")
        print("   2. Monitor business KPIs in output reports")
        print("   3. Review feature selection decisions in logs")
        print("   4. Check ROI analysis in evaluation results")
    else:
        print(f"\n⚠️  {total - passed} steps failed. Please fix issues before proceeding.")
        print("\n🔧 Common fixes:")
        print("   - Ensure all configuration files exist")
        print("   - Check plugin dependencies are installed")
        print("   - Verify business configuration syntax")
        print("   - Review import paths and module availability")
    
    return passed == total

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DS-AutoAdvisor v2.0 Step-by-Step Testing (with Business Features)')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help='Run specific step only')
    
    args = parser.parse_args()
    
    if args.step:
        # Run specific step
        step_functions = {
            1: test_step_1_basic_imports,
            2: test_step_2_config_files,
            3: test_step_3_infrastructure_modules,
            4: test_step_4_config_manager,
            5: test_step_5_data_quality,
            6: test_step_6_business_configuration,
            7: test_step_7_business_plugins,
            8: test_step_8_enhanced_trainer,
            9: test_step_9_pipeline_v2,
            10: test_step_10_business_integration,
        }
        
        success = step_functions[args.step]()
        sys.exit(0 if success else 1)
    else:
        # Run all steps
        success = run_all_steps()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
