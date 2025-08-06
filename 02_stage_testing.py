#!/usr/bin/env python3
"""
🧪 DS-AutoAdvisor: Step 2 - Modular Stage Testing
===============================================

WHAT IT DOES:
Tests individual pipeline stages in isolation for rapid development and debugging.
No redundant data loading or profiling - uses results from 01_data_discovery.py.
Fully configurable via unified_config_v2.yaml with testing mode overrides.

WHEN TO USE:
- After completing data discovery (Step 1)
- During development and debugging
- When testing specific pipeline components
- For rapid iteration on cleaning/training parameters

HOW TO USE:
Test specific stage:
    python 02_stage_testing.py --stage cleaning
    python 02_stage_testing.py --stage training --mode comprehensive
    python 02_stage_testing.py --stage evaluation --mode custom

Test multiple stages:
    python 02_stage_testing.py --stage cleaning,training --mode fast

Test all stages sequentially:
    python 02_stage_testing.py --stage all --mode comprehensive

Use discovery results:
    python 02_stage_testing.py --stage all --discovery-dir pipeline_outputs/01_discovery_dataset_20250804_143022

TESTING MODES:
- fast: Maximum speed - minimal models, reduced analysis (default)
- comprehensive: Thorough testing - more models, complete analysis
- custom: Use exact unified_config_v2.yaml settings without overrides

CONFIGURATION:
All settings controlled via unified_config_v2.yaml:
- stage_testing: Testing-specific overrides
- step_XX_*: Pipeline step configurations  
- global: Core pipeline settings

AVAILABLE STAGES:
- cleaning: Test data cleaning with YAML configuration
- advisory: Test ML advisory and model recommendations  
- training: Test model training with configurable limits
- evaluation: Test model evaluation and analysis
- all: Run all stages in sequence

INTERACTIVE FEATURES:
- Stage-by-stage execution with checkpoints
- Interactive parameter adjustment via config
- Quick iteration without full pipeline overhead
- Detailed stage-specific reporting
- Configuration validation and display

OUTPUTS:
✅ Stage-specific test results and logs
✅ Cleaned data for downstream testing
✅ Model artifacts for evaluation testing
✅ Performance metrics and timing analysis
✅ Configuration summary and validation report

NEXT STEP:
After successful stage testing, run: python 03_full_pipeline.py
"""

import sys
import os
import csv
import traceback
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

class ModularStageTester:
    """Modular testing of individual pipeline stages"""
    
    def __init__(self, discovery_dir: str = None, output_base: str = "pipeline_outputs", testing_mode: str = "fast"):
        """Initialize stage tester"""
        self.output_base = Path(output_base)
        self.discovery_dir = Path(discovery_dir) if discovery_dir else self._find_latest_discovery()
        self.testing_mode = testing_mode  # "fast", "comprehensive", "custom"
        
        # Create organized output structure
        self.testing_outputs = self._create_output_structure()
        
        # Load discovery results and configuration
        self.discovery_summary = self._load_discovery_results()
        self.config = self._load_configuration()
        
        # Apply testing mode overrides
        self._apply_testing_mode_overrides()
        
        # Display configuration source
        self._display_config_info()
        
        # Stage execution history
        self.stage_results = {}
        self.execution_log = []
    
    def _apply_testing_mode_overrides(self):
        """Apply testing mode specific overrides to configuration"""
        if self.testing_mode == "fast":
            # FAST MODE: Turn off most parameters for maximum speed testing
            stage_testing = self.config.setdefault('stage_testing', {})
            
            # Training overrides - minimal models, no advanced features
            training_config = stage_testing.setdefault('training', {})
            training_config.update({
                'max_models': 2,
                'include_ensemble': False,
                'include_advanced': False,
                'enable_tuning': False
            })
            
            # Evaluation overrides - disable expensive analysis
            eval_config = stage_testing.setdefault('evaluation', {})
            eval_config.update({
                'enable_shap': False,
                'enable_stability_analysis': False,
                'enable_interpretability': False,
                'enable_uncertainty_analysis': False,
                'enable_learning_curves': True,  # Keep basic analysis
                'enable_residual_analysis': True,  # Keep basic analysis
                'n_permutations': 5,  # Minimal permutations
                'max_shap_samples': 50,  # Minimal SHAP samples
                'plot_format': 'html'  # Fast format
            })
            
            print(f"⚡ Fast testing mode applied - minimal parameters for speed")
            
        elif self.testing_mode == "comprehensive":
            # COMPREHENSIVE MODE: Turn on most parameters for thorough testing
            stage_testing = self.config.setdefault('stage_testing', {})
            
            # Training overrides - more models, all advanced features
            training_config = stage_testing.setdefault('training', {})
            training_config.update({
                'max_models': 5,
                'include_ensemble': True,
                'include_advanced': True,
                'enable_tuning': True
            })
            
            # Evaluation overrides - enable all analysis types
            eval_config = stage_testing.setdefault('evaluation', {})
            eval_config.update({
                'enable_shap': True,
                'enable_stability_analysis': True,
                'enable_interpretability': True,
                'enable_uncertainty_analysis': True,
                'enable_learning_curves': True,
                'enable_residual_analysis': True,
                'n_permutations': 30,  # More thorough permutations
                'max_shap_samples': 1000,  # More SHAP samples
                'plot_format': 'both'  # Save both HTML and PNG
            })
            
            print(f"🔬 Comprehensive testing mode applied - all features enabled")
            
        # CUSTOM MODE: Use exact YAML config values without any overrides
        elif self.testing_mode == "custom":
            print(f"⚙️ Custom testing mode - using exact YAML configuration values")
            # No overrides applied - configuration loaded directly from YAML
            
            # Show what's currently configured in YAML
            stage_testing = self.config.get('stage_testing', {})
            training_config = stage_testing.get('training', {})
            eval_config = stage_testing.get('evaluation', {})
            
            print(f"   📋 Training config: max_models={training_config.get('max_models', 'from fallback')}, "
                  f"ensemble={training_config.get('include_ensemble', 'from fallback')}, "
                  f"advanced={training_config.get('include_advanced', 'from fallback')}")
            print(f"   📋 Evaluation config: shap={eval_config.get('enable_shap', 'from fallback')}, "
                  f"stability={eval_config.get('enable_stability_analysis', 'from fallback')}, "
                  f"interpretability={eval_config.get('enable_interpretability', 'from fallback')}")
        
        else:
            print(f"⚠️ Unknown testing mode: {self.testing_mode}. Using default behavior.")
    
    def _display_config_info(self):
        """Display information about the configuration being used"""
        print(f"\n📋 Configuration Summary:")
        
        # Show main settings
        global_config = self.config.get('global', {})
        print(f"   Project: {global_config.get('project_name', 'DS-AutoAdvisor')}")
        print(f"   Version: {global_config.get('version', 'Unknown')}")
        print(f"   Environment: {global_config.get('environment', 'development')}")
        print(f"   Data input: {global_config.get('data_input_path', 'Not specified')}")
        print(f"   Target column: {global_config.get('target_column', 'Not specified')}")
        print(f"   Random state: {global_config.get('random_state', 'Not set')}")
        print(f"   Testing mode: {self.testing_mode.upper()}")
        
        # Show current testing mode configuration
        stage_testing_config = self.config.get('stage_testing', {})
        if stage_testing_config:
            training_config = stage_testing_config.get('training', {})
            eval_config = stage_testing_config.get('evaluation', {})
            
            print(f"\n🧪 Testing Mode: {self.testing_mode.upper()}")
            if self.testing_mode == "fast":
                print(f"   ⚡ Speed optimized - minimal models and analysis")
            elif self.testing_mode == "comprehensive":
                print(f"   🔬 Full featured - all models and analysis enabled")
            elif self.testing_mode == "custom":
                print(f"   ⚙️  User configured - respecting YAML settings")
            
            print(f"   Training: max_models={training_config.get('max_models', 'default')}, "
                  f"ensemble={training_config.get('include_ensemble', 'default')}, "
                  f"tuning={training_config.get('enable_tuning', 'default')}")
            print(f"   Evaluation: shap={eval_config.get('enable_shap', 'default')}, "
                  f"stability={eval_config.get('enable_stability_analysis', 'default')}, "
                  f"interpretability={eval_config.get('enable_interpretability', 'default')}")
        
        # Check if unified config sections are available
        config_sections = {
            'Data Discovery': 'step_01_discovery',
            'Data Cleaning': 'step_02_cleaning', 
            'ML Advisory': 'step_03_advisory',
            'Model Training': 'step_04_training',
            'Model Evaluation': 'step_05_evaluation'
        }
        
        available_sections = []
        for name, key in config_sections.items():
            if key in self.config:
                available_sections.append(name)
        
        if available_sections:
            print(f"   Available config sections: {', '.join(available_sections)}")
        
        print()
    
    def _find_latest_discovery(self) -> Optional[Path]:
        """Find the latest discovery directory"""
        discovery_dirs = list(self.output_base.glob("01_discovery_*"))
        if discovery_dirs:
            # Sort by creation time and get the latest
            latest = max(discovery_dirs, key=lambda p: p.stat().st_mtime)
            print(f"🔍 Using latest discovery: {latest}")
            return latest
        else:
            print("❌ No discovery directory found. Run 01_data_discovery.py first.")
            return None
    
    def _create_output_structure(self) -> Dict[str, Path]:
        """Create organized output directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        outputs = {
            'base': self.output_base,
            'testing': self.output_base / f"02_stage_testing_{timestamp}",
            'cleaned_data': self.output_base / f"02_stage_testing_{timestamp}" / "cleaned_data",
            'models': self.output_base / f"02_stage_testing_{timestamp}" / "models",
            'evaluations': self.output_base / f"02_stage_testing_{timestamp}" / "evaluations",
            'logs': self.output_base / f"02_stage_testing_{timestamp}" / "logs",
            'reports': self.output_base / f"02_stage_testing_{timestamp}" / "reports"
        }
        
        # Create all directories
        for output_dir in outputs.values():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        return outputs
    
    def _load_discovery_results(self) -> Dict[str, Any]:
        """Load results from data discovery stage"""
        if not self.discovery_dir or not self.discovery_dir.exists():
            print("❌ Discovery directory not found")
            return {}
        
        summary_file = self.discovery_dir / "discovery_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        else:
            print("⚠️ Discovery summary not found, creating minimal configuration")
            return {}
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load pipeline configuration from unified_config_v2.yaml"""
        # Try to load from discovery results first
        if self.discovery_summary and 'generated_files' in self.discovery_summary:
            config_path = self.discovery_summary['generated_files'].get('pipeline_config')
            if config_path and Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    print(f"📋 Using discovery config: {config_path}")
                    return config
        
        # Use unified config as primary source
        unified_config_path = project_root / "config" / "unified_config_v2.yaml"
        if unified_config_path.exists():
            with open(unified_config_path, 'r') as f:
                config = yaml.safe_load(f)
                print(f"📋 Using unified config: {unified_config_path}")
                return config
        
        # Create minimal config as fallback
        print("⚠️ No configuration found, creating minimal config")
        return self._create_minimal_config()
    
    def _create_minimal_config(self) -> Dict[str, Any]:
        """Create minimal configuration for testing"""
        return {
            'global': {
                'data_input_path': 'data/telco_churn_data.csv',
                'target_column': 'Churn',
                'csv_delimiter': ',',
                'csv_encoding': 'utf-8',
                'output_base_dir': str(self.output_base)
            }
        }
    
    def _detect_target_column(self, df: pd.DataFrame, original_target: str) -> Optional[str]:
        """Detect the actual target column name after data cleaning transformations"""
        # Try original name first
        if original_target in df.columns:
            return original_target
        
        # Check if config has target column variants
        if 'target_column_variants' in self.config.get('global', {}):
            variants = self.config['global']['target_column_variants']
            for variant in variants:
                if variant in df.columns:
                    print(f"🔄 Found target column variant: {variant} (original: {original_target})")
                    return variant
        
        # Try common transformations of the original target name
        common_transformations = [
            f"{original_target}_binary_0",
            f"{original_target}_encoded", 
            f"{original_target}_label",
            f"{original_target}_target",
            original_target.lower(),
            original_target.upper()
        ]
        
        for transformed_name in common_transformations:
            if transformed_name in df.columns:
                print(f"🔄 Auto-detected target column: {transformed_name} (original: {original_target})")
                return transformed_name
        
        return None
    
    def test_stage(self, stage_name: str) -> bool:
        """Test a specific pipeline stage"""
        print(f"\n🧪 Testing Stage: {stage_name.upper()}")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Route to appropriate testing method
            if stage_name == "cleaning":
                success = self._test_cleaning_stage()
            elif stage_name == "advisory":
                success = self._test_advisory_stage()
            elif stage_name == "training":
                success = self._test_training_stage()
            elif stage_name == "evaluation":
                success = self._test_evaluation_stage()
            else:
                print(f"❌ Unknown stage: {stage_name}")
                return False
            
            # Log execution
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.execution_log.append({
                'stage': stage_name,
                'success': success,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            })
            
            if success:
                print(f"✅ Stage '{stage_name}' completed successfully ({execution_time:.2f}s)")
            else:
                print(f"❌ Stage '{stage_name}' failed ({execution_time:.2f}s)")
            
            return success
            
        except Exception as e:
            print(f"❌ Stage '{stage_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _test_cleaning_stage(self) -> bool:
        """Test data cleaning stage"""
        print("🧹 Testing Data Cleaning Stage")
        
        try:
            # Get cleaning configuration from unified config
            cleaning_config_path = None
            if self.discovery_summary and 'generated_files' in self.discovery_summary:
                cleaning_config_path = self.discovery_summary['generated_files'].get('cleaning_config')
            
            if not cleaning_config_path or not Path(cleaning_config_path).exists():
                print("⚠️ Cleaning config not found, using unified config settings")
                cleaning_config_path = None
            
            # Get data path from unified config
            data_path = self.config['global']['data_input_path']
            if not Path(data_path).exists():
                print(f"❌ Data file not found: {data_path}")
                return False
            
            # Import and configure cleaner
            try:
                sys.path.append(str(project_root / "src" / "2_data_cleaning"))
                from data_cleaner import DataCleaner, CleaningConfig
                
                # Create cleaning configuration using unified config settings
                output_path = self.testing_outputs['cleaned_data'] / "test_cleaned_data.csv"
                log_path = self.testing_outputs['logs'] / "cleaning_test_log.json"
                
                # Get cleaning settings from unified config
                stage_cleaning_config = self.config.get('stage_testing', {}).get('cleaning', {})
                step_cleaning_config = self.config.get('step_02_cleaning', {})
                
                cleaning_config = CleaningConfig(
                    input_path=data_path,
                    output_path=str(output_path),
                    log_path=str(log_path),
                    column_config_path=cleaning_config_path,
                    verbose=stage_cleaning_config.get('verbose', True),
                    # Test settings from unified config
                    remove_duplicates=step_cleaning_config.get('execution', {}).get('validate_before_cleaning', True),
                    outlier_removal=True,
                    outlier_method="iqr"
                )
                
                print(f"📊 Input data: {data_path}")
                print(f"📤 Output data: {output_path}")
                print(f"📋 Log file: {log_path}")
                if cleaning_config_path:
                    print(f"⚙️ Config template: {cleaning_config_path}")
                print(f"⚙️ Using unified config settings")
                
                # Run cleaning
                print("🔄 Running data cleaning...")
                cleaner = DataCleaner(cleaning_config)
                
                # Show cleaning steps
                steps = cleaner.list_steps()
                print(f"📋 Cleaning steps: {steps}")
                
                # Execute cleaning
                cleaned_df, cleaning_log = cleaner.clean()
                
                # Display results
                print(f"\n✅ Data cleaning completed successfully!")
                
                initial_shape = cleaning_log.get('initial_shape', (0, 0))
                final_shape = cleaning_log.get('final_shape', (0, 0))
                processing_time = cleaning_log.get('processing_time', 0)
                
                print(f"📊 Cleaning Results:")
                print(f"   Original shape: {initial_shape}")
                print(f"   Cleaned shape:  {final_shape}")
                print(f"   Rows removed:   {initial_shape[0] - final_shape[0]:,}")
                print(f"   Columns removed: {initial_shape[1] - final_shape[1]}")
                print(f"   Processing time: {processing_time:.2f}s")
                
                # Show actions performed
                actions = cleaning_log.get('actions', [])
                if actions:
                    print(f"\n🔧 Actions Performed ({len(actions)}):")
                    for i, action in enumerate(actions[:10], 1):
                        print(f"   {i:2d}. {action}")
                    if len(actions) > 10:
                        print(f"   ... and {len(actions) - 10} more actions")
                
                # Show warnings
                warnings = cleaning_log.get('warnings', [])
                if warnings:
                    print(f"\n⚠️ Warnings ({len(warnings)}):")
                    for warning in warnings[:5]:
                        print(f"   • {warning}")
                
                # Sample cleaned data
                print(f"\n📝 Cleaned Data Sample:")
                sample_cols = min(6, len(cleaned_df.columns))
                print(cleaned_df.head(3).iloc[:, :sample_cols].to_string(index=False))
                if len(cleaned_df.columns) > sample_cols:
                    print(f"   ... and {len(cleaned_df.columns) - sample_cols} more columns")
                
                # Store results for next stages
                self.stage_results['cleaning'] = {
                    'output_path': str(output_path),
                    'log_path': str(log_path),
                    'shape': final_shape,
                    'processing_time': processing_time,
                    'success': True
                }
                
                return True
                
            except ImportError as e:
                print(f"❌ Data cleaner not available: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Cleaning stage test failed: {e}")
            return False
    
    def _test_advisory_stage(self) -> bool:
        """Test ML advisory stage"""
        print("🤖 Testing ML Advisory Stage")
        
        try:
            # Get cleaned data path
            cleaned_data_path = None
            if 'cleaning' in self.stage_results:
                cleaned_data_path = self.stage_results['cleaning']['output_path']
            else:
                # Use original data if cleaning wasn't run
                cleaned_data_path = self.config['global']['data_input_path']
            
            if not Path(cleaned_data_path).exists():
                print(f"❌ Data file not found: {cleaned_data_path}")
                print("💡 Run cleaning stage first: python 02_stage_testing.py --stage cleaning")
                return False
            
            # Load data for advisory
            print(f"📊 Loading data: {cleaned_data_path}")
            df = pd.read_csv(cleaned_data_path)
            
            # Detect the actual target column name
            original_target = self.config['global']['target_column']
            actual_target = self._detect_target_column(df, original_target)
            
            if actual_target is None:
                print(f"❌ Target column '{original_target}' not found in data")
                print(f"Available columns: {list(df.columns)}")
                return False
            
            print(f"🎯 Target column: {actual_target}")
            if actual_target != original_target:
                print(f"   (transformed from original: {original_target})")
            print(f"📊 Data shape: {df.shape}")
            
            # Run advisory analysis
            try:
                sys.path.append(str(project_root / "src" / "3_advisory"))
                from assumption_checker import EnhancedAssumptionChecker, AssumptionConfig
                from model_recommender import recommend_model
                
                # Get advisory configuration from unified config
                stage_advisory_config = self.config.get('stage_testing', {}).get('advisory', {})
                step_advisory_config = self.config.get('step_03_advisory', {})
                assumption_config = stage_advisory_config.get('assumption_testing', step_advisory_config.get('assumption_testing', {}))
                
                # Configure assumption checker using unified config
                advisory_config = AssumptionConfig(
                    verbose=assumption_config.get('verbose', True),
                    generate_recommendations=assumption_config.get('generate_recommendations', True),
                    normality_alpha=assumption_config.get('normality_alpha', 0.05),
                    homo_alpha=assumption_config.get('homo_alpha', 0.05),
                    vif_threshold=assumption_config.get('vif_threshold', 10.0),
                    correlation_threshold=assumption_config.get('correlation_threshold', 0.95)
                )
                
                print("🔄 Running ML assumptions analysis...")
                print(f"⚙️ Using unified config settings:")
                print(f"   Normality alpha: {advisory_config.normality_alpha}")
                print(f"   VIF threshold: {advisory_config.vif_threshold}")
                print(f"   Correlation threshold: {advisory_config.correlation_threshold}")
                
                # Prepare data
                X = df.drop(columns=[actual_target])
                y = df[actual_target]
                
                # Run assumption checking using the correct method
                checker = EnhancedAssumptionChecker(advisory_config)
                assumptions_results = checker.run_all_checks(df, actual_target)
                
                print(f"✅ Assumptions analysis completed")
                print(f"📊 Assumptions checked: {len(assumptions_results)}")
                
                # Determine target type for model recommendations
                if y.dtype in ['object', 'category'] or y.nunique() < 20:
                    target_type = "classification"
                    print(f"� Problem type: Classification ({y.nunique()} classes)")
                else:
                    target_type = "regression"
                    print(f"📊 Problem type: Regression")
                
                # Get model recommendations using unified config settings
                model_rec_config = stage_advisory_config.get('model_recommendation', step_advisory_config.get('model_recommendation', {}))
                print("🔄 Generating model recommendations...")
                print(f"⚙️ Include ensemble suggestions: {model_rec_config.get('include_ensemble_suggestions', True)}")
                
                recommendations = recommend_model(
                    assumptions=assumptions_results,
                    target_type=target_type
                )
                
                print(f"✅ Model recommendations generated")
                
                # Display results
                print(f"\n📊 Advisory Results:")
                
                # Show assumption results summary - handle the actual structure from run_all_checks
                if isinstance(assumptions_results, dict):
                    # Count passed assumptions from the detailed results
                    passed_count = 0
                    total_count = 0
                    failed_assumptions = []
                    
                    # Check each assumption type
                    assumption_types = ['normality', 'homoscedasticity', 'multicollinearity', 'linearity', 'independence', 'class_balance']
                    for assumption_name in assumption_types:
                        if assumption_name in assumptions_results and assumptions_results[assumption_name] is not None:
                            total_count += 1
                            result = assumptions_results[assumption_name]
                            if isinstance(result, dict) and result.get('passed', False):
                                passed_count += 1
                            else:
                                failed_assumptions.append(assumption_name)
                    
                    print(f"   Assumptions passed: {passed_count}/{total_count}")
                    
                    if failed_assumptions:
                        print(f"   Failed assumptions: {', '.join(failed_assumptions)}")
                    
                    # Show any warnings or errors
                    if 'warnings' in assumptions_results and assumptions_results['warnings']:
                        print(f"   ⚠️ Warnings: {len(assumptions_results['warnings'])}")
                        for warning in assumptions_results['warnings'][:3]:
                            print(f"      • {warning}")
                    
                    if 'errors' in assumptions_results and assumptions_results['errors']:
                        print(f"   ❌ Errors: {len(assumptions_results['errors'])}")
                        for error in assumptions_results['errors'][:3]:
                            print(f"      • {error}")
                else:
                    passed_count = 0
                    total_count = 0
                    failed_assumptions = []
                    print("   Unable to parse assumption results")
                
                # Show top recommendations
                if isinstance(recommendations, dict):
                    if 'recommended_models' in recommendations:
                        models = recommendations['recommended_models']
                        if isinstance(models, list):
                            print(f"\n🎯 Model Recommendations:")
                            for i, model in enumerate(models[:3], 1):
                                if isinstance(model, dict):
                                    model_name = model.get('name', str(model))
                                    confidence = model.get('confidence', 'N/A')
                                    print(f"   {i}. {model_name} (confidence: {confidence})")
                                else:
                                    print(f"   {i}. {model}")
                    elif 'recommended_model' in recommendations:
                        model_name = recommendations.get('recommended_model', 'Unknown')
                        reason = recommendations.get('reason', 'No reason provided')
                        print(f"\n🎯 Recommended Model: {model_name}")
                        print(f"   Reason: {reason}")
                
                # Save advisory results to the reports directory
                advisory_results = {
                    'timestamp': datetime.now().isoformat(),
                    'data_shape': df.shape,
                    'target_column': actual_target,
                    'target_type': target_type,
                    'assumptions_results': assumptions_results,
                    'model_recommendations': recommendations,
                    'summary': {
                        'assumptions_passed': passed_count,
                        'total_assumptions': total_count,
                        'failed_assumptions': failed_assumptions,
                        'problem_type': target_type
                    }
                }
                
                advisory_output_path = self.testing_outputs['reports'] / "advisory_test_results.json"
                with open(advisory_output_path, 'w') as f:
                    json.dump(advisory_results, f, indent=2, default=str)
                
                print(f"📄 Advisory report: {advisory_output_path}")
                
                # Store results for next stages
                self.stage_results['advisory'] = {
                    'results_path': str(advisory_output_path),
                    'recommendations': recommendations,
                    'assumptions_passed': passed_count,
                    'target_type': target_type,
                    'success': True
                }
                
                return True
                
            except ImportError as e:
                print(f"❌ Advisory components not available: {e}")
                print("💡 Using basic advisory fallback...")
                return self._basic_advisory_fallback(df, actual_target)
                
        except Exception as e:
            print(f"❌ Advisory stage test failed: {e}")
            return False
    
    def _basic_advisory_fallback(self, df: pd.DataFrame, target_column: str) -> bool:
        """Basic advisory fallback when full components not available"""
        try:
            print("🔄 Running basic advisory analysis...")
            
            # Basic target analysis
            y = df[target_column]
            X = df.drop(columns=[target_column])
            
            # Determine problem type
            if y.dtype in ['object', 'category'] or y.nunique() < 20:
                problem_type = "classification"
                print(f"📊 Problem type: Classification ({y.nunique()} classes)")
            else:
                problem_type = "regression"
                print(f"📊 Problem type: Regression")
            
            # Basic data analysis
            print(f"📊 Feature analysis:")
            print(f"   Total features: {len(X.columns)}")
            print(f"   Numeric features: {len(X.select_dtypes(include=[np.number]).columns)}")
            print(f"   Categorical features: {len(X.select_dtypes(include=['object', 'category']).columns)}")
            print(f"   Missing values: {X.isnull().sum().sum()}")
            
            # Basic recommendations
            if problem_type == "classification":
                recommended_models = ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"]
            else:
                recommended_models = ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"]
            
            print(f"\n🎯 Basic Model Recommendations:")
            for i, model in enumerate(recommended_models, 1):
                print(f"   {i}. {model}")
            
            # Save basic results
            basic_results = {
                'timestamp': datetime.now().isoformat(),
                'problem_type': problem_type,
                'data_shape': df.shape,
                'feature_summary': {
                    'total_features': len(X.columns),
                    'numeric_features': len(X.select_dtypes(include=[np.number]).columns),
                    'categorical_features': len(X.select_dtypes(include=['object', 'category']).columns),
                    'missing_values': int(X.isnull().sum().sum())
                },
                'recommended_models': recommended_models
            }
            
            advisory_output_path = self.testing_outputs['reports'] / "basic_advisory_results.json"
            with open(advisory_output_path, 'w') as f:
                json.dump(basic_results, f, indent=2, default=str)
            
            # Store results for next stages
            self.stage_results['advisory'] = {
                'results_path': str(advisory_output_path),
                'recommendations': recommended_models,
                'problem_type': problem_type,
                'success': True
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Basic advisory failed: {e}")
            return False
    
    def _test_training_stage(self) -> bool:
        """Test model training stage (limited models for speed)"""
        print("🏋️ Testing Model Training Stage")
        
        try:
            # Get cleaned data path from previous cleaning stage
            cleaned_data_path = None
            if 'cleaning' in self.stage_results:
                cleaned_data_path = self.stage_results['cleaning']['output_path']
                print(f"📊 Using cleaned data from previous stage: {cleaned_data_path}")
            else:
                # Look for test_cleaned_data.csv from previous runs
                possible_paths = [
                    self.testing_outputs['cleaned_data'] / "test_cleaned_data.csv",
                    self.output_base / "test_cleaned_data.csv",
                    "data/cleaned_data.csv"
                ]
                for path in possible_paths:
                    if Path(path).exists():
                        cleaned_data_path = str(path)
                        print(f"📊 Found existing cleaned data: {cleaned_data_path}")
                        break
                
                if not cleaned_data_path:
                    print("❌ No cleaned data found. Please run cleaning stage first:")
                    print("💡 python 02_stage_testing.py --stage cleaning")
                    return False
            
            if not Path(cleaned_data_path).exists():
                print(f"❌ Cleaned data file not found: {cleaned_data_path}")
                print("💡 Run cleaning stage first: python 02_stage_testing.py --stage cleaning")
                return False
            
            # Load cleaned data
            print(f"📊 Loading cleaned data: {cleaned_data_path}")
            
            # Try different delimiters since data cleaner might use semicolon
            try:
                df = pd.read_csv(cleaned_data_path, delimiter=';')
                if len(df.columns) == 1:
                    # Try comma delimiter
                    df = pd.read_csv(cleaned_data_path, delimiter=',')
            except:
                # Fallback to standard read
                df = pd.read_csv(cleaned_data_path)
            
            # Check if we successfully parsed the file
            if len(df.columns) == 1:
                print(f"⚠️ Only 1 column detected, trying alternative parsing...")
                # Try auto-detecting delimiter
                import csv
                with open(cleaned_data_path, 'r') as f:
                    sample = f.read(1024)
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                    print(f"🔍 Detected delimiter: '{delimiter}'")
                
                df = pd.read_csv(cleaned_data_path, delimiter=delimiter)
            
            print(f"📊 Successfully loaded data with {len(df.columns)} columns")
            
            # Detect the actual target column name (may have been transformed during cleaning)
            original_target = self.config['global']['target_column']
            actual_target = self._detect_target_column(df, original_target)
            
            if actual_target is None:
                print(f"❌ Target column '{original_target}' not found in cleaned data")
                print(f"Available columns: {list(df.columns)}")
                return False
            
            print(f"🎯 Target column: {actual_target}")
            if actual_target != original_target:
                print(f"   (transformed from original: {original_target})")
            print(f"📊 Training data shape: {df.shape}")
            print(f"✅ Using pre-cleaned data (encoding/scaling already applied)")
            
            # Import trainer
            try:
                sys.path.append(str(project_root / "src" / "4_model_training"))
                from trainer import EnhancedModelTrainer, TrainerConfig
                
                # Get training configuration from unified config
                stage_training_config = self.config.get('stage_testing', {}).get('training', {})
                step_training_config = self.config.get('step_04_training', {})
                legacy_training_config = self.config.get('model_training', {})
                
                # Configure trainer using unified config with testing overrides
                # IMPORTANT: Disable preprocessing since data is already cleaned
                training_config = TrainerConfig(
                    test_size=stage_training_config.get('test_size', step_training_config.get('data_splitting', {}).get('test_size', 0.2)),
                    validation_size=stage_training_config.get('validation_size', step_training_config.get('data_splitting', {}).get('validation_size', 0.1)),
                    random_state=stage_training_config.get('random_state', step_training_config.get('data_splitting', {}).get('random_state', 42)),
                    max_models=stage_training_config.get('max_models', 
                               step_training_config.get('model_selection', {}).get('max_models', 
                               legacy_training_config.get('max_models', 10))),
                    include_ensemble=stage_training_config.get('include_ensemble', 
                                     step_training_config.get('model_selection', {}).get('include_ensemble', 
                                     legacy_training_config.get('include_ensemble', True))),
                    include_advanced=stage_training_config.get('include_advanced', 
                                     step_training_config.get('model_selection', {}).get('include_advanced', 
                                     legacy_training_config.get('include_advanced', True))),
                    enable_tuning=stage_training_config.get('enable_tuning', 
                                  step_training_config.get('hyperparameter_tuning', {}).get('enabled', 
                                  legacy_training_config.get('enable_tuning', True))),
                    verbose=stage_training_config.get('verbose', True),
                    save_models=stage_training_config.get('save_models', True),
                    model_dir=str(self.testing_outputs['models']),
                    parallel_jobs=stage_training_config.get('parallel_jobs', step_training_config.get('performance', {}).get('parallel_jobs', -1)),
                    # Disable preprocessing since data is already cleaned
                    encoding_strategy="none",  # Data already encoded
                    scaling_strategy="none",   # Data already scaled
                    enable_business_features=False  # Disable to avoid feature conflicts
                )
                
                print(f"⚙️ Training configuration (from unified config):")
                print(f"   Max models: {training_config.max_models}")
                print(f"   Test size: {training_config.test_size}")
                print(f"   Validation size: {training_config.validation_size}")
                print(f"   Hyperparameter tuning: {training_config.enable_tuning}")
                print(f"   Include ensemble: {training_config.include_ensemble}")
                print(f"   Include advanced: {training_config.include_advanced}")
                print(f"   Preprocessing: DISABLED (data already cleaned)")
                print(f"   Business features: DISABLED (for compatibility)")
                print(f"   Save models: {training_config.save_models}")
                print(f"   Model directory: {training_config.model_dir}")
                print(f"   Parallel jobs: {training_config.parallel_jobs}")
                
                # Initialize trainer
                trainer = EnhancedModelTrainer(training_config)
                
                print("🔄 Starting model training...")
                
                # Train models
                results = trainer.train_all_models(df, actual_target)
                
                if not results:
                    print("❌ No models trained successfully")
                    return False
                
                print(f"✅ Model training completed!")
                print(f"📊 Models trained: {len(results)}")
                
                # Generate training report
                target_type = trainer.infer_target_type(df[actual_target])
                report_path = str(self.testing_outputs['reports'] / "training_test_report.json")
                report = trainer.generate_report(results, target_type, report_path)
                
                # Display results
                print(f"\n📊 Training Results:")
                
                if report and 'summary' in report:
                    summary = report['summary']
                    print(f"   Best model: {summary.get('best_model', 'Unknown')}")
                    print(f"   Best score: {summary.get('best_score', 0):.4f}")
                    print(f"   Total training time: {summary.get('total_training_time', 0):.2f}s")
                
                # Show individual model results
                print(f"\n🏅 Model Performance:")
                for i, result in enumerate(results[:5], 1):  # Show top 5
                    primary_score = result.scores.get("test_accuracy" if target_type == "classification" else "test_r2", 0)
                    print(f"   {i}. {result.name}: {primary_score:.4f} (±{result.cv_scores.get('cv_accuracy_std', 0):.4f})")
                    print(f"      Training time: {result.training_time:.2f}s")
                
                # Show saved models
                model_files = list(self.testing_outputs['models'].glob("*.pkl"))
                if model_files:
                    print(f"\n💾 Saved Models ({len(model_files)}):")
                    for model_file in model_files[:5]:
                        file_size = model_file.stat().st_size / 1024
                        print(f"   📦 {model_file.name} ({file_size:.1f} KB)")
                
                # Store results for evaluation stage
                self.stage_results['training'] = {
                    'report_path': report_path,
                    'model_directory': str(self.testing_outputs['models']),
                    'best_model': report['summary'].get('best_model', 'Unknown') if report and 'summary' in report else 'Unknown',
                    'models_trained': len(results),
                    'success': True
                }
                
                return True
                
            except ImportError as e:
                print(f"❌ Model trainer not available: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Training stage test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _test_evaluation_stage(self) -> bool:
        """Test model evaluation stage"""
        print("📈 Testing Model Evaluation Stage")
        
        try:
            # Check if training was completed
            if 'training' not in self.stage_results:
                print("❌ Training stage not completed")
                print("💡 Run training stage first: python 02_stage_testing.py --stage training")
                return False
            
            training_results = self.stage_results['training']
            report_path = training_results['report_path']
            
            if not Path(report_path).exists():
                print(f"❌ Training report not found: {report_path}")
                return False
            
            print(f"📊 Loading training results: {report_path}")
            
            # Import evaluator
            try:
                sys.path.append(str(project_root / "src" / "5_evaluation"))
                from evaluator import ModelAnalyzer, AnalysisConfig
                
                # Get evaluation configuration from unified config
                stage_eval_config = self.config.get('stage_testing', {}).get('evaluation', {})
                step_eval_config = self.config.get('step_05_evaluation', {})
                legacy_eval_config = self.config.get('model_evaluation', {})
                
                # Configure evaluator using unified config with testing overrides
                eval_config = AnalysisConfig(
                    training_report_path=Path(report_path),
                    output_dir=self.testing_outputs['evaluations'],
                    enable_shap=stage_eval_config.get('enable_shap', 
                                step_eval_config.get('analysis_types', {}).get('enable_shap', 
                                legacy_eval_config.get('enable_shap', True))),
                    enable_learning_curves=stage_eval_config.get('enable_learning_curves', 
                                           step_eval_config.get('analysis_types', {}).get('enable_learning_curves', 
                                           legacy_eval_config.get('enable_learning_curves', True))),
                    enable_residual_analysis=stage_eval_config.get('enable_residual_analysis', 
                                             step_eval_config.get('analysis_types', {}).get('enable_residual_analysis', 
                                             legacy_eval_config.get('enable_residual_analysis', True))),
                    enable_stability_analysis=stage_eval_config.get('enable_stability_analysis', 
                                              step_eval_config.get('analysis_types', {}).get('enable_stability_analysis', 
                                              legacy_eval_config.get('enable_stability_analysis', True))),
                    enable_interpretability=stage_eval_config.get('enable_interpretability', 
                                            step_eval_config.get('analysis_types', {}).get('enable_interpretability', 
                                            legacy_eval_config.get('enable_interpretability', True))),
                    enable_uncertainty_analysis=stage_eval_config.get('enable_uncertainty_analysis', 
                                                step_eval_config.get('analysis_types', {}).get('enable_uncertainty_analysis', 
                                                legacy_eval_config.get('enable_uncertainty_analysis', True))),
                    verbose=stage_eval_config.get('verbose', True),
                    plot_format=stage_eval_config.get('plot_format', 
                               step_eval_config.get('visualization', {}).get('plot_format', 
                               legacy_eval_config.get('plot_format', 'html'))),
                    plot_dpi=stage_eval_config.get('plot_dpi', 
                             step_eval_config.get('visualization', {}).get('plot_dpi', 
                             legacy_eval_config.get('plot_dpi', 300))),
                    figure_size=tuple(stage_eval_config.get('figure_size', 
                                     step_eval_config.get('visualization', {}).get('figure_size', 
                                     legacy_eval_config.get('figure_size', [12, 8])))),
                    n_permutations=stage_eval_config.get('n_permutations', 
                                  step_eval_config.get('performance', {}).get('n_permutations', 
                                  legacy_eval_config.get('n_permutations', 50))),
                    max_shap_samples=stage_eval_config.get('max_shap_samples', 
                                     step_eval_config.get('performance', {}).get('max_shap_samples', 
                                     legacy_eval_config.get('max_shap_samples', 1000))),
                    n_models_to_evaluate=stage_eval_config.get('n_models_to_evaluate', 
                                         step_eval_config.get('performance', {}).get('n_models_to_evaluate', 
                                         legacy_eval_config.get('n_models_to_evaluate', 3)))
                )
                
                print(f"⚙️ Evaluation configuration (from unified config):")
                print(f"   Output directory: {eval_config.output_dir}")
                print(f"   SHAP analysis: {eval_config.enable_shap}")
                print(f"   Learning curves: {eval_config.enable_learning_curves}")
                print(f"   Residual analysis: {eval_config.enable_residual_analysis}")
                print(f"   Stability analysis: {eval_config.enable_stability_analysis}")
                print(f"   Interpretability: {eval_config.enable_interpretability}")
                print(f"   Uncertainty analysis: {eval_config.enable_uncertainty_analysis}")
                print(f"   Plot format: {eval_config.plot_format}")
                print(f"   Plot DPI: {eval_config.plot_dpi}")
                print(f"   Figure size: {eval_config.figure_size}")
                print(f"   N permutations: {eval_config.n_permutations}")
                print(f"   Max SHAP samples: {eval_config.max_shap_samples}")
                print(f"   N models to evaluate: {eval_config.n_models_to_evaluate}")
                
                # Debug: Show configuration sources
                print(f"\n🔍 Configuration Source Debug:")
                print(f"   stage_testing.evaluation exists: {'evaluation' in self.config.get('stage_testing', {})}")
                print(f"   stage_testing.evaluation.enable_shap: {stage_eval_config.get('enable_shap', 'NOT_FOUND')}")
                print(f"   step_05_evaluation exists: {'step_05_evaluation' in self.config}")
                print(f"   model_evaluation exists: {'model_evaluation' in self.config}")
                print(f"   model_evaluation.enable_shap: {legacy_eval_config.get('enable_shap', 'NOT_FOUND')}")
                print(f"   Testing mode: {self.testing_mode}")
                print(f"   Final enable_shap: {eval_config.enable_shap}")
                print(f"   Final plot_format: {eval_config.plot_format}")
                print(f"   Final n_permutations: {eval_config.n_permutations}")
                print(f"   Final max_shap_samples: {eval_config.max_shap_samples}")
                print(f"   Final n_models_to_evaluate: {eval_config.n_models_to_evaluate}")
                
                # Initialize analyzer
                analyzer = ModelAnalyzer(eval_config)
                
                print("🔄 Running model evaluation...")
                
                # Get test data for evaluation
                cleaned_data_path = None
                if 'cleaning' in self.stage_results:
                    cleaned_data_path = self.stage_results['cleaning']['output_path']
                else:
                    cleaned_data_path = self.config['global']['data_input_path']
                
                # Load test data
                print(f"📊 Loading test data: {cleaned_data_path}")
                
                # Try different delimiters since data cleaner might use semicolon
                try:
                    df = pd.read_csv(cleaned_data_path, delimiter=';')
                    if len(df.columns) == 1:
                        # Try comma delimiter
                        df = pd.read_csv(cleaned_data_path, delimiter=',')
                except:
                    # Fallback to standard read
                    df = pd.read_csv(cleaned_data_path)
                
                # Check if we successfully parsed the file
                if len(df.columns) == 1:
                    print(f"⚠️ Only 1 column detected, trying alternative parsing...")
                    # Try auto-detecting delimiter
                    with open(cleaned_data_path, 'r') as f:
                        sample = f.read(1024)
                        sniffer = csv.Sniffer()
                        delimiter = sniffer.sniff(sample).delimiter
                        print(f"🔍 Detected delimiter: '{delimiter}'")
                    
                    df = pd.read_csv(cleaned_data_path, delimiter=delimiter)
                
                print(f"📊 Successfully loaded data with {len(df.columns)} columns")
                
                # Detect the actual target column name (may have been transformed during cleaning)
                original_target = self.config['global']['target_column']
                actual_target = self._detect_target_column(df, original_target)
                
                if actual_target is None:
                    print(f"❌ Target column '{original_target}' not found in evaluation data")
                    print(f"Available columns: {list(df.columns)}")
                    return False
                
                print(f"🎯 Target column: {actual_target}")
                if actual_target != original_target:
                    print(f"   (transformed from original: {original_target})")
                
                X_test = df.drop(columns=[actual_target])
                y_test = df[actual_target]
                
                print(f"📊 Test data shape: {X_test.shape}")
                
                # Run evaluation analysis
                evaluation_results = analyzer.run_analysis(X_test, y_test)
                
                print(f"✅ Model evaluation completed!")
                
                # Display results
                print(f"\n📊 Evaluation Results:")
                
                if evaluation_results:
                    print(f"   Models evaluated: {len(evaluation_results)}")
                    
                    # Show performance summary
                    for model_name, results in list(evaluation_results.items())[:3]:
                        print(f"\n   {model_name}:")
                        if isinstance(results, dict) and 'metrics' in results:
                            metrics = results['metrics']
                            for metric, value in list(metrics.items())[:5]:
                                if isinstance(value, (int, float)):
                                    print(f"      {metric}: {value:.4f}")
                
                # Show generated files
                eval_files = list(self.testing_outputs['evaluations'].glob("*"))
                if eval_files:
                    print(f"\n📁 Generated Evaluation Files ({len(eval_files)}):")
                    for eval_file in eval_files[:10]:
                        file_size = eval_file.stat().st_size / 1024
                        print(f"   📄 {eval_file.name} ({file_size:.1f} KB)")
                
                # Store results
                self.stage_results['evaluation'] = {
                    'output_directory': str(self.testing_outputs['evaluations']),
                    'evaluation_results': evaluation_results,
                    'files_generated': len(eval_files),
                    'success': True
                }
                
                return True
                
            except ImportError as e:
                print(f"❌ Model evaluator not available: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Evaluation stage test failed: {e}")
            return False
    
    def test_all_stages(self) -> bool:
        """Test all stages in sequence"""
        print("🔄 Testing All Pipeline Stages")
        print("=" * 80)
        
        stages = ["cleaning", "advisory", "training", "evaluation"]
        
        for i, stage in enumerate(stages, 1):
            print(f"\n{'='*20} Stage {i}/{len(stages)}: {stage.upper()} {'='*20}")
            
            if not self.test_stage(stage):
                print(f"\n❌ Testing stopped at stage: {stage}")
                return False
            
            # Pause between stages for review
            if i < len(stages):
                response = input(f"\nContinue to next stage? (y/n/i for inspect): ").lower().strip()
                if response in ['n', 'no']:
                    print("Testing stopped by user")
                    return False
                elif response in ['i', 'inspect']:
                    self._inspect_stage_results(stage)
        
        print("\n" + "="*80)
        print("🎉 ALL STAGES TESTED SUCCESSFULLY!")
        print("="*80)
        
        self._save_testing_summary()
        return True
    
    def _inspect_stage_results(self, stage_name: str):
        """Interactive inspection of stage results"""
        print(f"\n🔍 Inspecting {stage_name.upper()} Results")
        print("Commands: 'summary', 'files', 'logs', 'next', 'quit'")
        
        while True:
            command = input(f"🔍 [{stage_name}] > ").strip().lower()
            
            if command in ['quit', 'next']:
                break
            elif command == 'summary':
                if stage_name in self.stage_results:
                    results = self.stage_results[stage_name]
                    print(f"\n📊 {stage_name.upper()} Summary:")
                    for key, value in results.items():
                        if key != 'success':
                            print(f"   {key}: {value}")
                else:
                    print(f"No results available for {stage_name}")
            elif command == 'files':
                # Show files in stage output directory
                stage_outputs = {
                    'cleaning': self.testing_outputs['cleaned_data'],
                    'advisory': self.testing_outputs['reports'],
                    'training': self.testing_outputs['models'],
                    'evaluation': self.testing_outputs['evaluations']
                }
                
                output_dir = stage_outputs.get(stage_name)
                if output_dir and output_dir.exists():
                    files = list(output_dir.glob("*"))
                    print(f"\n📁 {stage_name.upper()} Files ({len(files)}):")
                    for file_path in files:
                        file_size = file_path.stat().st_size / 1024
                        print(f"   📄 {file_path.name} ({file_size:.1f} KB)")
                else:
                    print(f"No output directory found for {stage_name}")
            elif command == 'logs':
                log_files = list(self.testing_outputs['logs'].glob(f"*{stage_name}*.json"))
                if log_files:
                    print(f"\n📋 {stage_name.upper()} Logs:")
                    for log_file in log_files:
                        print(f"   📄 {log_file}")
                else:
                    print(f"No log files found for {stage_name}")
            else:
                print("   Unknown command. Available: 'summary', 'files', 'logs', 'next', 'quit'")
    
    def _save_testing_summary(self):
        """Save comprehensive testing summary"""
        try:
            summary = {
                'testing_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'discovery_directory': str(self.discovery_dir) if self.discovery_dir else None,
                    'testing_directory': str(self.testing_outputs['testing'])
                },
                'execution_log': self.execution_log,
                'stage_results': self.stage_results,
                'testing_outputs': {k: str(v) for k, v in self.testing_outputs.items()},
                'summary': {
                    'total_stages_tested': len(self.execution_log),
                    'successful_stages': len([log for log in self.execution_log if log['success']]),
                    'total_execution_time': sum(log['execution_time'] for log in self.execution_log),
                    'all_stages_successful': all(log['success'] for log in self.execution_log)
                },
                'next_steps': [
                    "Review stage-specific results and outputs",
                    "Modify configurations as needed",
                    "Run full pipeline with: python 03_full_pipeline.py"
                ]
            }
            
            summary_path = self.testing_outputs['testing'] / "stage_testing_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"📄 Testing summary saved: {summary_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save testing summary: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='DS-AutoAdvisor Modular Stage Testing')
    parser.add_argument('--stage', type=str, required=True,
                       help='Stage(s) to test: cleaning, advisory, training, evaluation, all')
    parser.add_argument('--discovery-dir', type=str,
                       help='Path to discovery results directory')
    parser.add_argument('--output', type=str, default='pipeline_outputs',
                       help='Base output directory')
    parser.add_argument('--mode', type=str, default='fast', 
                       choices=['fast', 'comprehensive', 'custom'],
                       help='Testing mode: fast (minimal), comprehensive (thorough), custom (use exact config)')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ModularStageTester(
        discovery_dir=args.discovery_dir,
        output_base=args.output,
        testing_mode=args.mode
    )
    
    if not tester.discovery_dir:
        print("❌ No discovery directory available")
        print("💡 Run data discovery first: python 01_data_discovery.py")
        return 1
    
    print(f"🧪 Testing Mode: {args.mode.upper()}")
    if args.mode == "fast":
        print("   ⚡ Optimized for speed - minimal models and analysis")
    elif args.mode == "comprehensive":
        print("   🔬 Thorough testing - more models and complete analysis")
    elif args.mode == "custom":
        print("   ⚙️ Using exact unified_config_v2.yaml settings")
    
    # Parse stages
    if args.stage.lower() == 'all':
        success = tester.test_all_stages()
    else:
        stages = [s.strip() for s in args.stage.split(',')]
        success = True
        
        for stage in stages:
            if stage not in ['cleaning', 'advisory', 'training', 'evaluation']:
                print(f"❌ Unknown stage: {stage}")
                print("Available stages: cleaning, advisory, training, evaluation, all")
                return 1
            
            if not tester.test_stage(stage):
                success = False
                break
    
    if success:
        print(f"\n🎉 Stage Testing Completed Successfully!")
        print(f"📁 All outputs in: {tester.testing_outputs['testing']}")
        print(f"\n🚀 Next Steps:")
        print(f"   1. Review stage results: {tester.testing_outputs['testing']}")
        print(f"   2. Run full pipeline: python 03_full_pipeline.py")
        return 0
    else:
        print(f"\n❌ Stage Testing Failed")
        return 1


if __name__ == "__main__":
    exit(main())
