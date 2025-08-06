#!/usr/bin/env python3
"""
🧪 DS-AutoAdvisor: Step 2 - Pipeline Testing
==========================================

WHAT IT DOES:
Tests your complete ML pipeline step-by-step with interactive inspection.
Runs all 6 stages (profiling → cleaning → advisory → training → evaluation)
with ability to pause, inspect data, and adjust configuration at each stage.

WHEN TO USE:
- After data inspection (Step 1)
- Before running the full pipeline (Step 3)
- When you want to validate pipeline behavior
- When debugging pipeline issues

HOW TO USE:
Test with your data:
    python 2_test_pipeline.py --data data/your_data.csv

Auto-detect from config:
    python 2_test_pipeline.py

Start from specific stage:
    python 2_test_pipeline.py --start-stage 2

INTERACTIVE FEATURES:
During execution, you can:
- Type 'inspect' to explore data at current stage
- Use commands: 'stage <name>', 'columns <stage>', 'sample <stage>', 'stats <stage>'
- Continue (y), stop (n), or inspect (i) at each checkpoint

WHAT YOU GET:
✅ Step-by-step validation of all 6 pipeline stages
✅ Interactive data inspection at each stage
✅ Data quality assessment with v2.0 features
✅ Plugin results and feature selection insights
✅ Model training and evaluation testing
✅ Configuration adjustment recommendations

NEXT STEP:
After successful testing, run: python 3_run_pipeline.py
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# Import from the renamed pipeline file
import importlib.util
spec = importlib.util.spec_from_file_location("run_pipeline", project_root / "4_run_pipeline.py")
run_pipeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_pipeline_module)
DSAutoAdvisorPipelineV2 = run_pipeline_module.DSAutoAdvisorPipelineV2

from src.infrastructure.enhanced_config_manager import Environment

class StepByStepPipelineTester:
    """Step-by-step pipeline testing with data inspection"""
    
    def __init__(self, data_path: str = None, config_path: str = 'config/unified_config_v2.yaml'):
        """Initialize tester"""
        self.data_path = data_path
        self.config_path = config_path
        self.pipeline = None
        self.stage_data = {}  # Store data at each stage
        self.stage_reports = {}  # Store reports at each stage
        self.enhanced_profiling_outputs = {}  # Store enhanced profiling outputs
        
    def initialize_pipeline(self) -> bool:
        """Initialize the v2.0 pipeline"""
        print("🚀 Initializing DS-AutoAdvisor v2.0 Pipeline")
        print("=" * 60)
        
        try:
            self.pipeline = DSAutoAdvisorPipelineV2(
                config_path=self.config_path,
                environment=Environment.DEVELOPMENT,
                enable_v2_features=True
            )
            
            print("✅ Pipeline initialized successfully")
            print(f"   Pipeline Run ID: {self.pipeline.pipeline_run_id}")
            print(f"   V2 Features: {self.pipeline.enable_v2_features}")
            print(f"   MLflow Integration: {self.pipeline.mlflow_integration is not None}")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline initialization failed: {e}")
            return False
    
    def inspect_raw_data(self) -> bool:
        """Step 0: Inspect raw data before processing"""
        print("\n" + "="*60)
        print("📊 STEP 0: RAW DATA INSPECTION")
        print("="*60)
        
        try:
            # Get data path from config if not provided
            if not self.data_path:
                self.data_path = self.pipeline.config_manager.get_config('global', 'data_input_path')
            
            if not Path(self.data_path).exists():
                print(f"❌ Data file not found: {self.data_path}")
                return False
            
            # Load raw data
            print(f"📁 Loading data from: {self.data_path}")
            delimiter = self.pipeline.config_manager.get_config('global', 'csv_delimiter', ',')
            df = pd.read_csv(self.data_path, delimiter=delimiter)
            
            self.stage_data['raw'] = df.copy()
            
            # Display basic information
            print(f"\n📈 Raw Data Overview:")
            print(f"   Shape: {df.shape}")
            print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"   Delimiter used: '{delimiter}'")
            
            # Column information
            print(f"\n📋 Column Information:")
            for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                unique_count = df[col].nunique()
                
                print(f"   {i:2d}. {col:<20} | {str(dtype):<12} | Missing: {missing_count:4d} ({missing_pct:5.1f}%) | Unique: {unique_count:4d}")
            
            # Sample data
            print(f"\n📝 First 5 rows:")
            print(df.head().to_string(index=False))
            
            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"\n📊 Numeric Columns Statistics:")
                print(df[numeric_cols].describe().round(3).to_string())
            
            # Categorical columns summary
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                print(f"\n📊 Categorical Columns Summary:")
                for col in categorical_cols[:5]:  # Limit to first 5
                    value_counts = df[col].value_counts().head(3)
                    print(f"   {col}: {dict(value_counts)}")
            
            # Check target column
            target_col = self.pipeline.config_manager.get_config('global', 'target_column')
            if target_col and target_col in df.columns:
                print(f"\n🎯 Target Column '{target_col}':")
                if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() < 20:
                    print(f"   Distribution: {dict(df[target_col].value_counts())}")
                else:
                    print(f"   Statistics: {df[target_col].describe().round(3).to_dict()}")
            
            self._prompt_continue("Raw data inspection complete. Continue to data profiling?")
            return True
            
        except Exception as e:
            print(f"❌ Raw data inspection failed: {e}")
            return False
    
    def test_data_profiling(self) -> bool:
        """Step 1: Test data profiling stage"""
        print("\n" + "="*60)
        print("🔍 STEP 1: DATA PROFILING")
        print("="*60)
        
        try:
            # Run data quality assessment
            if self.pipeline.config_manager.is_feature_enabled('data_quality_v2'):
                print("📊 Running enhanced data quality assessment...")
                
                df = self.stage_data['raw']
                target_col = self.pipeline.config_manager.get_config('global', 'target_column')
                
                quality_report = self.pipeline.quality_assessor.assess_quality(df, target_col)
                self.stage_reports['profiling'] = quality_report
                
                print(f"\n📈 Data Quality Results:")
                print(f"   Overall Score: {quality_report.overall_score:.1f}/100")
                print(f"   Total Issues: {quality_report.total_issues}")
                
                # Show issues by severity
                for severity, count in quality_report.issues_by_severity.items():
                    if count > 0:
                        print(f"   {severity.upper()}: {count} issues")
                
                # Show top issues
                if quality_report.issues:
                    print(f"\n⚠️  Top Data Quality Issues:")
                    for i, issue in enumerate(quality_report.issues[:5], 1):
                        print(f"   {i}. [{issue.severity.upper()}] {issue.column or 'Dataset'}: {issue.description}")
                        print(f"      → {issue.suggested_action}")
                
                # Show recommendations
                if quality_report.recommendations:
                    print(f"\n💡 Recommendations:")
                    for i, rec in enumerate(quality_report.recommendations[:3], 1):
                        print(f"   {i}. {rec}")
            
            # Run Enhanced Data Profiling
            print(f"\n🔄 Running enhanced data profiling with machine-readable output...")
            
            try:
                # Import and run enhanced profiler
                import sys
                sys.path.append('src/1_data_profiling')
                from enhanced_data_profiler import EnhancedDataProfiler
                
                df = self.stage_data['raw']
                base_dir = Path().resolve()
                
                profiler = EnhancedDataProfiler(
                    input_path=self.data_path,
                    output_dir=base_dir / "docs",
                    dataset_name=Path(self.data_path).stem
                )
                
                # Run complete profiling with machine-readable output
                html_path, raw_data_path, config_path = profiler.run_complete_profiling()
                
                print("✅ Enhanced data profiling completed successfully")
                print(f"   📄 HTML Report: {html_path}")
                print(f"   📊 Raw Data: {raw_data_path}")
                print(f"   🔧 Config Template: {config_path}")
                
                # Show profiling summary
                if Path(raw_data_path).exists():
                    import json
                    with open(raw_data_path, 'r') as f:
                        profile_data = json.load(f)
                    
                    print(f"\n📈 Profiling Summary:")
                    print(f"   Columns analyzed: {len(profile_data.get('columns', {}))}")
                    print(f"   Missing data patterns: {len(profile_data.get('missing_data', {}))}")
                    print(f"   Correlation insights: {len(profile_data.get('correlations', {}))}")
                    
                    # Show column-specific recommendations
                    if Path(config_path).exists():
                        print(f"\n🎯 Column-Specific Recommendations Generated:")
                        print(f"   Review and modify: {config_path}")
                        print(f"   📋 Human intervention recommended before cleaning")
                        
                        # Pause for human review
                        print(f"\n⏸️  PIPELINE HALT: Review profiling results and configuration")
                        print(f"   1. Open HTML report: {html_path}")
                        print(f"   2. Review configuration: {config_path}")
                        print(f"   3. Modify settings as needed")
                        
                        response = input("\n   Ready to continue with data cleaning? (y/n/i): ").lower()
                        if response == 'n':
                            print("   Pipeline stopped for configuration review")
                            return False
                        elif response == 'i':
                            self._interactive_inspection()
                
                # Store paths for next stage
                self.enhanced_profiling_outputs = {
                    'html_report': html_path,
                    'raw_data': raw_data_path,
                    'config_template': config_path
                }
                
            except ImportError:
                print("⚠️  Enhanced profiler not available, running traditional profiling...")
                
                # Run traditional profiling
                print(f"\n🔄 Running traditional data profiling...")
                stage_result = self.pipeline._execute_stage("profiling")
                
                if stage_result.success:
                    print("✅ Data profiling completed successfully")
                    print(f"   Execution time: {stage_result.execution_time:.2f}s")
                    print(f"   Outputs created: {len(stage_result.outputs_created)}")
                    
                    for output in stage_result.outputs_created:
                        if Path(output).exists():
                            print(f"   📄 Created: {output}")
                else:
                    print(f"❌ Data profiling failed: {stage_result.error_message}")
                    return False
            
            self._prompt_continue("Data profiling complete. Continue to data cleaning?")
            return True
            
        except Exception as e:
            print(f"❌ Data profiling test failed: {e}")
            return False
    
    def test_data_cleaning(self) -> bool:
        """Step 2: Test data cleaning stage with enhanced column-specific cleaning"""
        print("\n" + "="*60)
        print("🧹 STEP 2: DATA CLEANING (Enhanced)")
        print("="*60)
        
        try:
            # Check if enhanced profiling was run and config exists
            config_path = None
            if hasattr(self, 'enhanced_profiling_outputs'):
                config_path = self.enhanced_profiling_outputs.get('config_template')
            
            if not config_path:
                # Look for default config template
                config_path = Path('config/cleaning_config_template.yaml')
            
            # Run Enhanced Data Cleaning
            print("🔄 Running enhanced column-specific data cleaning...")
            
            try:
                # Import and run enhanced cleaner
                import sys
                sys.path.append('src/2_data_cleaning')
                from data_cleaner import DataCleaner, CleaningConfig
                
                base_dir = Path().resolve()
                
                # Create enhanced cleaning configuration
                config_enhanced = CleaningConfig(
                    input_path=self.data_path,
                    output_path=base_dir / "data" / "cleaned_data_enhanced.csv",
                    log_path=base_dir / "docs" / "cleaning_log_enhanced.json",
                    column_config_path=str(config_path) if config_path and Path(config_path).exists() else None,
                    verbose=True,
                    **CleaningConfig.create_csv_preset("quoted_semicolon")
                )
                
                # Run enhanced cleaning
                cleaner_enhanced = DataCleaner(config_enhanced)
                print(f"📋 Pipeline steps: {cleaner_enhanced.list_steps()}")
                
                df_enhanced, log_enhanced = cleaner_enhanced.clean()
                
                print("✅ Enhanced data cleaning completed successfully")
                print(f"   Shape: {log_enhanced['initial_shape']} → {log_enhanced['final_shape']}")
                print(f"   Processing time: {log_enhanced['processing_time']:.2f}s")
                print(f"   Actions performed: {len(log_enhanced['actions'])}")
                
                # Show detailed cleaning actions
                if log_enhanced['actions']:
                    print(f"\n🔧 Cleaning Actions Performed:")
                    for i, action in enumerate(log_enhanced['actions'][:10], 1):  # Show first 10
                        print(f"   {i:2d}. {action}")
                    if len(log_enhanced['actions']) > 10:
                        print(f"   ... and {len(log_enhanced['actions']) - 10} more actions")
                
                # Show warnings if any
                if log_enhanced['warnings']:
                    print(f"\n⚠️  Warnings:")
                    for warning in log_enhanced['warnings'][:5]:
                        print(f"   • {warning}")
                
                # Store cleaned data
                self.stage_data['cleaned'] = df_enhanced
                
                # Compare with original data
                original_df = self.stage_data['raw']
                
                print(f"\n📊 Cleaning Results Summary:")
                print(f"   Original shape: {original_df.shape}")
                print(f"   Cleaned shape:  {df_enhanced.shape}")
                print(f"   Rows removed:   {original_df.shape[0] - df_enhanced.shape[0]:,}")
                print(f"   Columns removed: {original_df.shape[1] - df_enhanced.shape[1]}")
                
                # Compare missing data
                original_missing = original_df.isnull().sum().sum()
                cleaned_missing = df_enhanced.isnull().sum().sum()
                print(f"   Missing values: {original_missing:,} → {cleaned_missing:,}")
                
                # Show column-specific transformations
                print(f"\n📋 Column-Specific Transformations:")
                common_cols = set(original_df.columns) & set(df_enhanced.columns)
                transformed_cols = 0
                for col in list(common_cols)[:15]:  # Show first 15
                    orig_unique = original_df[col].nunique()
                    clean_unique = df_enhanced[col].nunique()
                    orig_missing = original_df[col].isnull().sum()
                    clean_missing = df_enhanced[col].isnull().sum()
                    
                    if orig_unique != clean_unique or orig_missing != clean_missing:
                        transformed_cols += 1
                        print(f"   {col}: Unique {orig_unique}→{clean_unique}, Missing {orig_missing}→{clean_missing}")
                
                if transformed_cols == 0:
                    print("   No significant column transformations detected")
                
                # Show sample of cleaned data
                print(f"\n📝 Cleaned Data Sample:")
                print(df_enhanced.head(3).to_string(index=False, max_cols=8))
                
            except ImportError as e:
                print(f"⚠️  Enhanced cleaner not available: {e}")
                print("Running traditional data cleaning...")
                
                # Fallback to traditional cleaning
                stage_result = self.pipeline._execute_stage("data_cleaning")
                
                if stage_result.success:
                    print("✅ Data cleaning completed successfully")
                    
                    # Load cleaned data
                    cleaned_path = self.pipeline.config_manager.get_config('data_cleaning', 'output_path', 'data/cleaned_data.csv')
                    if Path(cleaned_path).exists():
                        cleaned_df = pd.read_csv(cleaned_path)
                        self.stage_data['cleaned'] = cleaned_df
                        
                        original_df = self.stage_data['raw']
                        
                        print(f"\n📊 Cleaning Results:")
                        print(f"   Original shape: {original_df.shape}")
                        print(f"   Cleaned shape:  {cleaned_df.shape}")
                        print(f"   Rows removed:   {original_df.shape[0] - cleaned_df.shape[0]}")
                        print(f"   Columns removed: {original_df.shape[1] - cleaned_df.shape[1]}")
                else:
                    print(f"❌ Data cleaning failed: {stage_result.error_message}")
                    return False
            
            self._prompt_continue("Data cleaning complete. Continue to advisory?")
            return True
            
        except Exception as e:
            print(f"❌ Data cleaning test failed: {e}")
            return False
    
    def test_advisory(self) -> bool:
        """Step 3: Test advisory stage"""
        print("\n" + "="*60)
        print("🤖 STEP 3: ADVISORY")
        print("="*60)
        
        try:
            print("🔄 Running advisory stage...")
            stage_result = self.pipeline._execute_stage("advisory")
            
            if stage_result.success:
                print("✅ Advisory completed successfully")
                
                # Check for advisory outputs
                advisory_outputs = [output for output in stage_result.outputs_created 
                                 if 'advisory' in output.lower()]
                
                for output in advisory_outputs:
                    if Path(output).exists():
                        print(f"   📄 Created: {output}")
                        
                        # Try to show advisory content
                        if output.endswith('.json'):
                            try:
                                with open(output, 'r') as f:
                                    advisory_data = json.load(f)
                                print(f"   📊 Advisory insights generated: {len(advisory_data)} items")
                            except:
                                pass
                
                # Show plugin results if available
                if stage_result.artifacts:
                    plugin_results = {k: v for k, v in stage_result.artifacts.items() 
                                    if k.startswith('plugin_')}
                    if plugin_results:
                        print(f"\n🔌 Plugin Results:")
                        for plugin_key, result in plugin_results.items():
                            plugin_name = plugin_key.replace('plugin_', '').replace('_features', '')
                            if isinstance(result, list):
                                print(f"   {plugin_name}: {len(result)} features selected")
                                print(f"      Selected features: {result[:5]}{'...' if len(result) > 5 else ''}")
            else:
                print(f"❌ Advisory failed: {stage_result.error_message}")
                return False
            
            self._prompt_continue("Advisory complete. Continue to model training?")
            return True
            
        except Exception as e:
            print(f"❌ Advisory test failed: {e}")
            return False
    
    def test_model_training(self) -> bool:
        """Step 4: Test model training stage"""
        print("\n" + "="*60)
        print("🏋️ STEP 4: MODEL TRAINING")
        print("="*60)
        
        try:
            print("🔄 Running model training...")
            stage_result = self.pipeline._execute_stage("model_training")
            
            if stage_result.success:
                print("✅ Model training completed successfully")
                print(f"   Execution time: {stage_result.execution_time:.2f}s")
                
                # Show trained models
                model_files = [output for output in stage_result.outputs_created 
                             if output.endswith('.pkl')]
                print(f"   📦 Models trained: {len(model_files)}")
                
                for model_file in model_files[:5]:  # Show first 5
                    model_name = Path(model_file).stem
                    print(f"      • {model_name}")
                
                # Show training artifacts
                if stage_result.artifacts and 'training_results' in stage_result.artifacts:
                    training_results = stage_result.artifacts['training_results']
                    print(f"\n📊 Training Results:")
                    if isinstance(training_results, dict):
                        for model_name, results in list(training_results.items())[:3]:
                            if isinstance(results, dict) and 'metrics' in results:
                                metrics = results['metrics']
                                print(f"   {model_name}:")
                                for metric, value in metrics.items():
                                    print(f"      {metric}: {value:.4f}")
                
                # MLflow integration results
                if self.pipeline.mlflow_integration:
                    print(f"   🔬 Models logged to MLflow registry")
            else:
                print(f"❌ Model training failed: {stage_result.error_message}")
                return False
            
            self._prompt_continue("Model training complete. Continue to evaluation?")
            return True
            
        except Exception as e:
            print(f"❌ Model training test failed: {e}")
            return False
    
    def test_model_evaluation(self) -> bool:
        """Step 5: Test model evaluation stage"""
        print("\n" + "="*60)
        print("📈 STEP 5: MODEL EVALUATION")
        print("="*60)
        
        try:
            print("🔄 Running model evaluation...")
            stage_result = self.pipeline._execute_stage("model_evaluation")
            
            if stage_result.success:
                print("✅ Model evaluation completed successfully")
                
                # Show evaluation outputs
                eval_files = [output for output in stage_result.outputs_created]
                print(f"   📄 Evaluation files created: {len(eval_files)}")
                
                for eval_file in eval_files:
                    file_type = Path(eval_file).suffix
                    print(f"      • {Path(eval_file).name} ({file_type})")
                
                # Show evaluation artifacts
                if stage_result.artifacts and 'evaluation_results' in stage_result.artifacts:
                    eval_results = stage_result.artifacts['evaluation_results']
                    print(f"\n📊 Evaluation Summary:")
                    if isinstance(eval_results, dict):
                        for model_name, results in list(eval_results.items())[:3]:
                            print(f"   {model_name}:")
                            if isinstance(results, dict):
                                for metric, value in results.items():
                                    if isinstance(value, (int, float)):
                                        print(f"      {metric}: {value:.4f}")
                
                # Check for HTML reports
                html_files = [f for f in eval_files if f.endswith('.html')]
                if html_files:
                    print(f"\n🌐 HTML Reports Generated:")
                    for html_file in html_files:
                        print(f"   📄 {html_file}")
                        print(f"      Open in browser to view interactive plots")
            else:
                print(f"❌ Model evaluation failed: {stage_result.error_message}")
                return False
            
            print("\n🎉 Pipeline testing completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Model evaluation test failed: {e}")
            return False
    
    def _prompt_continue(self, message: str) -> bool:
        """Prompt user to continue or stop"""
        print(f"\n💭 {message}")
        while True:
            choice = input("   Continue? (y/n/inspect): ").lower().strip()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            elif choice in ['i', 'inspect']:
                self._interactive_inspection()
            else:
                print("   Please enter 'y' (yes), 'n' (no), or 'i' (inspect)")
    
    def _interactive_inspection(self):
        """Interactive data inspection"""
        print("\n🔍 Interactive Data Inspection")
        print("Available data stages:", list(self.stage_data.keys()))
        print("Commands: 'stage <name>', 'columns <stage>', 'sample <stage>', 'stats <stage>', 'quit'")
        
        while True:
            command = input("🔍 > ").strip().lower()
            
            if command == 'quit':
                break
            elif command.startswith('stage '):
                stage_name = command.split(' ', 1)[1]
                if stage_name in self.stage_data:
                    df = self.stage_data[stage_name]
                    print(f"   Stage '{stage_name}' - Shape: {df.shape}")
                    print(f"   Columns: {list(df.columns)}")
                else:
                    print(f"   Stage '{stage_name}' not found")
            elif command.startswith('columns '):
                stage_name = command.split(' ', 1)[1]
                if stage_name in self.stage_data:
                    df = self.stage_data[stage_name]
                    for col in df.columns:
                        print(f"   {col}: {df[col].dtype}")
                else:
                    print(f"   Stage '{stage_name}' not found")
            elif command.startswith('sample '):
                stage_name = command.split(' ', 1)[1]
                if stage_name in self.stage_data:
                    df = self.stage_data[stage_name]
                    print(df.head().to_string())
                else:
                    print(f"   Stage '{stage_name}' not found")
            elif command.startswith('stats '):
                stage_name = command.split(' ', 1)[1]
                if stage_name in self.stage_data:
                    df = self.stage_data[stage_name]
                    print(df.describe().to_string())
                else:
                    print(f"   Stage '{stage_name}' not found")
            else:
                print("   Unknown command")
    
    def run_complete_test(self) -> bool:
        """Run complete step-by-step test"""
        print("🚀 DS-AutoAdvisor v2.0 Complete Pipeline Testing")
        print("=" * 80)
        
        # Initialize pipeline
        if not self.initialize_pipeline():
            return False
        
        # Run each stage
        stages = [
            ("Raw Data Inspection", self.inspect_raw_data),
            ("Data Profiling", self.test_data_profiling),
            ("Data Cleaning", self.test_data_cleaning),
            ("Advisory", self.test_advisory),
            ("Model Training", self.test_model_training),
            ("Model Evaluation", self.test_model_evaluation),
        ]
        
        for stage_name, stage_func in stages:
            print(f"\n{'='*20} {stage_name} {'='*20}")
            if not stage_func():
                print(f"\n❌ Testing stopped at: {stage_name}")
                return False
        
        print("\n" + "="*80)
        print("🎉 ALL PIPELINE STAGES TESTED SUCCESSFULLY!")
        print("="*80)
        return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DS-AutoAdvisor v2.0 Step-by-Step Pipeline Testing')
    parser.add_argument('--data', type=str, help='Path to input data file')
    parser.add_argument('--config', type=str, default='config/unified_config_v2.yaml', 
                       help='Configuration file path')
    parser.add_argument('--start-stage', type=int, choices=[0, 1, 2, 3, 4, 5], 
                       help='Stage to start from (0=raw data, 1=profiling, etc.)')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = StepByStepPipelineTester(
        data_path=args.data,
        config_path=args.config
    )
    
    if args.start_stage is not None:
        print(f"🎯 Starting from stage {args.start_stage}")
        # TODO: Implement stage-specific starting
    
    # Run complete test
    success = tester.run_complete_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
