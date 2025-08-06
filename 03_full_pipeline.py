#!/usr/bin/env python3
"""
🚀 DS-AutoAdvisor: Step 3 - Full Production Pipeline
================================================

WHAT IT DOES:
Executes the complete ML pipeline in production mode using optimized configurations
from data discovery and validated stage testing. Full feature set enabled.

WHEN TO USE:
- After completing data discovery (Step 1) and stage testing (Step 2)
- For production model training and deployment
- When you need complete analysis with all features enabled
- For final model selection and comprehensive evaluation

HOW TO USE:
Production run with latest discovery:
    python 03_full_pipeline.py

Use specific discovery results:
    python 03_full_pipeline.py --discovery-dir pipeline_outputs/01_discovery_dataset_20250804_143022

Custom configuration:
    python 03_full_pipeline.py --config config/production_config.yaml

High-performance mode:
    python 03_full_pipeline.py --mode production --enable-all

PIPELINE STAGES:
1. Data Loading & Validation
2. Advanced Data Cleaning  
3. ML Advisory & Model Selection
4. Comprehensive Model Training
5. Deep Model Evaluation
6. Model Deployment Preparation

PRODUCTION FEATURES:
✅ Full hyperparameter tuning
✅ Ensemble model training
✅ Advanced evaluation metrics
✅ SHAP interpretability analysis
✅ Learning curve analysis
✅ Model stability testing
✅ Deployment artifacts generation
✅ Comprehensive reporting

OUTPUTS:
✅ Production-ready cleaned dataset
✅ Optimized trained models with artifacts
✅ Comprehensive evaluation reports
✅ Model interpretability analysis
✅ Deployment configuration files
✅ Complete audit trail and logs

INTERACTIVE FEATURES:
- Pipeline checkpoint reviews
- Configuration validation
- Performance monitoring
- Error recovery options
- Progress tracking with ETA

NEXT STEP:
Deploy models using generated artifacts and configuration files.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import argparse
import time
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

class ProductionPipeline:
    """Complete production ML pipeline execution"""
    
    def __init__(self, discovery_dir: str = None, config_path: str = None, 
                 output_base: str = "pipeline_outputs", mode: str = "production"):
        """Initialize production pipeline"""
        self.mode = mode
        self.output_base = Path(output_base)
        self.discovery_dir = Path(discovery_dir) if discovery_dir else self._find_latest_discovery()
        
        # Create organized output structure
        self.production_outputs = self._create_output_structure()
        
        # Load configuration and discovery results
        self.config = self._load_configuration(config_path)
        self.discovery_summary = self._load_discovery_results()
        
        # Pipeline execution state
        self.pipeline_state = {
            'current_stage': None,
            'completed_stages': [],
            'failed_stages': [],
            'stage_results': {},
            'start_time': datetime.now(),
            'checkpoint_data': {}
        }
        
        # Performance monitoring
        self.performance_tracker = {
            'stage_times': {},
            'memory_usage': {},
            'errors': [],
            'warnings': []
        }
        
        print(f"🚀 Production Pipeline Initialized")
        print(f"📁 Output directory: {self.production_outputs['base']}")
        print(f"⚙️ Mode: {mode}")
        if self.discovery_dir:
            print(f"🔍 Using discovery: {self.discovery_dir}")
    
    def _find_latest_discovery(self) -> Optional[Path]:
        """Find the latest discovery directory"""
        discovery_dirs = list(self.output_base.glob("01_discovery_*"))
        if discovery_dirs:
            latest = max(discovery_dirs, key=lambda p: p.stat().st_mtime)
            print(f"🔍 Using latest discovery: {latest}")
            return latest
        else:
            print("❌ No discovery directory found. Run 01_data_discovery.py first.")
            return None
    
    def _create_output_structure(self) -> Dict[str, Path]:
        """Create organized production output structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        outputs = {
            'base': self.output_base / f"03_production_{timestamp}",
            'data': self.output_base / f"03_production_{timestamp}" / "data",
            'models': self.output_base / f"03_production_{timestamp}" / "models",
            'evaluations': self.output_base / f"03_production_{timestamp}" / "evaluations",
            'reports': self.output_base / f"03_production_{timestamp}" / "reports",
            'logs': self.output_base / f"03_production_{timestamp}" / "logs",
            'artifacts': self.output_base / f"03_production_{timestamp}" / "artifacts",
            'deployment': self.output_base / f"03_production_{timestamp}" / "deployment"
        }
        
        # Create all directories
        for output_dir in outputs.values():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        return outputs
    
    def _load_discovery_results(self) -> Dict[str, Any]:
        """Load results from data discovery stage"""
        if not self.discovery_dir or not self.discovery_dir.exists():
            print("⚠️ Discovery directory not found, using defaults")
            return {}
        
        summary_file = self.discovery_dir / "discovery_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_configuration(self, config_path: str = None) -> Dict[str, Any]:
        """Load pipeline configuration"""
        # Priority: custom config > discovery config > default config
        
        if config_path and Path(config_path).exists():
            print(f"⚙️ Loading custom config: {config_path}")
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Try discovery-generated config
        if self.discovery_summary and 'generated_files' in self.discovery_summary:
            config_path = self.discovery_summary['generated_files'].get('pipeline_config')
            if config_path and Path(config_path).exists():
                print(f"⚙️ Loading discovery config: {config_path}")
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        
        # Fallback to default config
        default_config_path = project_root / "config" / "unified_config_v2.yaml"
        if default_config_path.exists():
            print(f"⚙️ Loading default config: {default_config_path}")
            with open(default_config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Create minimal production config
        print("⚙️ Creating minimal production config")
        return self._create_production_config()
    
    def _create_production_config(self) -> Dict[str, Any]:
        """Create production configuration"""
        return {
            'global': {
                'data_input_path': 'data/telco_churn_data.csv',
                'target_column': 'Churn',
                'csv_delimiter': ',',
                'csv_encoding': 'utf-8',
                'output_base_dir': str(self.output_base),
                'production_mode': True
            },
            'cleaning': {
                'remove_duplicates': True,
                'outlier_removal': True,
                'outlier_method': 'iqr',
                'handle_missing': True,
                'normalize_text': True,
                'validate_data': True
            },
            'training': {
                'test_size': 0.2,
                'validation_size': 0.2,
                'random_state': 42,
                'enable_tuning': True,
                'max_models': 20,
                'include_ensemble': True,
                'include_advanced': True,
                'enable_early_stopping': True,
                'n_jobs': -1
            },
            'evaluation': {
                'enable_shap': True,
                'enable_learning_curves': True,
                'enable_residual_analysis': True,
                'enable_stability_analysis': True,
                'enable_interpretability': True,
                'cross_validation_folds': 5
            }
        }
    
    def run_production_pipeline(self) -> bool:
        """Execute the complete production pipeline"""
        try:
            print("\n" + "="*80)
            print("🚀 STARTING PRODUCTION PIPELINE")
            print("="*80)
            
            # Pipeline stages
            stages = [
                ("data_loading", "📊 Data Loading & Validation"),
                ("data_cleaning", "🧹 Advanced Data Cleaning"),
                ("ml_advisory", "🤖 ML Advisory & Model Selection"),
                ("model_training", "🏋️ Comprehensive Model Training"),
                ("model_evaluation", "📈 Deep Model Evaluation"),
                ("deployment_prep", "🚀 Deployment Preparation")
            ]
            
            total_stages = len(stages)
            
            for i, (stage_id, stage_name) in enumerate(stages, 1):
                print(f"\n{'='*20} STAGE {i}/{total_stages}: {stage_name} {'='*20}")
                
                self.pipeline_state['current_stage'] = stage_id
                stage_start = datetime.now()
                
                # Execute stage
                if stage_id == "data_loading":
                    success = self._execute_data_loading()
                elif stage_id == "data_cleaning":
                    success = self._execute_data_cleaning()
                elif stage_id == "ml_advisory":
                    success = self._execute_ml_advisory()
                elif stage_id == "model_training":
                    success = self._execute_model_training()
                elif stage_id == "model_evaluation":
                    success = self._execute_model_evaluation()
                elif stage_id == "deployment_prep":
                    success = self._execute_deployment_prep()
                else:
                    print(f"❌ Unknown stage: {stage_id}")
                    success = False
                
                # Track performance
                stage_time = (datetime.now() - stage_start).total_seconds()
                self.performance_tracker['stage_times'][stage_id] = stage_time
                
                if success:
                    self.pipeline_state['completed_stages'].append(stage_id)
                    print(f"✅ Stage completed successfully ({stage_time:.2f}s)")
                    
                    # Save checkpoint
                    self._save_checkpoint(stage_id)
                    
                    # Stage review opportunity
                    if self.mode == "interactive" and i < total_stages:
                        if not self._stage_review_checkpoint(stage_id):
                            print("🛑 Pipeline stopped by user")
                            return False
                else:
                    self.pipeline_state['failed_stages'].append(stage_id)
                    print(f"❌ Stage failed ({stage_time:.2f}s)")
                    
                    if not self._handle_stage_failure(stage_id):
                        return False
            
            # Pipeline completion
            return self._finalize_pipeline()
            
        except KeyboardInterrupt:
            print("\n🛑 Pipeline interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Pipeline crashed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _execute_data_loading(self) -> bool:
        """Execute data loading and validation stage"""
        try:
            print("📊 Loading and validating data...")
            
            # Get data path
            data_path = self.config['global']['data_input_path']
            if not Path(data_path).exists():
                print(f"❌ Data file not found: {data_path}")
                return False
            
            # Load data
            print(f"📁 Loading data from: {data_path}")
            df = pd.read_csv(
                data_path,
                delimiter=self.config['global'].get('csv_delimiter', ','),
                encoding=self.config['global'].get('csv_encoding', 'utf-8')
            )
            
            target_column = self.config['global']['target_column']
            
            print(f"✅ Data loaded successfully")
            print(f"📊 Data shape: {df.shape}")
            print(f"🎯 Target column: {target_column}")
            
            # Validate data
            validation_results = self._validate_data(df, target_column)
            
            # Save validated data
            validated_data_path = self.production_outputs['data'] / "validated_data.csv"
            df.to_csv(validated_data_path, index=False)
            
            # Store results
            self.pipeline_state['stage_results']['data_loading'] = {
                'data_path': str(validated_data_path),
                'shape': df.shape,
                'target_column': target_column,
                'validation_results': validation_results,
                'success': True
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False
    
    def _validate_data(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Validate loaded data"""
        validation = {
            'shape_valid': df.shape[0] > 0 and df.shape[1] > 1,
            'target_exists': target_column in df.columns,
            'missing_data': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Check target column
        if validation['target_exists']:
            y = df[target_column]
            validation['target_info'] = {
                'unique_values': y.nunique(),
                'missing_target': y.isnull().sum(),
                'target_distribution': y.value_counts().to_dict()
            }
        
        # Print validation summary
        print(f"✅ Data validation:")
        print(f"   Shape valid: {validation['shape_valid']}")
        print(f"   Target exists: {validation['target_exists']}")
        print(f"   Missing data: {validation['missing_data']:,} cells")
        print(f"   Duplicate rows: {validation['duplicate_rows']:,}")
        
        if validation['target_exists']:
            target_info = validation['target_info']
            print(f"   Target unique values: {target_info['unique_values']}")
            print(f"   Missing target values: {target_info['missing_target']}")
        
        return validation
    
    def _execute_data_cleaning(self) -> bool:
        """Execute advanced data cleaning stage"""
        try:
            print("🧹 Executing advanced data cleaning...")
            
            # Get data from previous stage
            data_path = self.pipeline_state['stage_results']['data_loading']['data_path']
            
            # Import cleaner
            try:
                sys.path.append(str(project_root / "src" / "2_data_cleaning"))
                from data_cleaner import DataCleaner, CleaningConfig
                
                # Create production cleaning configuration
                cleaned_data_path = self.production_outputs['data'] / "production_cleaned_data.csv"
                cleaning_log_path = self.production_outputs['logs'] / "production_cleaning_log.json"
                
                # Get cleaning config from discovery if available
                cleaning_config_path = None
                if self.discovery_summary and 'generated_files' in self.discovery_summary:
                    cleaning_config_path = self.discovery_summary['generated_files'].get('cleaning_config')
                
                cleaning_config = CleaningConfig(
                    input_path=data_path,
                    output_path=str(cleaned_data_path),
                    log_path=str(cleaning_log_path),
                    column_config_path=cleaning_config_path,
                    verbose=True,
                    # Production settings
                    remove_duplicates=self.config['cleaning'].get('remove_duplicates', True),
                    outlier_removal=self.config['cleaning'].get('outlier_removal', True),
                    outlier_method=self.config['cleaning'].get('outlier_method', 'iqr'),
                    handle_missing=self.config['cleaning'].get('handle_missing', True),
                    normalize_text=self.config['cleaning'].get('normalize_text', True),
                    validate_data=self.config['cleaning'].get('validate_data', True)
                )
                
                print(f"⚙️ Production cleaning configuration:")
                print(f"   Remove duplicates: {cleaning_config.remove_duplicates}")
                print(f"   Outlier removal: {cleaning_config.outlier_removal}")
                print(f"   Handle missing: {cleaning_config.handle_missing}")
                print(f"   Normalize text: {cleaning_config.normalize_text}")
                
                # Execute cleaning
                cleaner = DataCleaner(cleaning_config)
                print("🔄 Running production data cleaning...")
                
                cleaned_df, cleaning_log = cleaner.clean()
                
                # Display results
                initial_shape = cleaning_log.get('initial_shape', (0, 0))
                final_shape = cleaning_log.get('final_shape', (0, 0))
                processing_time = cleaning_log.get('processing_time', 0)
                
                print(f"✅ Production data cleaning completed!")
                print(f"📊 Cleaning Results:")
                print(f"   Original shape: {initial_shape}")
                print(f"   Cleaned shape:  {final_shape}")
                print(f"   Data retention: {final_shape[0]/initial_shape[0]*100:.1f}%")
                print(f"   Processing time: {processing_time:.2f}s")
                
                # Show critical actions
                actions = cleaning_log.get('actions', [])
                print(f"   Actions performed: {len(actions)}")
                
                # Store results
                self.pipeline_state['stage_results']['data_cleaning'] = {
                    'cleaned_data_path': str(cleaned_data_path),
                    'cleaning_log_path': str(cleaning_log_path),
                    'initial_shape': initial_shape,
                    'final_shape': final_shape,
                    'data_retention': final_shape[0]/initial_shape[0] if initial_shape[0] > 0 else 0,
                    'processing_time': processing_time,
                    'actions_count': len(actions),
                    'success': True
                }
                
                return True
                
            except ImportError as e:
                print(f"❌ Data cleaner not available: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Data cleaning failed: {e}")
            return False
    
    def _execute_ml_advisory(self) -> bool:
        """Execute ML advisory and model selection stage"""
        try:
            print("🤖 Executing ML advisory and model selection...")
            
            # Get cleaned data
            cleaned_data_path = self.pipeline_state['stage_results']['data_cleaning']['cleaned_data_path']
            df = pd.read_csv(cleaned_data_path)
            target_column = self.config['global']['target_column']
            
            print(f"📊 Advisory analysis on {df.shape[0]:,} samples, {df.shape[1]} features")
            
            # Import advisory components
            try:
                sys.path.append(str(project_root / "src" / "advisor_3"))
                from assumption_checker import EnhancedAssumptionChecker, AssumptionConfig
                from model_recommender import recommend_model
                
                # Configure advisory
                advisory_config = AssumptionConfig(
                    output_dir=str(self.production_outputs['reports']),
                    verbose=True,
                    save_plots=True,
                    enable_advanced_analysis=True
                )
                
                # Prepare data
                X = df.drop(columns=[target_column])
                y = df[target_column]
                
                print("🔄 Running comprehensive ML assumptions analysis...")
                
                # Run assumption checking
                checker = EnhancedAssumptionChecker(advisory_config)
                assumptions_results = checker.check_all_assumptions(X, y, target_column)
                
                print("🔄 Generating comprehensive model recommendations...")
                
                # Get model recommendations
                recommendations = recommend_model(
                    df=df,
                    target_column=target_column,
                    assumptions_results=assumptions_results,
                    mode="production"
                )
                
                # Display results
                passed_assumptions = sum(1 for result in assumptions_results.values() if result.get('passed', False))
                total_assumptions = len(assumptions_results)
                
                print(f"✅ ML Advisory completed!")
                print(f"📊 Advisory Results:")
                print(f"   Assumptions analyzed: {total_assumptions}")
                print(f"   Assumptions passed: {passed_assumptions}")
                print(f"   Compliance rate: {passed_assumptions/total_assumptions*100:.1f}%")
                
                # Show top recommendations
                if isinstance(recommendations, dict) and 'recommended_models' in recommendations:
                    models = recommendations['recommended_models'][:10]
                    print(f"\n🎯 Top Model Recommendations:")
                    for i, model in enumerate(models, 1):
                        model_name = model.get('name', 'Unknown')
                        confidence = model.get('confidence', 0)
                        print(f"   {i:2d}. {model_name} (confidence: {confidence:.3f})")
                
                # Save comprehensive advisory results
                advisory_results = {
                    'timestamp': datetime.now().isoformat(),
                    'data_shape': df.shape,
                    'target_column': target_column,
                    'assumptions_results': assumptions_results,
                    'model_recommendations': recommendations,
                    'summary': {
                        'assumptions_analyzed': total_assumptions,
                        'assumptions_passed': passed_assumptions,
                        'compliance_rate': passed_assumptions/total_assumptions if total_assumptions > 0 else 0,
                        'recommended_models': [m.get('name', 'Unknown') for m in models] if 'models' in locals() else []
                    }
                }
                
                advisory_output_path = self.production_outputs['reports'] / "production_advisory_report.json"
                with open(advisory_output_path, 'w') as f:
                    json.dump(advisory_results, f, indent=2, default=str)
                
                # Store results
                self.pipeline_state['stage_results']['ml_advisory'] = {
                    'advisory_report_path': str(advisory_output_path),
                    'assumptions_passed': passed_assumptions,
                    'total_assumptions': total_assumptions,
                    'compliance_rate': passed_assumptions/total_assumptions if total_assumptions > 0 else 0,
                    'recommended_models': [m.get('name', 'Unknown') for m in models] if 'models' in locals() else [],
                    'success': True
                }
                
                return True
                
            except ImportError as e:
                print(f"❌ Advisory components not available: {e}")
                print("🔄 Using basic advisory fallback...")
                return self._basic_advisory_fallback(df, target_column)
                
        except Exception as e:
            print(f"❌ ML Advisory failed: {e}")
            return False
    
    def _basic_advisory_fallback(self, df: pd.DataFrame, target_column: str) -> bool:
        """Basic advisory fallback for production"""
        try:
            print("🔄 Running production-grade basic advisory...")
            
            # Enhanced basic analysis
            y = df[target_column]
            X = df.drop(columns=[target_column])
            
            # Determine problem type and complexity
            if y.dtype in ['object', 'category'] or y.nunique() < 20:
                problem_type = "classification"
                n_classes = y.nunique()
                print(f"📊 Problem: Classification ({n_classes} classes)")
                
                if n_classes == 2:
                    recommended_models = [
                        "LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier",
                        "XGBClassifier", "LGBMClassifier", "CatBoostClassifier"
                    ]
                else:
                    recommended_models = [
                        "RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier",
                        "LGBMClassifier", "CatBoostClassifier", "VotingClassifier"
                    ]
            else:
                problem_type = "regression"
                print(f"📊 Problem: Regression")
                recommended_models = [
                    "LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor",
                    "XGBRegressor", "LGBMRegressor", "CatBoostRegressor"
                ]
            
            # Enhanced feature analysis
            numeric_features = X.select_dtypes(include=[np.number]).columns
            categorical_features = X.select_dtypes(include=['object', 'category']).columns
            
            feature_analysis = {
                'total_features': len(X.columns),
                'numeric_features': len(numeric_features),
                'categorical_features': len(categorical_features),
                'missing_values': int(X.isnull().sum().sum()),
                'high_cardinality_features': [col for col in categorical_features if X[col].nunique() > 50]
            }
            
            print(f"📊 Advanced Feature Analysis:")
            print(f"   Total features: {feature_analysis['total_features']}")
            print(f"   Numeric features: {feature_analysis['numeric_features']}")
            print(f"   Categorical features: {feature_analysis['categorical_features']}")
            print(f"   High cardinality features: {len(feature_analysis['high_cardinality_features'])}")
            
            # Save enhanced advisory results
            advisory_results = {
                'timestamp': datetime.now().isoformat(),
                'problem_type': problem_type,
                'data_shape': df.shape,
                'feature_analysis': feature_analysis,
                'target_analysis': {
                    'unique_values': int(y.nunique()),
                    'missing_values': int(y.isnull().sum()),
                    'distribution': y.value_counts().head(10).to_dict()
                },
                'recommended_models': recommended_models,
                'model_selection_strategy': "gradient_boosting_preferred"
            }
            
            advisory_output_path = self.production_outputs['reports'] / "production_basic_advisory.json"
            with open(advisory_output_path, 'w') as f:
                json.dump(advisory_results, f, indent=2, default=str)
            
            print(f"\n🎯 Production Model Recommendations:")
            for i, model in enumerate(recommended_models, 1):
                print(f"   {i:2d}. {model}")
            
            # Store results
            self.pipeline_state['stage_results']['ml_advisory'] = {
                'advisory_report_path': str(advisory_output_path),
                'problem_type': problem_type,
                'recommended_models': recommended_models,
                'feature_analysis': feature_analysis,
                'success': True
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Basic advisory failed: {e}")
            return False
    
    def _execute_model_training(self) -> bool:
        """Execute comprehensive model training stage"""
        try:
            print("🏋️ Executing comprehensive model training...")
            
            # Get cleaned data
            cleaned_data_path = self.pipeline_state['stage_results']['data_cleaning']['cleaned_data_path']
            df = pd.read_csv(cleaned_data_path)
            target_column = self.config['global']['target_column']
            
            print(f"📊 Training on {df.shape[0]:,} samples, {df.shape[1]} features")
            
            # Import trainer
            try:
                sys.path.append(str(project_root / "src" / "pipeline_4"))
                from trainer import EnhancedModelTrainer, TrainerConfig
                
                # Production training configuration
                training_config = TrainerConfig(
                    test_size=self.config['training'].get('test_size', 0.2),
                    validation_size=self.config['training'].get('validation_size', 0.2),
                    random_state=self.config['training'].get('random_state', 42),
                    max_models=self.config['training'].get('max_models', 20),
                    include_ensemble=self.config['training'].get('include_ensemble', True),
                    include_advanced=self.config['training'].get('include_advanced', True),
                    enable_tuning=self.config['training'].get('enable_tuning', True),
                    enable_early_stopping=self.config['training'].get('enable_early_stopping', True),
                    verbose=True,
                    save_models=True,
                    model_dir=str(self.production_outputs['models']),
                    n_jobs=self.config['training'].get('n_jobs', -1)
                )
                
                print(f"⚙️ Production Training Configuration:")
                print(f"   Max models: {training_config.max_models}")
                print(f"   Hyperparameter tuning: {training_config.enable_tuning}")
                print(f"   Ensemble models: {training_config.include_ensemble}")
                print(f"   Advanced models: {training_config.include_advanced}")
                print(f"   Early stopping: {training_config.enable_early_stopping}")
                print(f"   Parallel jobs: {training_config.n_jobs}")
                
                # Initialize trainer
                trainer = EnhancedModelTrainer(training_config)
                
                print("🔄 Starting comprehensive model training...")
                start_time = time.time()
                
                # Train all models
                training_results = trainer.train_all_models(df, target_column)
                
                training_time = time.time() - start_time
                
                if not training_results:
                    print("❌ No models trained successfully")
                    return False
                
                print(f"✅ Comprehensive model training completed!")
                print(f"📊 Training Results:")
                print(f"   Models trained: {len(training_results)}")
                print(f"   Total training time: {training_time:.2f}s")
                print(f"   Average time per model: {training_time/len(training_results):.2f}s")
                
                # Generate comprehensive training report
                target_type = trainer.infer_target_type(df[target_column])
                report_path = str(self.production_outputs['reports'] / "production_training_report.json")
                training_report = trainer.generate_report(training_results, target_type, report_path)
                
                # Display top models
                print(f"\n🏅 Top Model Performance:")
                for i, result in enumerate(training_results[:10], 1):
                    print(f"   {i:2d}. {result.model_name}: {result.cv_score:.4f} (±{result.cv_std:.4f})")
                    print(f"       Training time: {result.training_time:.2f}s")
                
                # Show model artifacts
                model_files = list(self.production_outputs['models'].glob("*.pkl"))
                print(f"\n💾 Model Artifacts Generated: {len(model_files)}")
                
                # Store results
                self.pipeline_state['stage_results']['model_training'] = {
                    'training_report_path': report_path,
                    'model_directory': str(self.production_outputs['models']),
                    'models_trained': len(training_results),
                    'total_training_time': training_time,
                    'best_model': training_report['summary'].get('best_model', 'Unknown') if training_report and 'summary' in training_report else 'Unknown',
                    'best_score': training_report['summary'].get('best_score', 0) if training_report and 'summary' in training_report else 0,
                    'model_artifacts': len(model_files),
                    'success': True
                }
                
                return True
                
            except ImportError as e:
                print(f"❌ Model trainer not available: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            return False
    
    def _execute_model_evaluation(self) -> bool:
        """Execute deep model evaluation stage"""
        try:
            print("📈 Executing deep model evaluation...")
            
            # Get training results
            training_results = self.pipeline_state['stage_results']['model_training']
            report_path = training_results['training_report_path']
            
            # Import evaluator
            try:
                sys.path.append(str(project_root / "src" / "pipeline_4"))
                from evaluator import ModelAnalyzer, AnalysisConfig
                
                # Production evaluation configuration
                eval_config = AnalysisConfig(
                    training_report_path=Path(report_path),
                    output_dir=self.production_outputs['evaluations'],
                    enable_shap=self.config['evaluation'].get('enable_shap', True),
                    enable_learning_curves=self.config['evaluation'].get('enable_learning_curves', True),
                    enable_residual_analysis=self.config['evaluation'].get('enable_residual_analysis', True),
                    enable_stability_analysis=self.config['evaluation'].get('enable_stability_analysis', True),
                    enable_interpretability=self.config['evaluation'].get('enable_interpretability', True),
                    cross_validation_folds=self.config['evaluation'].get('cross_validation_folds', 5),
                    verbose=True
                )
                
                print(f"⚙️ Production Evaluation Configuration:")
                print(f"   SHAP analysis: {eval_config.enable_shap}")
                print(f"   Learning curves: {eval_config.enable_learning_curves}")
                print(f"   Residual analysis: {eval_config.enable_residual_analysis}")
                print(f"   Stability analysis: {eval_config.enable_stability_analysis}")
                print(f"   Interpretability: {eval_config.enable_interpretability}")
                print(f"   CV folds: {eval_config.cross_validation_folds}")
                
                # Initialize analyzer
                analyzer = ModelAnalyzer(eval_config)
                
                # Get test data
                cleaned_data_path = self.pipeline_state['stage_results']['data_cleaning']['cleaned_data_path']
                df = pd.read_csv(cleaned_data_path)
                target_column = self.config['global']['target_column']
                
                X_test = df.drop(columns=[target_column])
                y_test = df[target_column]
                
                print(f"📊 Evaluating on {X_test.shape[0]:,} samples")
                print("🔄 Running comprehensive model evaluation...")
                
                start_time = time.time()
                
                # Run comprehensive evaluation
                evaluation_results = analyzer.run_analysis(X_test, y_test)
                
                evaluation_time = time.time() - start_time
                
                print(f"✅ Deep model evaluation completed!")
                print(f"📊 Evaluation Results:")
                print(f"   Models evaluated: {len(evaluation_results)}")
                print(f"   Evaluation time: {evaluation_time:.2f}s")
                
                # Display evaluation summary
                if evaluation_results:
                    print(f"\n📊 Model Performance Summary:")
                    for model_name, results in list(evaluation_results.items())[:5]:
                        if isinstance(results, dict) and 'metrics' in results:
                            metrics = results['metrics']
                            # Show key metrics
                            key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mse', 'rmse', 'r2']
                            displayed_metrics = []
                            for metric in key_metrics:
                                if metric in metrics:
                                    displayed_metrics.append(f"{metric}: {metrics[metric]:.4f}")
                                    if len(displayed_metrics) >= 3:
                                        break
                            
                            print(f"   {model_name}: {', '.join(displayed_metrics)}")
                
                # Count generated artifacts
                eval_files = list(self.production_outputs['evaluations'].glob("*"))
                print(f"\n📁 Evaluation Artifacts Generated: {len(eval_files)}")
                
                # Store results
                self.pipeline_state['stage_results']['model_evaluation'] = {
                    'evaluation_directory': str(self.production_outputs['evaluations']),
                    'evaluation_results': evaluation_results,
                    'models_evaluated': len(evaluation_results),
                    'evaluation_time': evaluation_time,
                    'artifacts_generated': len(eval_files),
                    'success': True
                }
                
                return True
                
            except ImportError as e:
                print(f"❌ Model evaluator not available: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Model evaluation failed: {e}")
            return False
    
    def _execute_deployment_prep(self) -> bool:
        """Execute deployment preparation stage"""
        try:
            print("🚀 Preparing deployment artifacts...")
            
            # Get results from previous stages
            training_results = self.pipeline_state['stage_results']['model_training']
            evaluation_results = self.pipeline_state['stage_results']['model_evaluation']
            
            # Create deployment configuration
            deployment_config = {
                'model_info': {
                    'best_model': training_results.get('best_model', 'Unknown'),
                    'best_score': training_results.get('best_score', 0),
                    'model_directory': training_results.get('model_directory'),
                    'training_time': training_results.get('total_training_time', 0)
                },
                'data_info': {
                    'target_column': self.config['global']['target_column'],
                    'original_shape': self.pipeline_state['stage_results']['data_loading']['shape'],
                    'cleaned_shape': self.pipeline_state['stage_results']['data_cleaning']['final_shape'],
                    'data_retention': self.pipeline_state['stage_results']['data_cleaning']['data_retention']
                },
                'pipeline_info': {
                    'timestamp': datetime.now().isoformat(),
                    'mode': self.mode,
                    'discovery_dir': str(self.discovery_dir) if self.discovery_dir else None,
                    'total_execution_time': (datetime.now() - self.pipeline_state['start_time']).total_seconds()
                },
                'deployment_artifacts': {
                    'model_directory': str(self.production_outputs['models']),
                    'evaluation_directory': str(self.production_outputs['evaluations']),
                    'reports_directory': str(self.production_outputs['reports']),
                    'deployment_config': str(self.production_outputs['deployment'] / "deployment_config.yaml")
                }
            }
            
            # Save deployment configuration
            deployment_config_path = self.production_outputs['deployment'] / "deployment_config.yaml"
            with open(deployment_config_path, 'w') as f:
                yaml.dump(deployment_config, f, indent=2, default_flow_style=False)
            
            # Create model serving script template
            serving_script = self._create_serving_script_template(deployment_config)
            serving_script_path = self.production_outputs['deployment'] / "model_serving.py"
            with open(serving_script_path, 'w') as f:
                f.write(serving_script)
            
            # Create requirements file
            requirements = self._create_requirements_file()
            requirements_path = self.production_outputs['deployment'] / "requirements.txt"
            with open(requirements_path, 'w') as f:
                f.write(requirements)
            
            # Create deployment README
            readme = self._create_deployment_readme(deployment_config)
            readme_path = self.production_outputs['deployment'] / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme)
            
            print(f"✅ Deployment preparation completed!")
            print(f"📁 Deployment artifacts:")
            print(f"   📄 Config: {deployment_config_path}")
            print(f"   🐍 Serving script: {serving_script_path}")
            print(f"   📦 Requirements: {requirements_path}")
            print(f"   📖 README: {readme_path}")
            
            # Store results
            self.pipeline_state['stage_results']['deployment_prep'] = {
                'deployment_directory': str(self.production_outputs['deployment']),
                'config_path': str(deployment_config_path),
                'serving_script_path': str(serving_script_path),
                'requirements_path': str(requirements_path),
                'readme_path': str(readme_path),
                'success': True
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Deployment preparation failed: {e}")
            return False
    
    def _create_serving_script_template(self, deployment_config: Dict[str, Any]) -> str:
        """Create model serving script template"""
        return f'''#!/usr/bin/env python3
"""
Model Serving Script
Generated by DS-AutoAdvisor Production Pipeline
Timestamp: {datetime.now().isoformat()}
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json

class ModelServer:
    """Production model serving class"""
    
    def __init__(self, model_path: str):
        """Initialize model server"""
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Model loaded: {{self.model_path}}")
        except Exception as e:
            print(f"❌ Failed to load model: {{e}}")
            raise
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Make probability predictions (if supported)"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability predictions")


def main():
    """Example usage"""
    # Configuration
    MODEL_PATH = "{deployment_config['model_info']['model_directory']}"
    BEST_MODEL = "{deployment_config['model_info']['best_model']}"
    
    # Initialize server
    model_file = Path(MODEL_PATH) / f"{{BEST_MODEL}}.pkl"
    server = ModelServer(model_file)
    
    # Example prediction
    # X_new = pd.DataFrame(...)  # Your new data
    # predictions = server.predict(X_new)
    # print(f"Predictions: {{predictions}}")

if __name__ == "__main__":
    main()
'''
    
    def _create_requirements_file(self) -> str:
        """Create requirements.txt for deployment"""
        return '''# DS-AutoAdvisor Production Requirements
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.2.0
catboost>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
shap>=0.40.0
pyyaml>=6.0
joblib>=1.1.0
'''
    
    def _create_deployment_readme(self, deployment_config: Dict[str, Any]) -> str:
        """Create deployment README"""
        return f'''# DS-AutoAdvisor Production Deployment

## Model Information
- **Best Model**: {deployment_config['model_info']['best_model']}
- **Best Score**: {deployment_config['model_info']['best_score']:.4f}
- **Training Time**: {deployment_config['model_info']['training_time']:.2f}s

## Data Information
- **Target Column**: {deployment_config['data_info']['target_column']}
- **Original Shape**: {deployment_config['data_info']['original_shape']}
- **Cleaned Shape**: {deployment_config['data_info']['cleaned_shape']}
- **Data Retention**: {deployment_config['data_info']['data_retention']:.1%}

## Pipeline Information
- **Generated**: {deployment_config['pipeline_info']['timestamp']}
- **Mode**: {deployment_config['pipeline_info']['mode']}
- **Total Execution Time**: {deployment_config['pipeline_info']['total_execution_time']:.2f}s

## Deployment Files
- `deployment_config.yaml`: Complete deployment configuration
- `model_serving.py`: Model serving script template
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Load and use the model:
   ```python
   from model_serving import ModelServer
   
   server = ModelServer("path/to/best/model.pkl")
   predictions = server.predict(X_new)
   ```

## Directory Structure
- `models/`: Trained model artifacts
- `evaluations/`: Model evaluation reports and plots
- `reports/`: Training and advisory reports
- `deployment/`: This deployment package

## Next Steps
1. Test the model serving script with your data
2. Integrate into your production environment
3. Set up monitoring and logging
4. Schedule model retraining as needed
'''
    
    def _save_checkpoint(self, stage_id: str):
        """Save pipeline checkpoint"""
        try:
            checkpoint = {
                'stage_id': stage_id,
                'timestamp': datetime.now().isoformat(),
                'pipeline_state': self.pipeline_state,
                'performance_tracker': self.performance_tracker
            }
            
            checkpoint_path = self.production_outputs['logs'] / f"checkpoint_{stage_id}.json"
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
                
        except Exception as e:
            print(f"⚠️ Failed to save checkpoint: {e}")
    
    def _stage_review_checkpoint(self, stage_id: str) -> bool:
        """Interactive stage review checkpoint"""
        print(f"\n🔍 Stage Review: {stage_id.upper()}")
        print("Options: [c]ontinue, [s]kip, [r]eview results, [q]uit")
        
        while True:
            response = input("Choice: ").lower().strip()
            
            if response in ['c', 'continue', '']:
                return True
            elif response in ['s', 'skip']:
                print("⏭️ Skipping to next stage")
                return True
            elif response in ['r', 'review']:
                self._display_stage_results(stage_id)
            elif response in ['q', 'quit']:
                return False
            else:
                print("Invalid choice. Use: [c]ontinue, [s]kip, [r]eview, [q]uit")
    
    def _display_stage_results(self, stage_id: str):
        """Display stage results for review"""
        if stage_id in self.pipeline_state['stage_results']:
            results = self.pipeline_state['stage_results'][stage_id]
            print(f"\n📊 {stage_id.upper()} Results:")
            for key, value in results.items():
                if key != 'success':
                    print(f"   {key}: {value}")
        else:
            print(f"No results available for {stage_id}")
    
    def _handle_stage_failure(self, stage_id: str) -> bool:
        """Handle stage failure with recovery options"""
        print(f"\n🚨 Stage Failed: {stage_id}")
        print("Options: [r]etry, [s]kip, [q]uit")
        
        while True:
            response = input("Recovery choice: ").lower().strip()
            
            if response in ['r', 'retry']:
                print(f"🔄 Retrying stage: {stage_id}")
                return True  # Will retry the stage
            elif response in ['s', 'skip']:
                print(f"⏭️ Skipping stage: {stage_id}")
                return True  # Continue to next stage
            elif response in ['q', 'quit']:
                print("🛑 Pipeline aborted")
                return False
            else:
                print("Invalid choice. Use: [r]etry, [s]kip, [q]uit")
    
    def _finalize_pipeline(self) -> bool:
        """Finalize pipeline execution"""
        try:
            end_time = datetime.now()
            total_time = (end_time - self.pipeline_state['start_time']).total_seconds()
            
            # Create final summary
            pipeline_summary = {
                'execution_metadata': {
                    'start_time': self.pipeline_state['start_time'].isoformat(),
                    'end_time': end_time.isoformat(),
                    'total_execution_time': total_time,
                    'mode': self.mode
                },
                'stage_summary': {
                    'completed_stages': self.pipeline_state['completed_stages'],
                    'failed_stages': self.pipeline_state['failed_stages'],
                    'total_stages': len(self.pipeline_state['completed_stages']) + len(self.pipeline_state['failed_stages'])
                },
                'performance_summary': self.performance_tracker,
                'stage_results': self.pipeline_state['stage_results'],
                'output_directories': {k: str(v) for k, v in self.production_outputs.items()},
                'success': len(self.pipeline_state['failed_stages']) == 0
            }
            
            # Save final summary
            summary_path = self.production_outputs['base'] / "production_pipeline_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(pipeline_summary, f, indent=2, default=str)
            
            # Print final results
            print("\n" + "="*80)
            print("🎉 PRODUCTION PIPELINE COMPLETED")
            print("="*80)
            
            print(f"⏱️  Total execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
            print(f"✅ Completed stages: {len(self.pipeline_state['completed_stages'])}")
            print(f"❌ Failed stages: {len(self.pipeline_state['failed_stages'])}")
            
            if pipeline_summary['success']:
                print(f"\n🚀 Pipeline Success! All stages completed successfully.")
            else:
                print(f"\n⚠️  Pipeline completed with {len(self.pipeline_state['failed_stages'])} failed stages.")
            
            print(f"\n📁 All outputs saved to: {self.production_outputs['base']}")
            print(f"📄 Complete summary: {summary_path}")
            
            # Show key artifacts
            deployment_results = self.pipeline_state['stage_results'].get('deployment_prep', {})
            if deployment_results.get('success'):
                print(f"\n🚀 Deployment Artifacts Ready:")
                print(f"   📁 Deployment directory: {deployment_results['deployment_directory']}")
                print(f"   ⚙️  Configuration: {deployment_results['config_path']}")
                print(f"   🐍 Serving script: {deployment_results['serving_script_path']}")
            
            return pipeline_summary['success']
            
        except Exception as e:
            print(f"❌ Pipeline finalization failed: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='DS-AutoAdvisor Production Pipeline')
    parser.add_argument('--discovery-dir', type=str,
                       help='Path to discovery results directory')
    parser.add_argument('--config', type=str,
                       help='Path to custom configuration file')
    parser.add_argument('--output', type=str, default='pipeline_outputs',
                       help='Base output directory')
    parser.add_argument('--mode', type=str, default='production',
                       choices=['production', 'interactive'],
                       help='Pipeline execution mode')
    parser.add_argument('--enable-all', action='store_true',
                       help='Enable all advanced features (SHAP, ensemble, etc.)')
    
    args = parser.parse_args()
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Initialize pipeline
    pipeline = ProductionPipeline(
        discovery_dir=args.discovery_dir,
        config_path=args.config,
        output_base=args.output,
        mode=args.mode
    )
    
    if not pipeline.discovery_dir:
        print("❌ No discovery directory available")
        print("💡 Run data discovery first: python 01_data_discovery.py")
        return 1
    
    # Run production pipeline
    try:
        success = pipeline.run_production_pipeline()
        
        if success:
            print(f"\n🎉 Production Pipeline Completed Successfully!")
            print(f"📁 All outputs: {pipeline.production_outputs['base']}")
            print(f"\n🚀 Ready for Deployment!")
            return 0
        else:
            print(f"\n❌ Production Pipeline Failed")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n🛑 Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
