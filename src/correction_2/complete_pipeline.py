"""
Complete Data Science Pipeline Workflow
======================================

This shows how all the files work together in a typical ML project.

STAGE 1: DATA VALIDATION (enhanced_validation.py)
├── Check file exists, readable
├── Validate dataframe structure  
├── Check for obvious issues (empty, all nulls)
├── Memory usage warnings
└── Configuration validation

STAGE 2: BASIC DATA CLEANING (data_cleaner_improved.py) 
├── Remove duplicates
├── Handle missing values (imputation)
├── Remove/treat outliers  
├── Basic encoding (label, one-hot)
├── Scaling (standard, minmax, robust)
├── Type optimization
├── Date parsing
└── Save cleaned dataset

STAGE 3: ADVANCED ML PREPROCESSING (advanced_ml_preprocessing.py)
├── Feature Selection (chi2, mutual info, RFE)
├── Advanced Encoding (target, binary, hash)
├── Feature Engineering (polynomial, interactions)
├── Time Series Features (lags, cyclical)
├── Imbalanced Data Handling (SMOTE, undersampling)
├── Dimensionality Reduction (PCA, if needed)
└── Final ML-ready dataset

STAGE 4: MODEL TRAINING & EVALUATION
├── Train/validation/test split
├── Model selection & hyperparameter tuning
├── Cross-validation
├── Performance evaluation
└── Model deployment

"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, Any, Tuple

# Import our custom modules
from enhanced_validation import DataValidator, ValidationError, ErrorSeverity
from data_cleaner import DataCleaner, CleaningConfig
from advanced_ml_preprocessing import (
    AdvancedMLPreprocessor, FeatureSelectionStep, 
    AdvancedEncodingStep, TimeSeriesFeaturesStep, ImbalancedDataStep
)

class DataSciencePipeline:
    """Complete end-to-end data science pipeline"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.pipeline_log = {
            "validation": {},
            "cleaning": {},
            "ml_preprocessing": {},
            "final_summary": {}
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_complete_pipeline(self) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Execute the complete data science pipeline"""
        
        print("🚀 Starting Complete Data Science Pipeline...")
        
        # STAGE 1: VALIDATION
        print("\n📋 Stage 1: Data Validation")
        df_raw = self._stage1_validation()
        
        # STAGE 2: BASIC CLEANING  
        print("\n🧹 Stage 2: Basic Data Cleaning")
        df_clean = self._stage2_cleaning(df_raw)
        
        # STAGE 3: ADVANCED ML PREPROCESSING
        print("\n🔬 Stage 3: Advanced ML Preprocessing")
        X_final, y_final = self._stage3_ml_preprocessing(df_clean)
        
        # STAGE 4: FINAL SUMMARY
        print("\n📊 Stage 4: Pipeline Summary")
        self._stage4_summary(df_raw, X_final, y_final)
        
        return X_final, y_final, self.pipeline_log
    
    def _stage1_validation(self) -> pd.DataFrame:
        """Stage 1: Data validation and quality checks"""
        try:
            # Load raw data
            df = pd.read_csv(self.config['data']['input_path'])
            
            # Validate dataframe
            issues = DataValidator.validate_dataframe(df, self.config)
            
            # Validate configuration
            config_issues = DataValidator.validate_config(self.config['cleaning'])
            
            all_issues = issues + config_issues
            
            # Log issues by severity
            critical_issues = [i for i in all_issues if i['severity'] == ErrorSeverity.CRITICAL]
            error_issues = [i for i in all_issues if i['severity'] == ErrorSeverity.ERROR]
            warning_issues = [i for i in all_issues if i['severity'] == ErrorSeverity.WARNING]
            
            self.pipeline_log['validation'] = {
                'critical': critical_issues,
                'errors': error_issues, 
                'warnings': warning_issues,
                'total_issues': len(all_issues)
            }
            
            # Stop if critical issues
            if critical_issues:
                raise ValidationError(f"Critical validation errors: {[i['message'] for i in critical_issues]}")
            
            print(f"✅ Validation complete: {len(warning_issues)} warnings, {len(error_issues)} errors")
            
            return df
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            raise
    
    def _stage2_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 2: Basic data cleaning"""
        try:
            # Create cleaning configuration
            cleaning_config = CleaningConfig(**self.config['cleaning'])
            
            # Initialize and run cleaner
            cleaner = DataCleaner(cleaning_config)
            df_clean, cleaning_log = cleaner.clean()
            
            self.pipeline_log['cleaning'] = cleaning_log
            
            print(f"✅ Cleaning complete: {cleaning_log['initial_shape']} → {cleaning_log['final_shape']}")
            
            return df_clean
            
        except Exception as e:
            print(f"❌ Cleaning failed: {e}")
            raise
    
    def _stage3_ml_preprocessing(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Stage 3: Advanced ML preprocessing"""
        try:
            # Separate features and target
            target_col = self.config['ml_preprocessing']['target_column']
            
            if target_col and target_col in df.columns:
                X = df.drop(columns=[target_col])
                y = df[target_col]
            else:
                X = df
                y = None
                print("⚠️ No target column specified - proceeding without target")
            
            # Create ML preprocessing steps
            steps = []
            
            ml_config = self.config['ml_preprocessing']
            
            if ml_config.get('feature_selection', {}).get('enabled', False):
                fs_config = ml_config['feature_selection']
                steps.append(FeatureSelectionStep(
                    method=fs_config.get('method', 'mutual_info'),
                    k=fs_config.get('k', 10)
                ))
            
            if ml_config.get('advanced_encoding', {}).get('enabled', False):
                enc_config = ml_config['advanced_encoding']
                steps.append(AdvancedEncodingStep(
                    method=enc_config.get('method', 'target'),
                    max_cardinality=enc_config.get('max_cardinality', 50)
                ))
            
            if ml_config.get('time_series', {}).get('enabled', False):
                ts_config = ml_config['time_series']
                steps.append(TimeSeriesFeaturesStep(
                    lag_periods=ts_config.get('lag_periods', [1, 7])
                ))
            
            if ml_config.get('imbalanced_data', {}).get('enabled', False) and y is not None:
                ib_config = ml_config['imbalanced_data']
                steps.append(ImbalancedDataStep(
                    method=ib_config.get('method', 'smote')
                ))
            
            # Run ML preprocessing
            if steps:
                preprocessor = AdvancedMLPreprocessor(steps)
                X_processed, y_processed = preprocessor.fit_transform(X, y)
                
                self.pipeline_log['ml_preprocessing'] = preprocessor.log
                print(f"✅ ML preprocessing complete: {len(steps)} steps applied")
            else:
                X_processed, y_processed = X, y
                print("⚠️ No ML preprocessing steps configured")
            
            return X_processed, y_processed
            
        except Exception as e:
            print(f"❌ ML preprocessing failed: {e}")
            raise
    
    def _stage4_summary(self, df_raw: pd.DataFrame, X_final: pd.DataFrame, y_final: pd.Series):
        """Stage 4: Generate pipeline summary"""
        
        summary = {
            'initial_shape': df_raw.shape,
            'final_shape': (X_final.shape[0], X_final.shape[1] + (1 if y_final is not None else 0)),
            'features_created': X_final.shape[1] - (df_raw.shape[1] - (1 if y_final is not None else 0)),
            'rows_changed': X_final.shape[0] - df_raw.shape[0],
            'processing_stages': {
                'validation_issues': len(self.pipeline_log['validation'].get('warnings', [])) + 
                                   len(self.pipeline_log['validation'].get('errors', [])),
                'cleaning_actions': len(self.pipeline_log['cleaning'].get('actions', [])),
                'ml_preprocessing_actions': len(self.pipeline_log['ml_preprocessing'].get('actions', []))
            }
        }
        
        self.pipeline_log['final_summary'] = summary
        
        print("\n" + "="*50)
        print("📊 PIPELINE SUMMARY")
        print("="*50)
        print(f"Initial shape: {summary['initial_shape']}")
        print(f"Final shape: {summary['final_shape']}")
        print(f"Features created: {summary['features_created']}")
        print(f"Rows changed: {summary['rows_changed']}")
        print(f"Total processing actions: {sum(summary['processing_stages'].values())}")
        print("="*50)

# Example configuration file template
def create_example_config():
    """Create an example configuration file"""
    config = {
        'data': {
            'input_path': '../../data/bank.csv',
            'output_path': '../../data/bank_ml_ready.csv'
        },
        'cleaning': {
            'input_path': '../../data/bank.csv',
            'output_path': '../../data/bank_cleaned.csv', 
            'log_path': '../../docs/cleaning_log.json',
            'remove_duplicates': True,
            'outlier_removal': True,
            'outlier_method': 'iqr',
            'scaling': 'standard',
            'encoding': 'onehot',
            'verbose': True
        },
        'ml_preprocessing': {
            'target_column': 'target',
            'feature_selection': {
                'enabled': True,
                'method': 'mutual_info',
                'k': 15
            },
            'advanced_encoding': {
                'enabled': True,
                'method': 'target',
                'max_cardinality': 50
            },
            'time_series': {
                'enabled': False,
                'lag_periods': [1, 7, 30]
            },
            'imbalanced_data': {
                'enabled': True,
                'method': 'smote'
            }
        }
    }
    
    with open('../../config/pipeline_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print("✅ Example configuration created at ../../config/pipeline_config.yaml")

if __name__ == "__main__":
    # Create example config if it doesn't exist
    config_path = "../../config/pipeline_config.yaml"
    if not Path(config_path).exists():
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        create_example_config()
    
    # Run complete pipeline
    # pipeline = DataSciencePipeline(config_path)
    # X_final, y_final, log = pipeline.run_complete_pipeline()
    
    print("🎯 Complete pipeline framework ready!")
