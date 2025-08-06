#!/usr/bin/env python3
"""
🔧 DS-AutoAdvisor: Step 2 - Configure Column-Specific Cleaning
============================================================

WHAT IT DOES:
Generates machine-readable data profiling and creates customizable YAML configuration
for column-specific data cleaning. Allows human review and modification of cleaning
strategies before applying transformations.

WHEN TO USE:
- After data inspection (Step 1)
- Before testing the pipeline (Step 3)
- When you need column-specific control over data transformations
- For iterative configuration refinement

HOW TO USE:
Basic configuration:
    python 2_configure_cleaning.py --data data/your_data.csv

Review and retry:
    python 2_configure_cleaning.py --data data/your_data.csv --retry

Configure specific columns:
    python 2_configure_cleaning.py --data data/your_data.csv --columns age,balance,job

INTERACTIVE FEATURES:
- Automatic data profiling with machine-readable output
- Human-in-the-loop configuration review
- Column-specific transformation recommendations
- Iterative configuration refinement
- YAML template generation and validation
- Intelligent column dropping for unique identifiers

WHAT YOU GET:
✅ Comprehensive data profiling report (HTML)
✅ Machine-readable profiling data (JSON)
✅ Column-specific cleaning configuration (YAML)
✅ Human-reviewable transformation recommendations
✅ Validated configuration for next steps
✅ Automatic unique identifier detection and removal suggestions

NEXT STEP:
After configuration, run: python 3_test_pipeline.py
"""

# Add src to path for imports
import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# Add source directories to Python path
base_dir = Path(__file__).parent
sys.path.append(str(base_dir / "src" / "1_data_profiling"))
sys.path.append(str(base_dir / "src" / "2_data_cleaning"))

from enhanced_data_profiler import EnhancedDataProfiler
from data_cleaner import DataCleaner, CleaningConfig
from typing import List
import webbrowser
import subprocess

def _load_config():
    """Load configuration from unified_config_v2.yaml"""
    config_path = base_dir / "config" / "unified_config_v2.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"⚠️ Could not load config from {config_path}: {e}")
        return {}

def _open_html_report(html_path: Path):
    """Open HTML report in browser"""
    try:
        webbrowser.open(f"file://{html_path.absolute()}")
        print(f"   📖 Opening HTML report in browser...")
    except Exception as e:
        print(f"   ⚠️ Could not open browser: {e}")
        print(f"   📁 Manual open: {html_path}")

def _edit_yaml_config(yaml_path: Path):
    """Open YAML config in default editor"""
    try:
        # Try VS Code first, then system default
        try:
            subprocess.run(["code", str(yaml_path)], check=True)
            print(f"   📝 Opening in VS Code...")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to system default
            if os.name == 'nt':  # Windows
                os.startfile(str(yaml_path))
            elif os.name == 'posix':  # Linux/Mac
                subprocess.run(["xdg-open", str(yaml_path)])
            print(f"   📝 Opening in default editor...")
    except Exception as e:
        print(f"   ⚠️ Could not open editor: {e}")
        print(f"   📁 Manual edit: {yaml_path}")

def _validate_configuration(yaml_path: Path, data_path: Path, verbose: bool = True):
    """Validate YAML configuration against data"""
    try:
        import yaml
        import pandas as pd
        
        # Load global configuration to get CSV settings
        global_config = _load_config()
        csv_delimiter = global_config.get('global', {}).get('csv_delimiter', ',')
        csv_encoding = global_config.get('global', {}).get('csv_encoding', 'utf-8')
        
        # Load configuration
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load data sample with correct delimiter
        df = pd.read_csv(data_path, delimiter=csv_delimiter, encoding=csv_encoding).head(100)
        
        column_configs = config.get('column_transformations', {})
        data_columns = set(df.columns)
        config_columns = set(column_configs.keys())
        
        if verbose:
            print(f"   📊 Data columns: {len(data_columns)}")
            print(f"   ⚙️ Configured columns: {len(config_columns)}")
            print(f"   🔧 Using delimiter: '{csv_delimiter}', encoding: {csv_encoding}")
            
            missing_in_config = data_columns - config_columns
            missing_in_data = config_columns - data_columns
            
            if missing_in_config:
                print(f"   ⚠️ Columns without config: {len(missing_in_config)}")
                for col in list(missing_in_config)[:3]:
                    print(f"      - {col}")
                if len(missing_in_config) > 3:
                    print(f"      ... and {len(missing_in_config) - 3} more")
            
            if missing_in_data:
                print(f"   ❌ Config for missing columns: {len(missing_in_data)}")
                for col in list(missing_in_data)[:3]:
                    print(f"      - {col}")
        
        return len(missing_in_data) == 0
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Validation error: {e}")
        return False

def _preview_cleaning_effects(data_path: Path, yaml_path: Path):
    """Preview effects of cleaning configuration on small sample"""
    try:
        print(f"   🔍 Generating cleaning preview...")
        # This would implement a preview of cleaning effects
        # For now, just show that it would work
        print(f"   📊 Preview generated (feature under development)")
        print(f"   💡 Tip: Review configuration and test with small sample")
    except Exception as e:
        print(f"   ⚠️ Preview error: {e}")

def _apply_cleaning_and_save(data_path: Path, yaml_path: Path, output_path: Path):
    """Apply YAML configuration to clean data and save cleaned dataset"""
    try:
        print(f"   🧹 Applying cleaning configuration...")
        
        # Load global configuration for CSV settings
        global_config = _load_config()
        csv_delimiter = global_config.get('global', {}).get('csv_delimiter', ',')
        csv_encoding = global_config.get('global', {}).get('csv_encoding', 'utf-8')
        
        # Load original data
        print(f"   📊 Loading data: {data_path}")
        df = pd.read_csv(data_path, delimiter=csv_delimiter, encoding=csv_encoding)
        original_shape = df.shape
        print(f"   📈 Original data: {original_shape}")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        cleaned_file = output_path / "cleaned_data.csv"
        log_file = output_path / "cleaning_log.json"
        
        # Create CleaningConfig with paths to YAML configuration
        print(f"   ⚙️ Loading configuration: {yaml_path}")
        cleaning_config = CleaningConfig(
            input_path=str(data_path),
            output_path=str(cleaned_file),
            log_path=str(log_file),
            column_config_path=str(yaml_path),  # Point to our YAML config
            remove_duplicates=True,
            outlier_removal=True,
            outlier_method="iqr"
        )
        
        # Create data cleaner instance
        cleaner = DataCleaner(cleaning_config)
        
        # Apply cleaning transformations
        print(f"   🔧 Applying transformations...")
        cleaned_df, cleaning_log = cleaner.clean()
        
        # Show cleaning results
        cleaned_shape = cleaned_df.shape
        print(f"   ✅ Cleaned data: {cleaned_shape}")
        print(f"   📊 Shape change: {original_shape} → {cleaned_shape}")
        
        if original_shape[1] != cleaned_shape[1]:
            dropped_cols = original_shape[1] - cleaned_shape[1]
            print(f"   🗑️ Columns dropped: {dropped_cols}")
        
        if original_shape[0] != cleaned_shape[0]:
            dropped_rows = original_shape[0] - cleaned_shape[0]
            print(f"   📉 Rows removed: {dropped_rows} ({dropped_rows/original_shape[0]*100:.1f}%)")
        
        # Save cleaned dataset (should already be saved by cleaner.clean())
        print(f"   💾 Saved cleaned dataset: {cleaned_file}")
        
        # Show sample of cleaned data
        print(f"\n   📋 Cleaned Data Sample (first 5 rows):")
        print(f"   " + "="*60)
        sample_df = cleaned_df.head(5)
        for col in sample_df.columns[:5]:  # Show first 5 columns
            print(f"   {col}: {sample_df[col].tolist()}")
        if len(cleaned_df.columns) > 5:
            print(f"   ... and {len(cleaned_df.columns) - 5} more columns")
            
        # Show data types after cleaning
        print(f"\n   📊 Data Types After Cleaning:")
        print(f"   " + "="*40)
        dtype_counts = cleaned_df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {str(dtype):15} : {count:3d} columns")
            
        # Summary statistics for numeric columns
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n   📈 Numeric Columns Summary:")
            print(f"   " + "="*35)
            for col in numeric_cols[:3]:  # Show first 3 numeric columns
                stats = cleaned_df[col].describe()
                print(f"   {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
            if len(numeric_cols) > 3:
                print(f"   ... and {len(numeric_cols) - 3} more numeric columns")
        
        # Show cleaning log summary
        if cleaning_log and 'actions' in cleaning_log:
            print(f"\n   📋 Cleaning Actions Summary:")
            print(f"   " + "="*35)
            for action in cleaning_log['actions'][:5]:  # Show first 5 actions
                print(f"   - {action}")
            if len(cleaning_log['actions']) > 5:
                print(f"   ... and {len(cleaning_log['actions']) - 5} more actions")
        
        print(f"\n   🎯 Cleaning completed successfully!")
        print(f"   📁 Cleaned dataset: {cleaned_file}")
        print(f"   📋 Cleaning log: {log_file}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Cleaning failed: {e}")
        import traceback
        print(f"   🔍 Error details:")
        traceback.print_exc()
        return False

def configure_column_cleaning(data_path: str, 
                             output_dir: str = "config_outputs",
                             retry_mode: bool = False,
                             specific_columns: List[str] = None):
    """
    Interactive column-specific cleaning configuration
    
    Args:
        data_path: Path to input CSV file
        output_dir: Directory for configuration outputs
        retry_mode: Whether this is a retry/refinement session
        specific_columns: Focus on specific columns only
    """
    
    print("� DS-AutoAdvisor: Column-Specific Cleaning Configuration")
    print("=" * 70)
    
    # Set up paths
    base_dir = Path(__file__).parent
    data_file = Path(data_path)
    output_path = base_dir / output_dir
    config_file = base_dir / "config" / "cleaning_config_template.yaml"
    
    # Load global configuration for CSV settings
    global_config = _load_config()
    csv_delimiter = global_config.get('global', {}).get('csv_delimiter', ',')
    csv_encoding = global_config.get('global', {}).get('csv_encoding', 'utf-8')
    
    print(f"📁 Input file: {data_file}")
    print(f"📂 Output directory: {output_path}")
    print(f"⚙️ Configuration file: {config_file}")
    print(f"🔧 CSV settings: delimiter='{csv_delimiter}', encoding={csv_encoding}")
    
    # Check if this is a retry session
    if retry_mode and config_file.exists():
        print(f"\n🔄 RETRY MODE: Existing configuration found")
        print(f"   Previous config: {config_file}")
        
        choice = input("   Load existing config for modification? (y/n): ").lower().strip()
        if choice == 'y':
            print("   Loading existing configuration for refinement...")
        else:
            print("   Starting fresh configuration...")
            retry_mode = False
    
    # =========================================================================
    # STAGE 1: Enhanced Data Profiling (unless retry with existing config)
    # =========================================================================
    if not retry_mode or not config_file.exists():
        print(f"\n🔍 STAGE 1: Enhanced Data Profiling")
        print("-" * 50)
        
        # Create enhanced profiler
        profiler = EnhancedDataProfiler(
            data_path=str(data_file),
            output_dir=str(output_path / "profiling"),
            generate_html=True,
            save_raw_profile=True
        )
        
        # Run profiling with focus on specific columns if provided
        try:
            enhanced_profile, json_path, yaml_path = profiler.run_complete_profiling(
                delimiter=csv_delimiter,
                encoding=csv_encoding
            )
            
            print(f"✅ Profiling completed successfully!")
            print(f"   📊 Dataset: {enhanced_profile.shape}")
            print(f"   📄 HTML report: {output_path / 'profiling' / 'data_profiling_report.html'}")
            print(f"   💾 Raw data: {json_path}")
            print(f"   ⚙️ Config template: {yaml_path}")
            
        except Exception as e:
            print(f"❌ Profiling failed: {e}")
            return False
    else:
        print(f"\n🔄 STAGE 1: Using Existing Profiling Data")
        print("-" * 50)
        yaml_path = config_file
        json_path = output_path / "profiling" / "raw_profiling_data.json"
        print(f"   ⚙️ Existing config: {yaml_path}")
        print(f"   💾 Existing data: {json_path}")
    
    # =========================================================================
    # STAGE 2: Interactive Configuration Review & Modification
    # =========================================================================
    print(f"\n⚙️ STAGE 2: Configuration Review & Customization")
    print("-" * 50)
    
    # Show column-specific recommendations
    if config_file.exists():
        try:
            import yaml
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            columns = list(config_data.get('column_transformations', {}).keys())
            if specific_columns:
                columns = [col for col in columns if col in specific_columns]
            
            print(f"📋 Configuration includes {len(columns)} columns:")
            for i, col in enumerate(columns[:10], 1):  # Show first 10
                print(f"   {i:2d}. {col}")
            if len(columns) > 10:
                print(f"   ... and {len(columns) - 10} more columns")
            
            # Show sample configuration for first few columns
            print(f"\n📝 Sample Configuration Options:")
            for col in columns[:3]:
                col_config = config_data['column_transformations'][col]
                print(f"\n   {col}:")
                for key, value in list(col_config.items())[:5]:  # Show first 5 options
                    print(f"     {key}: {value}")
                if len(col_config) > 5:
                    print(f"     ... and {len(col_config) - 5} more options")
            
        except Exception as e:
            print(f"⚠️ Could not parse configuration: {e}")
    
    # Interactive configuration guidance
    print(f"\n🎯 CONFIGURATION GUIDANCE:")
    print(f"   📖 Open the HTML report to understand your data")
    print(f"   ⚙️ Edit the YAML config to customize transformations")
    print(f"   🔧 Focus on columns with data quality issues")
    print(f"   💡 Use the comprehensive options for data type conversion")
    print(f"   📊 Configure column-specific outlier handling and scaling")
    
    print(f"\n📁 Key Files to Review:")
    print(f"   1. HTML Report: {output_path / 'profiling' / 'data_profiling_report.html'}")
    print(f"   2. YAML Config: {config_file}")
    print(f"   3. Raw Data: {json_path}")
    
    # Enhanced configuration options display
    print(f"\n🛠️ AVAILABLE CONFIGURATION OPTIONS:")
    print(f"   Column Management: drop_column (remove entire column)")
    print(f"   Drop Reasons: unique_identifier, irrelevant, redundant, high_cardinality, low_variance")
    print(f"   Data Type Conversion: int, float, str, bool, category, datetime")
    print(f"   Missing Value Imputation: mean, median, most_frequent, constant, knn, iterative")
    print(f"   Outlier Treatment: remove, cap, winsorize, iqr_cap, zscore_cap, isolation_forest")
    print(f"   Text Operations: strip, lower, upper, remove_digits, remove_punctuation")
    print(f"   Encoding: onehot, label, target, ordinal, binary, frequency")
    print(f"   Scaling: standard, minmax, robust, maxabs, quantile_uniform")
    print(f"   Transformations: log, sqrt, boxcox, yeojohnson, polynomial")
    print(f"   Feature Engineering: binning, feature_interactions, date_features")
    
    # Interactive menu
    while True:
        print(f"\n🔧 CONFIGURATION ACTIONS:")
        print(f"   1. View HTML profiling report")
        print(f"   2. Edit YAML configuration")
        print(f"   3. Validate current configuration") 
        print(f"   4. Preview cleaning effects")
        print(f"   5. Apply cleaning and generate cleaned dataset")
        print(f"   6. Continue to next step")
        print(f"   7. Exit configuration")
        
        choice = input(f"\n   Select action (1-7): ").strip()
        
        if choice == '1':
            _open_html_report(output_path / 'profiling' / 'data_profiling_report.html')
        elif choice == '2':
            _edit_yaml_config(config_file)
        elif choice == '3':
            _validate_configuration(config_file, data_file)
        elif choice == '4':
            _preview_cleaning_effects(data_file, config_file)
        elif choice == '5':
            _apply_cleaning_and_save(data_file, config_file, output_path)
        elif choice == '6':
            print(f"✅ Configuration complete! Ready for next step.")
            break
        elif choice == '7':
            print(f"🚪 Exiting configuration. Run again to continue.")
            return False
        else:
            print(f"   Invalid choice. Please select 1-7.")
    
    # =========================================================================
    # STAGE 3: Configuration Validation & Summary
    # =========================================================================
    print(f"\n✅ STAGE 3: Configuration Summary")
    print("-" * 50)
    
    # Validate final configuration
    validation_results = _validate_configuration(config_file, data_file, verbose=False)
    
    print(f"📊 Configuration Status:")
    print(f"   Valid configuration: {'✅' if validation_results else '❌'}")
    print(f"   Configuration file: {config_file}")
    print(f"   Columns configured: {len(config_data.get('column_transformations', {}))}")
    
    print(f"\n🔄 Next Steps:")
    print(f"   1. Run: python 3_test_pipeline.py --data {data_file}")
    print(f"   2. Or retry config: python 2_configure_cleaning.py --data {data_file} --retry")
    
    return True
def main():
    """Main function with CLI support for interactive configuration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Configure column-specific data cleaning')
    parser.add_argument('--data', type=str, help='Path to input CSV file')
    parser.add_argument('--retry', action='store_true', help='Retry/refine existing configuration')
    parser.add_argument('--columns', type=str, help='Comma-separated list of specific columns to configure')
    parser.add_argument('--output-dir', type=str, default='config_outputs', help='Output directory for configurations')
    
    args = parser.parse_args()
    
    # If no data file specified, use example
    if not args.data:
        base_dir = Path(__file__).parent
        data_file = base_dir / "data" / "telco_churn_data.csv"
        
        # Check for available data files
        if not data_file.exists():
            print(f"❌ Default data file not found: {data_file}")
            print(f"� Available data files:")
            data_dir = base_dir / "data"
            if data_dir.exists():
                csv_files = list(data_dir.glob("*.csv"))
                if csv_files:
                    for file in csv_files:
                        print(f"   - {file.name}")
                    # Use first available CSV file
                    data_file = csv_files[0]
                    print(f"🎯 Using: {data_file.name}")
                else:
                    print(f"   No CSV files found in data directory")
                    return False
            else:
                print(f"   Data directory not found")
                return False
        else:
            print(f"🎯 Using default data file: {data_file.name}")
    else:
        data_file = Path(args.data)
        if not data_file.exists():
            print(f"❌ Data file not found: {data_file}")
            return False
    
    # Parse specific columns if provided
    specific_columns = None
    if args.columns:
        specific_columns = [col.strip() for col in args.columns.split(',')]
        print(f"🎯 Focusing on columns: {specific_columns}")
    
    # Run the interactive configuration
    success = configure_column_cleaning(
        data_path=str(data_file),
        output_dir=args.output_dir,
        retry_mode=args.retry,
        specific_columns=specific_columns
    )
    
    if success:
        print(f"\n🎯 Column cleaning configuration completed!")
        print(f"📋 Next step: python 3_test_pipeline.py --data {data_file}")
    else:
        print(f"\n❌ Configuration incomplete. Run again to continue.")
        
    return success

if __name__ == "__main__":
    main()
