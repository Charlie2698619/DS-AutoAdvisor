#!/usr/bin/env python3
"""
🧹 Simple Data Cleaning Test
==========================

Test just the data cleaning stage using your YAML configuration.
This bypasses the complex pipeline and focuses on cleaning functionality.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

# Add src to path
base_dir = Path(__file__).parent
sys.path.append(str(base_dir / "src" / "2_data_cleaning"))

from data_cleaner import DataCleaner, CleaningConfig


def test_cleaning_only(data_path: str, config_path: str = None):
    """Test data cleaning with YAML configuration"""
    
    print("🧹 Testing Data Cleaning with YAML Configuration")
    print("=" * 60)
    
    # Paths
    data_file = Path(data_path)
    config_file = Path(config_path) if config_path else Path("config/cleaning_config_template.yaml")
    output_dir = base_dir / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Data file: {data_file}")
    print(f"⚙️ Config file: {config_file}")
    print(f"📤 Output directory: {output_dir}")
    
    # Check files exist
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return False
        
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        print(f"💡 Run: python 2_configure_cleaning.py --data {data_file}")
        return False
    
    try:
        # Load original data
        print(f"\n📊 Loading original data...")
        
        # Load global config for CSV settings
        global_config_path = base_dir / "config" / "unified_config_v2.yaml"
        csv_delimiter = ','
        csv_encoding = 'utf-8'
        
        if global_config_path.exists():
            with open(global_config_path, 'r') as f:
                global_config = yaml.safe_load(f)
                csv_delimiter = global_config.get('global', {}).get('csv_delimiter', ',')
                csv_encoding = global_config.get('global', {}).get('csv_encoding', 'utf-8')
        
        df_original = pd.read_csv(data_file, delimiter=csv_delimiter, encoding=csv_encoding)
        original_shape = df_original.shape
        
        print(f"   Original shape: {original_shape}")
        print(f"   Columns: {list(df_original.columns)}")
        print(f"   Missing values: {df_original.isnull().sum().sum()}")
        print(f"   Memory usage: {df_original.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Create cleaning configuration
        print(f"\n⚙️ Setting up cleaning configuration...")
        
        cleaned_file = output_dir / "cleaned_data.csv"
        log_file = output_dir / "cleaning_log.json"
        
        cleaning_config = CleaningConfig(
            input_path=str(data_file),
            output_path=str(cleaned_file),
            log_path=str(log_file),
            column_config_path=str(config_file),
            remove_duplicates=True,
            outlier_removal=True,
            outlier_method="iqr",
            verbose=True
        )
        
        print(f"   Input: {cleaning_config.input_path}")
        print(f"   Output: {cleaning_config.output_path}")
        print(f"   Log: {cleaning_config.log_path}")
        print(f"   Column config: {cleaning_config.column_config_path}")
        
        # Create and run cleaner
        print(f"\n🔧 Running data cleaner...")
        
        cleaner = DataCleaner(cleaning_config)
        
        # List cleaning steps
        steps = cleaner.list_steps()
        print(f"   Cleaning steps: {steps}")
        
        # Run cleaning
        cleaned_df, cleaning_log = cleaner.clean()
        
        # Show results
        print(f"\n✅ Cleaning completed successfully!")
        
        cleaned_shape = cleaned_df.shape
        print(f"\n📊 Cleaning Results:")
        print(f"   Original shape: {original_shape}")
        print(f"   Cleaned shape:  {cleaned_shape}")
        print(f"   Rows removed:   {original_shape[0] - cleaned_shape[0]:,} ({((original_shape[0] - cleaned_shape[0]) / original_shape[0] * 100):.1f}%)")
        print(f"   Columns removed: {original_shape[1] - cleaned_shape[1]}")
        
        # Missing data comparison
        original_missing = df_original.isnull().sum().sum()
        cleaned_missing = cleaned_df.isnull().sum().sum()
        print(f"   Missing values: {original_missing:,} → {cleaned_missing:,}")
        
        # Processing time
        if 'processing_time' in cleaning_log:
            print(f"   Processing time: {cleaning_log['processing_time']:.2f}s")
        
        # Show actions performed
        if 'actions' in cleaning_log and cleaning_log['actions']:
            print(f"\n🔧 Actions Performed ({len(cleaning_log['actions'])}):")
            for i, action in enumerate(cleaning_log['actions'][:10], 1):
                print(f"   {i:2d}. {action}")
            if len(cleaning_log['actions']) > 10:
                print(f"   ... and {len(cleaning_log['actions']) - 10} more actions")
        
        # Show warnings
        if 'warnings' in cleaning_log and cleaning_log['warnings']:
            print(f"\n⚠️ Warnings ({len(cleaning_log['warnings'])}):")
            for warning in cleaning_log['warnings'][:5]:
                print(f"   • {warning}")
        
        # Show column transformations
        print(f"\n📋 Column-Specific Transformations:")
        common_cols = set(df_original.columns) & set(cleaned_df.columns)
        transformed_count = 0
        
        for col in list(common_cols)[:10]:  # Show first 10 columns
            orig_unique = df_original[col].nunique()
            clean_unique = cleaned_df[col].nunique()
            orig_missing = df_original[col].isnull().sum()
            clean_missing = cleaned_df[col].isnull().sum()
            orig_dtype = str(df_original[col].dtype)
            clean_dtype = str(cleaned_df[col].dtype)
            
            changes = []
            if orig_unique != clean_unique:
                changes.append(f"unique: {orig_unique}→{clean_unique}")
            if orig_missing != clean_missing:
                changes.append(f"missing: {orig_missing}→{clean_missing}")
            if orig_dtype != clean_dtype:
                changes.append(f"dtype: {orig_dtype}→{clean_dtype}")
            
            if changes:
                transformed_count += 1
                print(f"   {col}: {', '.join(changes)}")
        
        if transformed_count == 0:
            print("   No significant transformations detected in first 10 columns")
        
        # Show dropped columns
        dropped_cols = set(df_original.columns) - set(cleaned_df.columns)
        if dropped_cols:
            print(f"\n🗑️ Dropped Columns ({len(dropped_cols)}):")
            for col in list(dropped_cols)[:10]:
                print(f"   • {col}")
            if len(dropped_cols) > 10:
                print(f"   ... and {len(dropped_cols) - 10} more")
        
        # Data types summary
        print(f"\n📊 Data Types After Cleaning:")
        dtype_counts = cleaned_df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {str(dtype):15}: {count:3d} columns")
        
        # Sample of cleaned data
        print(f"\n📝 Cleaned Data Sample:")
        sample_cols = min(8, len(cleaned_df.columns))
        print(cleaned_df.head(3).iloc[:, :sample_cols].to_string(index=False))
        if len(cleaned_df.columns) > sample_cols:
            print(f"   ... and {len(cleaned_df.columns) - sample_cols} more columns")
        
        # Files created
        print(f"\n📁 Output Files:")
        print(f"   📄 Cleaned data: {cleaned_file}")
        print(f"   📋 Cleaning log: {log_file}")
        
        if cleaned_file.exists():
            file_size = cleaned_file.stat().st_size / 1024**2
            print(f"   📊 Cleaned file size: {file_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Cleaning failed: {e}")
        import traceback
        print(f"\n🔍 Error details:")
        traceback.print_exc()
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test data cleaning with YAML configuration')
    parser.add_argument('--data', type=str, default='data/telco_churn_data.csv', 
                       help='Path to input data file')
    parser.add_argument('--config', type=str, default='config/cleaning_config_template.yaml',
                       help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    success = test_cleaning_only(args.data, args.config)
    print(f"\n{'🎉 Success!' if success else '❌ Failed!'}")
    return success


if __name__ == "__main__":
    main()
