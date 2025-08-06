#!/usr/bin/env python3
"""
🔍 DS-AutoAdvisor: Step 1 - Data Inspection 
(independent from the main pipeline, this script does not import any other modules from the project only the config manager)
==========================================

WHAT IT DOES:
Quick data inspection and quality analysis before running the pipeline.
Gets basic stats, detects issues, and suggests configuration settings.

WHEN TO USE:
- First time with new dataset
- When you want to understand data characteristics
- Before configuring the pipeline

HOW TO USE:
Basic inspection:
    python 1_inspect_data.py --data data/your_data.csv

Detailed analysis:
    python 1_inspect_data.py --data data/your_data.csv --detailed

Auto-detect from config:
    python 1_inspect_data.py --detailed

WHAT YOU GET:
✅ Dataset overview (shape, types, memory usage)
✅ Missing data analysis with severity levels
✅ Data quality issues and recommendations
✅ Configuration suggestions for YAML files
✅ Column-by-column detailed analysis (--detailed mode)
✅ Correlation analysis for numeric columns

NEXT STEP:
After inspection, run: python 2_test_pipeline.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

class QuickDataInspector:
    """Quick data inspection utility"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None # DataFrame to hold loaded data
        
    def load_data(self) -> bool:
        """Load the data file"""
        try:
            if not Path(self.data_path).exists():
                print(f"❌ Data file not found: {self.data_path}")
                return False
            
            # Try different delimiters
            delimiters = [',', ';', '\t', '|']
            for delimiter in delimiters:
                try:
                    self.df = pd.read_csv(self.data_path, delimiter=delimiter, nrows=5)
                    if self.df.shape[1] > 1:  # More than one column suggests correct delimiter
                        self.df = pd.read_csv(self.data_path, delimiter=delimiter)
                        print(f"✅ Data loaded successfully using delimiter: '{delimiter}'")
                        return True
                except:
                    continue
            
            # Fallback to default comma
            self.df = pd.read_csv(self.data_path)
            print("✅ Data loaded with default settings")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False
    
    def basic_info(self):
        """Display basic dataset information"""
        print("\n" + "="*60)
        print("📊 BASIC DATASET INFORMATION")
        print("="*60)
        
        print(f"📁 File: {self.data_path}")
        print(f"📏 Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"💾 Memory: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types summary
        dtype_counts = self.df.dtypes.value_counts()
        print(f"\n📋 Data Types:")
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
    
    def column_analysis(self):
        """Analyze each column in detail"""
        print("\n" + "="*60)
        print("📋 DETAILED COLUMN ANALYSIS")
        print("="*60)
        
        for i, col in enumerate(self.df.columns, 1):
            print(f"\n{i:2d}. Column: '{col}'")
            print(f"    Type: {self.df[col].dtype}")
            print(f"    Non-null: {self.df[col].count():,} ({self.df[col].count()/len(self.df)*100:.1f}%)")
            print(f"    Missing: {self.df[col].isnull().sum():,} ({self.df[col].isnull().sum()/len(self.df)*100:.1f}%)")
            print(f"    Unique: {self.df[col].nunique():,}")
            
            # Type-specific analysis
            if self.df[col].dtype in ['int64', 'float64']:
                self._analyze_numeric_column(col)
            elif self.df[col].dtype in ['object', 'category']:
                self._analyze_categorical_column(col)
    
    def _analyze_numeric_column(self, col: str):
        """Analyze numeric column"""
        series = self.df[col].dropna()
        if len(series) == 0:
            return
            
        print(f"    Range: {series.min():.3f} to {series.max():.3f}")
        print(f"    Mean: {series.mean():.3f}, Median: {series.median():.3f}")
        print(f"    Std: {series.std():.3f}")
        
        # Check for potential outliers using IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            print(f"    Potential outliers (IQR method): {outliers} ({outliers/len(series)*100:.1f}%)")
    
    def _analyze_categorical_column(self, col: str):
        """Analyze categorical column"""
        series = self.df[col].dropna()
        if len(series) == 0:
            return
            
        value_counts = series.value_counts()
        print(f"    Most frequent: '{value_counts.index[0]}' ({value_counts.iloc[0]} times)")
        
        if len(value_counts) <= 10:
            print(f"    All values: {dict(value_counts)}")
        else:
            print(f"    Top 5 values: {dict(value_counts.head())}")
            
        # Check for high cardinality
        cardinality_ratio = len(value_counts) / len(series)
        if cardinality_ratio > 0.8:
            print(f"    ⚠️  High cardinality: {cardinality_ratio:.1%} (might need special handling)")
    
    def missing_data_analysis(self):
        """Analyze missing data patterns"""
        print("\n" + "="*60)
        print("🕳️  MISSING DATA ANALYSIS")
        print("="*60)
        
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) == 0:
            print("✅ No missing data found!")
            return
        
        print(f"📊 Columns with missing data: {len(missing_data)}")
        print(f"📊 Total missing values: {missing_data.sum():,}")
        
        for col, count in missing_data.items():
            percentage = count / len(self.df) * 100
            severity = "🔴" if percentage > 50 else "🟡" if percentage > 20 else "🟢"
            print(f"   {severity} {col:<20}: {count:6,} ({percentage:5.1f}%)")
    
    def correlation_analysis(self):
        """Analyze correlations between numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("\n⚠️  Not enough numeric columns for correlation analysis")
            return
        
        print("\n" + "="*60)
        print("🔗 CORRELATION ANALYSIS")
        print("="*60)
        
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_val))
        
        if high_corr_pairs:
            print("🔍 High correlations (|r| > 0.7):")
            for col1, col2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"   {col1} ↔ {col2}: {corr_val:.3f}")
        else:
            print("✅ No high correlations found")
    
    def data_quality_summary(self):
        """Provide data quality summary and recommendations"""
        print("\n" + "="*60)
        print("📋 DATA QUALITY SUMMARY & RECOMMENDATIONS")
        print("="*60)
        
        issues = []
        recommendations = []
        
        # Check dataset size
        if len(self.df) < 100:
            issues.append("Very small dataset (< 100 rows)")
            recommendations.append("Consider collecting more data for reliable model training")
        elif len(self.df) < 1000:
            issues.append("Small dataset (< 1,000 rows)")
            recommendations.append("Use cross-validation for model evaluation")
        
        # Check missing data
        missing_percentage = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        if missing_percentage > 20:
            issues.append(f"High missing data percentage: {missing_percentage:.1f}%")
            recommendations.append("Consider data imputation or collection improvement")
        
        # Check data types
        object_cols = len(self.df.select_dtypes(include=['object']).columns)
        if object_cols / len(self.df.columns) > 0.8:
            issues.append("Many text/object columns detected")
            recommendations.append("Consider encoding categorical variables and feature engineering")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate rows found: {duplicates}")
            recommendations.append("Remove duplicate rows before training")
        
        # Check for constant columns
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        if constant_cols:
            issues.append(f"Constant columns: {len(constant_cols)}")
            recommendations.append(f"Remove constant columns: {constant_cols}")
        
        # Display results
        if issues:
            print("⚠️  Issues Found:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        else:
            print("✅ No major data quality issues detected")
        
        if recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
    
    def generate_config_suggestions(self):
        """Generate configuration suggestions based on data analysis"""
        print("\n" + "="*60)
        print("⚙️  CONFIGURATION SUGGESTIONS")
        print("="*60)
        
        suggestions = {}
        
        # Suggest target column (last column by default, or most likely candidate)
        if len(self.df.columns) > 1:
            last_col = self.df.columns[-1]
            if self.df[last_col].nunique() < 20:  # Likely categorical target
                suggestions['target_column'] = last_col
                print(f"💡 Suggested target column: '{last_col}'")
                print(f"   (Has {self.df[last_col].nunique()} unique values)")
        
        # Suggest CSV delimiter
        suggestions['csv_delimiter'] = ','  # Default, could be improved with detection
        
        # Suggest data cleaning settings
        high_missing_cols = self.df.columns[self.df.isnull().sum() / len(self.df) > 0.5].tolist()
        if high_missing_cols:
            suggestions['columns_to_drop'] = high_missing_cols
            print(f"💡 Suggested columns to drop (>50% missing): {high_missing_cols}")
        
        # Suggest encoding for categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        low_cardinality_cols = [col for col in categorical_cols if self.df[col].nunique() < 10]
        if low_cardinality_cols:
            suggestions['categorical_encoding'] = low_cardinality_cols
            print(f"💡 Suggested one-hot encoding for: {low_cardinality_cols}")
        
        return suggestions
    
    def run_complete_inspection(self, detailed: bool = False):
        """Run complete data inspection"""
        print("🔍 DS-AutoAdvisor v2.0 - Quick Data Inspector")
        print("=" * 80)
        
        if not self.load_data():
            return False
        
        # Basic analysis (always run)
        self.basic_info()
        self.missing_data_analysis()
        self.data_quality_summary()
        
        if detailed:
            # Detailed analysis
            self.column_analysis()
            self.correlation_analysis()
        
        # Configuration suggestions
        self.generate_config_suggestions()
        
        print("\n" + "="*80)
        print("✅ Data inspection complete!")
        print("💡 Next step: python 2_test_pipeline.py")
        print("="*80)
        
        return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Data Inspector for DS-AutoAdvisor v2.0')
    parser.add_argument('--data', type=str, help='Path to data file to inspect')
    parser.add_argument('--detailed', action='store_true', help='Run detailed analysis')
    
    args = parser.parse_args()
    
    # Get data path
    data_path = args.data
    if not data_path:
        # Try to get from config
        try:
            sys.path.append(str(Path(__file__).parent / "src"))
            from src.infrastructure.enhanced_config_manager import get_config_manager, Environment
            config_manager = get_config_manager('config/unified_config_v2.yaml', Environment.DEVELOPMENT)
            data_path = config_manager.get_config('global', 'data_input_path')
            print(f"📁 Using data path from config: {data_path}")
        except:
            print("❌ Please specify data path with --data parameter")
            sys.exit(1)
    
    # Run inspection
    inspector = QuickDataInspector(data_path)
    success = inspector.run_complete_inspection(detailed=args.detailed)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
