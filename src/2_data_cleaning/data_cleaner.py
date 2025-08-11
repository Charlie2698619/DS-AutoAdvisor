import pandas as pd
import numpy as np
import csv
import os
import json
import warnings
import yaml
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, OrdinalEncoder, 
    MinMaxScaler, StandardScaler, PowerTransformer, RobustScaler,
    MaxAbsScaler, QuantileTransformer
)
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from scipy.stats import skew, boxcox
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class ColumnCleaningConfig:
    """Configuration for column-specific cleaning operations"""
    # Column-level actions
    drop_column: bool = False  # Whether to drop this entire column from the dataset
    drop_reason: Optional[str] = None  # Reason for dropping: 'unique_identifier', 'irrelevant', 'redundant', 'high_cardinality', 'low_variance'
    
    # Data Type Conversion
    dtype_conversion: Optional[str] = None  # int, float, str, bool, category, datetime
    datetime_format: Optional[str] = None  # for datetime conversion: '%Y-%m-%d', '%d/%m/%Y', etc.
    
    # Missing Value Handling
    imputation: Optional[str] = None  # mean, median, most_frequent, constant, knn, iterative, interpolate, forward_fill, backward_fill
    imputation_value: Any = None  # for constant imputation
    
    # Outlier Treatment
    outlier_treatment: Optional[str] = None  # remove, cap, winsorize, null, iqr_cap, zscore_cap, isolation_forest
    outlier_threshold: Optional[float] = None  # custom threshold for zscore (default: 3.0) or IQR factor (default: 1.5)
    
    # Text/String Cleaning
    text_cleaning: bool = False
    text_operations: Optional[List[str]] = None  # strip, lower, upper, title, remove_digits, remove_punctuation, remove_extra_spaces
    regex_replace: Optional[Dict[str, str]] = None  # {pattern: replacement}
    
    # Categorical Encoding
    encoding: Optional[str] = None  # onehot, label, target, ordinal, binary, frequency, mean_encoding
    encoding_categories: Optional[List[str]] = None  # explicit category order for ordinal encoding
    max_categories: Optional[int] = None  # limit for high cardinality features
    
    # Numerical Transformations
    scaling: Optional[str] = None  # standard, minmax, robust, maxabs, quantile_uniform, quantile_normal
    transformation: Optional[str] = None  # log, log1p, sqrt, square, reciprocal, boxcox, yeojohnson, polynomial
    polynomial_degree: Optional[int] = None  # for polynomial transformation
    
    # Feature Engineering
    binning: Optional[str] = None  # equal_width, equal_frequency, kmeans, quantile
    n_bins: Optional[int] = None  # number of bins for binning
    feature_interactions: Optional[List[str]] = None  # create interaction features with specified columns
    
    # Date/Time Features
    date_conversion: bool = False
    extract_date_features: Optional[List[str]] = None  # year, month, day, dayofweek, quarter, is_weekend, hour, minute
    
    # Data Validation
    value_constraints: Optional[Dict[str, Any]] = None  # min_value, max_value, allowed_values, regex_pattern
    
    # Custom Operations
    custom_function: Optional[str] = None  # name of custom function to apply
    custom_parameters: Optional[Dict[str, Any]] = None  # parameters for custom function

@dataclass
class CleaningConfig:
    """Type-safe configuration class with column-specific support"""
    # Data I/O
    input_path: str
    output_path: str
    log_path: str
    chunk_size: Optional[int] = None  # For large datasets
    
    # Enhanced configuration paths
    column_config_path: Optional[str] = None  # Path to column-specific YAML config
    profiling_data_path: Optional[str] = None  # Path to raw profiling JSON data
    
    # CSV Output Format Settings 
    output_delimiter: str = ";"           # Delimiter: ',', ';', '\t', '|'
    output_quoting: int = csv.QUOTE_ALL   # Quoting style: 0=MINIMAL, 1=ALL, 2=NONNUMERIC, 3=NONE
    output_quotechar: str = '"'           # Quote character: '"', "'"
    output_escapechar: Optional[str] = None  # Escape character: '\\', None
    output_encoding: str = 'utf-8'        # File encoding
    output_lineterminator: str = '\n'     # Line ending: '\n', '\r\n'
    output_doublequote: bool = True       # Handle "" as escaped quotes
    
    # Global duplicate handling
    remove_duplicates: bool = True
    duplicate_subset: Optional[List[str]] = None
    
    # Global low variance
    remove_low_variance: bool = True
    low_variance_thresh: int = 1
    
    # Global missing data (fallback when column-specific not defined)
    drop_high_miss_cols: bool = True
    missing_col_thresh: float = 0.3
    drop_high_miss_rows: bool = False
    missing_row_thresh: float = 0.5
    
    # Global imputation (fallback)
    impute_num: str = "auto"  # auto, mean, median, knn, iterative
    impute_cat: str = "auto"  # auto, most_frequent, constant
    impute_const: Any = 0
    
    # Global outliers (fallback)
    outlier_removal: bool = True
    outlier_method: str = "iqr"  # iqr, isoforest, zscore
    iqr_factor: float = 1.5
    iforest_contam: float = 0.01
    zscore_thresh: float = 3.0
    
    # Global transformations (fallback)
    skew_correction: bool = True
    skew_thresh: float = 1.0
    skew_method: str = "yeo-johnson"
    
    # Global scaling (fallback)
    scaling: str = "standard"  # standard, minmax, robust, none
    
    # Global encoding (fallback)
    encoding: str = "onehot"  # onehot, label, ordinal, target
    max_cardinality: int = 20
    
    # Date handling
    date_parse: bool = True
    extract_date_features: bool = True
    date_formats: Optional[List[str]] = None
    
    # Type optimization
    downcast_types: bool = True
    
    # Text handling
    text_policy: str = "warn"  # warn, drop, ignore, vectorize
    text_vectorizer: str = "tfidf"  # tfidf, count, hash
    
    # Advanced options
    target_column: Optional[str] = None
    feature_selection: bool = False
    correlation_thresh: float = 0.95
    vif_thresh: float = 10.0
    
    # Performance
    n_jobs: int = -1 # Use all available cores
    random_state: int = 42 
    verbose: bool = True # Enable verbose logging
    
    # Column-specific configurations (loaded from YAML)
    column_configs: Dict[str, ColumnCleaningConfig] = None

    def __post_init__(self):
        if self.column_configs is None:
            self.column_configs = {}
        
        # Load column-specific configuration if path provided
        if self.column_config_path and Path(self.column_config_path).exists():
            self._load_column_configs()
    
    def _load_column_configs(self):
        """Load column-specific configurations from YAML file"""
        try:
            with open(self.column_config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            if 'column_transformations' in yaml_data:
                for col_name, col_config in yaml_data['column_transformations'].items():
                    self.column_configs[col_name] = ColumnCleaningConfig(**col_config)
            
            # Update global settings if present
            if 'global_settings' in yaml_data:
                global_settings = yaml_data['global_settings']
                for key, value in global_settings.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                        
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Could not load column configs from {self.column_config_path}: {e}")

    @classmethod
    def from_yaml_config(cls, yaml_config: Dict[str, Any], input_path: str, output_path: str, log_path: str, **overrides):
        """Create CleaningConfig from YAML configuration with required paths"""
        # Extract relevant settings from YAML
        config_params = {
            'input_path': input_path,
            'output_path': output_path,
            'log_path': log_path,
            'verbose': yaml_config.get('verbose', True),
            'remove_duplicates': yaml_config.get('remove_duplicates', True),
            'remove_low_variance': yaml_config.get('remove_low_variance', True),
            'drop_high_miss_cols': yaml_config.get('drop_high_miss_cols', True),
            'missing_col_thresh': yaml_config.get('missing_col_thresh', 0.3),
            'drop_high_miss_rows': yaml_config.get('drop_high_miss_rows', False),
            'missing_row_thresh': yaml_config.get('missing_row_thresh', 0.5),
            'impute_num': yaml_config.get('impute_num', 'auto'),
            'impute_cat': yaml_config.get('impute_cat', 'auto'),
            'outlier_removal': yaml_config.get('outlier_removal', True),
            'outlier_method': yaml_config.get('outlier_method', 'iqr'),
            'scaling': yaml_config.get('scaling', 'standard'),
            'encoding': yaml_config.get('encoding', 'onehot'),
            'target_column': yaml_config.get('target_column', None)
        }
        
        # Apply any overrides
        config_params.update(overrides)
        
        return cls(**config_params)

    @classmethod
    def create_csv_preset(cls, preset: str, **kwargs):
        """Create config with predefined CSV output presets"""
        presets = {
            "quoted_semicolon": {
                "output_delimiter": ";",
                "output_quoting": csv.QUOTE_ALL,
                "output_quotechar": '"',
                "output_escapechar": None,
                "output_doublequote": True
            },
            "clean_semicolon": {
                "output_delimiter": ";",
                "output_quoting": csv.QUOTE_MINIMAL,
                "output_quotechar": '"',
                "output_escapechar": None,
                "output_doublequote": True
            },
            "standard_csv": {
                "output_delimiter": ",",
                "output_quoting": csv.QUOTE_MINIMAL,
                "output_quotechar": '"',
                "output_escapechar": None,
                "output_doublequote": True
            },
            "tab_separated": {
                "output_delimiter": "\t",
                "output_quoting": csv.QUOTE_MINIMAL,
                "output_quotechar": '"',
                "output_escapechar": None,
                "output_doublequote": True
            },
            "pipe_separated": {
                "output_delimiter": "|",
                "output_quoting": csv.QUOTE_MINIMAL,
                "output_quotechar": '"',
                "output_escapechar": None,
                "output_doublequote": True
            },
            "no_quotes": {
                "output_delimiter": ";",
                "output_quoting": csv.QUOTE_NONE,
                "output_quotechar": '"',
                "output_escapechar": "\\",
                "output_doublequote": False
            }
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        # Merge preset with custom kwargs
        config_dict = {**presets[preset], **kwargs}
        return config_dict

class CleaningStep(ABC):
    """Base class for all cleaning steps"""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def execute(self, df: pd.DataFrame, config: CleaningConfig, log: Dict) -> pd.DataFrame:
        pass
        
    def log_action(self, log: Dict, message: str, step_type: str = "action"):
        log[f"{step_type}s"].append(f"[{self.name}] {message}")

class DeduplicationStep(CleaningStep):
    def execute(self, df: pd.DataFrame, config: CleaningConfig, log: Dict) -> pd.DataFrame:
        if not config.remove_duplicates:
            return df
            
        before = len(df)
        subset = config.duplicate_subset if config.duplicate_subset else None
        df = df.drop_duplicates(subset=subset)
        removed = before - len(df)
        
        if removed > 0:
            self.log_action(log, f"Removed {removed} duplicate rows")
        return df

class OutlierRemovalStep(CleaningStep):
    def execute(self, df: pd.DataFrame, config: CleaningConfig, log: Dict) -> pd.DataFrame:
        if not config.outlier_removal:
            return df
            
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return df
            
        before = len(df)
        
        if config.outlier_method == "iqr":
            df = self._remove_iqr_outliers(df, num_cols, config.iqr_factor)
        elif config.outlier_method == "isoforest":
            df = self._remove_isoforest_outliers(df, num_cols, config.iforest_contam, config.random_state)
        elif config.outlier_method == "zscore":
            df = self._remove_zscore_outliers(df, num_cols, config.zscore_thresh)
            
        removed = before - len(df)
        if removed > 0:
            self.log_action(log, f"Removed {removed} outliers using {config.outlier_method}")
        return df
    
    def _remove_iqr_outliers(self, df: pd.DataFrame, num_cols: List[str], factor: float) -> pd.DataFrame:
        mask = pd.Series([True] * len(df), index=df.index)
        for col in num_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            col_mask = (df[col] >= Q1 - factor * IQR) & (df[col] <= Q3 + factor * IQR)
            mask = mask & col_mask
        return df[mask]
    
    def _remove_isoforest_outliers(self, df: pd.DataFrame, num_cols: List[str], 
                                 contamination: float, random_state: int) -> pd.DataFrame:
        iso = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
        mask = iso.fit_predict(df[num_cols]) == 1
        return df[mask]
    
    def _remove_zscore_outliers(self, df: pd.DataFrame, num_cols: List[str], thresh: float) -> pd.DataFrame:
        mask = pd.Series([True] * len(df), index=df.index)
        for col in num_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            col_mask = z_scores <= thresh
            mask = mask & col_mask
        return df[mask]

class ColumnSpecificCleaningStep(CleaningStep):
    """Apply column-specific transformations based on configuration"""
    
    def execute(self, df: pd.DataFrame, config: CleaningConfig, log: Dict) -> pd.DataFrame:
        if not config.column_configs:
            self.log_action(log, "No column-specific configurations found, skipping")
            return df
        
        df_cleaned = df.copy()
        columns_to_drop = []  # Track columns marked for dropping
        
        for col_name, col_config in config.column_configs.items():
            if col_name not in df.columns:
                self.log_action(log, f"Column '{col_name}' not found in dataset", "warning")
                continue
            
            # Check if column should be dropped
            if col_config.drop_column:
                columns_to_drop.append(col_name)
                reason = col_config.drop_reason or "not specified"
                self.log_action(log, f"Column '{col_name}' marked for removal (reason: {reason})")
                continue
            
            try:
                df_cleaned = self._apply_column_transformations(
                    df_cleaned, col_name, col_config, config, log
                )
            except Exception as e:
                self.log_action(log, f"Failed to process column '{col_name}': {str(e)}", "error")
                continue
        
        # Drop columns after all transformations
        if columns_to_drop:
            initial_shape = df_cleaned.shape
            df_cleaned = df_cleaned.drop(columns=columns_to_drop)
            self.log_action(log, f"Dropped {len(columns_to_drop)} columns: {columns_to_drop}")
            self.log_action(log, f"Shape after column removal: {initial_shape} → {df_cleaned.shape}")
        
        return df_cleaned
    
    def _apply_column_transformations(self, df: pd.DataFrame, col_name: str, 
                                    col_config: ColumnCleaningConfig, 
                                    global_config: CleaningConfig, log: Dict) -> pd.DataFrame:
        """Apply all transformations for a specific column"""
        
        # 0. Check if column should be dropped
        if col_config.drop_column:
            reason = col_config.drop_reason or "not specified"
            self.log_action(log, f"Marking column '{col_name}' for removal (reason: {reason})")
            # Note: Column will be dropped in a separate step to avoid modifying df during iteration
            return df
        
        # 1. Data type conversion (must be first)
        if col_config.dtype_conversion:
            df = self._convert_dtype(df, col_name, col_config, log)
        
        # 2. Text cleaning and operations
        if col_config.text_cleaning:
            df = self._clean_text(df, col_name, col_config, log)
        
        # 3. Data validation
        if col_config.value_constraints:
            df = self._validate_data(df, col_name, col_config, log)
        
        # 4. Date conversion and feature extraction
        if col_config.date_conversion:
            df = self._convert_to_datetime(df, col_name, col_config, log)
        
        # 5. Handle missing values
        if col_config.imputation:
            df = self._impute_missing(df, col_name, col_config, global_config, log)
        
        # 6. Handle outliers
        if col_config.outlier_treatment:
            df = self._handle_outliers(df, col_name, col_config, global_config, log)
        
        # 7. Apply transformations (log, sqrt, etc.)
        if col_config.transformation:
            df = self._apply_transformation(df, col_name, col_config, log)
        
        # 8. Apply binning
        if col_config.binning:
            df = self._apply_binning(df, col_name, col_config, log)
        
        # 9. Apply scaling
        if col_config.scaling:
            df = self._apply_scaling(df, col_name, col_config, log)
        
        # 10. Apply encoding
        if col_config.encoding:
            df = self._apply_encoding(df, col_name, col_config, global_config, log)
        
        # 11. Feature interactions (if specified)
        if col_config.feature_interactions:
            df = self._create_feature_interactions(df, col_name, col_config, log)
        
        return df
    
    def _clean_text(self, df: pd.DataFrame, col_name: str, 
                   col_config: ColumnCleaningConfig, log: Dict) -> pd.DataFrame:
        """Enhanced text cleaning with multiple operations"""
        if df[col_name].dtype == 'object':
            before_unique = df[col_name].nunique()
            df[col_name] = df[col_name].astype(str)
            
            # Apply text operations if specified
            operations = col_config.text_operations or ['strip']
            operations_applied = []
            
            for operation in operations:
                if operation == 'strip':
                    df[col_name] = df[col_name].str.strip()
                    operations_applied.append('strip')
                elif operation == 'lower':
                    df[col_name] = df[col_name].str.lower()
                    operations_applied.append('lower')
                elif operation == 'upper':
                    df[col_name] = df[col_name].str.upper()
                    operations_applied.append('upper')
                elif operation == 'title':
                    df[col_name] = df[col_name].str.title()
                    operations_applied.append('title')
                elif operation == 'remove_digits':
                    df[col_name] = df[col_name].str.replace(r'\d+', '', regex=True)
                    operations_applied.append('remove_digits')
                elif operation == 'remove_punctuation':
                    df[col_name] = df[col_name].str.replace(r'[^\w\s]', '', regex=True)
                    operations_applied.append('remove_punctuation')
                elif operation == 'remove_extra_spaces':
                    df[col_name] = df[col_name].str.replace(r'\s+', ' ', regex=True)
                    operations_applied.append('remove_extra_spaces')
            
            # Apply regex replacements if specified
            if col_config.regex_replace:
                for pattern, replacement in col_config.regex_replace.items():
                    df[col_name] = df[col_name].str.replace(pattern, replacement, regex=True)
                    operations_applied.append(f'regex_replace({pattern})')
            
            after_unique = df[col_name].nunique()
            
            if before_unique != after_unique or operations_applied:
                self.log_action(log, f"Text cleaning on '{col_name}': {operations_applied}, {before_unique} → {after_unique} unique values")
        
        return df
    
    def _convert_to_datetime(self, df: pd.DataFrame, col_name: str, 
                           col_config: ColumnCleaningConfig, log: Dict) -> pd.DataFrame:
        """Convert column to datetime and extract features"""
        try:
            # Convert to datetime
            if col_config.datetime_format:
                df[col_name] = pd.to_datetime(df[col_name], format=col_config.datetime_format, errors='coerce')
            else:
                df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
            
            self.log_action(log, f"Converted '{col_name}' to datetime")
            
            # Extract date features if specified
            if col_config.extract_date_features:
                features_created = []
                
                for feature in col_config.extract_date_features:
                    if feature == 'year':
                        df[f"{col_name}_year"] = df[col_name].dt.year
                        features_created.append('year')
                    elif feature == 'month':
                        df[f"{col_name}_month"] = df[col_name].dt.month
                        features_created.append('month')
                    elif feature == 'day':
                        df[f"{col_name}_day"] = df[col_name].dt.day
                        features_created.append('day')
                    elif feature == 'dayofweek':
                        df[f"{col_name}_dayofweek"] = df[col_name].dt.dayofweek
                        features_created.append('dayofweek')
                    elif feature == 'quarter':
                        df[f"{col_name}_quarter"] = df[col_name].dt.quarter
                        features_created.append('quarter')
                    elif feature == 'is_weekend':
                        df[f"{col_name}_is_weekend"] = (df[col_name].dt.dayofweek >= 5).astype(int)
                        features_created.append('is_weekend')
                    elif feature == 'hour':
                        df[f"{col_name}_hour"] = df[col_name].dt.hour
                        features_created.append('hour')
                    elif feature == 'minute':
                        df[f"{col_name}_minute"] = df[col_name].dt.minute
                        features_created.append('minute')
                
                if features_created:
                    self.log_action(log, f"Extracted date features from '{col_name}': {features_created}")
            
        except Exception as e:
            self.log_action(log, f"Failed to convert '{col_name}' to datetime: {str(e)}", "warning")
        
        return df
    
    def _impute_missing(self, df: pd.DataFrame, col_name: str, 
                       col_config: ColumnCleaningConfig, 
                       global_config: CleaningConfig, log: Dict) -> pd.DataFrame:
        """Impute missing values for specific column"""
        missing_count = df[col_name].isnull().sum()
        if missing_count == 0:
            return df
        
        method = col_config.imputation
        
        try:
            if method == "mean":
                df[col_name].fillna(df[col_name].mean(), inplace=True)
            elif method == "median":
                df[col_name].fillna(df[col_name].median(), inplace=True)
            elif method == "most_frequent":
                df[col_name].fillna(df[col_name].mode()[0], inplace=True)
            elif method == "constant":
                fill_value = col_config.imputation_value if col_config.imputation_value is not None else 0
                df[col_name].fillna(fill_value, inplace=True)
            elif method == "interpolate":
                df[col_name] = df[col_name].interpolate()
            elif method == "forward_fill":
                df[col_name] = df[col_name].fillna(method='ffill')
            elif method == "backward_fill":
                df[col_name] = df[col_name].fillna(method='bfill')
            elif method == "knn":
                # Use KNN imputer for this column
                imputer = KNNImputer(n_neighbors=5)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if col_name in numeric_cols and len(numeric_cols) > 1:
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            elif method == "iterative":
                # Use iterative imputer
                imputer = IterativeImputer(random_state=global_config.random_state)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if col_name in numeric_cols and len(numeric_cols) > 1:
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
            self.log_action(log, f"Imputed {missing_count} missing values in '{col_name}' using {method}")
            
        except Exception as e:
            self.log_action(log, f"Imputation failed for '{col_name}': {str(e)}", "warning")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, col_name: str, 
                        col_config: ColumnCleaningConfig, 
                        global_config: CleaningConfig, log: Dict) -> pd.DataFrame:
        """Handle outliers for specific column"""
        if df[col_name].dtype not in [np.number, 'int64', 'float64']:
            return df
        
        treatment = col_config.outlier_treatment
        before_count = len(df)
        
        try:
            iqr_factor = col_config.outlier_threshold or 1.5
            zscore_threshold = col_config.outlier_threshold or 3.0
            
            if treatment == "remove":
                # Remove outliers using IQR method
                Q1, Q3 = df[col_name].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_factor * IQR
                upper_bound = Q3 + iqr_factor * IQR
                
                outlier_mask = (df[col_name] < lower_bound) | (df[col_name] > upper_bound)
                outlier_count = outlier_mask.sum()
                df = df[~outlier_mask]
                
                self.log_action(log, f"Removed {outlier_count} outliers from '{col_name}' using IQR")
                
            elif treatment == "cap":
                # Cap outliers to percentile bounds
                lower_bound = df[col_name].quantile(0.01)
                upper_bound = df[col_name].quantile(0.99)
                
                capped_count = ((df[col_name] < lower_bound) | (df[col_name] > upper_bound)).sum()
                df[col_name] = df[col_name].clip(lower_bound, upper_bound)
                
                self.log_action(log, f"Capped {capped_count} outliers in '{col_name}' to percentiles")
                
            elif treatment == "winsorize":
                # Winsorize to 1st and 99th percentiles
                from scipy.stats import mstats
                df[col_name] = mstats.winsorize(df[col_name], limits=[0.01, 0.01])
                self.log_action(log, f"Winsorized '{col_name}' to 1st-99th percentiles")
                
            elif treatment == "iqr_cap":
                # Cap using IQR bounds
                Q1, Q3 = df[col_name].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_factor * IQR
                upper_bound = Q3 + iqr_factor * IQR
                
                capped_count = ((df[col_name] < lower_bound) | (df[col_name] > upper_bound)).sum()
                df[col_name] = df[col_name].clip(lower_bound, upper_bound)
                
                self.log_action(log, f"Capped {capped_count} outliers in '{col_name}' using IQR bounds")
                
            elif treatment == "zscore_cap":
                # Cap using Z-score
                mean_val = df[col_name].mean()
                std_val = df[col_name].std()
                lower_bound = mean_val - zscore_threshold * std_val
                upper_bound = mean_val + zscore_threshold * std_val
                
                capped_count = ((df[col_name] < lower_bound) | (df[col_name] > upper_bound)).sum()
                df[col_name] = df[col_name].clip(lower_bound, upper_bound)
                
                self.log_action(log, f"Capped {capped_count} outliers in '{col_name}' using Z-score")
                
            elif treatment == "isolation_forest":
                # Use Isolation Forest
                from sklearn.ensemble import IsolationForest
                iso = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso.fit_predict(df[[col_name]])
                outlier_mask = outlier_labels == -1
                outlier_count = outlier_mask.sum()
                df = df[~outlier_mask]
                
                self.log_action(log, f"Removed {outlier_count} outliers from '{col_name}' using Isolation Forest")
                
            elif treatment == "null":
                # Set outliers to null for later imputation
                Q1, Q3 = df[col_name].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_factor * IQR
                upper_bound = Q3 + iqr_factor * IQR
                
                outlier_mask = (df[col_name] < lower_bound) | (df[col_name] > upper_bound)
                outlier_count = outlier_mask.sum()
                df.loc[outlier_mask, col_name] = np.nan
                
                self.log_action(log, f"Set {outlier_count} outliers to null in '{col_name}'")
                
        except Exception as e:
            self.log_action(log, f"Outlier handling failed for '{col_name}': {str(e)}", "warning")
        
        return df
    
    def _apply_transformation(self, df: pd.DataFrame, col_name: str, 
                            col_config: ColumnCleaningConfig, log: Dict) -> pd.DataFrame:
        """Apply mathematical transformations"""
        if df[col_name].dtype not in [np.number, 'int64', 'float64']:
            return df
        
        transformation = col_config.transformation
        
        try:
            if transformation == "log":
                # Ensure positive values for log transform
                if (df[col_name] <= 0).any():
                    df[col_name] = df[col_name] - df[col_name].min() + 1
                df[col_name] = np.log(df[col_name])
                
            elif transformation == "log1p":
                # Log(1 + x) transformation - handles zeros better
                df[col_name] = np.log1p(df[col_name])
                
            elif transformation == "sqrt":
                # Ensure non-negative values
                if (df[col_name] < 0).any():
                    df[col_name] = df[col_name] - df[col_name].min()
                df[col_name] = np.sqrt(df[col_name])
                
            elif transformation == "square":
                # Square transformation
                df[col_name] = np.square(df[col_name])
                
            elif transformation == "reciprocal":
                # Reciprocal transformation (1/x)
                # Avoid division by zero
                df[col_name] = df[col_name].replace(0, np.nan)
                df[col_name] = 1 / df[col_name]
                
            elif transformation == "polynomial":
                # Polynomial transformation
                degree = col_config.polynomial_degree or 2
                df[col_name] = np.power(df[col_name], degree)
                
            elif transformation == "boxcox":
                from scipy.stats import boxcox
                if (df[col_name] > 0).all():
                    df[col_name], _ = boxcox(df[col_name])
                else:
                    self.log_action(log, f"BoxCox requires positive values for '{col_name}' - skipping", "warning")
                    return df
                    
            elif transformation == "yeojohnson":
                transformer = PowerTransformer(method='yeo-johnson')
                df[col_name] = transformer.fit_transform(df[[col_name]]).flatten()
            
            self.log_action(log, f"Applied {transformation} transformation to '{col_name}'")
            
        except Exception as e:
            self.log_action(log, f"Transformation failed for '{col_name}': {str(e)}", "warning")
        
        return df
    
    def _apply_scaling(self, df: pd.DataFrame, col_name: str, 
                      col_config: ColumnCleaningConfig, log: Dict) -> pd.DataFrame:
        """Apply scaling to numerical columns"""
        if df[col_name].dtype not in [np.number, 'int64', 'float64']:
            return df
        
        scaling_method = col_config.scaling
        
        try:
            if scaling_method == "standard":
                scaler = StandardScaler()
            elif scaling_method == "minmax":
                scaler = MinMaxScaler()
            elif scaling_method == "robust":
                scaler = RobustScaler()
            elif scaling_method == "maxabs":
                from sklearn.preprocessing import MaxAbsScaler
                scaler = MaxAbsScaler()
            elif scaling_method == "quantile_uniform":
                from sklearn.preprocessing import QuantileTransformer
                scaler = QuantileTransformer(output_distribution='uniform')
            elif scaling_method == "quantile_normal":
                from sklearn.preprocessing import QuantileTransformer
                scaler = QuantileTransformer(output_distribution='normal')
            else:
                return df
            
            df[col_name] = scaler.fit_transform(df[[col_name]]).flatten()
            self.log_action(log, f"Applied {scaling_method} scaling to '{col_name}'")
            
        except Exception as e:
            self.log_action(log, f"Scaling failed for '{col_name}': {str(e)}", "warning")
        
        return df
    
    def _apply_encoding(self, df: pd.DataFrame, col_name: str, 
                       col_config: ColumnCleaningConfig, 
                       global_config: CleaningConfig, log: Dict) -> pd.DataFrame:
        """Apply encoding to categorical columns"""
        if df[col_name].dtype == 'object' or df[col_name].dtype.name == 'category':
            
            encoding_method = col_config.encoding
            
            try:
                if encoding_method == "onehot":
                    # One-hot encoding
                    dummies = pd.get_dummies(df[col_name], prefix=col_name)
                    df = df.drop(col_name, axis=1)
                    df = pd.concat([df, dummies], axis=1)
                    self.log_action(log, f"One-hot encoded '{col_name}' into {len(dummies.columns)} columns")
                    
                elif encoding_method == "label":
                    # Label encoding
                    encoder = LabelEncoder()
                    df[col_name] = encoder.fit_transform(df[col_name].astype(str))
                    self.log_action(log, f"Label encoded '{col_name}'")
                    
                elif encoding_method == "ordinal":
                    # Ordinal encoding with explicit category order
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    if col_config.encoding_categories:
                        # Use custom category order
                        encoder = OrdinalEncoder(
                            categories=[col_config.encoding_categories],
                            handle_unknown='use_encoded_value', 
                            unknown_value=-1
                        )
                    df[col_name] = encoder.fit_transform(df[[col_name]]).flatten()
                    self.log_action(log, f"Ordinal encoded '{col_name}'")
                    
                elif encoding_method == "binary":
                    # Binary encoding - convert to binary representation
                    unique_vals = df[col_name].unique()
                    n_bits = int(np.ceil(np.log2(len(unique_vals))))
                    
                    # Create label encoder first
                    le = LabelEncoder()
                    encoded_labels = le.fit_transform(df[col_name].astype(str))
                    
                    # Convert to binary representation
                    for bit in range(n_bits):
                        df[f"{col_name}_binary_{bit}"] = (encoded_labels >> bit) & 1
                    
                    # Drop original column
                    df = df.drop(col_name, axis=1)
                    self.log_action(log, f"Binary encoded '{col_name}' into {n_bits} binary columns")
                    
                elif encoding_method == "frequency":
                    # Frequency encoding - replace with frequency of occurrence
                    freq_map = df[col_name].value_counts().to_dict()
                    df[col_name] = df[col_name].map(freq_map)
                    self.log_action(log, f"Frequency encoded '{col_name}'")
                    
                elif encoding_method == "mean_encoding":
                    # Mean encoding (same as target encoding but more explicit name)
                    if global_config.target_column and global_config.target_column in df.columns:
                        target_means = df.groupby(col_name)[global_config.target_column].mean()
                        df[col_name] = df[col_name].map(target_means)
                        self.log_action(log, f"Mean encoded '{col_name}' using target column")
                    else:
                        self.log_action(log, f"Mean encoding for '{col_name}' requires target column", "warning")
                        
                elif encoding_method == "target":
                    # Target encoding (requires target column)
                    if global_config.target_column and global_config.target_column in df.columns:
                        target_means = df.groupby(col_name)[global_config.target_column].mean()
                        df[col_name] = df[col_name].map(target_means)
                        self.log_action(log, f"Target encoded '{col_name}'")
                    else:
                        self.log_action(log, f"Target encoding for '{col_name}' requires target column", "warning")
                        
            except Exception as e:
                self.log_action(log, f"Encoding failed for '{col_name}': {str(e)}", "warning")
        
        return df
    
    def _convert_dtype(self, df: pd.DataFrame, col_name: str, 
                      col_config: ColumnCleaningConfig, log: Dict) -> pd.DataFrame:
        """Convert data type of column"""
        try:
            target_dtype = col_config.dtype_conversion
            
            if target_dtype == "int":
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype('Int64')
            elif target_dtype == "float":
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            elif target_dtype == "str":
                df[col_name] = df[col_name].astype(str)
            elif target_dtype == "bool":
                df[col_name] = df[col_name].astype(bool)
            elif target_dtype == "category":
                df[col_name] = df[col_name].astype('category')
            elif target_dtype == "datetime":
                format_str = col_config.datetime_format
                if format_str:
                    df[col_name] = pd.to_datetime(df[col_name], format=format_str, errors='coerce')
                else:
                    df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
            
            self.log_action(log, f"Converted '{col_name}' to {target_dtype}")
            
        except Exception as e:
            self.log_action(log, f"Data type conversion failed for '{col_name}': {str(e)}", "warning")
        
        return df
    
    def _validate_data(self, df: pd.DataFrame, col_name: str, 
                      col_config: ColumnCleaningConfig, log: Dict) -> pd.DataFrame:
        """Apply data validation constraints"""
        try:
            constraints = col_config.value_constraints
            violations = 0
            
            if 'min_value' in constraints:
                mask = df[col_name] < constraints['min_value']
                violations += mask.sum()
                df.loc[mask, col_name] = constraints['min_value']
            
            if 'max_value' in constraints:
                mask = df[col_name] > constraints['max_value']
                violations += mask.sum()
                df.loc[mask, col_name] = constraints['max_value']
            
            if 'allowed_values' in constraints:
                allowed = constraints['allowed_values']
                mask = ~df[col_name].isin(allowed)
                violations += mask.sum()
                df.loc[mask, col_name] = np.nan  # Set invalid values to NaN
            
            if violations > 0:
                self.log_action(log, f"Fixed {violations} constraint violations in '{col_name}'")
                
        except Exception as e:
            self.log_action(log, f"Data validation failed for '{col_name}': {str(e)}", "warning")
        
        return df
    
    def _apply_binning(self, df: pd.DataFrame, col_name: str, 
                      col_config: ColumnCleaningConfig, log: Dict) -> pd.DataFrame:
        """Apply binning to numerical columns"""
        if df[col_name].dtype not in [np.number, 'int64', 'float64']:
            return df
        
        try:
            binning_method = col_config.binning
            n_bins = col_config.n_bins or 5
            
            if binning_method == "equal_width":
                df[f"{col_name}_binned"] = pd.cut(df[col_name], bins=n_bins, labels=False)
                
            elif binning_method == "equal_frequency":
                df[f"{col_name}_binned"] = pd.qcut(df[col_name], q=n_bins, labels=False, duplicates='drop')
                
            elif binning_method == "quantile":
                quantiles = np.linspace(0, 1, n_bins + 1)
                df[f"{col_name}_binned"] = pd.cut(df[col_name], 
                                                 bins=df[col_name].quantile(quantiles).values, 
                                                 labels=False, include_lowest=True)
                                                 
            elif binning_method == "kmeans":
                # KMeans-based binning
                from sklearn.cluster import KMeans
                data_reshaped = df[col_name].values.reshape(-1, 1)
                kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(data_reshaped)
                
                # Sort clusters by their centers for meaningful ordering
                centers = kmeans.cluster_centers_.flatten()
                cluster_order = np.argsort(centers)
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_order)}
                df[f"{col_name}_binned"] = np.array([label_mapping[label] for label in cluster_labels])
            
            self.log_action(log, f"Applied {binning_method} binning to '{col_name}' with {n_bins} bins")
            
        except Exception as e:
            self.log_action(log, f"Binning failed for '{col_name}': {str(e)}", "warning")
        
        return df
    
    def _create_feature_interactions(self, df: pd.DataFrame, col_name: str, 
                                   col_config: ColumnCleaningConfig, log: Dict) -> pd.DataFrame:
        """Create interaction features"""
        try:
            interaction_cols = col_config.feature_interactions
            
            for other_col in interaction_cols:
                if other_col in df.columns:
                    # Create multiplication interaction
                    interaction_name = f"{col_name}_x_{other_col}"
                    if df[col_name].dtype in [np.number, 'int64', 'float64'] and \
                       df[other_col].dtype in [np.number, 'int64', 'float64']:
                        df[interaction_name] = df[col_name] * df[other_col]
                        self.log_action(log, f"Created interaction feature '{interaction_name}'")
            
        except Exception as e:
            self.log_action(log, f"Feature interaction failed for '{col_name}': {str(e)}", "warning")
        
        return df

class DataCleaner:
    """Main data cleaner class with enhanced column-specific architecture"""
    
    def __init__(self, config: CleaningConfig):
        self.config = config
        self.log = {
            "initial_shape": None,
            "final_shape": None,
            "actions": [],
            "warnings": [],
            "errors": [],
            "processing_time": None,
            "column_transformations": {}
        }
        
        # Initialize cleaning steps
        self.steps = [
            DeduplicationStep("Deduplication"),
            OutlierRemovalStep("OutlierRemoval"),
            ColumnSpecificCleaningStep("ColumnSpecificCleaning"),  # NEW: Column-specific transformations
            # Add more steps as needed
        ]
    
    def clean(self) -> Tuple[pd.DataFrame, Dict]:
        """Main cleaning pipeline with chunked processing support"""
        import time
        start_time = time.time()
        
        try:
            if self.config.chunk_size:
                df = self._process_in_chunks()
            else:
                df = self._process_full_dataset()
                
            self.log["final_shape"] = df.shape
            self.log["processing_time"] = time.time() - start_time
            
            # Save results
            self._save_results(df)
            
            return df, self.log
            
        except Exception as e:
            self.log["errors"].append(f"Pipeline failed: {repr(e)}")
            self._save_log()
            raise
    
    def add_custom_step(self, step: CleaningStep, position: Optional[int] = None):
        """Add a custom cleaning step to the pipeline"""
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)
    
    def remove_step(self, step_name: str):
        """Remove a cleaning step by name"""
        self.steps = [step for step in self.steps if step.name != step_name]
    
    def list_steps(self) -> List[str]:
        """List all cleaning steps in order"""
        return [step.name for step in self.steps]
    
    def _process_full_dataset(self) -> pd.DataFrame:
        """Process entire dataset in memory"""
        # Determine CSV reading parameters based on file extension or configuration
        read_kwargs = {}
        
        # Check if input is semicolon-delimited (common in European CSV files)
        try:
            # Try to detect delimiter by reading first line
            with open(self.config.input_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line.count(';') > first_line.count(','):
                    read_kwargs['delimiter'] = ';'
                elif first_line.count(',') > first_line.count(';'):
                    read_kwargs['delimiter'] = ','
                else:
                    # Default to comma
                    read_kwargs['delimiter'] = ','
        except:
            # Default to comma if detection fails
            read_kwargs['delimiter'] = ','
        
        df = pd.read_csv(self.config.input_path, **read_kwargs)
        self.log["initial_shape"] = df.shape
        
        if self.config.verbose:
            print(f"📊 Loaded data with shape {df.shape} using delimiter '{read_kwargs.get('delimiter', ',')}'")
        
        for step in self.steps:
            try:
                df = step.execute(df, self.config, self.log)
            except Exception as e:
                self.log["errors"].append(f"{step.name} failed: {repr(e)}")
                if self.config.verbose:
                    print(f"⚠️ {step.name} failed: {e}")
                continue
                
        return df
    
    def _process_in_chunks(self) -> pd.DataFrame:
        """Process large datasets in chunks"""
        chunks = []
        
        # Determine CSV reading parameters
        read_kwargs = {'chunksize': self.config.chunk_size}
        
        # Try to detect delimiter
        try:
            with open(self.config.input_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line.count(';') > first_line.count(','):
                    read_kwargs['delimiter'] = ';'
                elif first_line.count(',') > first_line.count(';'):
                    read_kwargs['delimiter'] = ','
                else:
                    read_kwargs['delimiter'] = ','
        except:
            read_kwargs['delimiter'] = ','
        
        for chunk in pd.read_csv(self.config.input_path, **read_kwargs):
            if self.log["initial_shape"] is None:
                self.log["initial_shape"] = (0, len(chunk.columns))
            
            self.log["initial_shape"] = (
                self.log["initial_shape"][0] + len(chunk),
                self.log["initial_shape"][1]
            )
            
            # Process chunk
            for step in self.steps:
                try:
                    chunk = step.execute(chunk, self.config, self.log)
                except Exception as e:
                    self.log["errors"].append(f"{step.name} failed on chunk: {repr(e)}")
                    continue
            
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True)
    
    def _save_results(self, df: pd.DataFrame):
        """Save cleaned data with customizable CSV format"""
        # Ensure output directory exists
        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data based on quoting style
        df_output = df.copy()
        
        # Handle special cases based on quoting style
        if self.config.output_quoting == csv.QUOTE_NONE:
            # For no quotes, ensure no delimiter conflicts in data
            string_cols = df_output.select_dtypes(include=['object']).columns
            for col in string_cols:
                # Replace delimiter with underscore if found in data
                df_output[col] = df_output[col].astype(str).str.replace(
                    self.config.output_delimiter, '_', regex=False
                )
                # Remove quotes that might cause issues
                df_output[col] = df_output[col].str.replace('"', '', regex=False)
                df_output[col] = df_output[col].str.replace("'", '', regex=False)
        
        # Save with customizable settings
        try:
            df_output.to_csv(
                self.config.output_path,
                index=False,
                encoding=self.config.output_encoding,
                sep=self.config.output_delimiter,
                quoting=self.config.output_quoting,
                quotechar=self.config.output_quotechar,
                escapechar=self.config.output_escapechar,
                doublequote=self.config.output_doublequote,
                lineterminator=self.config.output_lineterminator
            )
        except Exception as e:
            # Fallback to safe settings if custom settings fail
            self.log["warnings"].append(f"Custom CSV settings failed: {e}. Using fallback.")
            df_output.to_csv(
                self.config.output_path,
                index=False,
                encoding='utf-8',
                sep=',',
                quoting=csv.QUOTE_MINIMAL
            )
        
        # Save log
        self._save_log()
        
        if self.config.verbose:
            print(f"✅ Cleaned data saved to {self.config.output_path}")
            print(f"📊 Shape: {self.log['initial_shape']} → {self.log['final_shape']}")
            
            # Describe the output format
            quote_styles = {
                0: "QUOTE_MINIMAL", 1: "QUOTE_ALL", 
                2: "QUOTE_NONNUMERIC", 3: "QUOTE_NONE"
            }
            print(f"💾 Format: {self.config.output_delimiter}-delimited, {quote_styles.get(self.config.output_quoting, 'UNKNOWN')} quoting")
            
            # Show sample of actual output
            try:
                with open(self.config.output_path, 'r', encoding=self.config.output_encoding) as f:
                    first_line = f.readline().strip()
                    second_line = f.readline().strip()
                    print(f"📝 Header: {first_line}")
                    print(f"📝 Sample: {second_line[:100]}...")
            except Exception as e:
                print(f"⚠️ Could not read sample output: {e}")

    
    def _save_log(self):
        """Save the cleaning log to JSON file"""
        try:
            # Ensure log directory exists
            Path(self.config.log_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save log with proper JSON serialization
            with open(self.config.log_path, 'w', encoding='utf-8') as f:
                json.dump(self.log, f, indent=2, default=str, ensure_ascii=False)
                
            if self.config.verbose:
                print(f"📋 Cleaning log saved to {self.config.log_path}")
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ Could not save log: {e}")



# Factory function for backward compatibility
def create_cleaner_from_dict(config_dict: Dict) -> DataCleaner:
    """Create DataCleaner from dictionary config (backward compatibility)"""
    config = CleaningConfig(**config_dict)
    return DataCleaner(config)

# Example usage
if __name__ == "__main__":
    base_dir = "/mnt/c/Users/tony3/Desktop/tidytuesday/ds-autoadvisor"
    
    # Example 1: Enhanced column-specific cleaning
    print("🔧 Example 1: Column-Specific Cleaning")
    print("="*50)
    
    # Configuration with column-specific settings
    config_enhanced = CleaningConfig(
        input_path=f"{base_dir}/data/bank.csv",
        output_path=f"{base_dir}/data/bank_cleaned_enhanced.csv",
        log_path=f"{base_dir}/docs/cleaning_log_enhanced.json",
        column_config_path=f"{base_dir}/config/cleaning_config_template.yaml",  # Generated by enhanced profiler
        profiling_data_path=f"{base_dir}/docs/raw_profiling_data.json",      # Generated by enhanced profiler
        verbose=True,
        **CleaningConfig.create_csv_preset("quoted_semicolon")
    )
    
    # Run enhanced cleaning
    try:
        cleaner_enhanced = DataCleaner(config_enhanced)
        print(f"📋 Pipeline steps: {cleaner_enhanced.list_steps()}")
        df_enhanced, log_enhanced = cleaner_enhanced.clean()
        
        print(f"✅ Enhanced cleaning completed!")
        print(f"   Shape: {log_enhanced['initial_shape']} → {log_enhanced['final_shape']}")
        print(f"   Processing time: {log_enhanced['processing_time']:.2f}s")
        print(f"   Actions performed: {len(log_enhanced['actions'])}")
        
    except Exception as e:
        print(f"⚠️ Enhanced cleaning failed (config files may not exist): {e}")
        print("💡 Run the enhanced profiler first to generate configuration files")
    
    print("\n" + "="*50)
    
    # Example 2: Traditional cleaning (backward compatibility)
    print("🔧 Example 2: Traditional Global Cleaning")
    print("="*50)
    
    config_traditional = CleaningConfig(
        input_path=f"{base_dir}/data/bank.csv",
        output_path=f"{base_dir}/data/bank_cleaned_traditional.csv",
        log_path=f"{base_dir}/docs/cleaning_log_traditional.json",
        verbose=True,
        **CleaningConfig.create_csv_preset("clean_semicolon")
    )
    
    cleaner_traditional = DataCleaner(config_traditional)
    df_traditional, log_traditional = cleaner_traditional.clean()
    
    print(f"✅ Traditional cleaning completed!")
    print(f"   Shape: {log_traditional['initial_shape']} → {log_traditional['final_shape']}")
    
    # Example 3: Custom cleaning pipeline
    print("\n" + "="*50)
    print("🔧 Example 3: Custom Pipeline")
    print("="*50)
    
    # Create custom step
    class CustomTextCleaningStep(CleaningStep):
        def execute(self, df: pd.DataFrame, config: CleaningConfig, log: Dict) -> pd.DataFrame:
            # Custom logic for specific business needs
            text_cols = df.select_dtypes(include=['object']).columns
            for col in text_cols:
                # Remove special characters, convert to lowercase
                df[col] = df[col].astype(str).str.replace(r'[^\w\s]', '', regex=True).str.lower()
            self.log_action(log, f"Applied custom text cleaning to {len(text_cols)} columns")
            return df
    
    config_custom = CleaningConfig(
        input_path=f"{base_dir}/data/bank.csv",
        output_path=f"{base_dir}/data/bank_cleaned_custom.csv",
        log_path=f"{base_dir}/docs/cleaning_log_custom.json",
        verbose=True,
        **CleaningConfig.create_csv_preset("standard_csv")
    )
    
    cleaner_custom = DataCleaner(config_custom)
    cleaner_custom.add_custom_step(CustomTextCleaningStep("CustomTextCleaning"), position=1)
    print(f"📋 Custom pipeline steps: {cleaner_custom.list_steps()}")
    
    df_custom, log_custom = cleaner_custom.clean()
    print(f"✅ Custom cleaning completed!")
    
    print("\n🎉 All cleaning examples completed!")
    print("💡 To use column-specific cleaning:")
    print("   1. Run enhanced_data_profiler.py to generate raw profiling data")
    print("   2. Review and modify the generated cleaning_config_template.yaml")
    print("   3. Run this cleaner with column_config_path pointing to your YAML")
    
    # Show configuration file format
    print("\n� Sample Column Configuration YAML:")
    print("""
column_transformations:
  age:
    imputation: 'median'
    outlier_treatment: 'cap'
    scaling: 'standard'
    transformation: null
  
  job:
    imputation: 'most_frequent'
    encoding: 'onehot'
    text_cleaning: true
    
  balance:
    imputation: 'median'
    outlier_treatment: 'remove'
    transformation: 'log'
    scaling: 'robust'
""")
