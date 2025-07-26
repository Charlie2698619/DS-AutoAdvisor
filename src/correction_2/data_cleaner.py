import pandas as pd
import numpy as np
import csv
import os
import json
import warnings
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, OrdinalEncoder, 
    MinMaxScaler, StandardScaler, PowerTransformer, RobustScaler
)
from sklearn.ensemble import IsolationForest
from scipy.stats import skew
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class CleaningConfig:
    """Type-safe configuration class"""
    # Data I/O
    input_path: str
    output_path: str
    log_path: str
    chunk_size: Optional[int] = None  # For large datasets
    
    # CSV Output Format Settings 
    output_delimiter: str = ";"           # Delimiter: ',', ';', '\t', '|'
    output_quoting: int = csv.QUOTE_ALL   # Quoting style: 0=MINIMAL, 1=ALL, 2=NONNUMERIC, 3=NONE
    output_quotechar: str = '"'           # Quote character: '"', "'"
    output_escapechar: Optional[str] = None  # Escape character: '\\', None
    output_encoding: str = 'utf-8'        # File encoding
    output_lineterminator: str = '\n'     # Line ending: '\n', '\r\n'
    output_doublequote: bool = True       # Handle "" as escaped quotes
    
    # Duplicate handling
    remove_duplicates: bool = True
    duplicate_subset: Optional[List[str]] = None
    
    # Low variance
    remove_low_variance: bool = True
    low_variance_thresh: int = 1
    
    # Missing data
    drop_high_miss_cols: bool = True
    missing_col_thresh: float = 0.3
    drop_high_miss_rows: bool = False
    missing_row_thresh: float = 0.5
    
    # Imputation
    impute_num: str = "auto"  # auto, mean, median, knn, iterative
    impute_cat: str = "auto"  # auto, most_frequent, constant
    impute_const: Any = 0
    
    # Outliers
    outlier_removal: bool = True
    outlier_method: str = "iqr"  # iqr, isoforest, zscore
    iqr_factor: float = 1.5
    iforest_contam: float = 0.01
    zscore_thresh: float = 3.0
    
    # Transformations
    skew_correction: bool = True
    skew_thresh: float = 1.0
    skew_method: str = "yeo-johnson"
    
    # Scaling
    scaling: str = "standard"  # standard, minmax, robust, none
    
    # Encoding
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

class DataCleaner:
    """Main data cleaner class with improved architecture"""
    
    def __init__(self, config: CleaningConfig):
        self.config = config
        self.log = {
            "initial_shape": None,
            "final_shape": None,
            "actions": [],
            "warnings": [],
            "errors": [],
            "processing_time": None
        }
        
        # Initialize cleaning steps
        self.steps = [
            DeduplicationStep("Deduplication"),
            OutlierRemovalStep("OutlierRemoval"),
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
    
    def _process_full_dataset(self) -> pd.DataFrame:
        """Process entire dataset in memory"""
        df = pd.read_csv(self.config.input_path)
        self.log["initial_shape"] = df.shape
        
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
        
        for chunk in pd.read_csv(self.config.input_path, chunksize=self.config.chunk_size):
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
    
    # Example 1: Quoted semicolon (your current preference)
    config1 = CleaningConfig(
        input_path=f"{base_dir}/data/bank.csv",
        output_path=f"{base_dir}/data/bank_cleaned_quoted.csv",
        log_path=f"{base_dir}/docs/cleaning_log_quoted.json",
        verbose=True,
        **CleaningConfig.create_csv_preset("quoted_semicolon")
    )
    
    # Example 2: Clean semicolon (minimal quotes)
    config2 = CleaningConfig(
        input_path=f"{base_dir}/data/bank.csv",
        output_path=f"{base_dir}/data/bank_cleaned_clean.csv",
        log_path=f"{base_dir}/docs/cleaning_log_clean.json",
        verbose=True,
        **CleaningConfig.create_csv_preset("clean_semicolon")
    )
    
    # Example 3: Standard CSV (comma-separated)
    config3 = CleaningConfig(
        input_path=f"{base_dir}/data/bank.csv",
        output_path=f"{base_dir}/data/bank_cleaned_standard.csv",
        log_path=f"{base_dir}/docs/cleaning_log_standard.json",
        verbose=True,
        **CleaningConfig.create_csv_preset("standard_csv")
    )
    
    # Example 4: Custom format
    config4 = CleaningConfig(
        input_path=f"{base_dir}/data/bank.csv",
        output_path=f"{base_dir}/data/bank_cleaned_custom.csv",
        log_path=f"{base_dir}/docs/cleaning_log_custom.json",
        verbose=True,
        # Custom CSV settings
        output_delimiter="|",
        output_quoting=csv.QUOTE_NONNUMERIC,
        output_quotechar="'",
        output_encoding="utf-8"
    )
    
    # Choose which config to use
    config = config1  # Change this to test different formats
    
    print(f"🔧 Using CSV format preset with settings:")
    print(f"   Delimiter: '{config.output_delimiter}'")
    print(f"   Quoting: {config.output_quoting}")
    print(f"   Quote char: '{config.output_quotechar}'")
    print(f"   Encoding: {config.output_encoding}")
    
    cleaner = DataCleaner(config)
    df, log = cleaner.clean()
    
    # Verify the output format
    print(f"\n🔍 Verifying output format...")
    try:
        test_df = pd.read_csv(
            config.output_path, 
            delimiter=config.output_delimiter,
            quotechar=config.output_quotechar,
            encoding=config.output_encoding
        )
        print(f"✅ Verification successful: {test_df.shape}")
        print(f"✅ Columns: {list(test_df.columns)}")
        print(f"✅ Ready for trainer!")
    except Exception as e:
        print(f"❌ Verification failed: {e}")
