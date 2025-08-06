"""
Enhanced error handling and validation patterns
"""

import os
import traceback
from enum import Enum
from typing import List, Dict, Any
import pandas as pd

class ValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class ErrorSeverity(Enum):
    WARNING = "warning"
    ERROR = "error" 
    CRITICAL = "critical"

class DataValidator:
    """Pre-processing data validation"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, config: dict) -> List[Dict[str, Any]]:
        """Comprehensive dataframe validation"""
        issues = []
        
        # Check basic properties
        if df.empty:
            issues.append({
                "severity": ErrorSeverity.CRITICAL,
                "message": "DataFrame is empty",
                "suggestion": "Check data loading process"
            })
        
        # Check for all-null columns
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            issues.append({
                "severity": ErrorSeverity.ERROR,
                "message": f"Columns with all null values: {null_cols}",
                "suggestion": "Consider dropping these columns"
            })
        
        # Check memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        if memory_mb > 1000:  # > 1GB
            issues.append({
                "severity": ErrorSeverity.WARNING,
                "message": f"Large dataset: {memory_mb:.1f}MB",
                "suggestion": "Consider chunked processing"
            })
        
        # Check for potential ID columns (high cardinality, unique values)
        for col in df.columns:
            if df[col].nunique() == len(df) and len(df) > 100:
                issues.append({
                    "severity": ErrorSeverity.WARNING,
                    "message": f"Potential ID column: {col}",
                    "suggestion": "Consider excluding from ML features"
                })
        
        return issues
    
    @staticmethod
    def validate_config(config: dict) -> List[Dict[str, Any]]:
        """Validate configuration parameters"""
        issues = []
        
        # Check file paths exist
        if not os.path.exists(config.get("input_path", "")):
            issues.append({
                "severity": ErrorSeverity.CRITICAL,
                "message": "Input file does not exist",
                "suggestion": "Check input_path configuration"
            })
        
        # Check parameter ranges
        if not 0 <= config.get("missing_col_thresh", 0.3) <= 1:
            issues.append({
                "severity": ErrorSeverity.ERROR,
                "message": "missing_col_thresh must be between 0 and 1",
                "suggestion": "Set to value between 0.0 and 1.0"
            })
        
        return issues

# Enhanced error context manager
class ErrorContext:
    """Context manager for better error reporting"""
    
    def __init__(self, operation: str, log: dict):
        self.operation = operation
        self.log = log
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_msg = f"{self.operation} failed: {exc_val}"
            self.log["errors"].append({
                "operation": self.operation,
                "error": str(exc_val),
                "type": exc_type.__name__,
                "traceback": traceback.format_exc()
            })
            print(f"❌ {error_msg}")
            return False  # Don't suppress the exception
        return True

# Usage example:
# with ErrorContext("Outlier Removal", log):
#     df = remove_outliers(df, config)
