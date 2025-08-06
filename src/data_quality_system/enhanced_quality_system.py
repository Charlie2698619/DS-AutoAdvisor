"""
Enhanced Data Quality System for DS-AutoAdvisor v2.0
===================================================

This module provides advanced data quality features including:
- Automated data type enforcement
- Advanced outlier detection with ensemble methods
- Data quality scoring and recommendations
- Interactive data validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import warnings
warnings.filterwarnings('ignore')

class DataType(Enum):
    """Supported data types for enforcement"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"  
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"

class OutlierMethod(Enum):
    """Outlier detection methods"""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    Z_SCORE = "z_score"
    IQR = "iqr"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"

@dataclass
class DataQualityIssue:
    """Represents a data quality issue"""
    issue_type: str
    severity: str  # "low", "medium", "high", "critical"
    column: Optional[str]
    description: str
    suggested_action: str
    affected_rows: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment report"""
    overall_score: float
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues: List[DataQualityIssue]
    column_scores: Dict[str, float]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TypeEnforcementRule:
    """Rule for data type enforcement"""
    column: str
    target_type: DataType
    confidence_threshold: float = 0.8
    auto_correct: bool = True
    validation_rules: Dict[str, Any] = field(default_factory=dict)

class DataTypeInferrer:
    """
    Intelligent data type inference with confidence scoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def infer_types(self, df: pd.DataFrame) -> Dict[str, Tuple[DataType, float]]:
        """
        Infer data types for all columns with confidence scores
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary mapping column names to (inferred_type, confidence)
        """
        type_inferences = {}
        
        for column in df.columns:
            inferred_type, confidence = self._infer_column_type(df[column])
            type_inferences[column] = (inferred_type, confidence)
        
        return type_inferences
    
    def _infer_column_type(self, series: pd.Series) -> Tuple[DataType, float]:
        """
        Infer data type for a single column
        
        Args:
            series: Pandas series to analyze
            
        Returns:
            Tuple of (inferred_type, confidence_score)
        """
        # Remove null values for analysis
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return DataType.TEXT, 0.0
        
        # Check for boolean type
        if self._is_boolean(non_null_series):
            return DataType.BOOLEAN, 0.95
        
        # Check for datetime type
        datetime_confidence = self._check_datetime(non_null_series)
        if datetime_confidence > 0.8:
            return DataType.DATETIME, datetime_confidence
        
        # Check for numeric type
        numeric_confidence = self._check_numeric(non_null_series)
        if numeric_confidence > 0.8:
            return DataType.NUMERIC, numeric_confidence
        
        # Check for categorical type
        categorical_confidence = self._check_categorical(non_null_series)
        if categorical_confidence > 0.7:
            return DataType.CATEGORICAL, categorical_confidence
        
        # Default to text
        return DataType.TEXT, 0.6
    
    def _is_boolean(self, series: pd.Series) -> bool:
        """Check if series contains boolean values"""
        unique_values = set(series.astype(str).str.lower().unique())
        boolean_patterns = [
            {'true', 'false'},
            {'yes', 'no'},
            {'y', 'n'},
            {'1', '0'},
            {'on', 'off'}
        ]
        
        return any(unique_values == pattern for pattern in boolean_patterns)
    
    def _check_datetime(self, series: pd.Series) -> float:
        """Check confidence that series contains datetime values"""
        try:
            # Try parsing as datetime
            parsed = pd.to_datetime(series, errors='coerce')
            success_rate = (parsed.notna()).sum() / len(series)
            return success_rate
        except:
            return 0.0
    
    def _check_numeric(self, series: pd.Series) -> float:
        """Check confidence that series contains numeric values"""
        try:
            # Try converting to numeric
            converted = pd.to_numeric(series, errors='coerce')
            success_rate = (converted.notna()).sum() / len(series)
            return success_rate
        except:
            return 0.0
    
    def _check_categorical(self, series: pd.Series) -> float:
        """Check confidence that series is categorical"""
        unique_count = series.nunique()
        total_count = len(series)
        
        # Calculate uniqueness ratio
        uniqueness_ratio = unique_count / total_count
        
        # High uniqueness suggests not categorical
        if uniqueness_ratio > 0.5:
            return 0.3
        
        # Very low uniqueness suggests categorical
        if uniqueness_ratio < 0.1:
            return 0.9
        
        # Medium uniqueness - check other factors
        # If all values are strings and relatively short, likely categorical
        if series.dtype == 'object':
            avg_length = series.astype(str).str.len().mean()
            if avg_length < 50:  # Short strings are likely categorical
                return 0.8 - (uniqueness_ratio * 0.5)
        
        return 0.5

class TypeEnforcer:
    """
    Automated data type enforcement with validation
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize type enforcer
        
        Args:
            strict_mode: If True, raises exceptions on type enforcement failures
        """
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(__name__)
    
    def enforce_types(self, df: pd.DataFrame, 
                     type_rules: Union[List[TypeEnforcementRule], Dict[str, str]]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[DataQualityIssue]]]:
        """
        Enforce data types according to rules
        
        Args:
            df: Input dataframe
            type_rules: List of TypeEnforcementRule objects or dictionary mapping column names to type strings
            
        Returns:
            If type_rules is a dictionary: corrected_dataframe
            If type_rules is a list: Tuple of (corrected_dataframe, list_of_issues)
        """
        # Handle dictionary input for convenience
        if isinstance(type_rules, dict):
            return self._enforce_types_from_dict(df, type_rules)
        
        # Handle list of TypeEnforcementRule objects
        corrected_df = df.copy()
        issues = []
        
        for rule in type_rules:
            if rule.column not in corrected_df.columns:
                issues.append(DataQualityIssue(
                    issue_type="missing_column",
                    severity="high",
                    column=rule.column,
                    description=f"Column '{rule.column}' not found in dataset",
                    suggested_action="Check column names or data source"
                ))
                continue
            
            column_issues = self._enforce_column_type(corrected_df, rule)
            issues.extend(column_issues)
        
        return corrected_df, issues
    
    def _enforce_types_from_dict(self, df: pd.DataFrame, type_dict: Dict[str, str]) -> pd.DataFrame:
        """Convenience method to enforce types from a dictionary"""
        corrected_df = df.copy()
        
        for column, dtype_str in type_dict.items():
            if column not in corrected_df.columns:
                continue
                
            try:
                if dtype_str == 'float64' or dtype_str == 'numeric':
                    corrected_df[column] = pd.to_numeric(corrected_df[column], errors='coerce')
                elif dtype_str == 'int64':
                    corrected_df[column] = pd.to_numeric(corrected_df[column], errors='coerce').fillna(0).astype('int64')
                elif dtype_str == 'category' or dtype_str == 'categorical':
                    corrected_df[column] = corrected_df[column].astype('category')
                elif dtype_str == 'datetime':
                    corrected_df[column] = pd.to_datetime(corrected_df[column], errors='coerce')
                elif dtype_str == 'bool' or dtype_str == 'boolean':
                    # Simple boolean conversion
                    bool_map = {'true': True, 'false': False, '1': True, '0': False, 
                               'yes': True, 'no': False, 'y': True, 'n': False}
                    corrected_df[column] = corrected_df[column].astype(str).str.lower().map(bool_map)
                else:
                    corrected_df[column] = corrected_df[column].astype(dtype_str)
            except Exception as e:
                self.logger.warning(f"Failed to convert column {column} to {dtype_str}: {e}")
        
        return corrected_df
    
    def _enforce_column_type(self, df: pd.DataFrame, 
                           rule: TypeEnforcementRule) -> List[DataQualityIssue]:
        """Enforce type for a single column"""
        issues = []
        column = rule.column
        
        try:
            if rule.target_type == DataType.NUMERIC:
                df[column], column_issues = self._enforce_numeric(df[column], rule)
            elif rule.target_type == DataType.CATEGORICAL:
                df[column], column_issues = self._enforce_categorical(df[column], rule)
            elif rule.target_type == DataType.DATETIME:
                df[column], column_issues = self._enforce_datetime(df[column], rule)
            elif rule.target_type == DataType.BOOLEAN:
                df[column], column_issues = self._enforce_boolean(df[column], rule)
            else:  # TEXT
                df[column], column_issues = self._enforce_text(df[column], rule)
            
            issues.extend(column_issues)
            
        except Exception as e:
            issues.append(DataQualityIssue(
                issue_type="type_enforcement_error",
                severity="high",
                column=column,
                description=f"Failed to enforce type {rule.target_type.value}: {str(e)}",
                suggested_action="Review data and type enforcement rules"
            ))
        
        return issues
    
    def _enforce_numeric(self, series: pd.Series, 
                        rule: TypeEnforcementRule) -> Tuple[pd.Series, List[DataQualityIssue]]:
        """Enforce numeric type"""
        issues = []
        
        # Convert to numeric, marking errors
        converted = pd.to_numeric(series, errors='coerce')
        failed_indices = converted.isna() & series.notna()
        
        if failed_indices.any():
            issues.append(DataQualityIssue(
                issue_type="type_conversion_failure",
                severity="medium",
                column=rule.column,
                description=f"{failed_indices.sum()} values could not be converted to numeric",
                suggested_action="Review non-numeric values or use different data type",
                affected_rows=series[failed_indices].index.tolist()
            ))
        
        return converted, issues
    
    def _enforce_categorical(self, series: pd.Series, 
                           rule: TypeEnforcementRule) -> Tuple[pd.Series, List[DataQualityIssue]]:
        """Enforce categorical type"""
        issues = []
        
        # Convert to categorical
        max_categories = rule.validation_rules.get('max_categories', 100)
        unique_count = series.nunique()
        
        if unique_count > max_categories:
            issues.append(DataQualityIssue(
                issue_type="high_cardinality",
                severity="medium",
                column=rule.column,
                description=f"Column has {unique_count} unique values (limit: {max_categories})",
                suggested_action="Consider using different data type or reducing cardinality"
            ))
        
        categorical_series = series.astype('category')
        return categorical_series, issues
    
    def _enforce_datetime(self, series: pd.Series, 
                         rule: TypeEnforcementRule) -> Tuple[pd.Series, List[DataQualityIssue]]:
        """Enforce datetime type"""
        issues = []
        
        # Try different datetime formats
        datetime_formats = rule.validation_rules.get('formats', [None])
        
        converted = None
        for fmt in datetime_formats:
            try:
                converted = pd.to_datetime(series, format=fmt, errors='coerce')
                break
            except:
                continue
        
        if converted is None:
            converted = pd.to_datetime(series, errors='coerce')
        
        failed_indices = converted.isna() & series.notna()
        
        if failed_indices.any():
            issues.append(DataQualityIssue(
                issue_type="datetime_conversion_failure",
                severity="medium",
                column=rule.column,
                description=f"{failed_indices.sum()} values could not be converted to datetime",
                suggested_action="Review date formats or use different data type",
                affected_rows=series[failed_indices].index.tolist()
            ))
        
        return converted, issues
    
    def _enforce_boolean(self, series: pd.Series, 
                        rule: TypeEnforcementRule) -> Tuple[pd.Series, List[DataQualityIssue]]:
        """Enforce boolean type"""
        issues = []
        
        # Map common boolean representations
        boolean_mapping = {
            'true': True, 'false': False,
            'yes': True, 'no': False,
            'y': True, 'n': False,
            '1': True, '0': False,
            'on': True, 'off': False,
            1: True, 0: False
        }
        
        # Apply mapping
        mapped_series = series.astype(str).str.lower().map(boolean_mapping)
        failed_indices = mapped_series.isna() & series.notna()
        
        if failed_indices.any():
            issues.append(DataQualityIssue(
                issue_type="boolean_conversion_failure",
                severity="medium",
                column=rule.column,
                description=f"{failed_indices.sum()} values could not be converted to boolean",
                suggested_action="Review boolean representations or use different data type",
                affected_rows=series[failed_indices].index.tolist()
            ))
        
        return mapped_series, issues
    
    def _enforce_text(self, series: pd.Series, 
                     rule: TypeEnforcementRule) -> Tuple[pd.Series, List[DataQualityIssue]]:
        """Enforce text type"""
        issues = []
        
        # Convert to string
        text_series = series.astype(str)
        
        # Check for length constraints
        max_length = rule.validation_rules.get('max_length')
        if max_length:
            long_values = text_series.str.len() > max_length
            if long_values.any():
                issues.append(DataQualityIssue(
                    issue_type="text_length_violation",
                    severity="low",
                    column=rule.column,
                    description=f"{long_values.sum()} values exceed maximum length {max_length}",
                    suggested_action="Consider truncating or using different constraints",
                    affected_rows=series[long_values].index.tolist()
                ))
        
        return text_series, issues

class AdvancedOutlierDetector:
    """
    Advanced outlier detection with ensemble methods and interactive feedback
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize outlier detector
        
        Args:
            contamination: Expected proportion of outliers
        """
        self.contamination = contamination
        self.logger = logging.getLogger(__name__)
        self.detectors = {}
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize outlier detection algorithms"""
        self.detectors = {
            OutlierMethod.ISOLATION_FOREST: IsolationForest(
                contamination=self.contamination,
                random_state=42
            ),
            OutlierMethod.LOCAL_OUTLIER_FACTOR: LocalOutlierFactor(
                contamination=self.contamination
            ),
            OutlierMethod.ELLIPTIC_ENVELOPE: EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
        }
    
    def detect_outliers(self, df: pd.DataFrame, 
                       methods: List[OutlierMethod] = None,
                       ensemble_voting: bool = True) -> Dict[str, Any]:
        """
        Detect outliers using multiple methods
        
        Args:
            df: Input dataframe
            methods: List of detection methods to use
            ensemble_voting: Whether to use ensemble voting
            
        Returns:
            Dictionary containing outlier detection results
        """
        if methods is None:
            methods = [OutlierMethod.ISOLATION_FOREST, OutlierMethod.Z_SCORE]
        
        # Prepare numeric data
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            return {"outliers": pd.Series(False, index=df.index), "scores": {}}
        
        X = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Scale data for algorithms that require it
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        outlier_results = {}
        outlier_scores = {}
        
        for method in methods:
            if method == OutlierMethod.Z_SCORE:
                outliers, scores = self._z_score_detection(X)
            elif method == OutlierMethod.IQR:
                outliers, scores = self._iqr_detection(X)
            else:
                outliers, scores = self._sklearn_detection(X_scaled, method)
            
            outlier_results[method.value] = outliers
            outlier_scores[method.value] = scores
        
        # Ensemble voting if multiple methods
        if ensemble_voting and len(methods) > 1:
            final_outliers = self._ensemble_voting(outlier_results)
        else:
            # Use first method result
            final_outliers = list(outlier_results.values())[0]
        
        return {
            "outliers": pd.Series(final_outliers, index=df.index),
            "scores": outlier_scores,
            "individual_results": outlier_results
        }
    
    def _z_score_detection(self, X: pd.DataFrame, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Z-score based outlier detection"""
        z_scores = np.abs((X - X.mean()) / X.std())
        outliers = (z_scores > threshold).any(axis=1)
        scores = z_scores.max(axis=1)  # Max z-score across features
        
        return outliers.values, scores.values
    
    def _iqr_detection(self, X: pd.DataFrame, factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """IQR based outlier detection"""
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = ((X < lower_bound) | (X > upper_bound)).any(axis=1)
        
        # Calculate outlier scores as distance from IQR bounds
        scores = np.maximum(
            (lower_bound - X).clip(lower=0),
            (X - upper_bound).clip(lower=0)
        ).max(axis=1)
        
        return outliers.values, scores.values
    
    def _sklearn_detection(self, X: np.ndarray, method: OutlierMethod) -> Tuple[np.ndarray, np.ndarray]:
        """Scikit-learn based outlier detection"""
        detector = self.detectors[method]
        
        if method == OutlierMethod.LOCAL_OUTLIER_FACTOR:
            predictions = detector.fit_predict(X)
            scores = -detector.negative_outlier_factor_
        else:
            predictions = detector.fit_predict(X)
            scores = detector.score_samples(X)
        
        outliers = predictions == -1
        
        return outliers, scores
    
    def _ensemble_voting(self, outlier_results: Dict[str, np.ndarray]) -> np.ndarray:
        """Ensemble voting for outlier detection"""
        votes = np.stack(list(outlier_results.values()))
        vote_counts = votes.sum(axis=0)
        
        # Majority voting
        threshold = len(outlier_results) / 2
        final_outliers = vote_counts > threshold
        
        return final_outliers

class DataQualityAssessor:
    """
    Comprehensive data quality assessment and scoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.type_inferrer = DataTypeInferrer()
        self.outlier_detector = AdvancedOutlierDetector()
    
    def assess_quality(self, df: pd.DataFrame, 
                      target_column: str = None) -> DataQualityReport:
        """
        Comprehensive data quality assessment
        
        Args:
            df: Input dataframe
            target_column: Target variable column name
            
        Returns:
            DataQualityReport with comprehensive assessment
        """
        issues = []
        column_scores = {}
        
        # Check basic data integrity
        issues.extend(self._check_basic_integrity(df))
        
        # Check missing data
        issues.extend(self._check_missing_data(df))
        
        # Check data types and consistency
        issues.extend(self._check_data_types(df))
        
        # Check for duplicates
        issues.extend(self._check_duplicates(df))
        
        # Check outliers
        issues.extend(self._check_outliers(df))
        
        # Check target variable quality (if specified)
        if target_column and target_column in df.columns:
            issues.extend(self._check_target_variable(df, target_column))
        
        # Calculate column-wise quality scores
        for column in df.columns:
            column_scores[column] = self._calculate_column_score(df[column], issues)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(issues, len(df))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues)
        
        # Categorize issues by severity
        issues_by_severity = {
            'low': len([i for i in issues if i.severity == 'low']),
            'medium': len([i for i in issues if i.severity == 'medium']),
            'high': len([i for i in issues if i.severity == 'high']),
            'critical': len([i for i in issues if i.severity == 'critical'])
        }
        
        return DataQualityReport(
            overall_score=overall_score,
            total_issues=len(issues),
            issues_by_severity=issues_by_severity,
            issues=issues,
            column_scores=column_scores,
            recommendations=recommendations,
            metadata={
                'dataset_shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum(),
                'assessment_timestamp': pd.Timestamp.now().isoformat()
            }
        )
    
    def _check_basic_integrity(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Check basic data integrity"""
        issues = []
        
        # Check if dataframe is empty
        if df.empty:
            issues.append(DataQualityIssue(
                issue_type="empty_dataset",
                severity="critical",
                column=None,
                description="Dataset is empty",
                suggested_action="Check data source and loading process"
            ))
        
        # Check for unnamed columns
        unnamed_columns = [col for col in df.columns if str(col).startswith('Unnamed:')]
        if unnamed_columns:
            issues.append(DataQualityIssue(
                issue_type="unnamed_columns",
                severity="medium",
                column=None,
                description=f"Found {len(unnamed_columns)} unnamed columns",
                suggested_action="Provide meaningful column names",
                metadata={"unnamed_columns": unnamed_columns}
            ))
        
        return issues
    
    def _check_missing_data(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Check for missing data patterns"""
        issues = []
        
        missing_stats = df.isnull().sum()
        total_rows = len(df)
        
        for column, missing_count in missing_stats.items():
            if missing_count > 0:
                missing_percentage = (missing_count / total_rows) * 100
                
                if missing_percentage >= 80:
                    severity = "critical"
                elif missing_percentage >= 50:
                    severity = "high"
                elif missing_percentage >= 20:
                    severity = "medium"
                else:
                    severity = "low"
                
                issues.append(DataQualityIssue(
                    issue_type="missing_data",
                    severity=severity,
                    column=column,
                    description=f"Column has {missing_count} missing values ({missing_percentage:.1f}%)",
                    suggested_action="Consider imputation, removal, or data collection improvement",
                    affected_rows=df[df[column].isnull()].index.tolist(),
                    metadata={"missing_percentage": missing_percentage}
                ))
        
        return issues
    
    def _check_data_types(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Check data type consistency and appropriateness"""
        issues = []
        
        # Infer optimal data types
        type_inferences = self.type_inferrer.infer_types(df)
        
        for column, (inferred_type, confidence) in type_inferences.items():
            current_dtype = str(df[column].dtype)
            
            # Check if current type is suboptimal
            if confidence > 0.8:
                if (inferred_type == DataType.NUMERIC and 
                    current_dtype == 'object'):
                    issues.append(DataQualityIssue(
                        issue_type="suboptimal_data_type",
                        severity="medium",
                        column=column,
                        description=f"Column appears numeric but stored as {current_dtype}",
                        suggested_action="Convert to numeric type for better performance",
                        metadata={"suggested_type": inferred_type.value, "confidence": confidence}
                    ))
                
                elif (inferred_type == DataType.CATEGORICAL and 
                      current_dtype == 'object' and 
                      df[column].nunique() / len(df) < 0.1):
                    issues.append(DataQualityIssue(
                        issue_type="suboptimal_data_type",
                        severity="low",
                        column=column,
                        description=f"Column appears categorical but stored as {current_dtype}",
                        suggested_action="Convert to categorical type for memory efficiency",
                        metadata={"suggested_type": inferred_type.value, "confidence": confidence}
                    ))
        
        return issues
    
    def _check_duplicates(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Check for duplicate rows"""
        issues = []
        
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100
            
            severity = "high" if duplicate_percentage > 10 else "medium"
            
            issues.append(DataQualityIssue(
                issue_type="duplicate_rows",
                severity=severity,
                column=None,
                description=f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.1f}%)",
                suggested_action="Remove duplicates or investigate data source",
                affected_rows=df[df.duplicated()].index.tolist(),
                metadata={"duplicate_percentage": duplicate_percentage}
            ))
        
        return issues
    
    def _check_outliers(self, df: pd.DataFrame) -> List[DataQualityIssue]:
        """Check for outliers in numeric columns"""
        issues = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            outlier_results = self.outlier_detector.detect_outliers(df[numeric_columns])
            outliers = outlier_results["outliers"]
            
            outlier_count = outliers.sum()
            if outlier_count > 0:
                outlier_percentage = (outlier_count / len(df)) * 100
                
                severity = "high" if outlier_percentage > 20 else "medium"
                
                issues.append(DataQualityIssue(
                    issue_type="outliers_detected",
                    severity=severity,
                    column=None,
                    description=f"Found {outlier_count} potential outliers ({outlier_percentage:.1f}%)",
                    suggested_action="Investigate outliers and consider removal or transformation",
                    affected_rows=df[outliers].index.tolist(),
                    metadata={"outlier_percentage": outlier_percentage}
                ))
        
        return issues
    
    def _check_target_variable(self, df: pd.DataFrame, target_column: str) -> List[DataQualityIssue]:
        """Check target variable quality"""
        issues = []
        
        target_series = df[target_column]
        
        # Check for missing targets
        missing_targets = target_series.isnull().sum()
        if missing_targets > 0:
            issues.append(DataQualityIssue(
                issue_type="missing_target_values",
                severity="critical",
                column=target_column,
                description=f"Target variable has {missing_targets} missing values",
                suggested_action="Remove rows with missing targets or investigate data quality",
                affected_rows=df[target_series.isnull()].index.tolist()
            ))
        
        # Check class imbalance (for classification)
        if target_series.dtype == 'object' or target_series.nunique() < 20:
            value_counts = target_series.value_counts()
            class_ratios = value_counts / len(target_series)
            
            min_class_ratio = class_ratios.min()
            if min_class_ratio < 0.05:  # Less than 5% representation
                issues.append(DataQualityIssue(
                    issue_type="class_imbalance",
                    severity="medium",
                    column=target_column,
                    description=f"Severe class imbalance detected (min class: {min_class_ratio:.1%})",
                    suggested_action="Consider resampling techniques or different algorithms",
                    metadata={"class_distribution": value_counts.to_dict()}
                ))
        
        return issues
    
    def _calculate_column_score(self, series: pd.Series, issues: List[DataQualityIssue]) -> float:
        """Calculate quality score for a single column"""
        base_score = 100.0
        
        # Get issues affecting this column
        column_issues = [issue for issue in issues if issue.column == series.name]
        
        for issue in column_issues:
            if issue.severity == "critical":
                base_score -= 30
            elif issue.severity == "high":
                base_score -= 20
            elif issue.severity == "medium":
                base_score -= 10
            else:  # low
                base_score -= 5
        
        return max(0.0, min(100.0, base_score))
    
    def _calculate_overall_score(self, issues: List[DataQualityIssue], num_rows: int) -> float:
        """Calculate overall data quality score"""
        base_score = 100.0
        
        for issue in issues:
            weight = 1.0
            
            # Weight by number of affected rows
            if issue.affected_rows:
                weight = len(issue.affected_rows) / num_rows
            
            if issue.severity == "critical":
                base_score -= 25 * weight
            elif issue.severity == "high":
                base_score -= 15 * weight
            elif issue.severity == "medium":
                base_score -= 8 * weight
            else:  # low
                base_score -= 3 * weight
        
        return max(0.0, min(100.0, base_score))
    
    def _generate_recommendations(self, issues: List[DataQualityIssue]) -> List[str]:
        """Generate actionable recommendations based on issues"""
        recommendations = []
        
        # Aggregate recommendations by type
        issue_types = {}
        for issue in issues:
            if issue.issue_type not in issue_types:
                issue_types[issue.issue_type] = []
            issue_types[issue.issue_type].append(issue)
        
        # Generate recommendations based on issue patterns
        for issue_type, type_issues in issue_types.items():
            if issue_type == "missing_data":
                high_missing_columns = [i.column for i in type_issues if i.severity in ["high", "critical"]]
                if high_missing_columns:
                    recommendations.append(
                        f"Consider removing columns with excessive missing data: {', '.join(high_missing_columns)}"
                    )
            
            elif issue_type == "duplicate_rows":
                recommendations.append("Remove duplicate rows to improve data quality and model performance")
            
            elif issue_type == "outliers_detected":
                recommendations.append("Investigate outliers and consider robust scaling or outlier removal")
            
            elif issue_type == "class_imbalance":
                recommendations.append("Address class imbalance using resampling techniques or cost-sensitive learning")
            
            elif issue_type == "suboptimal_data_type":
                recommendations.append("Optimize data types for better memory usage and performance")
        
        # Add general recommendations
        if len(issues) > 10:
            recommendations.append("High number of data quality issues detected - consider comprehensive data cleaning")
        
        return recommendations
