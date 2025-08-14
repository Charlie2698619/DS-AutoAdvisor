#!/usr/bin/env python3
"""
📊 Quality Metrics Calculator for DS-AutoAdvisor
===============================================

Comprehensive data quality metrics calculation including:
- Completeness (missing data analysis)
- Consistency (format and value consistency)
- Accuracy (pattern compliance and validation)
- Validity (business rule compliance)
- Uniqueness (duplicate and identifier analysis)
- Timeliness (data freshness and temporal patterns)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import yaml
from pathlib import Path

@dataclass
class QualityScore:
    """Quality score with detailed breakdown"""
    overall_score: float
    dimension_scores: Dict[str, float]
    issues_count: int
    total_checks: int
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class QualityDimension:
    """Individual quality dimension assessment"""
    name: str
    score: float
    weight: float
    issues: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    recommendations: List[str]

class QualityMetricsCalculator:
    """
    Comprehensive data quality metrics calculator with YAML configuration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize quality metrics calculator"""
        self.config = self._load_config(config_path)
        self.quality_dimensions = self._initialize_dimensions()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load quality metrics configuration from YAML"""
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "unified_config_v3.yaml"
        
        default_config = {
            'quality_metrics': {
                'dimensions': {
                    'completeness': {
                        'weight': 0.25,
                        'thresholds': {
                            'excellent': 0.95,
                            'good': 0.85,
                            'acceptable': 0.70,
                            'poor': 0.50
                        }
                    },
                    'consistency': {
                        'weight': 0.20,
                        'thresholds': {
                            'excellent': 0.98,
                            'good': 0.90,
                            'acceptable': 0.75,
                            'poor': 0.60
                        }
                    },
                    'accuracy': {
                        'weight': 0.20,
                        'thresholds': {
                            'excellent': 0.95,
                            'good': 0.85,
                            'acceptable': 0.70,
                            'poor': 0.55
                        }
                    },
                    'validity': {
                        'weight': 0.15,
                        'thresholds': {
                            'excellent': 0.98,
                            'good': 0.90,
                            'acceptable': 0.80,
                            'poor': 0.65
                        }
                    },
                    'uniqueness': {
                        'weight': 0.10,
                        'thresholds': {
                            'excellent': 0.99,
                            'good': 0.95,
                            'acceptable': 0.85,
                            'poor': 0.70
                        }
                    },
                    'timeliness': {
                        'weight': 0.10,
                        'thresholds': {
                            'excellent': 0.95,
                            'good': 0.85,
                            'acceptable': 0.70,
                            'poor': 0.50
                        }
                    }
                },
                'scoring': {
                    'excellent_score': 100,
                    'good_score': 80,
                    'acceptable_score': 60,
                    'poor_score': 40,
                    'failing_score': 20
                },
                'validation_rules': {
                    'numeric_ranges': True,
                    'categorical_domains': True,
                    'date_ranges': True,
                    'business_rules': True
                }
            }
        }
        
        if Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    
                    # Priority 1: Check global section first
                    if 'quality_metrics' in loaded_config:
                        print("📋 Quality Metrics: Using global configuration")
                        return loaded_config['quality_metrics']
                    
                    # Priority 2: Check custom_mode
                    if 'custom_mode' in loaded_config and 'data_discovery' in loaded_config['custom_mode']:
                        discovery_config = loaded_config['custom_mode']['data_discovery']
                        if 'quality_metrics' in discovery_config:
                            print("📋 Quality Metrics: Using custom_mode configuration")
                            return discovery_config['quality_metrics']
                    
                    # Priority 3: Check fast_mode
                    if 'fast_mode' in loaded_config and 'data_discovery' in loaded_config['fast_mode']:
                        discovery_config = loaded_config['fast_mode']['data_discovery']
                        if 'quality_metrics' in discovery_config:
                            print("📋 Quality Metrics: Using fast_mode configuration")
                            return discovery_config['quality_metrics']
                
                print("📋 Quality Metrics: No config found, using defaults")
                return default_config['quality_metrics']
            except Exception as e:
                print(f"⚠️ Error loading quality metrics config: {e}")
                return default_config['quality_metrics']
        else:
            return default_config['quality_metrics']
    
    def _initialize_dimensions(self) -> Dict[str, QualityDimension]:
        """Initialize quality dimension objects"""
        dimensions = {}
        
        for dim_name, dim_config in self.config['dimensions'].items():
            dimensions[dim_name] = QualityDimension(
                name=dim_name,
                score=0.0,
                weight=dim_config['weight'],
                issues=[],
                metrics={},
                recommendations=[]
            )
        
        return dimensions
    
    def calculate_quality_score(self, df: pd.DataFrame, 
                              target_column: Optional[str] = None) -> QualityScore:
        """Calculate comprehensive quality score for the dataset"""
        print(f"📊 Calculating quality metrics for dataset ({df.shape[0]} rows, {df.shape[1]} columns)...")
        
        # Reset dimensions
        self.quality_dimensions = self._initialize_dimensions()
        
        # Calculate each dimension
        self._calculate_completeness(df)
        self._calculate_consistency(df)
        self._calculate_accuracy(df)
        self._calculate_validity(df, target_column)
        self._calculate_uniqueness(df)
        self._calculate_timeliness(df)
        
        # Calculate overall score
        overall_score, total_issues, total_checks = self._calculate_overall_score()
        
        # Calculate confidence based on data size and completeness
        confidence = self._calculate_confidence(df)
        
        # Compile metadata
        metadata = {
            'dataset_shape': df.shape,
            'calculation_timestamp': datetime.now().isoformat(),
            'target_column': target_column,
            'dimension_weights': {name: dim.weight for name, dim in self.quality_dimensions.items()}
        }
        
        return QualityScore(
            overall_score=overall_score,
            dimension_scores={name: dim.score for name, dim in self.quality_dimensions.items()},
            issues_count=total_issues,
            total_checks=total_checks,
            confidence=confidence,
            metadata=metadata
        )
    
    def _calculate_completeness(self, df: pd.DataFrame):
        """Calculate completeness dimension (missing data analysis)"""
        dimension = self.quality_dimensions['completeness']
        
        # Overall missing data ratio
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness_ratio = 1 - (missing_cells / total_cells)
        
        # Column-wise completeness analysis
        column_completeness = {}
        critical_missing_columns = []
        
        for column in df.columns:
            col_missing_ratio = df[column].isnull().sum() / len(df)
            col_completeness = 1 - col_missing_ratio
            column_completeness[column] = col_completeness
            
            # Identify critical issues
            if col_missing_ratio > 0.5:
                critical_missing_columns.append(column)
                dimension.issues.append({
                    'type': 'high_missing_data',
                    'column': column,
                    'severity': 'high',
                    'missing_ratio': col_missing_ratio,
                    'description': f"Column '{column}' has {col_missing_ratio:.1%} missing values"
                })
            elif col_missing_ratio > 0.2:
                dimension.issues.append({
                    'type': 'moderate_missing_data',
                    'column': column,
                    'severity': 'medium',
                    'missing_ratio': col_missing_ratio,
                    'description': f"Column '{column}' has {col_missing_ratio:.1%} missing values"
                })
        
        # Calculate score based on thresholds
        thresholds = self.config['dimensions']['completeness']['thresholds']
        scoring = self.config['scoring']
        
        if completeness_ratio >= thresholds['excellent']:
            dimension.score = scoring['excellent_score']
        elif completeness_ratio >= thresholds['good']:
            dimension.score = scoring['good_score']
        elif completeness_ratio >= thresholds['acceptable']:
            dimension.score = scoring['acceptable_score']
        elif completeness_ratio >= thresholds['poor']:
            dimension.score = scoring['poor_score']
        else:
            dimension.score = scoring['failing_score']
        
        # Store metrics
        dimension.metrics = {
            'overall_completeness': completeness_ratio,
            'total_missing_cells': missing_cells,
            'columns_with_missing': len([c for c in column_completeness.values() if c < 1.0]),
            'critical_missing_columns': len(critical_missing_columns),
            'column_completeness': column_completeness
        }
        
        # Generate recommendations
        if critical_missing_columns:
            dimension.recommendations.append(f"Consider dropping {len(critical_missing_columns)} columns with >50% missing data")
        if completeness_ratio < 0.8:
            dimension.recommendations.append("Implement comprehensive missing data imputation strategy")
        if len([c for c in column_completeness.values() if c < 0.9]) > 5:
            dimension.recommendations.append("Review data collection process - many columns have missing values")
    
    def _calculate_consistency(self, df: pd.DataFrame):
        """Calculate consistency dimension (format and value consistency)"""
        dimension = self.quality_dimensions['consistency']
        
        consistency_scores = []
        format_issues = 0
        
        for column in df.columns:
            col_series = df[column].dropna()
            if len(col_series) == 0:
                continue
            
            # Format consistency for text columns
            if col_series.dtype == 'object':
                format_score = self._assess_format_consistency(col_series, column, dimension)
                consistency_scores.append(format_score)
            else:
                # For numeric columns, check for data type consistency
                numeric_score = self._assess_numeric_consistency(col_series, column, dimension)
                consistency_scores.append(numeric_score)
        
        # Calculate overall consistency score
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        
        # Apply scoring thresholds
        thresholds = self.config['dimensions']['consistency']['thresholds']
        scoring = self.config['scoring']
        
        if avg_consistency >= thresholds['excellent']:
            dimension.score = scoring['excellent_score']
        elif avg_consistency >= thresholds['good']:
            dimension.score = scoring['good_score']
        elif avg_consistency >= thresholds['acceptable']:
            dimension.score = scoring['acceptable_score']
        elif avg_consistency >= thresholds['poor']:
            dimension.score = scoring['poor_score']
        else:
            dimension.score = scoring['failing_score']
        
        dimension.metrics = {
            'average_consistency': avg_consistency,
            'format_issues_count': format_issues,
            'columns_analyzed': len(consistency_scores)
        }
        
        # Generate recommendations
        if avg_consistency < 0.8:
            dimension.recommendations.append("Standardize data formats and implement validation rules")
        if format_issues > 0:
            dimension.recommendations.append(f"Address {format_issues} format consistency issues found")
    
    def _assess_format_consistency(self, series: pd.Series, column: str, 
                                 dimension: QualityDimension) -> float:
        """Assess format consistency for text columns"""
        if len(series) == 0:
            return 1.0
        
        # Check case consistency
        has_upper = series.str.contains(r'[A-Z]', na=False).sum()
        has_lower = series.str.contains(r'[a-z]', na=False).sum()
        
        case_consistency = 1.0
        if has_upper > 0 and has_lower > 0:
            # Mixed case - check if it's consistent pattern
            case_ratio = min(has_upper, has_lower) / len(series)
            if case_ratio > 0.1:  # Significant inconsistency
                case_consistency = 1 - case_ratio
                dimension.issues.append({
                    'type': 'case_inconsistency',
                    'column': column,
                    'severity': 'low',
                    'description': f"Inconsistent case formatting in '{column}'"
                })
        
        # Check whitespace consistency
        has_leading_space = series.str.startswith(' ').sum()
        has_trailing_space = series.str.endswith(' ').sum()
        
        whitespace_consistency = 1.0
        if has_leading_space > 0 or has_trailing_space > 0:
            whitespace_ratio = (has_leading_space + has_trailing_space) / len(series)
            if whitespace_ratio > 0.05:  # More than 5% have whitespace issues
                whitespace_consistency = 1 - whitespace_ratio
                dimension.issues.append({
                    'type': 'whitespace_inconsistency',
                    'column': column,
                    'severity': 'low',
                    'description': f"Inconsistent whitespace in '{column}'"
                })
        
        return (case_consistency + whitespace_consistency) / 2
    
    def _assess_numeric_consistency(self, series: pd.Series, column: str,
                                  dimension: QualityDimension) -> float:
        """Assess consistency for numeric columns"""
        # Check for mixed data types that pandas might have missed
        string_series = series.astype(str)
        
        # Look for non-numeric patterns in supposedly numeric data
        non_numeric_pattern = string_series.str.contains(r'[^0-9.\-+eE]', na=False)
        non_numeric_count = non_numeric_pattern.sum()
        
        if non_numeric_count > 0:
            inconsistency_ratio = non_numeric_count / len(series)
            dimension.issues.append({
                'type': 'numeric_inconsistency',
                'column': column,
                'severity': 'medium',
                'inconsistency_ratio': inconsistency_ratio,
                'description': f"Found {non_numeric_count} non-numeric values in numeric column '{column}'"
            })
            return 1 - inconsistency_ratio
        
        return 1.0
    
    def _calculate_accuracy(self, df: pd.DataFrame):
        """Calculate accuracy dimension (pattern compliance and validation)"""
        dimension = self.quality_dimensions['accuracy']
        
        accuracy_scores = []
        
        for column in df.columns:
            col_series = df[column].dropna()
            if len(col_series) == 0:
                continue
            
            # Pattern-based accuracy assessment
            pattern_score = self._assess_pattern_accuracy(col_series, column, dimension)
            accuracy_scores.append(pattern_score)
        
        # Calculate overall accuracy
        avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 1.0
        
        # Apply scoring thresholds
        thresholds = self.config['dimensions']['accuracy']['thresholds']
        scoring = self.config['scoring']
        
        if avg_accuracy >= thresholds['excellent']:
            dimension.score = scoring['excellent_score']
        elif avg_accuracy >= thresholds['good']:
            dimension.score = scoring['good_score']
        elif avg_accuracy >= thresholds['acceptable']:
            dimension.score = scoring['acceptable_score']
        elif avg_accuracy >= thresholds['poor']:
            dimension.score = scoring['poor_score']
        else:
            dimension.score = scoring['failing_score']
        
        dimension.metrics = {
            'average_accuracy': avg_accuracy,
            'columns_validated': len(accuracy_scores)
        }
        
        if avg_accuracy < 0.8:
            dimension.recommendations.append("Implement data validation rules and fix accuracy issues")
    
    def _assess_pattern_accuracy(self, series: pd.Series, column: str,
                               dimension: QualityDimension) -> float:
        """Assess pattern accuracy for a column"""
        # Simple pattern validation based on column name hints
        column_lower = column.lower()
        
        if 'email' in column_lower:
            return self._validate_email_pattern(series, column, dimension)
        elif 'phone' in column_lower:
            return self._validate_phone_pattern(series, column, dimension)
        elif 'date' in column_lower or 'time' in column_lower:
            return self._validate_date_pattern(series, column, dimension)
        else:
            # Generic validation - check for obviously invalid values
            return self._validate_generic_patterns(series, column, dimension)
    
    def _validate_email_pattern(self, series: pd.Series, column: str,
                              dimension: QualityDimension) -> float:
        """Validate email patterns"""
        import re
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        valid_count = 0
        for value in series:
            if email_pattern.match(str(value)):
                valid_count += 1
        
        accuracy = valid_count / len(series)
        
        if accuracy < 0.9:
            dimension.issues.append({
                'type': 'invalid_email_format',
                'column': column,
                'severity': 'medium',
                'accuracy': accuracy,
                'description': f"Email column '{column}' has {(1-accuracy):.1%} invalid email formats"
            })
        
        return accuracy
    
    def _validate_phone_pattern(self, series: pd.Series, column: str,
                              dimension: QualityDimension) -> float:
        """Validate phone number patterns"""
        import re
        # Basic phone pattern - can be made more sophisticated
        phone_pattern = re.compile(r'^[\+]?[1-9]?[0-9\s\-\(\)]{7,15}$')
        
        valid_count = 0
        for value in series:
            if phone_pattern.match(str(value).strip()):
                valid_count += 1
        
        accuracy = valid_count / len(series)
        
        if accuracy < 0.9:
            dimension.issues.append({
                'type': 'invalid_phone_format',
                'column': column,
                'severity': 'medium',
                'accuracy': accuracy,
                'description': f"Phone column '{column}' has {(1-accuracy):.1%} invalid phone formats"
            })
        
        return accuracy
    
    def _validate_date_pattern(self, series: pd.Series, column: str,
                             dimension: QualityDimension) -> float:
        """Validate date patterns"""
        try:
            converted_dates = pd.to_datetime(series, errors='coerce')
            valid_count = converted_dates.notna().sum()
            accuracy = valid_count / len(series)
            
            if accuracy < 0.9:
                dimension.issues.append({
                    'type': 'invalid_date_format',
                    'column': column,
                    'severity': 'medium',
                    'accuracy': accuracy,
                    'description': f"Date column '{column}' has {(1-accuracy):.1%} invalid date formats"
                })
            
            return accuracy
        except:
            return 0.5  # Default if conversion fails
    
    def _validate_generic_patterns(self, series: pd.Series, column: str,
                                 dimension: QualityDimension) -> float:
        """Generic pattern validation"""
        # Check for obviously invalid values (empty strings, suspicious patterns)
        string_series = series.astype(str)
        
        # Count suspicious values
        suspicious_count = 0
        suspicious_count += (string_series == '').sum()  # Empty strings
        suspicious_count += (string_series == 'nan').sum()  # String 'nan'
        suspicious_count += (string_series == 'null').sum()  # String 'null'
        suspicious_count += (string_series == 'None').sum()  # String 'None'
        
        accuracy = 1 - (suspicious_count / len(series))
        
        if accuracy < 0.95 and suspicious_count > 0:
            dimension.issues.append({
                'type': 'suspicious_values',
                'column': column,
                'severity': 'low',
                'suspicious_count': suspicious_count,
                'description': f"Found {suspicious_count} suspicious values in '{column}'"
            })
        
        return accuracy
    
    def _calculate_validity(self, df: pd.DataFrame, target_column: Optional[str]):
        """Calculate validity dimension (business rule compliance)"""
        dimension = self.quality_dimensions['validity']
        
        validity_scores = []
        
        # Check for basic validity rules
        for column in df.columns:
            col_series = df[column].dropna()
            if len(col_series) == 0:
                continue
            
            validity_score = self._assess_column_validity(col_series, column, dimension)
            validity_scores.append(validity_score)
        
        # Target column specific validation
        if target_column and target_column in df.columns:
            target_validity = self._assess_target_validity(df[target_column], dimension)
            validity_scores.append(target_validity)
        
        # Calculate overall validity
        avg_validity = np.mean(validity_scores) if validity_scores else 1.0
        
        # Apply scoring thresholds
        thresholds = self.config['dimensions']['validity']['thresholds']
        scoring = self.config['scoring']
        
        if avg_validity >= thresholds['excellent']:
            dimension.score = scoring['excellent_score']
        elif avg_validity >= thresholds['good']:
            dimension.score = scoring['good_score']
        elif avg_validity >= thresholds['acceptable']:
            dimension.score = scoring['acceptable_score']
        elif avg_validity >= thresholds['poor']:
            dimension.score = scoring['poor_score']
        else:
            dimension.score = scoring['failing_score']
        
        dimension.metrics = {
            'average_validity': avg_validity,
            'columns_validated': len(validity_scores)
        }
        
        if avg_validity < 0.9:
            dimension.recommendations.append("Implement business rule validation and fix validity issues")
    
    def _assess_column_validity(self, series: pd.Series, column: str,
                              dimension: QualityDimension) -> float:
        """Assess validity for a single column"""
        # Basic validity checks
        validity_score = 1.0
        
        # Check for negative values in columns that shouldn't have them
        if series.dtype in ['int64', 'float64']:
            if 'age' in column.lower() or 'count' in column.lower() or 'amount' in column.lower():
                negative_count = (series < 0).sum()
                if negative_count > 0:
                    validity_score -= negative_count / len(series)
                    dimension.issues.append({
                        'type': 'invalid_negative_values',
                        'column': column,
                        'severity': 'medium',
                        'negative_count': negative_count,
                        'description': f"Found {negative_count} negative values in '{column}' (should be positive)"
                    })
        
        # Check for impossible values
        if 'percentage' in column.lower() or 'percent' in column.lower():
            if series.dtype in ['int64', 'float64']:
                invalid_percentage = ((series < 0) | (series > 100)).sum()
                if invalid_percentage > 0:
                    validity_score -= invalid_percentage / len(series)
                    dimension.issues.append({
                        'type': 'invalid_percentage_range',
                        'column': column,
                        'severity': 'high',
                        'invalid_count': invalid_percentage,
                        'description': f"Found {invalid_percentage} percentage values outside 0-100 range in '{column}'"
                    })
        
        return max(0.0, validity_score)
    
    def _assess_target_validity(self, target_series: pd.Series, dimension: QualityDimension) -> float:
        """Assess validity of target column"""
        validity_score = 1.0
        
        # Check target distribution
        if target_series.dtype in ['int64', 'float64']:
            # For numeric targets, check for reasonable distribution
            if target_series.nunique() == 2:  # Binary classification
                value_counts = target_series.value_counts()
                minority_ratio = value_counts.min() / value_counts.sum()
                
                if minority_ratio < 0.01:  # Less than 1% minority class
                    dimension.issues.append({
                        'type': 'extreme_class_imbalance',
                        'column': 'target',
                        'severity': 'high',
                        'minority_ratio': minority_ratio,
                        'description': f"Extreme class imbalance: {minority_ratio:.2%} minority class"
                    })
                    validity_score = 0.7
        
        return validity_score
    
    def _calculate_uniqueness(self, df: pd.DataFrame):
        """Calculate uniqueness dimension (duplicate and identifier analysis)"""
        dimension = self.quality_dimensions['uniqueness']
        
        # Overall duplicate analysis
        duplicate_rows = df.duplicated().sum()
        duplicate_ratio = duplicate_rows / len(df)
        
        # Column-wise uniqueness analysis
        uniqueness_scores = []
        
        for column in df.columns:
            col_series = df[column].dropna()
            if len(col_series) == 0:
                continue
            
            unique_ratio = col_series.nunique() / len(col_series)
            uniqueness_scores.append(unique_ratio)
            
            # Check for potential identifier columns
            if unique_ratio > 0.95:
                dimension.issues.append({
                    'type': 'potential_identifier',
                    'column': column,
                    'severity': 'info',
                    'unique_ratio': unique_ratio,
                    'description': f"Column '{column}' appears to be a unique identifier ({unique_ratio:.1%} unique)"
                })
        
        # Calculate uniqueness score
        uniqueness_score = 1 - duplicate_ratio
        
        # Apply scoring thresholds
        thresholds = self.config['dimensions']['uniqueness']['thresholds']
        scoring = self.config['scoring']
        
        if uniqueness_score >= thresholds['excellent']:
            dimension.score = scoring['excellent_score']
        elif uniqueness_score >= thresholds['good']:
            dimension.score = scoring['good_score']
        elif uniqueness_score >= thresholds['acceptable']:
            dimension.score = scoring['acceptable_score']
        elif uniqueness_score >= thresholds['poor']:
            dimension.score = scoring['poor_score']
        else:
            dimension.score = scoring['failing_score']
        
        dimension.metrics = {
            'duplicate_ratio': duplicate_ratio,
            'duplicate_rows': duplicate_rows,
            'average_uniqueness': np.mean(uniqueness_scores) if uniqueness_scores else 1.0
        }
        
        if duplicate_ratio > 0.05:
            dimension.recommendations.append(f"Remove {duplicate_rows} duplicate rows ({duplicate_ratio:.1%} of data)")
    
    def _calculate_timeliness(self, df: pd.DataFrame):
        """Calculate timeliness dimension (data freshness and temporal patterns)"""
        dimension = self.quality_dimensions['timeliness']
        
        # Look for date/time columns
        datetime_columns = []
        for column in df.columns:
            if any(keyword in column.lower() for keyword in ['date', 'time', 'created', 'updated', 'timestamp']):
                try:
                    converted = pd.to_datetime(df[column], errors='coerce')
                    if converted.notna().sum() / len(df) > 0.8:  # At least 80% valid dates
                        datetime_columns.append(column)
                except:
                    continue
        
        if not datetime_columns:
            # No datetime columns found - assign default score
            dimension.score = self.config['scoring']['good_score']
            dimension.metrics = {'datetime_columns_found': 0}
            dimension.recommendations.append("No datetime columns found - consider adding timestamps for data lineage")
            return
        
        # Analyze temporal patterns
        timeliness_scores = []
        
        for col in datetime_columns:
            datetime_series = pd.to_datetime(df[col], errors='coerce').dropna()
            if len(datetime_series) == 0:
                continue
            
            # Check data freshness
            latest_date = datetime_series.max()
            current_date = datetime.now()
            
            if pd.notna(latest_date):
                days_old = (current_date - latest_date).days
                
                # Score based on data age
                if days_old <= 1:
                    freshness_score = 1.0
                elif days_old <= 7:
                    freshness_score = 0.9
                elif days_old <= 30:
                    freshness_score = 0.8
                elif days_old <= 90:
                    freshness_score = 0.6
                else:
                    freshness_score = 0.4
                
                timeliness_scores.append(freshness_score)
                
                if days_old > 30:
                    dimension.issues.append({
                        'type': 'stale_data',
                        'column': col,
                        'severity': 'medium' if days_old < 90 else 'high',
                        'days_old': days_old,
                        'description': f"Data in '{col}' is {days_old} days old"
                    })
        
        # Calculate overall timeliness score
        avg_timeliness = np.mean(timeliness_scores) if timeliness_scores else 0.8
        
        # Apply scoring thresholds
        thresholds = self.config['dimensions']['timeliness']['thresholds']
        scoring = self.config['scoring']
        
        if avg_timeliness >= thresholds['excellent']:
            dimension.score = scoring['excellent_score']
        elif avg_timeliness >= thresholds['good']:
            dimension.score = scoring['good_score']
        elif avg_timeliness >= thresholds['acceptable']:
            dimension.score = scoring['acceptable_score']
        elif avg_timeliness >= thresholds['poor']:
            dimension.score = scoring['poor_score']
        else:
            dimension.score = scoring['failing_score']
        
        dimension.metrics = {
            'datetime_columns_found': len(datetime_columns),
            'average_timeliness': avg_timeliness,
            'analyzed_columns': datetime_columns
        }
        
        if avg_timeliness < 0.8:
            dimension.recommendations.append("Update data sources to improve timeliness")
    
    def _calculate_overall_score(self) -> Tuple[float, int, int]:
        """Calculate weighted overall quality score"""
        weighted_score = 0.0
        total_weight = 0.0
        total_issues = 0
        total_checks = 0
        
        for dimension in self.quality_dimensions.values():
            weighted_score += dimension.score * dimension.weight
            total_weight += dimension.weight
            total_issues += len(dimension.issues)
            total_checks += 1
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return overall_score, total_issues, total_checks
    
    def _calculate_confidence(self, df: pd.DataFrame) -> float:
        """Calculate confidence in quality assessment based on data characteristics"""
        # Base confidence on data size
        if len(df) < 100:
            size_confidence = 0.6
        elif len(df) < 1000:
            size_confidence = 0.8
        else:
            size_confidence = 1.0
        
        # Factor in completeness
        completeness_ratio = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
        completeness_confidence = completeness_ratio
        
        # Overall confidence
        confidence = (size_confidence + completeness_confidence) / 2
        
        return confidence
    
    def generate_quality_report(self, quality_score: QualityScore) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        report = {
            'summary': {
                'overall_score': quality_score.overall_score,
                'grade': self._get_quality_grade(quality_score.overall_score),
                'confidence': quality_score.confidence,
                'total_issues': quality_score.issues_count,
                'assessment_timestamp': datetime.now().isoformat()
            },
            'dimensions': {},
            'issues': {
                'high': [],
                'medium': [],
                'low': [],
                'info': []
            },
            'recommendations': [],
            'metadata': quality_score.metadata
        }
        
        # Compile dimension details
        for name, dimension in self.quality_dimensions.items():
            report['dimensions'][name] = {
                'score': dimension.score,
                'weight': dimension.weight,
                'grade': self._get_quality_grade(dimension.score),
                'metrics': dimension.metrics,
                'issues_count': len(dimension.issues),
                'recommendations': dimension.recommendations
            }
            
            # Categorize issues by severity
            for issue in dimension.issues:
                severity = issue.get('severity', 'info')
                report['issues'][severity].append(issue)
            
            # Add recommendations
            report['recommendations'].extend(dimension.recommendations)
        
        return report
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'


def main():
    """Test the quality metrics calculator"""
    # Create sample data with quality issues
    np.random.seed(42)
    
    data = {
        'id': list(range(1000)) + [999],  # Duplicate ID
        'name': ['User_' + str(i) for i in range(1000)] + ['user_duplicate'],  # Case inconsistency
        'email': ['user{}@example.com'.format(i) for i in range(990)] + ['invalid_email'] * 11,
        'age': list(np.random.randint(18, 80, 990)) + [-5, 150, 200, 300, 400, 500, 600, 700, 800, 900],  # Invalid ages
        'salary': list(np.random.normal(50000, 15000, 990)) + [None] * 10,  # Missing values
        'date_joined': pd.date_range('2020-01-01', periods=1000, freq='D'),
        'percentage': list(np.random.uniform(0, 100, 990)) + [150, 200, -10, -20, 300, 400, 500, 600, 700, 800]  # Invalid percentages
    }
    
    df = pd.DataFrame(data)
    
    # Initialize calculator
    calculator = QualityMetricsCalculator()
    
    # Calculate quality score
    quality_score = calculator.calculate_quality_score(df, target_column=None)
    
    # Generate report
    report = calculator.generate_quality_report(quality_score)
    
    # Print results
    print("\n📊 Data Quality Assessment Results")
    print("=" * 50)
    print(f"Overall Score: {quality_score.overall_score:.1f}/100 (Grade: {report['summary']['grade']})")
    print(f"Confidence: {quality_score.confidence:.2f}")
    print(f"Total Issues: {quality_score.issues_count}")
    
    print("\nDimension Scores:")
    for name, score in quality_score.dimension_scores.items():
        grade = report['dimensions'][name]['grade']
        print(f"   {name.capitalize()}: {score:.1f}/100 (Grade: {grade})")
    
    print(f"\nIssues by Severity:")
    for severity, issues in report['issues'].items():
        if issues:
            print(f"   {severity.upper()}: {len(issues)}")


if __name__ == "__main__":
    main()
