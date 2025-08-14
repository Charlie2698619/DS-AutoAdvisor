#!/usr/bin/env python3
"""
🧠 Data Type Inference Engine for DS-AutoAdvisor
==============================================

Multi-stage heuristic classification with confidence scoring for intelligent data type detection.

Features:
- Pattern matching for datetime formats
- Statistical tests for numerical vs categorical
- Cardinality-based classification rules
- Confidence scoring for inference quality
- YAML-controlled thresholds and parameters
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass
class InferenceResult:
    """Data type inference result with confidence"""
    inferred_type: str
    confidence: float
    original_type: str
    sample_values: List[Any]
    patterns_detected: List[str]
    statistics: Dict[str, Any]
    recommendations: List[str]

class DataTypeInferenceEngine:
    """
    Advanced data type inference with multi-stage heuristic classification
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize inference engine with YAML configuration"""
        self.config = self._load_config(config_path)
        self._compile_patterns()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            # Try to find unified_config_v3.yaml
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "unified_config_v3.yaml"
        
        default_config = {
            'data_type_inference': {
                'confidence_thresholds': {
                    'high': 0.8,
                    'medium': 0.6,
                    'low': 0.4
                },
                'cardinality_thresholds': {
                    'unique_ratio_for_id': 0.95,
                    'unique_ratio_for_categorical': 0.5,
                    'max_categories': 50
                },
                'datetime_patterns': [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                    r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                    r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
                ],
                'numeric_patterns': [
                    r'^[-+]?\d*\.?\d+([eE][-+]?\d+)?$',  # Scientific notation
                    r'^[-+]?\d+$',                       # Integer
                    r'^[-+]?\d*\.\d+$',                  # Decimal
                ],
                'statistical_tests': {
                    'enable_normality_test': True,
                    'enable_chi_square_test': True,
                    'sample_size_for_tests': 1000
                }
            }
        }
        
        if Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    
                    # Priority 1: Check global section first
                    if 'data_type_inference' in loaded_config:
                        print("📋 Data Type Inference: Using global configuration")
                        return loaded_config['data_type_inference']
                    
                    # Priority 2: Check custom_mode
                    if 'custom_mode' in loaded_config and 'data_discovery' in loaded_config['custom_mode']:
                        discovery_config = loaded_config['custom_mode']['data_discovery']
                        if 'data_type_inference' in discovery_config:
                            print("📋 Data Type Inference: Using custom_mode configuration")
                            return discovery_config['data_type_inference']
                    
                    # Priority 3: Check fast_mode
                    if 'fast_mode' in loaded_config and 'data_discovery' in loaded_config['fast_mode']:
                        discovery_config = loaded_config['fast_mode']['data_discovery']
                        if 'data_type_inference' in discovery_config:
                            print("📋 Data Type Inference: Using fast_mode configuration")
                            return discovery_config['data_type_inference']
                
                print("📋 Data Type Inference: No config found, using defaults")
                return default_config['data_type_inference']
            except Exception as e:
                print(f"⚠️ Error loading config {config_path}: {e}")
                return default_config['data_type_inference']
        else:
            return default_config['data_type_inference']
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching"""
        self.datetime_patterns = [
            re.compile(pattern) for pattern in self.config['datetime_patterns']
        ]
        self.numeric_patterns = [
            re.compile(pattern) for pattern in self.config['numeric_patterns']
        ]
    
    def infer_column_type(self, series: pd.Series, column_name: str) -> InferenceResult:
        """
        Infer data type for a single column using multi-stage analysis
        
        Args:
            series: Pandas series to analyze
            column_name: Name of the column
            
        Returns:
            InferenceResult with inferred type and confidence
        """
        # Stage 1: Basic information gathering
        original_type = str(series.dtype)
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return InferenceResult(
                inferred_type="empty",
                confidence=1.0,
                original_type=original_type,
                sample_values=[],
                patterns_detected=["all_null"],
                statistics={"total_count": len(series), "non_null_count": 0},
                recommendations=["Consider dropping this column - all values are null"]
            )
        
        # Sample values for analysis
        sample_size = min(len(non_null_series), self.config['statistical_tests']['sample_size_for_tests'])
        sample_values = non_null_series.sample(n=sample_size).tolist()
        
        # Stage 2: Pattern-based analysis
        patterns_detected = []
        confidence_scores = []
        
        # Check for datetime patterns
        datetime_confidence = self._check_datetime_patterns(sample_values, patterns_detected)
        if datetime_confidence > 0:
            confidence_scores.append(('datetime', datetime_confidence))
        
        # Check for numeric patterns  
        numeric_confidence = self._check_numeric_patterns(sample_values, patterns_detected)
        if numeric_confidence > 0:
            confidence_scores.append(('numeric', numeric_confidence))
        
        # Stage 3: Cardinality-based analysis
        cardinality_result = self._analyze_cardinality(non_null_series, patterns_detected)
        if cardinality_result:
            confidence_scores.append(cardinality_result)
        
        # Stage 4: Statistical analysis
        statistical_result = self._perform_statistical_tests(non_null_series, patterns_detected)
        if statistical_result:
            confidence_scores.append(statistical_result)
        
        # Stage 5: Final type inference
        inferred_type, final_confidence = self._determine_final_type(
            confidence_scores, original_type, non_null_series
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            inferred_type, final_confidence, original_type, non_null_series
        )
        
        # Compile statistics
        statistics = self._compile_statistics(non_null_series, inferred_type)
        
        return InferenceResult(
            inferred_type=inferred_type,
            confidence=final_confidence,
            original_type=original_type,
            sample_values=sample_values[:10],  # Limit sample values
            patterns_detected=patterns_detected,
            statistics=statistics,
            recommendations=recommendations
        )
    
    def _check_datetime_patterns(self, sample_values: List[Any], patterns_detected: List[str]) -> float:
        """Check for datetime patterns in sample values"""
        if not sample_values:
            return 0.0
        
        # Convert to strings for pattern matching
        string_values = [str(val) for val in sample_values]
        
        pattern_matches = 0
        total_patterns = len(self.datetime_patterns)
        
        for pattern in self.datetime_patterns:
            matches = sum(1 for val in string_values if pattern.search(val))
            if matches > 0:
                match_ratio = matches / len(string_values)
                pattern_matches += match_ratio
                patterns_detected.append(f"datetime_pattern_{pattern.pattern}")
        
        # Try pandas datetime conversion
        try:
            converted = pd.to_datetime(string_values, errors='coerce')
            valid_dates = converted.notna().sum()
            conversion_ratio = valid_dates / len(string_values)
            
            if conversion_ratio > 0.7:
                patterns_detected.append("pandas_datetime_convertible")
                return min(0.9, conversion_ratio + 0.1)
        except:
            pass
        
        # Return confidence based on pattern matches
        if pattern_matches > 0:
            return min(0.8, pattern_matches / total_patterns)
        
        return 0.0
    
    def _check_numeric_patterns(self, sample_values: List[Any], patterns_detected: List[str]) -> float:
        """Check for numeric patterns in sample values"""
        if not sample_values:
            return 0.0
        
        string_values = [str(val).strip() for val in sample_values]
        numeric_matches = 0
        
        for pattern in self.numeric_patterns:
            matches = sum(1 for val in string_values if pattern.match(val))
            if matches > 0:
                match_ratio = matches / len(string_values)
                numeric_matches += match_ratio
                patterns_detected.append(f"numeric_pattern_{pattern.pattern[:20]}")
        
        # Try pandas numeric conversion
        try:
            converted = pd.to_numeric(string_values, errors='coerce')
            valid_numbers = converted.notna().sum()
            conversion_ratio = valid_numbers / len(string_values)
            
            if conversion_ratio > 0.8:
                patterns_detected.append("pandas_numeric_convertible")
                return min(0.9, conversion_ratio)
        except:
            pass
        
        return min(0.8, numeric_matches)
    
    def _analyze_cardinality(self, series: pd.Series, patterns_detected: List[str]) -> Optional[Tuple[str, float]]:
        """Analyze cardinality to determine categorical vs continuous nature"""
        unique_count = series.nunique()
        total_count = len(series)
        unique_ratio = unique_count / total_count
        
        thresholds = self.config['cardinality_thresholds']
        
        # Check for unique identifier
        if unique_ratio >= thresholds['unique_ratio_for_id']:
            patterns_detected.append(f"unique_identifier_{unique_ratio:.2f}")
            return ('identifier', 0.9)
        
        # Check for categorical
        if (unique_ratio <= thresholds['unique_ratio_for_categorical'] and 
            unique_count <= thresholds['max_categories']):
            patterns_detected.append(f"categorical_{unique_count}_categories")
            return ('categorical', 0.8)
        
        # High cardinality continuous
        if unique_ratio > 0.8:
            patterns_detected.append(f"continuous_high_cardinality_{unique_ratio:.2f}")
            return ('continuous', 0.7)
        
        return None
    
    def _perform_statistical_tests(self, series: pd.Series, patterns_detected: List[str]) -> Optional[Tuple[str, float]]:
        """Perform statistical tests for type classification"""
        if not self.config['statistical_tests']['enable_normality_test']:
            return None
        
        # Only perform on numeric-like data
        try:
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) < 10:
                return None
            
            # Test for normality (indicates continuous numeric)
            from scipy import stats
            _, p_value = stats.normaltest(numeric_series) # very sensitive to outliers
            
            if p_value > 0.05:  # Normal distribution
                patterns_detected.append(f"normal_distribution_p{p_value:.3f}")
                return ('continuous_normal', 0.7)
            else:
                patterns_detected.append(f"non_normal_distribution_p{p_value:.3f}")
                return ('continuous_skewed', 0.6)
                
        except ImportError:
            # scipy not available
            patterns_detected.append("statistical_tests_unavailable")
            return None
        except Exception as e:
            patterns_detected.append(f"statistical_test_error_{str(e)[:20]}")
            return None
    
    def _determine_final_type(self, confidence_scores: List[Tuple[str, float]], 
                            original_type: str, series: pd.Series) -> Tuple[str, float]:
        """Determine final type based on all confidence scores"""
        if not confidence_scores:
            # Fallback based on original type
            if 'int' in original_type or 'float' in original_type:
                return 'numeric', 0.5
            elif 'object' in original_type:
                return 'text', 0.5
            else:
                return 'unknown', 0.3
        
        # Find highest confidence score
        best_type, best_confidence = max(confidence_scores, key=lambda x: x[1])
        
        # Apply business rules for final decision
        if best_type == 'identifier' and best_confidence > 0.8:
            return 'identifier', best_confidence
        elif best_type == 'datetime' and best_confidence > 0.7:
            return 'datetime', best_confidence
        elif best_type in ['numeric', 'continuous_normal', 'continuous_skewed'] and best_confidence > 0.6:
            return 'numeric', best_confidence
        elif best_type == 'categorical' and best_confidence > 0.6:
            return 'categorical', best_confidence
        else:
            # Default to text if no clear classification
            return 'text', max(0.3, best_confidence * 0.5)
    
    def _generate_recommendations(self, inferred_type: str, confidence: float, 
                                original_type: str, series: pd.Series) -> List[str]:
        """Generate actionable recommendations based on inference"""
        recommendations = []
        
        # Confidence-based recommendations
        if confidence < self.config['confidence_thresholds']['low']:
            recommendations.append("⚠️ Low confidence in type inference - manual review recommended")
        
        # Type-specific recommendations
        if inferred_type == 'identifier':
            recommendations.append("🔑 Consider dropping for modeling (likely unique identifier)")
        elif inferred_type == 'datetime':
            recommendations.append("📅 Convert to datetime and extract temporal features")
        elif inferred_type == 'numeric' and original_type == 'object':
            recommendations.append("🔢 Convert to numeric type for better performance")
        elif inferred_type == 'categorical':
            unique_count = series.nunique()
            if unique_count > 20:
                recommendations.append(f"📊 High cardinality categorical ({unique_count} categories) - consider encoding strategies")
            else:
                recommendations.append("📊 Apply appropriate categorical encoding (one-hot, target, etc.)")
        elif inferred_type == 'text':
            recommendations.append("📝 Consider text preprocessing or feature extraction")
        
        # Missing values recommendation
        missing_ratio = series.isnull().sum() / len(series)
        if missing_ratio > 0.1:
            recommendations.append(f"🔧 Handle missing values ({missing_ratio:.1%} missing)")
        
        return recommendations
    
    def _compile_statistics(self, series: pd.Series, inferred_type: str) -> Dict[str, Any]:
        """Compile comprehensive statistics for the series"""
        stats = {
            'total_count': len(series),
            'non_null_count': series.count(),
            'null_count': series.isnull().sum(),
            'missing_ratio': series.isnull().sum() / len(series),
            'unique_count': series.nunique(),
            'unique_ratio': series.nunique() / len(series)
        }
        
        # Type-specific statistics
        if inferred_type == 'numeric':
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                stats.update({
                    'mean': float(numeric_series.mean()),
                    'std': float(numeric_series.std()),
                    'min': float(numeric_series.min()),
                    'max': float(numeric_series.max()),
                    'median': float(numeric_series.median()),
                    'skewness': float(numeric_series.skew()),
                    'kurtosis': float(numeric_series.kurtosis())
                })
            except:
                pass
        elif inferred_type == 'categorical':
            value_counts = series.value_counts().head(10)
            stats['top_categories'] = value_counts.to_dict()
            stats['category_distribution'] = {
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            }
        
        return stats
    
    def infer_dataframe_types(self, df: pd.DataFrame) -> Dict[str, InferenceResult]:
        """Infer types for all columns in a DataFrame"""
        results = {}
        
        print(f"🧠 Running data type inference on {len(df.columns)} columns...")
        
        for i, column in enumerate(df.columns, 1):
            print(f"   [{i:2d}/{len(df.columns):2d}] Analyzing {column}...")
            results[column] = self.infer_column_type(df[column], column)
        
        return results
    
    def generate_inference_report(self, inference_results: Dict[str, InferenceResult]) -> Dict[str, Any]:
        """Generate comprehensive inference report"""
        report = {
            'summary': {
                'total_columns': len(inference_results),
                'inference_timestamp': datetime.now().isoformat(),
                'confidence_distribution': {},
                'type_distribution': {}
            },
            'columns': {},
            'recommendations': {
                'high_priority': [],
                'medium_priority': [],
                'low_priority': []
            }
        }
        
        # Analyze confidence distribution
        confidences = [result.confidence for result in inference_results.values()]
        thresholds = self.config['confidence_thresholds']
        
        report['summary']['confidence_distribution'] = {
            'high': sum(1 for c in confidences if c >= thresholds['high']),
            'medium': sum(1 for c in confidences if thresholds['medium'] <= c < thresholds['high']),
            'low': sum(1 for c in confidences if c < thresholds['medium'])
        }
        
        # Analyze type distribution
        types = [result.inferred_type for result in inference_results.values()]
        report['summary']['type_distribution'] = {
            type_name: types.count(type_name) for type_name in set(types)
        }
        
        # Compile column details and recommendations
        for column, result in inference_results.items():
            report['columns'][column] = {
                'inferred_type': result.inferred_type,
                'confidence': result.confidence,
                'original_type': result.original_type,
                'patterns_detected': result.patterns_detected,
                'statistics': result.statistics,
                'recommendations': result.recommendations
            }
            
            # Categorize recommendations by priority
            if result.confidence < thresholds['low']:
                report['recommendations']['high_priority'].extend([
                    f"{column}: {rec}" for rec in result.recommendations
                ])
            elif result.confidence < thresholds['medium']:
                report['recommendations']['medium_priority'].extend([
                    f"{column}: {rec}" for rec in result.recommendations
                ])
            else:
                report['recommendations']['low_priority'].extend([
                    f"{column}: {rec}" for rec in result.recommendations
                ])
        
        return report


def main():
    """Test the data type inference engine"""
    # Create sample data for testing
    data = {
        'id': range(1000),
        'date_str': ['2023-01-01', '2023-01-02', '2023-01-03'] * 333 + ['2023-01-01'],
        'numeric_str': ['123', '456', '789'] * 333 + ['123'],
        'category': ['A', 'B', 'C'] * 333 + ['A'],
        'text': ['Hello', 'World', 'Test'] * 333 + ['Hello'],
        'mixed': [1, 'two', 3.0, None] * 250
    }
    
    df = pd.DataFrame(data)
    
    # Initialize inference engine
    engine = DataTypeInferenceEngine()
    
    # Run inference
    results = engine.infer_dataframe_types(df)
    
    # Generate report
    report = engine.generate_inference_report(results)
    
    # Print results
    print("\n🧠 Data Type Inference Results")
    print("=" * 50)
    
    for column, result in results.items():
        print(f"\n📊 {column}")
        print(f"   Original: {result.original_type} → Inferred: {result.inferred_type}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Patterns: {', '.join(result.patterns_detected)}")
        if result.recommendations:
            print(f"   Recommendations: {result.recommendations[0]}")


if __name__ == "__main__":
    main()
