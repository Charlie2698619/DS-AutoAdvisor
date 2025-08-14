#!/usr/bin/env python3
"""
🔍 Pattern Detection Module for DS-AutoAdvisor
=============================================

Advanced pattern detection for data quality assessment including:
- Business patterns (emails, phone numbers, IDs)
- Formatting patterns (dates, currencies, codes)
- Anomaly patterns (outliers, inconsistencies)
- Statistical patterns (distributions, correlations)
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import yaml
from pathlib import Path

@dataclass
class PatternMatch:
    """Represents a detected pattern with metadata"""
    pattern_name: str
    pattern_type: str
    confidence: float
    matches_count: int
    total_count: int
    match_ratio: float
    examples: List[str]
    description: str

class PatternDetector:
    """
    Advanced pattern detection for data quality analysis
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize pattern detector with YAML configuration"""
        self.config = self._load_config(config_path)
        self._compile_patterns()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load pattern detection configuration from YAML"""
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "unified_config_v3.yaml"
        
        default_config = {
            'pattern_detection': {
                'business_patterns': {
                    'email': {
                        'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                        'confidence_threshold': 0.8,
                        'description': 'Email address format'
                    },
                    'phone': {
                        'pattern': r'^[\+]?[1-9]?[0-9]{7,15}$',
                        'confidence_threshold': 0.7,
                        'description': 'Phone number format'
                    },
                    'ssn': {
                        'pattern': r'^\d{3}-\d{2}-\d{4}$',
                        'confidence_threshold': 0.9,
                        'description': 'Social Security Number'
                    },
                    'credit_card': {
                        'pattern': r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$',
                        'confidence_threshold': 0.8,
                        'description': 'Credit card number'
                    }
                },
                'formatting_patterns': {
                    'currency': {
                        'pattern': r'^\$?\d{1,3}(,\d{3})*(\.\d{2})?$',
                        'confidence_threshold': 0.7,
                        'description': 'Currency format'
                    },
                    'percentage': {
                        'pattern': r'^\d{1,3}(\.\d+)?%$',
                        'confidence_threshold': 0.8,
                        'description': 'Percentage format'
                    },
                    'zip_code': {
                        'pattern': r'^\d{5}(-\d{4})?$',
                        'confidence_threshold': 0.8,
                        'description': 'ZIP code format'
                    },
                    'iso_date': {
                        'pattern': r'^\d{4}-\d{2}-\d{2}$',
                        'confidence_threshold': 0.9,
                        'description': 'ISO date format (YYYY-MM-DD)'
                    }
                },
                'id_patterns': {
                    'uuid': {
                        'pattern': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                        'confidence_threshold': 0.95,
                        'description': 'UUID format'
                    },
                    'sequential_id': {
                        'pattern': 'SEQUENTIAL_CHECK',  # Special handling
                        'confidence_threshold': 0.8,
                        'description': 'Sequential identifier'
                    }
                },
                'anomaly_detection': {
                    'enable_outlier_detection': True,
                    'outlier_methods': ['iqr', 'zscore'],
                    'outlier_threshold': 3.0,
                    'enable_inconsistency_detection': True
                },
                'statistical_patterns': {
                    'enable_distribution_analysis': True,
                    'enable_correlation_analysis': True,
                    'correlation_threshold': 0.8
                }
            }
        }
        
        if Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    
                    # Priority 1: Check global section first
                    if 'pattern_detection' in loaded_config:
                        print("📋 Pattern Detection: Using global configuration")
                        return loaded_config['pattern_detection']
                    
                    # Priority 2: Check custom_mode
                    if 'custom_mode' in loaded_config and 'data_discovery' in loaded_config['custom_mode']:
                        discovery_config = loaded_config['custom_mode']['data_discovery']
                        if 'pattern_detection' in discovery_config:
                            print("📋 Pattern Detection: Using custom_mode configuration")
                            return discovery_config['pattern_detection']
                    
                    # Priority 3: Check fast_mode
                    if 'fast_mode' in loaded_config and 'data_discovery' in loaded_config['fast_mode']:
                        discovery_config = loaded_config['fast_mode']['data_discovery']
                        if 'pattern_detection' in discovery_config:
                            print("📋 Pattern Detection: Using fast_mode configuration")
                            return discovery_config['pattern_detection']
                
                print("📋 Pattern Detection: No config found, using defaults")
                return default_config['pattern_detection']
            except Exception as e:
                print(f"⚠️ Error loading pattern config: {e}")
                return default_config['pattern_detection']
        else:
            return default_config['pattern_detection']
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching"""
        self.compiled_patterns = {}
        
        # Compile business patterns
        for pattern_name, pattern_config in self.config['business_patterns'].items():
            if pattern_config['pattern'] != 'SEQUENTIAL_CHECK':
                self.compiled_patterns[f"business_{pattern_name}"] = {
                    'regex': re.compile(pattern_config['pattern'], re.IGNORECASE),
                    'config': pattern_config
                }
        
        # Compile formatting patterns
        for pattern_name, pattern_config in self.config['formatting_patterns'].items():
            self.compiled_patterns[f"formatting_{pattern_name}"] = {
                'regex': re.compile(pattern_config['pattern']),
                'config': pattern_config
            }
        
        # Compile ID patterns (except sequential)
        for pattern_name, pattern_config in self.config['id_patterns'].items():
            if pattern_config['pattern'] != 'SEQUENTIAL_CHECK':
                self.compiled_patterns[f"id_{pattern_name}"] = {
                    'regex': re.compile(pattern_config['pattern'], re.IGNORECASE),
                    'config': pattern_config
                }
    
    def detect_patterns_in_series(self, series: pd.Series, column_name: str) -> List[PatternMatch]:
        """Detect all patterns in a pandas series"""
        patterns_found = []
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return patterns_found
        
        # Convert to strings for pattern matching
        string_values = non_null_series.astype(str).tolist()
        
        # Test regex patterns
        for pattern_key, pattern_info in self.compiled_patterns.items():
            matches = self._test_regex_pattern(
                string_values, pattern_info['regex'], pattern_info['config']
            )
            if matches:
                patterns_found.append(matches)
        
        # Test special patterns
        special_patterns = self._detect_special_patterns(non_null_series, column_name)
        patterns_found.extend(special_patterns)
        
        # Test statistical patterns
        if self.config['statistical_patterns']['enable_distribution_analysis']:
            statistical_patterns = self._detect_statistical_patterns(non_null_series)
            patterns_found.extend(statistical_patterns)
        
        # Test anomaly patterns
        if self.config['anomaly_detection']['enable_outlier_detection']:
            anomaly_patterns = self._detect_anomaly_patterns(non_null_series)
            patterns_found.extend(anomaly_patterns)
        
        return patterns_found
    
    def _test_regex_pattern(self, string_values: List[str], regex: re.Pattern, 
                          config: Dict[str, Any]) -> Optional[PatternMatch]:
        """Test a regex pattern against string values"""
        matches = 0
        examples = []
        
        for value in string_values:
            if regex.match(value.strip()):
                matches += 1
                if len(examples) < 5:  # Collect up to 5 examples
                    examples.append(value)
        
        total_count = len(string_values)
        match_ratio = matches / total_count
        
        if match_ratio >= config['confidence_threshold']:
            return PatternMatch(
                pattern_name=regex.pattern,
                pattern_type="regex",
                confidence=min(0.95, match_ratio),
                matches_count=matches,
                total_count=total_count,
                match_ratio=match_ratio,
                examples=examples,
                description=config['description']
            )
        
        return None
    
    def _detect_special_patterns(self, series: pd.Series, column_name: str) -> List[PatternMatch]:
        """Detect special patterns that require custom logic"""
        patterns = []
        
        # Sequential ID detection
        if self._is_sequential_id(series):
            patterns.append(PatternMatch(
                pattern_name="sequential_id",
                pattern_type="special",
                confidence=0.9,
                matches_count=len(series),
                total_count=len(series),
                match_ratio=1.0,
                examples=series.head(3).astype(str).tolist(),
                description="Sequential identifier pattern detected"
            ))
        
        # Constant value detection
        if series.nunique() == 1:
            patterns.append(PatternMatch(
                pattern_name="constant_value",
                pattern_type="special",
                confidence=1.0,
                matches_count=len(series),
                total_count=len(series),
                match_ratio=1.0,
                examples=[str(series.iloc[0])],
                description="All values are identical (constant)"
            ))
        
        # High cardinality unique identifier
        unique_ratio = series.nunique() / len(series)
        if unique_ratio > 0.95:
            patterns.append(PatternMatch(
                pattern_name="unique_identifier",
                pattern_type="special",
                confidence=unique_ratio,
                matches_count=series.nunique(),
                total_count=len(series),
                match_ratio=unique_ratio,
                examples=series.head(3).astype(str).tolist(),
                description=f"High cardinality unique values ({unique_ratio:.1%} unique)"
            ))
        
        return patterns
    
    def _is_sequential_id(self, series: pd.Series) -> bool:
        """Check if series contains sequential identifiers"""
        try:
            # Convert to numeric and check for sequence
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) < len(series) * 0.8:  # At least 80% should be numeric
                return False
            
            # Check if values are sequential
            sorted_values = numeric_series.sort_values()
            diffs = sorted_values.diff().dropna()
            
            # Allow for some gaps but mostly sequential
            if len(diffs) > 0:
                most_common_diff = diffs.mode().iloc[0] if len(diffs.mode()) > 0 else 1
                sequential_ratio = (diffs == most_common_diff).sum() / len(diffs)
                return sequential_ratio > 0.7
                
        except Exception:
            pass
        
        return False
    
    def _detect_statistical_patterns(self, series: pd.Series) -> List[PatternMatch]:
        """Detect statistical patterns in the data"""
        patterns = []
        
        try:
            # Convert to numeric for analysis
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) < 10:
                return patterns
            
            # Normal distribution detection
            try:
                from scipy import stats
                _, p_value = stats.normaltest(numeric_series)
                
                if p_value > 0.05:
                    patterns.append(PatternMatch(
                        pattern_name="normal_distribution",
                        pattern_type="statistical",
                        confidence=min(0.9, 1 - p_value),
                        matches_count=len(numeric_series),
                        total_count=len(series),
                        match_ratio=len(numeric_series) / len(series),
                        examples=[f"mean={numeric_series.mean():.2f}", f"std={numeric_series.std():.2f}"],
                        description="Data follows normal distribution"
                    ))
            except ImportError:
                pass
            
            # Uniform distribution detection (rough check)
            hist, _ = np.histogram(numeric_series, bins=10)
            if np.std(hist) / np.mean(hist) < 0.3:  # Low coefficient of variation
                patterns.append(PatternMatch(
                    pattern_name="uniform_distribution",
                    pattern_type="statistical",
                    confidence=0.7,
                    matches_count=len(numeric_series),
                    total_count=len(series),
                    match_ratio=len(numeric_series) / len(series),
                    examples=[f"range=({numeric_series.min():.2f}, {numeric_series.max():.2f})"],
                    description="Data appears uniformly distributed"
                ))
            
        except Exception as e:
            pass
        
        return patterns
    
    def _detect_anomaly_patterns(self, series: pd.Series) -> List[PatternMatch]:
        """Detect anomaly patterns (outliers, inconsistencies)"""
        patterns = []
        
        try:
            # Numeric outlier detection
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) >= 10:
                outliers = self._detect_outliers(numeric_series)
                if len(outliers) > 0:
                    outlier_ratio = len(outliers) / len(numeric_series)
                    patterns.append(PatternMatch(
                        pattern_name="numeric_outliers",
                        pattern_type="anomaly",
                        confidence=min(0.9, outlier_ratio * 2),
                        matches_count=len(outliers),
                        total_count=len(numeric_series),
                        match_ratio=outlier_ratio,
                        examples=[f"{val:.2f}" for val in outliers[:3]],
                        description=f"Detected {len(outliers)} numeric outliers"
                    ))
            
            # Inconsistent formatting detection
            if series.dtype == 'object':
                inconsistencies = self._detect_format_inconsistencies(series)
                if inconsistencies:
                    patterns.append(inconsistencies)
                    
        except Exception:
            pass
        
        return patterns
    
    def _detect_outliers(self, numeric_series: pd.Series) -> List[float]:
        """Detect numeric outliers using IQR method"""
        Q1 = numeric_series.quantile(0.25)
        Q3 = numeric_series.quantile(0.75)
        IQR = Q3 - Q1
        
        threshold = self.config['anomaly_detection']['outlier_threshold']
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = numeric_series[(numeric_series < lower_bound) | (numeric_series > upper_bound)]
        return outliers.tolist()
    
    def _detect_format_inconsistencies(self, series: pd.Series) -> Optional[PatternMatch]:
        """Detect formatting inconsistencies in text data"""
        string_series = series.astype(str)
        
        # Check for mixed case patterns
        has_upper = string_series.str.contains(r'[A-Z]').sum()
        has_lower = string_series.str.contains(r'[a-z]').sum()
        total_count = len(string_series)
        
        if has_upper > 0 and has_lower > 0:
            inconsistency_ratio = min(has_upper, has_lower) / total_count
            if inconsistency_ratio > 0.1:  # At least 10% inconsistency
                return PatternMatch(
                    pattern_name="case_inconsistency",
                    pattern_type="anomaly",
                    confidence=inconsistency_ratio,
                    matches_count=int(min(has_upper, has_lower)),
                    total_count=total_count,
                    match_ratio=inconsistency_ratio,
                    examples=string_series.head(3).tolist(),
                    description="Inconsistent case formatting detected"
                )
        
        return None
    
    def detect_dataframe_patterns(self, df: pd.DataFrame) -> Dict[str, List[PatternMatch]]:
        """Detect patterns across all columns in a DataFrame"""
        results = {}
        
        print(f"🔍 Running pattern detection on {len(df.columns)} columns...")
        
        for i, column in enumerate(df.columns, 1):
            print(f"   [{i:2d}/{len(df.columns):2d}] Detecting patterns in {column}...")
            results[column] = self.detect_patterns_in_series(df[column], column)
        
        # Cross-column pattern detection
        cross_patterns = self._detect_cross_column_patterns(df)
        if cross_patterns:
            results['__cross_column__'] = cross_patterns
        
        return results
    
    def _detect_cross_column_patterns(self, df: pd.DataFrame) -> List[PatternMatch]:
        """Detect patterns across multiple columns"""
        patterns = []
        
        if not self.config['statistical_patterns']['enable_correlation_analysis']:
            return patterns
        
        try:
            # Correlation analysis for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                corr_matrix = numeric_df.corr()
                
                # Find high correlations
                threshold = self.config['statistical_patterns']['correlation_threshold']
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = abs(corr_matrix.iloc[i, j])
                        if corr_value >= threshold:
                            col1 = corr_matrix.columns[i]
                            col2 = corr_matrix.columns[j]
                            high_corr_pairs.append((col1, col2, corr_value))
                
                if high_corr_pairs:
                    patterns.append(PatternMatch(
                        pattern_name="high_correlation",
                        pattern_type="cross_column",
                        confidence=0.8,
                        matches_count=len(high_corr_pairs),
                        total_count=len(numeric_df.columns),
                        match_ratio=len(high_corr_pairs) / len(numeric_df.columns),
                        examples=[f"{pair[0]} ↔ {pair[1]} (r={pair[2]:.2f})" for pair in high_corr_pairs[:3]],
                        description=f"Found {len(high_corr_pairs)} highly correlated column pairs"
                    ))
        
        except Exception:
            pass
        
        return patterns
    
    def generate_pattern_report(self, pattern_results: Dict[str, List[PatternMatch]]) -> Dict[str, Any]:
        """Generate comprehensive pattern detection report"""
        report = {
            'summary': {
                'total_columns_analyzed': len([k for k in pattern_results.keys() if k != '__cross_column__']),
                'patterns_detected': sum(len(patterns) for patterns in pattern_results.values()),
                'analysis_timestamp': datetime.now().isoformat(),
                'pattern_type_distribution': {},
                'high_confidence_patterns': 0
            },
            'columns': {},
            'cross_column_patterns': [],
            'recommendations': []
        }
        
        all_patterns = []
        
        # Process column-specific patterns
        for column, patterns in pattern_results.items():
            if column == '__cross_column__':
                report['cross_column_patterns'] = [
                    {
                        'pattern_name': p.pattern_name,
                        'confidence': p.confidence,
                        'description': p.description,
                        'examples': p.examples
                    } for p in patterns
                ]
                continue
            
            report['columns'][column] = [
                {
                    'pattern_name': p.pattern_name,
                    'pattern_type': p.pattern_type,
                    'confidence': p.confidence,
                    'match_ratio': p.match_ratio,
                    'description': p.description,
                    'examples': p.examples
                } for p in patterns
            ]
            
            all_patterns.extend(patterns)
        
        # Calculate summary statistics
        if all_patterns:
            pattern_types = [p.pattern_type for p in all_patterns]
            report['summary']['pattern_type_distribution'] = {
                ptype: pattern_types.count(ptype) for ptype in set(pattern_types)
            }
            
            report['summary']['high_confidence_patterns'] = sum(
                1 for p in all_patterns if p.confidence >= 0.8
            )
        
        # Generate recommendations
        report['recommendations'] = self._generate_pattern_recommendations(all_patterns)
        
        return report
    
    def _generate_pattern_recommendations(self, patterns: List[PatternMatch]) -> List[str]:
        """Generate actionable recommendations based on detected patterns"""
        recommendations = []
        
        # Group patterns by type
        pattern_groups = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
            pattern_groups[pattern_type].append(pattern)
        
        # Generate type-specific recommendations
        if 'special' in pattern_groups:
            unique_id_patterns = [p for p in pattern_groups['special'] if 'unique' in p.pattern_name or 'sequential' in p.pattern_name]
            if unique_id_patterns:
                recommendations.append(f"🔑 Found {len(unique_id_patterns)} potential identifier columns - consider dropping for modeling")
        
        if 'anomaly' in pattern_groups:
            outlier_patterns = [p for p in pattern_groups['anomaly'] if 'outlier' in p.pattern_name]
            if outlier_patterns:
                recommendations.append(f"⚠️ Detected outliers in {len(outlier_patterns)} columns - review and handle appropriately")
        
        if 'regex' in pattern_groups:
            business_patterns = [p for p in pattern_groups['regex'] if any(bp in p.pattern_name for bp in ['email', 'phone', 'ssn'])]
            if business_patterns:
                recommendations.append(f"🏢 Found {len(business_patterns)} columns with business data patterns - ensure proper privacy handling")
        
        if 'cross_column' in pattern_groups:
            correlation_patterns = [p for p in pattern_groups['cross_column'] if 'correlation' in p.pattern_name]
            if correlation_patterns:
                recommendations.append("📊 High correlations detected - consider feature selection to reduce multicollinearity")
        
        return recommendations


def main():
    """Test the pattern detector"""
    # Create sample data for testing
    data = {
        'id': range(1, 1001),
        'email': ['user{}@example.com'.format(i) for i in range(1000)],
        'phone': ['555-{:04d}'.format(i) for i in range(1000)],
        'amount': ['${:.2f}'.format(i * 10.5) for i in range(1000)],
        'constant': ['SAME_VALUE'] * 1000,
        'normal_dist': np.random.normal(100, 15, 1000),
        'with_outliers': list(np.random.normal(50, 10, 990)) + [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
    }
    
    df = pd.DataFrame(data)
    
    # Initialize pattern detector
    detector = PatternDetector()
    
    # Run pattern detection
    results = detector.detect_dataframe_patterns(df)
    
    # Generate report
    report = detector.generate_pattern_report(results)
    
    # Print results
    print("\n🔍 Pattern Detection Results")
    print("=" * 50)
    
    for column, patterns in results.items():
        if column == '__cross_column__':
            continue
        print(f"\n📊 {column}")
        for pattern in patterns:
            print(f"   Pattern: {pattern.pattern_name}")
            print(f"   Type: {pattern.pattern_type}")
            print(f"   Confidence: {pattern.confidence:.2f}")
            print(f"   Description: {pattern.description}")
            if pattern.examples:
                print(f"   Examples: {', '.join(pattern.examples[:2])}")


if __name__ == "__main__":
    main()
