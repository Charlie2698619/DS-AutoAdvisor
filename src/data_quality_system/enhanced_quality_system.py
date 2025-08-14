#!/usr/bin/env python3
"""
🏆 Enhanced Data Quality System for DS-AutoAdvisor
=================================================

Comprehensive data quality assessment system combining:
- Multi-stage data type inference with confidence scoring
- Advanced pattern detection for business and anomaly patterns
- Comprehensive quality metrics across 6 dimensions
- YAML-controlled configuration and thresholds
- Detailed reporting with actionable recommendations

This system serves as the in-depth quality checker integrated with the profiling pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml
import json
from pathlib import Path

# Import our quality system components
try:
    from .data_type_inference import DataTypeInferenceEngine, InferenceResult
    from .pattern_detectors import PatternDetector, PatternMatch
    from .quality_metrics import QualityMetricsCalculator, QualityScore, QualityDimension
except ImportError:
    # Fallback for direct execution
    from data_type_inference import DataTypeInferenceEngine, InferenceResult
    from pattern_detectors import PatternDetector, PatternMatch
    from quality_metrics import QualityMetricsCalculator, QualityScore, QualityDimension

@dataclass
class QualityIssue:
    """Represents a data quality issue"""
    severity: str  # 'high', 'medium', 'low', 'info'
    column: str
    issue_type: str
    description: str
    suggested_action: str
    affected_rows: List[int]
    metadata: Dict[str, Any]

@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    overall_score: float
    total_issues: int
    issues_by_severity: Dict[str, int]
    column_scores: Dict[str, float]
    recommendations: List[str]
    issues: List[QualityIssue]
    metadata: Dict[str, Any]
    
    # Additional components
    type_inference_results: Dict[str, Any]
    pattern_detection_results: Dict[str, Any]
    quality_metrics_results: Dict[str, Any]

class DataQualityAssessor:
    """
    Main enhanced data quality assessment system
    
    Integrates data type inference, pattern detection, and quality metrics
    for comprehensive quality analysis with YAML configuration control.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the enhanced data quality assessor"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize component systems
        self.type_inference_engine = DataTypeInferenceEngine(config_path)
        self.pattern_detector = PatternDetector(config_path)
        self.quality_calculator = QualityMetricsCalculator(config_path)
        
        print("🏆 Enhanced Data Quality System initialized")
        print(f"   Configuration: {self._get_config_source()}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if self.config_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.config_path = project_root / "config" / "unified_config_v3.yaml"
        
        default_config = {
            'enhanced_quality_system': {
                'components': {
                    'enable_type_inference': True,
                    'enable_pattern_detection': True,
                    'enable_quality_metrics': True,
                    'enable_cross_validation': True
                },
                'reporting': {
                    'generate_detailed_report': True,
                    'include_recommendations': True,
                    'include_sample_data': True,
                    'max_sample_size': 100
                },
                'thresholds': {
                    'critical_score_threshold': 40,
                    'warning_score_threshold': 70,
                    'good_score_threshold': 85
                },
                'integration': {
                    'use_yaml_config': True,
                    'respect_mode_settings': True,
                    'output_format': 'comprehensive' 
                }
            }
        }
        
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    
                    # Check for global enhanced_quality_system config first
                    if 'enhanced_quality_system' in loaded_config:
                        enhanced_config = loaded_config['enhanced_quality_system']
                        
                        # Merge with defaults, preserving nested structure
                        config = default_config['enhanced_quality_system'].copy()
                        
                        # Deep merge the configurations
                        for key, value in enhanced_config.items():
                            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                                config[key].update(value)
                            else:
                                config[key] = value
                        
                        return config
                    
                    # Fallback: Extract quality system config based on mode (legacy support)
                    enhanced_config = None
                    
                    # Try custom mode first
                    if 'custom_mode' in loaded_config and 'data_discovery' in loaded_config['custom_mode']:
                        discovery_config = loaded_config['custom_mode']['data_discovery']
                        if 'quality_assessment' in discovery_config:
                            quality_config = discovery_config['quality_assessment']
                            # Look for enhanced_quality_system config
                            if 'enhanced_quality_system' in quality_config:
                                enhanced_config = quality_config['enhanced_quality_system']
                    
                    # Fall back to fast mode
                    if enhanced_config is None and 'fast_mode' in loaded_config and 'data_discovery' in loaded_config['fast_mode']:
                        discovery_config = loaded_config['fast_mode']['data_discovery']
                        if 'quality_assessment' in discovery_config:
                            quality_config = discovery_config['quality_assessment']
                            if 'enhanced_quality_system' in quality_config:
                                enhanced_config = quality_config['enhanced_quality_system']
                    
                    
                    
    
                    if enhanced_config:
                        # Merge with defaults, preserving nested structure
                        config = default_config['enhanced_quality_system'].copy()
                        
                        # Deep merge the configurations
                        for key, value in enhanced_config.items():
                            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                                config[key].update(value)
                            else:
                                config[key] = value
                        
                        return config
                
                return default_config['enhanced_quality_system']
            except Exception as e:
                print(f"⚠️ Error loading quality system config: {e}")
                return default_config['enhanced_quality_system']
        else:
            return default_config['enhanced_quality_system']
    
    def _get_config_source(self) -> str:
        """Get description of configuration source"""
        if Path(self.config_path).exists():
            return f"YAML ({self.config_path})"
        else:
            return "Default (YAML not found)"
    
    def assess_quality(self, df: pd.DataFrame, 
                      target_column: Optional[str] = None,
                      enable_sampling: bool = True) -> QualityReport:
        """
        Perform comprehensive quality assessment
        
        Args:
            df: DataFrame to assess
            target_column: Optional target column name
            enable_sampling: Whether to use sampling for large datasets
            
        Returns:
            QualityReport with comprehensive assessment results
        """
        print(f"🏆 Starting Enhanced Data Quality Assessment")
        print(f"   Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"   Target column: {target_column or 'Auto-detect'}")
        
        # Sample data if necessary for performance
        assessment_df = self._prepare_assessment_data(df, enable_sampling)
        
        # Initialize results containers
        all_issues = []
        recommendations = []
        metadata = {
            'assessment_timestamp': datetime.now().isoformat(),
            'original_shape': df.shape,
            'assessment_shape': assessment_df.shape,
            'target_column': target_column,
            'sampling_used': len(assessment_df) < len(df)
        }
        
        # Component 1: Data Type Inference
        type_inference_results = {}
        if self.config['components']['enable_type_inference']:
            print("   🧠 Running data type inference...")
            type_results = self.type_inference_engine.infer_dataframe_types(assessment_df)
            type_inference_results = self.type_inference_engine.generate_inference_report(type_results)
            
            # Convert type inference issues to quality issues
            type_issues = self._convert_type_inference_issues(type_results)
            all_issues.extend(type_issues)
            
            # Add type inference recommendations
            for priority, recs in type_inference_results.get('recommendations', {}).items():
                recommendations.extend(recs)
        
        # Component 2: Pattern Detection
        pattern_detection_results = {}
        if self.config['components']['enable_pattern_detection']:
            print("   🔍 Running pattern detection...")
            pattern_results = self.pattern_detector.detect_dataframe_patterns(assessment_df)
            pattern_detection_results = self.pattern_detector.generate_pattern_report(pattern_results)
            
            # Convert pattern detection issues to quality issues
            pattern_issues = self._convert_pattern_detection_issues(pattern_results)
            all_issues.extend(pattern_issues)
            
            # Add pattern detection recommendations
            recommendations.extend(pattern_detection_results.get('recommendations', []))
        
        # Component 3: Quality Metrics
        quality_metrics_results = {}
        if self.config['components']['enable_quality_metrics']:
            print("   📊 Calculating quality metrics...")
            quality_score = self.quality_calculator.calculate_quality_score(assessment_df, target_column)
            quality_metrics_results = self.quality_calculator.generate_quality_report(quality_score)
            
            # Convert quality metrics issues to quality issues
            metrics_issues = self._convert_quality_metrics_issues(quality_metrics_results)
            all_issues.extend(metrics_issues)
            
            # Add quality metrics recommendations
            recommendations.extend(quality_metrics_results.get('recommendations', []))
        
        # Component 4: Cross-validation and Integration
        if self.config['components']['enable_cross_validation']:
            print("   🔄 Running cross-validation checks...")
            cross_validation_issues = self._perform_cross_validation(
                assessment_df, type_inference_results, pattern_detection_results, quality_metrics_results
            )
            all_issues.extend(cross_validation_issues)
        
        # Calculate overall scores and metrics
        overall_score, column_scores, issues_by_severity = self._calculate_comprehensive_scores(
            all_issues, assessment_df, quality_metrics_results
        )
        
        # Generate final recommendations
        final_recommendations = self._generate_final_recommendations(
            overall_score, issues_by_severity, recommendations
        )
        
        # Create comprehensive report
        report = QualityReport(
            overall_score=overall_score,
            total_issues=len(all_issues),
            issues_by_severity=issues_by_severity,
            column_scores=column_scores,
            recommendations=final_recommendations,
            issues=all_issues,
            metadata=metadata,
            type_inference_results=type_inference_results,
            pattern_detection_results=pattern_detection_results,
            quality_metrics_results=quality_metrics_results
        )
        
        print(f"✅ Quality assessment completed")
        print(f"   Overall Score: {overall_score:.1f}/100")
        print(f"   Total Issues: {len(all_issues)}")
        print(f"   High Priority: {issues_by_severity.get('high', 0)}")
        print(f"   Medium Priority: {issues_by_severity.get('medium', 0)}")
        
        return report
    
    def _prepare_assessment_data(self, df: pd.DataFrame, enable_sampling: bool) -> pd.DataFrame:
        """Prepare data for assessment (sampling if needed)"""
        if not enable_sampling or len(df) <= 10000:
            return df
        
        # Sample for large datasets
        sample_size = min(10000, len(df))
        print(f"   📊 Sampling {sample_size} rows from {len(df)} for performance...")
        
        # Stratified sampling if target column is identified
        return df.sample(n=sample_size, random_state=42)
    
    def _convert_type_inference_issues(self, type_results: Dict[str, InferenceResult]) -> List[QualityIssue]:
        """Convert type inference results to quality issues"""
        issues = []
        
        for column, result in type_results.items():
            # Low confidence inference
            if result.confidence < 0.6:
                issues.append(QualityIssue(
                    severity='medium' if result.confidence < 0.4 else 'low',
                    column=column,
                    issue_type='low_confidence_type_inference',
                    description=f"Low confidence ({result.confidence:.2f}) in type inference for '{column}'",
                    suggested_action="Manual review of column data type recommended",
                    affected_rows=[],
                    metadata={
                        'inferred_type': result.inferred_type,
                        'original_type': result.original_type,
                        'confidence': result.confidence,
                        'patterns': result.patterns_detected
                    }
                ))
            
            # Type mismatch issues (these are preprocessing recommendations, not quality problems)
            if result.inferred_type != result.original_type and result.confidence > 0.7:
                # Type recommendations should be info/low severity, not high priority quality issues
                severity = 'low' if result.confidence > 0.8 else 'info'
                issues.append(QualityIssue(
                    severity=severity,
                    column=column,
                    issue_type='type_mismatch',
                    description=f"Column '{column}' stored as {result.original_type} but appears to be {result.inferred_type}",
                    suggested_action=f"Consider converting to {result.inferred_type} type",
                    affected_rows=[],
                    metadata={
                        'inferred_type': result.inferred_type,
                        'original_type': result.original_type,
                        'confidence': result.confidence
                    }
                ))
        
        return issues
    
    def _convert_pattern_detection_issues(self, pattern_results: Dict[str, List[PatternMatch]]) -> List[QualityIssue]:
        """Convert pattern detection results to quality issues"""
        issues = []
        
        for column, patterns in pattern_results.items():
            if column == '__cross_column__':
                continue
            
            for pattern in patterns:
                if pattern.pattern_type == 'anomaly':
                    # Anomaly patterns are quality issues
                    severity = 'high' if pattern.confidence > 0.8 else 'medium'
                    issues.append(QualityIssue(
                        severity=severity,
                        column=column,
                        issue_type=f'anomaly_{pattern.pattern_name}',
                        description=f"Anomaly detected in '{column}': {pattern.description}",
                        suggested_action="Review and clean anomalous values",
                        affected_rows=[],
                        metadata={
                            'pattern_name': pattern.pattern_name,
                            'confidence': pattern.confidence,
                            'match_ratio': pattern.match_ratio,
                            'examples': pattern.examples
                        }
                    ))
                
                elif pattern.pattern_type == 'special' and 'unique' in pattern.pattern_name:
                    # Unique identifier detection
                    issues.append(QualityIssue(
                        severity='info',
                        column=column,
                        issue_type='potential_identifier',
                        description=f"Column '{column}' appears to be a unique identifier",
                        suggested_action="Consider dropping for modeling if not needed",
                        affected_rows=[],
                        metadata={
                            'pattern_name': pattern.pattern_name,
                            'confidence': pattern.confidence
                        }
                    ))
        
        return issues
    
    def _convert_quality_metrics_issues(self, metrics_results: Dict[str, Any]) -> List[QualityIssue]:
        """Convert quality metrics results to quality issues"""
        issues = []
        
        # Process dimension-specific issues
        if 'dimensions' in metrics_results:
            for dimension_name, dimension_data in metrics_results['dimensions'].items():
                dimension_score = dimension_data.get('score', 100)
                
                # Flag dimensions with poor scores
                if dimension_score < 60:
                    severity = 'high' if dimension_score < 40 else 'medium'
                    issues.append(QualityIssue(
                        severity=severity,
                        column='__dataset__',
                        issue_type=f'poor_{dimension_name}_quality',
                        description=f"Poor {dimension_name} quality (score: {dimension_score:.1f}/100)",
                        suggested_action=f"Focus on improving {dimension_name} dimension",
                        affected_rows=[],
                        metadata={
                            'dimension': dimension_name,
                            'score': dimension_score,
                            'metrics': dimension_data.get('metrics', {})
                        }
                    ))
        
        # Process categorized issues
        if 'issues' in metrics_results:
            for severity, severity_issues in metrics_results['issues'].items():
                for issue in severity_issues:
                    issues.append(QualityIssue(
                        severity=severity,
                        column=issue.get('column', 'unknown'),
                        issue_type=issue.get('type', 'quality_metric_issue'),
                        description=issue.get('description', 'Quality metric issue detected'),
                        suggested_action=issue.get('recommendation', 'Review and address issue'),
                        affected_rows=[],
                        metadata=issue
                    ))
        
        return issues
    
    def _perform_cross_validation(self, df: pd.DataFrame, 
                                type_results: Dict[str, Any],
                                pattern_results: Dict[str, Any],
                                metrics_results: Dict[str, Any]) -> List[QualityIssue]:
        """Perform cross-validation between different assessment components"""
        issues = []
        
        # Cross-validate type inference with pattern detection
        if type_results and pattern_results:
            type_columns = type_results.get('columns', {})
            pattern_columns = pattern_results.get('columns', {})
            
            for column in df.columns:
                if column in type_columns and column in pattern_columns:
                    inferred_type = type_columns[column].get('inferred_type')
                    patterns = pattern_columns[column]
                    
                    # Check for conflicts
                    if inferred_type == 'numeric' and any('email' in p.get('pattern_name', '') or 'phone' in p.get('pattern_name', '') for p in patterns):
                        issues.append(QualityIssue(
                            severity='medium',
                            column=column,
                            issue_type='type_pattern_conflict',
                            description=f"Type inference suggests numeric but patterns suggest business data for '{column}'",
                            suggested_action="Manual review of column content recommended",
                            affected_rows=[],
                            metadata={
                                'inferred_type': inferred_type,
                                'detected_patterns': [p.get('pattern_name', '') for p in patterns]
                            }
                        ))
        
        return issues
    
    def _calculate_comprehensive_scores(self, issues: List[QualityIssue], df: pd.DataFrame,
                                      metrics_results: Dict[str, Any]) -> Tuple[float, Dict[str, float], Dict[str, int]]:
        """Calculate comprehensive quality scores"""
        # Start with metrics-based overall score if available
        if metrics_results and 'summary' in metrics_results:
            base_score = metrics_results['summary'].get('overall_score', 80)
        else:
            base_score = 80  # Default base score
        
        # Penalize based on issues (only actual quality problems, not preprocessing recommendations)
        issue_penalty = 0
        issues_by_severity = {'high': 0, 'medium': 0, 'low': 0, 'info': 0}
        
        for issue in issues:
            severity = issue.severity
            issues_by_severity[severity] += 1
            
            # Only penalize for actual data quality problems, not preprocessing recommendations
            if issue.issue_type in ['type_mismatch', 'encoding_needed']:
                # Type mismatches and encoding needs are preprocessing steps, not quality issues
                # Reduce penalty significantly or skip
                if severity == 'high':
                    issue_penalty += 1  # Reduced from 10
                elif severity == 'medium':
                    issue_penalty += 0.5  # Reduced from 5
                # Skip low and info for preprocessing
            else:
                # Apply full penalties for actual quality issues
                if severity == 'high':
                    issue_penalty += 10
                elif severity == 'medium':
                    issue_penalty += 5
                elif severity == 'low':
                    issue_penalty += 2
            # Info issues don't penalize score
        
        # Calculate final overall score
        overall_score = max(0, base_score - issue_penalty)
        
        # Calculate column-specific scores
        column_scores = {}
        for column in df.columns:
            column_issues = [issue for issue in issues if issue.column == column]
            column_penalty = sum(10 if issue.severity == 'high' else 5 if issue.severity == 'medium' else 2 if issue.severity == 'low' else 0 for issue in column_issues)
            column_scores[column] = max(0, 100 - column_penalty)
        
        return overall_score, column_scores, issues_by_severity
    
    def _generate_final_recommendations(self, overall_score: float, 
                                      issues_by_severity: Dict[str, int],
                                      component_recommendations: List[str]) -> List[str]:
        """Generate final prioritized recommendations"""
        recommendations = []
        
        # Access thresholds from config (handle nested structure)
        thresholds = self.config.get('thresholds', {})
        critical_threshold = thresholds.get('critical_score_threshold', 30)
        warning_threshold = thresholds.get('warning_score_threshold', 70)
        good_threshold = thresholds.get('good_score_threshold', 85)
        
        # Priority recommendations based on overall score
        if overall_score < critical_threshold:
            recommendations.append("🚨 CRITICAL: Comprehensive data quality remediation required before modeling")
        elif overall_score < warning_threshold:
            recommendations.append("⚠️ WARNING: Significant quality issues detected - address before production use")
        elif overall_score < good_threshold:
            recommendations.append("ℹ️ INFO: Minor quality issues present - consider addressing for optimal results")
        else:
            recommendations.append("✅ GOOD: Data quality is acceptable for modeling")
        
        # Issue-specific recommendations
        if issues_by_severity['high'] > 0:
            recommendations.append(f"🔴 Address {issues_by_severity['high']} high-priority quality issues immediately")
        
        if issues_by_severity['medium'] > 5:
            recommendations.append(f"🟡 Consider addressing {issues_by_severity['medium']} medium-priority issues")
        
        # Add unique component recommendations
        unique_recommendations = list(set(component_recommendations))
        recommendations.extend(unique_recommendations[:10])  # Limit to top 10
        
        return recommendations
    
    def save_assessment_report(self, report: QualityReport, output_path: str) -> bool:
        """Save comprehensive assessment report to file"""
        try:
            # Convert dataclass to dict for serialization
            report_dict = {
                'summary': {
                    'overall_score': report.overall_score,
                    'total_issues': report.total_issues,
                    'issues_by_severity': report.issues_by_severity,
                    'recommendations_count': len(report.recommendations)
                },
                'detailed_scores': {
                    'column_scores': report.column_scores
                },
                'issues': [
                    {
                        'severity': issue.severity,
                        'column': issue.column,
                        'issue_type': issue.issue_type,
                        'description': issue.description,
                        'suggested_action': issue.suggested_action,
                        'metadata': issue.metadata
                    } for issue in report.issues
                ],
                'recommendations': report.recommendations,
                'metadata': report.metadata,
                'component_results': {
                    'type_inference': report.type_inference_results,
                    'pattern_detection': report.pattern_detection_results,
                    'quality_metrics': report.quality_metrics_results
                }
            }
            
            # Save to JSON file
            output_file = Path(output_path)
            with open(output_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            print(f"📄 Quality assessment report saved: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save assessment report: {e}")
            return False
    
    def generate_summary_report(self, report: QualityReport) -> str:
        """Generate human-readable summary report"""
        summary = []
        summary.append("🏆 ENHANCED DATA QUALITY ASSESSMENT SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Overall Quality Score: {report.overall_score:.1f}/100")
        summary.append(f"Total Issues Found: {report.total_issues}")
        summary.append("")
        
        # Issues breakdown
        summary.append("📊 Issues by Severity:")
        for severity, count in report.issues_by_severity.items():
            if count > 0:
                summary.append(f"   {severity.upper()}: {count}")
        summary.append("")
        
        # Top recommendations
        summary.append("💡 Key Recommendations:")
        for i, rec in enumerate(report.recommendations[:5], 1):
            summary.append(f"   {i}. {rec}")
        
        if len(report.recommendations) > 5:
            summary.append(f"   ... and {len(report.recommendations) - 5} more")
        
        summary.append("")
        summary.append(f"📅 Assessment completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(summary)


def main():
    """Test the enhanced data quality system"""
    # Create comprehensive test dataset
    np.random.seed(42)
    
    test_data = {
        # Good quality columns
        'id': range(1, 1001),
        'name': [f'Customer_{i}' for i in range(1, 1001)],
        'email': [f'user{i}@example.com' for i in range(1, 1001)],
        
        # Columns with quality issues
        'age': list(np.random.randint(18, 80, 990)) + [-5, 150, 200, 300, 400, 500, 600, 700, 800, 900],  # Invalid ages
        'salary': list(np.random.normal(50000, 15000, 980)) + [None] * 20,  # Missing values
        'phone': ['555-' + str(i).zfill(4) for i in range(990)] + ['invalid'] * 10,  # Invalid phones
        'percentage': list(np.random.uniform(0, 100, 980)) + [150, -10] * 10,  # Invalid percentages
        
        # Date columns
        'date_joined': pd.date_range('2020-01-01', periods=1000, freq='D'),
        'last_login': pd.date_range('2023-01-01', periods=1000, freq='H'),
        
        # Categorical with issues
        'category': ['A', 'B', 'C'] * 333 + ['invalid_cat'],
        'status': ['active', 'ACTIVE', 'Active', 'inactive'] * 250  # Case inconsistency
    }
    
    df = pd.DataFrame(test_data)
    
    # Initialize enhanced quality assessor
    assessor = DataQualityAssessor()
    
    # Run comprehensive assessment
    quality_report = assessor.assess_quality(df, target_column='category')
    
    # Generate and print summary
    summary = assessor.generate_summary_report(quality_report)
    print(summary)
    
    # Save detailed report
    assessor.save_assessment_report(quality_report, 'test_quality_report.json')


if __name__ == "__main__":
    main()
