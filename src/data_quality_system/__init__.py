"""
Enhanced Data Quality System for DS-AutoAdvisor
===============================================

A comprehensive data quality assessment system with advanced features:
- Multi-stage heuristic data type inference
- Confidence-based quality scoring
- Pattern matching for datetime and numerical detection
- Statistical tests for data classification
- Cardinality-based classification rules
- YAML-controlled configuration
"""

from .enhanced_quality_system import DataQualityAssessor, QualityReport, QualityIssue
from .data_type_inference import DataTypeInferenceEngine
from .quality_metrics import QualityMetricsCalculator
from .pattern_detectors import PatternDetector

__version__ = "3.0.0"
__all__ = [
    "DataQualityAssessor",
    "QualityReport", 
    "QualityIssue",
    "DataTypeInferenceEngine",
    "QualityMetricsCalculator",
    "PatternDetector"
]
