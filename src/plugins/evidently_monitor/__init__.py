"""
📊 Evidently Monitor Plugin for DS-AutoAdvisor
=============================================

Monitoring plugin that provides drift detection and performance monitoring.

Features:
- Train vs. production data comparison
- Drift detection reports
- Performance monitoring
- Cronable CLI functionality
- Slack-ready HTML output
- Synthetic drift demo

Usage:
    from src.plugins.evidently_monitor import EvidentiallyMonitor
    
    monitor = EvidentiallyMonitor(config)
    report = monitor.generate_drift_report(train_data, production_data)
"""

from .monitor_manager import EvidentiallyMonitor
from .drift_detector import DriftDetector
from .report_generator import ReportGenerator
from .cli_monitor import CLIMonitor

__all__ = ['EvidentiallyMonitor', 'DriftDetector', 'ReportGenerator', 'CLIMonitor']
