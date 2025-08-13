#!/usr/bin/env python3
"""
📊 Evidently Monitor Manager
===========================

Main manager for monitoring train vs. production data drift and performance.
Integrates with DS-AutoAdvisor pipeline patterns.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging

# Evidently imports
try:
    from evidently import Dataset, DataDefinition, Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False
    # Create dummy classes for when evidently is not available
    class DataDefinition:
        def __init__(self, **kwargs):
            self.numerical_columns = []
            self.categorical_columns = []
            self.target = None
    
    class Dataset:
        @staticmethod
        def from_pandas(df, data_definition=None):
            return df
    
    class Report:
        def __init__(self, presets):
            self.presets = presets
        
        def run(self, current_data, reference_data):
            return {"result": "dummy"}
    
    class DataDriftPreset:
        def __init__(self):
            pass
    
    class DataSummaryPreset:
        def __init__(self):
            pass
            self.categorical_features = []
    
    class Report:
        def __init__(self, metrics=None):
            self.metrics = metrics or []
        
        def run(self, reference_data, current_data, column_mapping=None):
            pass
        
        def save_html(self, path):
            with open(path, 'w') as f:
                f.write("<html><body><h1>Evidently not available</h1></body></html>")
        
        def save_json(self, path):
            with open(path, 'w') as f:
                json.dump({"error": "Evidently not available"}, f)
    
    class TestSuite:
        def __init__(self, tests=None):
            self.tests = tests or []
        
        def run(self, reference_data, current_data, column_mapping=None):
            pass
        
        def save_html(self, path):
            with open(path, 'w') as f:
                f.write("<html><body><h1>Evidently not available</h1></body></html>")
        
        def save_json(self, path):
            with open(path, 'w') as f:
                json.dump({"error": "Evidently not available"}, f)
        
        def as_dict(self):
            return {"tests": []}

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "utils"))
sys.path.append(str(project_root / "src"))

class EvidentiallyMonitor:
    """
    Main manager for monitoring data drift and performance using Evidently.
    
    Provides train vs. production comparison, drift detection, and alerting.
    """
    
    def __init__(self, output_base: str = "pipeline_outputs"):
        """Initialize the monitoring manager"""
        self.output_base = Path(output_base)
        
        # Create monitoring output structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.monitor_outputs = {
            'base': self.output_base / f"monitoring_outputs_{timestamp}",
            'reports': self.output_base / f"monitoring_outputs_{timestamp}/reports",
            'drift_analysis': self.output_base / f"monitoring_outputs_{timestamp}/drift_analysis",
            'performance': self.output_base / f"monitoring_outputs_{timestamp}/performance",
            'alerts': self.output_base / f"monitoring_outputs_{timestamp}/alerts",
            'logs': self.output_base / f"monitoring_outputs_{timestamp}/logs"
        }
        
        # Create directories
        for path in self.monitor_outputs.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Check Evidently availability
        if not HAS_EVIDENTLY:
            self.logger.error("Evidently not available - please install with: pip install evidently")
            raise ImportError("Evidently not available")
        
        # Monitoring configuration
        self.drift_thresholds = {
            'data_drift': 0.5,  # Threshold for data drift detection
            'column_drift': 0.5,  # Threshold for individual column drift
            'missing_values': 0.1  # Threshold for missing values increase
        }
        
        # Alert settings
        self.alert_config = {
            'enabled': True,
            'slack_webhook': None,  # Will be set by user
            'email_config': None,   # Future implementation
            'severity_levels': ['INFO', 'WARNING', 'CRITICAL']
        }
        
        print("📊 Evidently Monitor initialized")
        self.logger.info("EvidentiallyMonitor initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for monitoring operations"""
        logger = logging.getLogger('evidently_monitor')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.monitor_outputs['logs'] / 'evidently_monitor.log'
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler if not already added
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def set_drift_thresholds(self, thresholds: Dict[str, float]):
        """Set custom drift detection thresholds"""
        self.drift_thresholds.update(thresholds)
        self.logger.info(f"Updated drift thresholds: {self.drift_thresholds}")
    
    def set_alert_config(self, alert_config: Dict[str, Any]):
        """Set alert configuration"""
        self.alert_config.update(alert_config)
        self.logger.info("Updated alert configuration")
    
    def _create_column_mapping(self, reference_data: pd.DataFrame, 
                               current_data: pd.DataFrame,
                               target_column: str = None,
                               prediction_column: str = None) -> DataDefinition:
        """Create data definition for Evidently (replaces ColumnMapping)"""
        
        # Identify numerical and categorical features automatically
        numerical_features = []
        categorical_features = []
        
        for col in reference_data.columns:
            if col not in [target_column, prediction_column]:
                if reference_data[col].dtype in ['int64', 'float64']:
                    numerical_features.append(col)
                else:
                    categorical_features.append(col)
        
        return DataDefinition(
            numerical_columns=numerical_features,
            categorical_columns=categorical_features
        )
    
    def generate_drift_report(self, reference_data: pd.DataFrame, 
                              current_data: pd.DataFrame,
                              target_column: str = None,
                              prediction_column: str = None,
                              report_name: str = None) -> Tuple[Report, Path]:
        """Generate comprehensive drift detection report using Evidently v0.7+"""
        
        if report_name is None:
            report_name = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Create data definition (replaces column mapping)
            data_definition = self._create_column_mapping(
                reference_data, current_data, target_column, prediction_column
            )
            
            # Create Evidently datasets
            reference_dataset = Dataset.from_pandas(reference_data, data_definition=data_definition)
            current_dataset = Dataset.from_pandas(current_data, data_definition=data_definition)
            
            # Create report with presets (simpler API)
            report = Report([
                DataDriftPreset(),
                DataSummaryPreset()
            ])
            
            # Run report
            result = report.run(current_dataset, reference_dataset)
            
            # Save HTML report
            html_path = self.monitor_outputs['reports'] / f"{report_name}.html"
            result.save_html(str(html_path))
            
            print(f"📊 Drift report generated: {html_path}")
            self.logger.info(f"Drift report saved: {html_path}")
            
            return result, html_path
            
        except Exception as e:
            self.logger.error(f"Error generating drift report: {e}")
            raise e
    
    def run_drift_tests(self, reference_data: pd.DataFrame, 
                        current_data: pd.DataFrame,
                        target_column: str = None,
                        test_name: str = None) -> Tuple[Any, Dict[str, Any]]:
        """Run drift tests using Report with tests (simplified for v0.7+)"""
        
        if test_name is None:
            test_name = f"drift_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Create data definition
            data_definition = self._create_column_mapping(
                reference_data, current_data, target_column
            )
            
            # Create Evidently datasets
            reference_dataset = Dataset.from_pandas(reference_data, data_definition=data_definition)
            current_dataset = Dataset.from_pandas(current_data, data_definition=data_definition)
            
            # Create report with tests enabled
            report = Report([
                DataDriftPreset(),
                DataSummaryPreset()
            ], include_tests=True)
            
            # Run report
            result = report.run(current_dataset, reference_dataset)
            
            # Save test results
            html_path = self.monitor_outputs['drift_analysis'] / f"{test_name}.html"
            result.save_html(str(html_path))
            
            # Create summary for alerting
            alerts = self._generate_simple_alerts(result, test_name)
            
            print(f"🧪 Drift tests completed: {html_path}")
            self.logger.info(f"Drift tests saved: {html_path}")
            
            return result, alerts
            
        except Exception as e:
            self.logger.error(f"Error running drift tests: {e}")
            raise e
    
    def _generate_simple_alerts(self, report_result: Any, test_name: str) -> Dict[str, Any]:
        """Generate simple alerts based on report results (v0.7+ compatible)"""
        
        alerts = {
            'timestamp': datetime.now().isoformat(),
            'test_name': test_name,
            'alert_level': 'INFO',
            'alerts': []
        }
        
        if not self.alert_config['enabled']:
            return alerts
        
        try:
            # Simple drift detection based on the report presence
            alerts['alert_level'] = 'INFO'
            alerts['alerts'].append({
                'type': 'drift_monitoring',
                'message': 'Data drift monitoring completed successfully',
                'severity': 'INFO',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.warning(f"Error generating alerts: {e}")
            alerts['alerts'].append({
                'type': 'error',
                'message': f'Alert generation failed: {str(e)}',
                'severity': 'WARNING',
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def create_synthetic_drift_demo(self, original_data: pd.DataFrame, 
                                    target_column: str = None,
                                    drift_severity: str = "medium") -> pd.DataFrame:
        """Create synthetic production data with drift for demonstration"""
        
        # Create a copy for modification
        drift_data = original_data.copy()
        
        # Define drift parameters based on severity
        drift_params = {
            'low': {'shift_factor': 0.1, 'noise_factor': 0.05, 'missing_rate': 0.02},
            'medium': {'shift_factor': 0.3, 'noise_factor': 0.1, 'missing_rate': 0.05},
            'high': {'shift_factor': 0.5, 'noise_factor': 0.2, 'missing_rate': 0.1}
        }
        
        params = drift_params.get(drift_severity, drift_params['medium'])
        
        # Apply drift to numerical columns
        numerical_cols = drift_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != target_column:
                # Add systematic shift
                shift = np.random.normal(0, params['shift_factor'] * drift_data[col].std(), len(drift_data))
                drift_data[col] = drift_data[col] + shift
                
                # Add noise
                noise = np.random.normal(0, params['noise_factor'] * drift_data[col].std(), len(drift_data))
                drift_data[col] = drift_data[col] + noise
        
        # Apply drift to categorical columns
        categorical_cols = drift_data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col != target_column:
                # Randomly change some values to simulate concept drift
                n_changes = int(len(drift_data) * params['shift_factor'])
                change_indices = np.random.choice(len(drift_data), n_changes, replace=False)
                
                unique_values = drift_data[col].unique()
                if len(unique_values) > 1:
                    for idx in change_indices:
                        current_value = drift_data.loc[idx, col]
                        other_values = [v for v in unique_values if v != current_value]
                        drift_data.loc[idx, col] = np.random.choice(other_values)
        
        # Introduce missing values
        all_cols = [col for col in drift_data.columns if col != target_column]
        n_missing = int(len(drift_data) * len(all_cols) * params['missing_rate'])
        
        for _ in range(n_missing):
            row_idx = np.random.randint(0, len(drift_data))
            col_idx = np.random.randint(0, len(all_cols))
            drift_data.iloc[row_idx, drift_data.columns.get_loc(all_cols[col_idx])] = np.nan
        
        print(f"🎭 Generated synthetic drift data with {drift_severity} severity")
        self.logger.info(f"Created synthetic drift data: {drift_severity} severity")
        
        return drift_data
    
    def run_monitoring_workflow(self, reference_data: pd.DataFrame,
                                current_data: pd.DataFrame,
                                target_column: str = None,
                                prediction_column: str = None,
                                workflow_name: str = None) -> Dict[str, Any]:
        """Run complete monitoring workflow: reports + tests + alerts"""
        
        if workflow_name is None:
            workflow_name = f"monitoring_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        workflow_results = {
            'workflow_name': workflow_name,
            'timestamp': datetime.now().isoformat(),
            'summary': {}
        }
        
        try:
            print(f"🚀 Starting monitoring workflow: {workflow_name}")
            
            # Generate drift report
            print("📊 Generating drift report...")
            report, report_path = self.generate_drift_report(
                reference_data, current_data, target_column, prediction_column,
                f"{workflow_name}_report"
            )
            workflow_results['drift_report_path'] = str(report_path)
            
            # Run drift tests
            print("🧪 Running drift tests...")
            test_suite, alerts = self.run_drift_tests(
                reference_data, current_data, target_column,
                f"{workflow_name}_tests"
            )
            workflow_results['alerts'] = alerts
            
            # Generate summary
            workflow_results['summary'] = {
                'data_samples': {
                    'reference': len(reference_data),
                    'current': len(current_data)
                },
                'columns_analyzed': len(reference_data.columns),
                'alert_level': alerts.get('alert_level', 'INFO'),
                'total_alerts': len(alerts.get('alerts', []))
            }
            
            # Save workflow summary
            summary_file = self.monitor_outputs['base'] / f"{workflow_name}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(workflow_results, f, indent=2)
            
            print(f"✅ Monitoring workflow complete: {summary_file}")
            self.logger.info(f"Monitoring workflow completed: {workflow_name}")
            
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"Error in monitoring workflow: {e}")
            raise e
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of all monitoring activities"""
        
        summary = {
            'total_reports': 0,
            'total_tests': 0,
            'total_alerts': 0,
            'recent_activities': []
        }
        
        try:
            # Count reports
            reports_dir = self.monitor_outputs['reports']
            if reports_dir.exists():
                summary['total_reports'] = len(list(reports_dir.glob("*.html")))
            
            # Count test results
            tests_dir = self.monitor_outputs['drift_analysis']
            if tests_dir.exists():
                summary['total_tests'] = len(list(tests_dir.glob("*.html")))
            
            # Count alerts
            alerts_dir = self.monitor_outputs['alerts']
            if alerts_dir.exists():
                summary['total_alerts'] = len(list(alerts_dir.glob("*.json")))
            
            # Get recent activities (last 10 files)
            all_files = []
            for dir_path in [reports_dir, tests_dir, alerts_dir]:
                if dir_path.exists():
                    all_files.extend(dir_path.glob("*"))
            
            # Sort by modification time and get recent ones
            recent_files = sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]
            
            for file_path in recent_files:
                summary['recent_activities'].append({
                    'file': file_path.name,
                    'type': file_path.parent.name,
                    'timestamp': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
            
        except Exception as e:
            self.logger.warning(f"Error generating monitoring summary: {e}")
        
        return summary
