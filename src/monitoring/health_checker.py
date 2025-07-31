"""
Industrial-Grade Health Checking and Monitoring
=============================================

Comprehensive health monitoring for DS-AutoAdvisor pipeline.
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import json

@dataclass
class HealthMetric:
    """Health metric definition"""
    name: str
    value: float
    threshold: float
    status: str  # "healthy", "warning", "critical"
    timestamp: datetime
    details: Optional[Dict] = None

class PipelineHealthChecker:
    """Comprehensive pipeline health monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Start health monitoring"""
        self.monitoring_active = True
        self.logger.info("Health monitoring started")
        
        # Perform initial health check
        initial_metrics = self.check_system_health()
        for metric in initial_metrics:
            if metric.status == "critical":
                self.logger.error(f"Critical health issue detected: {metric.name} = {metric.value}")
            elif metric.status == "warning":
                self.logger.warning(f"Health warning: {metric.name} = {metric.value}")
        
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        self.logger.info("Health monitoring stopped")
        
    def check_system_health(self) -> List[HealthMetric]:
        """Check system-level health metrics"""
        metrics = []
        
        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            threshold=80.0,
            status="healthy" if cpu_percent < 80 else "warning" if cpu_percent < 95 else "critical",
            timestamp=datetime.now(),
            details={"cores": psutil.cpu_count()}
        ))
        
        # Memory Usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        metrics.append(HealthMetric(
            name="memory_usage",
            value=memory_percent,
            threshold=85.0,
            status="healthy" if memory_percent < 85 else "warning" if memory_percent < 95 else "critical",
            timestamp=datetime.now(),
            details={"total_gb": round(memory.total / (1024**3), 2)}
        ))
        
        # Disk Usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        metrics.append(HealthMetric(
            name="disk_usage", 
            value=disk_percent,
            threshold=90.0,
            status="healthy" if disk_percent < 90 else "warning" if disk_percent < 98 else "critical",
            timestamp=datetime.now(),
            details={"free_gb": round(disk.free / (1024**3), 2)}
        ))
        
        return metrics
    
    def check_data_health(self, df: pd.DataFrame, schema: Optional[Dict] = None) -> List[HealthMetric]:
        """Check data quality health metrics"""
        metrics = []
        
        # Data completeness
        completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        metrics.append(HealthMetric(
            name="data_completeness",
            value=completeness,
            threshold=95.0,
            status="healthy" if completeness >= 95 else "warning" if completeness >= 90 else "critical",
            timestamp=datetime.now(),
            details={"missing_values": int(df.isnull().sum().sum())}
        ))
        
        # Data freshness (if timestamp column exists)
        if 'timestamp' in df.columns or 'date' in df.columns:
            timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'date'
            try:
                latest_record = pd.to_datetime(df[timestamp_col]).max()
                hours_since_latest = (datetime.now() - latest_record).total_seconds() / 3600
                
                metrics.append(HealthMetric(
                    name="data_freshness_hours",
                    value=hours_since_latest,
                    threshold=24.0,  # 24 hours
                    status="healthy" if hours_since_latest < 24 else "warning" if hours_since_latest < 48 else "critical",
                    timestamp=datetime.now(),
                    details={"latest_record": latest_record.isoformat()}
                ))
            except:
                pass
        
        # Schema drift detection
        if schema:
            expected_columns = set(schema.get('columns', []))
            actual_columns = set(df.columns)
            
            missing_columns = expected_columns - actual_columns
            extra_columns = actual_columns - expected_columns
            
            schema_match_score = len(expected_columns.intersection(actual_columns)) / len(expected_columns) * 100
            
            metrics.append(HealthMetric(
                name="schema_compatibility",
                value=schema_match_score,
                threshold=100.0,
                status="healthy" if schema_match_score == 100 else "warning" if schema_match_score >= 90 else "critical",
                timestamp=datetime.now(),
                details={
                    "missing_columns": list(missing_columns),
                    "extra_columns": list(extra_columns)
                }
            ))
        
        return metrics
    
    def check_model_health(self, model_metrics: Dict[str, float]) -> List[HealthMetric]:
        """Check model performance health"""
        metrics = []
        
        # Model accuracy/performance
        if 'accuracy' in model_metrics:
            accuracy = model_metrics['accuracy']
            metrics.append(HealthMetric(
                name="model_accuracy",
                value=accuracy,
                threshold=0.80,
                status="healthy" if accuracy >= 0.80 else "warning" if accuracy >= 0.70 else "critical",
                timestamp=datetime.now()
            ))
        
        # Model inference time
        if 'inference_time_ms' in model_metrics:
            inference_time = model_metrics['inference_time_ms']
            metrics.append(HealthMetric(
                name="model_inference_time",
                value=inference_time,
                threshold=1000.0,  # 1 second
                status="healthy" if inference_time < 1000 else "warning" if inference_time < 5000 else "critical",
                timestamp=datetime.now()
            ))
        
        return metrics
    
    def generate_health_report(self, include_system=True, include_data=None, include_model=None) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "metrics": [],
            "recommendations": [],
            "alerts": []
        }
        
        all_metrics = []
        
        if include_system:
            all_metrics.extend(self.check_system_health())
        
        if include_data is not None:
            all_metrics.extend(self.check_data_health(include_data))
        
        if include_model is not None:
            all_metrics.extend(self.check_model_health(include_model))
        
        # Determine overall status
        critical_count = sum(1 for m in all_metrics if m.status == "critical")
        warning_count = sum(1 for m in all_metrics if m.status == "warning")
        
        if critical_count > 0:
            report["overall_status"] = "critical"
        elif warning_count > 0:
            report["overall_status"] = "warning"
        
        # Convert metrics to dict format
        report["metrics"] = [
            {
                "name": m.name,
                "value": m.value,
                "threshold": m.threshold,
                "status": m.status,
                "timestamp": m.timestamp.isoformat(),
                "details": m.details
            }
            for m in all_metrics
        ]
        
        # Generate recommendations
        for metric in all_metrics:
            if metric.status in ["warning", "critical"]:
                report["recommendations"].append(self._get_recommendation(metric))
        
        # Store for history
        self.metrics_history.append(report)
        
        return report
    
    def _get_recommendation(self, metric: HealthMetric) -> str:
        """Generate recommendation based on metric"""
        recommendations = {
            "cpu_usage": "Consider scaling up CPU resources or optimizing computational workloads",
            "memory_usage": "Consider increasing memory allocation or optimizing memory usage",
            "disk_usage": "Consider cleaning up old files or expanding disk space",
            "data_completeness": "Investigate data quality issues and implement data validation",
            "data_freshness_hours": "Check data ingestion pipeline for delays",
            "schema_compatibility": "Review schema changes and update data contracts",
            "model_accuracy": "Retrain model or investigate data drift issues",
            "model_inference_time": "Optimize model architecture or increase computational resources"
        }
        
        return recommendations.get(metric.name, f"Monitor {metric.name} closely and take corrective action")

# Example usage in pipeline
if __name__ == "__main__":
    # Initialize health checker
    health_checker = PipelineHealthChecker({})
    
    # Generate sample report
    report = health_checker.generate_health_report()
    print(json.dumps(report, indent=2))
