"""
DS-AutoAdvisor v2.0 MLflow Integration
=====================================

This module integrates MLflow for comprehensive model versioning, experiment tracking,
and model registry management in DS-AutoAdvisor v2.0.

Features:
- Experiment tracking for all pipeline runs
- Model versioning with automatic registration
- Performance comparison across runs
- Model deployment lifecycle management
- Artifact storage and retrieval
- Integration with metadata system
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# Optional imports for different ML frameworks
try:
    import mlflow.xgboost
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import mlflow.lightgbm
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import mlflow.pytorch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

try:
    import mlflow.tensorflow
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field

@dataclass
class MLflowConfig:
    """Configuration for MLflow integration"""
    tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    experiment_name: str = "ds-autoadvisor"
    artifact_location: str = "./mlruns"
    model_registry_uri: str = None  # Uses tracking_uri if None
    auto_log: bool = True
    log_system_metrics: bool = True
    log_artifacts: bool = True

@dataclass
class ModelInfo:
    """Information about a registered model"""
    name: str
    version: int
    stage: str  # "Staging", "Production", "Archived"
    run_id: str
    model_uri: str
    metrics: Dict[str, float]
    tags: Dict[str, str]
    creation_timestamp: datetime
    last_updated_timestamp: datetime

class MLflowManager:
    """
    Central MLflow management for DS-AutoAdvisor v2.0
    
    Features:
    - Experiment and run management
    - Model registration and versioning
    - Artifact logging and retrieval
    - Performance tracking and comparison
    - Model lifecycle management
    """
    
    def __init__(self, config: MLflowConfig):
        """
        Initialize MLflow manager
        
        Args:
            config: MLflow configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set up MLflow
        self._setup_mlflow()
        
        # Initialize MLflow client
        self.client = MlflowClient(
            tracking_uri=config.tracking_uri,
            registry_uri=config.model_registry_uri or config.tracking_uri
        )
        
        # Set or create experiment
        self.experiment = self._setup_experiment()
        
        self.logger.info(f"MLflow manager initialized with experiment: {self.experiment.name}")
    
    def _setup_mlflow(self):
        """Setup MLflow configuration"""
        # Set tracking URI
        mlflow.set_tracking_uri(self.config.tracking_uri)
        
        # Enable autologging if configured
        if self.config.auto_log:
            mlflow.sklearn.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True
            )
            
            # Enable autologging for other frameworks if available
            try:
                if HAS_XGBOOST:
                    mlflow.xgboost.autolog()
                if HAS_LIGHTGBM:
                    mlflow.lightgbm.autolog()
            except Exception as e:
                self.logger.debug(f"Optional autolog setup failed: {e}")
        
        # Configure system metrics logging
        if self.config.log_system_metrics:
            mlflow.enable_system_metrics_logging()
    
    def _setup_experiment(self):
        """Setup or get MLflow experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=self.config.experiment_name,
                    artifact_location=self.config.artifact_location
                )
                experiment = mlflow.get_experiment(experiment_id)
            
            mlflow.set_experiment(self.config.experiment_name)
            return experiment
            
        except Exception as e:
            self.logger.error(f"Failed to setup MLflow experiment: {e}")
            raise
    
    def start_pipeline_run(self, pipeline_config: Dict[str, Any], 
                          run_name: Optional[str] = None) -> str:
        """
        Start a new MLflow run for pipeline execution
        
        Args:
            pipeline_config: Pipeline configuration
            run_name: Optional name for the run
            
        Returns:
            MLflow run ID
        """
        run_name = run_name or f"ds-autoadvisor-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Start MLflow run
        run = mlflow.start_run(run_name=run_name)
        
        # Log pipeline configuration
        mlflow.log_params({
            "pipeline_version": pipeline_config.get('global', {}).get('version', '2.0'),
            "data_input_path": pipeline_config.get('global', {}).get('data_input_path'),
            "target_column": pipeline_config.get('global', {}).get('target_column'),
            "random_state": pipeline_config.get('global', {}).get('random_state'),
        })
        
        # Log stage configurations
        for stage in ['profiling', 'data_cleaning', 'ml_advisory', 'model_training', 'model_evaluation']:
            if stage in pipeline_config:
                stage_config = pipeline_config[stage]
                for key, value in stage_config.items():
                    if isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(f"{stage}_{key}", value)
        
        # Set tags
        mlflow.set_tags({
            "pipeline_version": "2.0",
            "framework": "ds-autoadvisor",
            "run_type": "full_pipeline"
        })
        
        self.logger.info(f"Started MLflow run: {run.info.run_id}")
        return run.info.run_id
    
    def log_data_quality_metrics(self, quality_report):
        """Log data quality assessment metrics"""
        if not mlflow.active_run():
            self.logger.warning("No active MLflow run for logging data quality metrics")
            return
        
        # Log quality scores
        mlflow.log_metric("data_quality_overall_score", quality_report.overall_score)
        mlflow.log_metric("data_quality_total_issues", quality_report.total_issues)
        
        # Log issues by severity
        for severity, count in quality_report.issues_by_severity.items():
            mlflow.log_metric(f"data_quality_issues_{severity}", count)
        
        # Log column-wise scores
        for column, score in quality_report.column_scores.items():
            mlflow.log_metric(f"column_quality_{column}", score)
        
        # Log quality report as artifact
        quality_report_path = "data_quality_report.json"
        with open(quality_report_path, 'w') as f:
            json.dump({
                "overall_score": quality_report.overall_score,
                "total_issues": quality_report.total_issues,
                "issues_by_severity": quality_report.issues_by_severity,
                "column_scores": quality_report.column_scores,
                "recommendations": quality_report.recommendations,
                "metadata": quality_report.metadata
            }, f, indent=2, default=str)
        
        mlflow.log_artifact(quality_report_path, "data_quality")
        os.remove(quality_report_path)  # Clean up
    
    def log_stage_execution(self, stage_name: str, stage_result, execution_time: float):
        """Log individual stage execution metrics"""
        if not mlflow.active_run():
            self.logger.warning(f"No active MLflow run for logging stage {stage_name}")
            return
        
        # Log stage metrics
        mlflow.log_metric(f"stage_{stage_name}_execution_time", execution_time)
        mlflow.log_metric(f"stage_{stage_name}_success", 1 if stage_result.success else 0)
        mlflow.log_metric(f"stage_{stage_name}_outputs_count", len(stage_result.outputs_created))
        
        # Log stage artifacts
        for output_path in stage_result.outputs_created:
            if Path(output_path).exists():
                mlflow.log_artifact(output_path, f"stage_outputs/{stage_name}")
        
        # Log stage-specific metrics
        if stage_result.artifacts:
            for key, value in stage_result.artifacts.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"stage_{stage_name}_{key}", value)
                elif isinstance(value, str):
                    mlflow.log_param(f"stage_{stage_name}_{key}", value)
    
    def log_model_training_results(self, training_results: List[Any]) -> Dict[str, str]:
        """
        Log model training results and register models
        
        Args:
            training_results: List of training result objects
            
        Returns:
            Dictionary mapping model names to registered model URIs
        """
        if not mlflow.active_run():
            self.logger.warning("No active MLflow run for logging models")
            return {}
        
        registered_models = {}
        
        for result in training_results:
            try:
                model_name = result.model_name
                model = result.model
                metrics = result.metrics
                
                # Log model metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"model_{model_name}_{metric_name}", metric_value)
                
                # Log model hyperparameters
                if hasattr(result, 'hyperparameters'):
                    for param_name, param_value in result.hyperparameters.items():
                        mlflow.log_param(f"model_{model_name}_{param_name}", param_value)
                
                # Log model
                model_uri = self._log_model(model, model_name, result)
                if model_uri:
                    registered_models[model_name] = model_uri
                
            except Exception as e:
                self.logger.error(f"Failed to log model {getattr(result, 'model_name', 'unknown')}: {e}")
        
        return registered_models
    
    def _log_model(self, model, model_name: str, result) -> Optional[str]:
        """Log individual model to MLflow"""
        try:
            # Determine model type and log accordingly
            model_class = type(model).__name__
            
            if 'XGB' in model_class and HAS_XGBOOST:
                model_info = mlflow.xgboost.log_model(
                    model, 
                    f"models/{model_name}",
                    registered_model_name=f"ds_autoadvisor_{model_name}"
                )
            elif ('LightGBM' in model_class or 'LGBM' in model_class) and HAS_LIGHTGBM:
                model_info = mlflow.lightgbm.log_model(
                    model,
                    f"models/{model_name}",
                    registered_model_name=f"ds_autoadvisor_{model_name}"
                )
            else:
                # Default to sklearn
                model_info = mlflow.sklearn.log_model(
                    model,
                    f"models/{model_name}",
                    registered_model_name=f"ds_autoadvisor_{model_name}"
                )
            
            self.logger.info(f"Logged model {model_name} with URI: {model_info.model_uri}")
            return model_info.model_uri
            
        except Exception as e:
            self.logger.error(f"Failed to log model {model_name}: {e}")
            return None
    
    def log_evaluation_results(self, evaluation_results: Dict[str, Any]):
        """Log model evaluation results"""
        if not mlflow.active_run():
            self.logger.warning("No active MLflow run for logging evaluation results")
            return
        
        # Log evaluation metrics
        for model_name, model_results in evaluation_results.items():
            if isinstance(model_results, dict):
                for metric_name, metric_value in model_results.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(f"eval_{model_name}_{metric_name}", metric_value)
        
        # Log evaluation artifacts (plots, reports)
        evaluation_dir = Path("evaluation_results")
        if evaluation_dir.exists():
            for eval_file in evaluation_dir.glob("*"):
                if eval_file.is_file():
                    mlflow.log_artifact(str(eval_file), "evaluation")
    
    def end_pipeline_run(self, success: bool, error_message: Optional[str] = None):
        """End the current pipeline run"""
        if not mlflow.active_run():
            self.logger.warning("No active MLflow run to end")
            return
        
        # Log final pipeline status
        mlflow.log_metric("pipeline_success", 1 if success else 0)
        
        if error_message:
            mlflow.set_tag("error_message", error_message)
            mlflow.set_tag("status", "failed")
        else:
            mlflow.set_tag("status", "completed")
        
        # End the run
        mlflow.end_run()
        self.logger.info("Ended MLflow run")
    
    def get_best_models(self, metric_name: str = "accuracy", 
                       top_k: int = 5) -> List[ModelInfo]:
        """
        Get top performing models from the registry
        
        Args:
            metric_name: Metric to rank by
            top_k: Number of top models to return
            
        Returns:
            List of top performing models
        """
        try:
            # Get all registered models
            registered_models = self.client.search_registered_models()
            
            model_infos = []
            
            for rm in registered_models:
                for mv in rm.latest_versions:
                    if mv.current_stage in ["Staging", "Production"]:
                        # Get run metrics
                        run = self.client.get_run(mv.run_id)
                        metrics = run.data.metrics
                        
                        if metric_name in metrics:
                            model_info = ModelInfo(
                                name=rm.name,
                                version=int(mv.version),
                                stage=mv.current_stage,
                                run_id=mv.run_id,
                                model_uri=mv.source,
                                metrics=metrics,
                                tags=mv.tags or {},
                                creation_timestamp=datetime.fromtimestamp(mv.creation_timestamp / 1000),
                                last_updated_timestamp=datetime.fromtimestamp(mv.last_updated_timestamp / 1000)
                            )
                            model_infos.append(model_info)
            
            # Sort by metric and return top k
            model_infos.sort(key=lambda x: x.metrics.get(metric_name, 0), reverse=True)
            return model_infos[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to get best models: {e}")
            return []
    
    def compare_pipeline_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple pipeline runs
        
        Args:
            run_ids: List of MLflow run IDs to compare
            
        Returns:
            DataFrame with comparison results
        """
        try:
            comparison_data = []
            
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                
                run_data = {
                    "run_id": run_id,
                    "run_name": run.data.tags.get("mlflow.runName", "Unknown"),
                    "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
                    "status": run.info.status.string,
                    **run.data.metrics,
                    **{f"param_{k}": v for k, v in run.data.params.items()}
                }
                
                comparison_data.append(run_data)
            
            return pd.DataFrame(comparison_data)
            
        except Exception as e:
            self.logger.error(f"Failed to compare runs: {e}")
            return pd.DataFrame()
    
    def promote_model_to_production(self, model_name: str, version: int):
        """Promote a model version to production"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            self.logger.info(f"Promoted {model_name} v{version} to Production")
            
        except Exception as e:
            self.logger.error(f"Failed to promote model: {e}")
    
    def archive_model(self, model_name: str, version: int):
        """Archive a model version"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Archived"
            )
            self.logger.info(f"Archived {model_name} v{version}")
            
        except Exception as e:
            self.logger.error(f"Failed to archive model: {e}")
    
    def load_model(self, model_name: str, version: Optional[int] = None, 
                   stage: Optional[str] = None):
        """
        Load a model from the registry
        
        Args:
            model_name: Name of the registered model
            version: Specific version to load
            stage: Stage to load from ("Production", "Staging")
            
        Returns:
            Loaded model object
        """
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            # Try different loading methods based on model type
            try:
                return mlflow.sklearn.load_model(model_uri)
            except:
                if HAS_XGBOOST:
                    try:
                        return mlflow.xgboost.load_model(model_uri)
                    except:
                        pass
                if HAS_LIGHTGBM:
                    try:
                        return mlflow.lightgbm.load_model(model_uri)
                    except:
                        pass
                # Fallback to pyfunc
                return mlflow.pyfunc.load_model(model_uri)
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def get_model_lineage(self, model_name: str, version: int) -> Dict[str, Any]:
        """Get lineage information for a specific model"""
        try:
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)
            
            lineage = {
                "model_name": model_name,
                "version": version,
                "run_id": model_version.run_id,
                "creation_timestamp": model_version.creation_timestamp,
                "stage": model_version.current_stage,
                "run_metrics": run.data.metrics,
                "run_params": run.data.params,
                "run_tags": run.data.tags,
                "artifacts": []
            }
            
            # Get artifacts
            try:
                artifacts = self.client.list_artifacts(model_version.run_id)
                lineage["artifacts"] = [artifact.path for artifact in artifacts]
            except Exception as e:
                self.logger.warning(f"Could not get artifacts for {model_name} v{version}: {e}")
            
            return lineage
            
        except Exception as e:
            self.logger.error(f"Failed to get model lineage: {e}")
            return {}
    
    def search_experiments(self, filter_string: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search experiments with optional filtering"""
        try:
            experiments = self.client.search_experiments(
                view_type=ViewType.ACTIVE_ONLY,
                filter_string=filter_string
            )
            
            return [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "creation_time": datetime.fromtimestamp(exp.creation_time / 1000) if exp.creation_time else None,
                    "last_update_time": datetime.fromtimestamp(exp.last_update_time / 1000) if exp.last_update_time else None,
                    "tags": exp.tags
                }
                for exp in experiments
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to search experiments: {e}")
            return []


class MLflowIntegration:
    """
    High-level MLflow integration for DS-AutoAdvisor pipeline
    """
    
    def __init__(self, config_manager):
        """
        Initialize MLflow integration
        
        Args:
            config_manager: Enhanced configuration manager
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Get MLflow configuration
        mlflow_config = self._get_mlflow_config()
        
        # Initialize MLflow manager
        self.mlflow_manager = MLflowManager(mlflow_config)
        
        self.current_run_id = None
    
    def _get_mlflow_config(self) -> MLflowConfig:
        """Get MLflow configuration from config manager"""
        mlflow_settings = self.config_manager.get_config('infrastructure', 'mlflow', {})
        
        return MLflowConfig(
            tracking_uri=mlflow_settings.get('tracking_uri', "sqlite:///mlruns/mlflow.db"),
            experiment_name=mlflow_settings.get('experiment_name', "ds-autoadvisor"),
            artifact_location=mlflow_settings.get('artifact_location', "./mlruns"),
            model_registry_uri=mlflow_settings.get('model_registry_uri'),
            auto_log=mlflow_settings.get('auto_log', True),
            log_system_metrics=mlflow_settings.get('log_system_metrics', True),
            log_artifacts=mlflow_settings.get('log_artifacts', True)
        )
    
    def start_pipeline_tracking(self, pipeline_config: Dict[str, Any], 
                              run_name: Optional[str] = None) -> str:
        """Start MLflow tracking for pipeline run"""
        if self.config_manager.is_feature_enabled('mlflow_tracking'):
            self.current_run_id = self.mlflow_manager.start_pipeline_run(
                pipeline_config, run_name
            )
            return self.current_run_id
        return None
    
    def log_data_quality(self, quality_report):
        """Log data quality metrics if tracking enabled"""
        if self.config_manager.is_feature_enabled('mlflow_tracking') and self.current_run_id:
            self.mlflow_manager.log_data_quality_metrics(quality_report)
    
    def log_stage_result(self, stage_name: str, stage_result, execution_time: float):
        """Log stage execution results if tracking enabled"""
        if self.config_manager.is_feature_enabled('mlflow_tracking') and self.current_run_id:
            self.mlflow_manager.log_stage_execution(stage_name, stage_result, execution_time)
    
    def log_training_results(self, training_results: List[Any]) -> Dict[str, str]:
        """Log model training results and register models if enabled"""
        if (self.config_manager.is_feature_enabled('model_versioning') and 
            self.config_manager.is_feature_enabled('mlflow_tracking') and 
            self.current_run_id):
            return self.mlflow_manager.log_model_training_results(training_results)
        return {}
    
    def log_evaluation_results(self, evaluation_results: Dict[str, Any]):
        """Log evaluation results if tracking enabled"""
        if self.config_manager.is_feature_enabled('mlflow_tracking') and self.current_run_id:
            self.mlflow_manager.log_evaluation_results(evaluation_results)
    
    def end_pipeline_tracking(self, success: bool, error_message: Optional[str] = None):
        """End MLflow tracking for pipeline run"""
        if self.config_manager.is_feature_enabled('mlflow_tracking') and self.current_run_id:
            self.mlflow_manager.end_pipeline_run(success, error_message)
            self.current_run_id = None
    
    def get_historical_performance(self, metric_name: str = "accuracy") -> List[ModelInfo]:
        """Get historical model performance for meta-learning"""
        if self.config_manager.is_feature_enabled('model_versioning'):
            return self.mlflow_manager.get_best_models(metric_name)
        return []
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple pipeline runs"""
        if self.config_manager.is_feature_enabled('mlflow_tracking'):
            return self.mlflow_manager.compare_pipeline_runs(run_ids)
        return pd.DataFrame()


# Global MLflow integration instance
_mlflow_integration: Optional[MLflowIntegration] = None

def get_mlflow_integration(config_manager) -> MLflowIntegration:
    """Get global MLflow integration instance"""
    global _mlflow_integration
    
    if _mlflow_integration is None:
        _mlflow_integration = MLflowIntegration(config_manager)
    
    return _mlflow_integration
