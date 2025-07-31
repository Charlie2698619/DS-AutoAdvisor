"""
Industrial-Grade Model Lifecycle Management
==========================================

Comprehensive model management including versioning, registry, deployment,
monitoring, and automated lifecycle operations.
"""

import os
import json
import pickle
import joblib
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import shutil
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.tracking import MlflowClient

class ModelStatus(Enum):
    """Model status in lifecycle"""
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    RETIRED = "retired"
    FAILED = "failed"

class ModelType(Enum):
    """Types of models supported"""
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    CUSTOM = "custom"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: Optional[float] = None
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    model_size: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

@dataclass
class ModelArtifact:
    """Model artifact information"""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    file_path: Optional[str] = None
    config_path: Optional[str] = None
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    data_schema: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.model_id:
            self.model_id = self._generate_model_id()
    
    def _generate_model_id(self) -> str:
        """Generate unique model ID"""
        timestamp = int(datetime.now().timestamp())
        content = f"{self.name}_{self.version}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

class ModelRegistry:
    """
    Central model registry for managing ML models
    
    Features:
    - Model versioning and storage
    - Metadata management
    - Performance tracking
    - Deployment status
    - Model comparison
    """
    
    def __init__(self, 
                 registry_path: str,
                 mlflow_tracking_uri: Optional[str] = None,
                 enable_mlflow: bool = True):
        
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.registry_path / "models"
        self.metadata_dir = self.registry_path / "metadata"
        self.configs_dir = self.registry_path / "configs"
        
        # Create directories
        for dir_path in [self.models_dir, self.metadata_dir, self.configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # MLflow integration
        self.enable_mlflow = enable_mlflow
        if enable_mlflow:
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            self.mlflow_client = MlflowClient()
    
    def register_model(self, 
                      model: Any,
                      name: str,
                      version: str,
                      model_type: ModelType,
                      metrics: ModelMetrics,
                      config: Optional[Dict[str, Any]] = None,
                      tags: Optional[List[str]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> ModelArtifact:
        """
        Register a new model in the registry
        
        Args:
            model: The trained model object
            name: Model name
            version: Model version
            model_type: Type of model
            metrics: Performance metrics
            config: Model configuration
            tags: Model tags
            metadata: Additional metadata
            
        Returns:
            ModelArtifact with registration details
        """
        
        # Create model artifact
        artifact = ModelArtifact(
            model_id="",  # Will be generated
            name=name,
            version=version,
            model_type=model_type,
            status=ModelStatus.VALIDATION,
            metrics=metrics,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Save model file
        model_filename = f"{artifact.model_id}.pkl"
        model_path = self.models_dir / model_filename
        
        try:
            if model_type in [ModelType.SKLEARN, ModelType.XGBOOST, ModelType.LIGHTGBM]:
                joblib.dump(model, model_path)
            else:
                # Use pickle as fallback
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            artifact.file_path = str(model_path)
            artifact.metadata['model_size'] = model_path.stat().st_size
            
            self.logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
        
        # Save configuration
        if config:
            config_filename = f"{artifact.model_id}_config.json"
            config_path = self.configs_dir / config_filename
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            artifact.config_path = str(config_path)
        
        # Save metadata
        metadata_filename = f"{artifact.model_id}_metadata.json"
        metadata_path = self.metadata_dir / metadata_filename
        
        with open(metadata_path, 'w') as f:
            json.dump(asdict(artifact), f, indent=2, default=str)
        
        # MLflow logging
        if self.enable_mlflow:
            self._log_to_mlflow(artifact, model, config)
        
        self.logger.info(f"Model {name} v{version} registered successfully with ID {artifact.model_id}")
        return artifact
    
    def _log_to_mlflow(self, artifact: ModelArtifact, model: Any, config: Optional[Dict]):
        """Log model to MLflow"""
        try:
            with mlflow.start_run(run_name=f"{artifact.name}_v{artifact.version}"):
                # Log parameters
                if config:
                    mlflow.log_params(config)
                
                # Log metrics
                mlflow.log_metrics(artifact.metrics.to_dict())
                
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=artifact.name
                )
                
                # Log tags
                for tag in artifact.tags:
                    mlflow.set_tag(tag, True)
                
                mlflow.set_tag("model_id", artifact.model_id)
                mlflow.set_tag("status", artifact.status.value)
                
        except Exception as e:
            self.logger.warning(f"Failed to log to MLflow: {e}")
    
    def load_model(self, model_id: str) -> Tuple[Any, ModelArtifact]:
        """
        Load model and its metadata
        
        Args:
            model_id: Model identifier
            
        Returns:
            Tuple of (model_object, model_artifact)
        """
        # Load metadata
        metadata_path = self.metadata_dir / f"{model_id}_metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Model {model_id} not found in registry")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Reconstruct artifact
        artifact = ModelArtifact(**metadata)
        
        # Load model
        if not artifact.file_path or not Path(artifact.file_path).exists():
            raise FileNotFoundError(f"Model file not found: {artifact.file_path}")
        
        try:
            if artifact.model_type in [ModelType.SKLEARN, ModelType.XGBOOST, ModelType.LIGHTGBM]:
                model = joblib.load(artifact.file_path)
            else:
                with open(artifact.file_path, 'rb') as f:
                    model = pickle.load(f)
            
            self.logger.info(f"Model {model_id} loaded successfully")
            return model, artifact
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def list_models(self, 
                   status: Optional[ModelStatus] = None,
                   name_pattern: Optional[str] = None) -> List[ModelArtifact]:
        """
        List models in registry
        
        Args:
            status: Filter by status
            name_pattern: Filter by name pattern
            
        Returns:
            List of model artifacts
        """
        models = []
        
        for metadata_file in self.metadata_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                artifact = ModelArtifact(**metadata)
                
                # Apply filters
                if status and artifact.status != status:
                    continue
                
                if name_pattern and name_pattern not in artifact.name:
                    continue
                
                models.append(artifact)
                
            except Exception as e:
                self.logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        
        # Sort by creation time (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        return models
    
    def promote_model(self, model_id: str, target_status: ModelStatus) -> bool:
        """
        Promote model to new status
        
        Args:
            model_id: Model identifier
            target_status: Target status
            
        Returns:
            Success status
        """
        try:
            _, artifact = self.load_model(model_id)
            
            # Validation logic
            if target_status == ModelStatus.PRODUCTION:
                if artifact.status != ModelStatus.STAGING:
                    raise ValueError("Can only promote staging models to production")
                
                # Additional production readiness checks
                if not self._validate_production_readiness(artifact):
                    raise ValueError("Model not ready for production")
            
            # Update status
            artifact.status = target_status
            
            # Save updated metadata
            metadata_path = self.metadata_dir / f"{model_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(artifact), f, indent=2, default=str)
            
            # Update MLflow if enabled
            if self.enable_mlflow:
                try:
                    # Update model stage in MLflow registry
                    model_version = self.mlflow_client.get_latest_versions(
                        artifact.name, stages=["None", "Staging", "Production"]
                    )[0]
                    
                    stage_mapping = {
                        ModelStatus.STAGING: "Staging",
                        ModelStatus.PRODUCTION: "Production",
                        ModelStatus.RETIRED: "Archived"
                    }
                    
                    if target_status in stage_mapping:
                        self.mlflow_client.transition_model_version_stage(
                            name=artifact.name,
                            version=model_version.version,
                            stage=stage_mapping[target_status]
                        )
                        
                except Exception as e:
                    self.logger.warning(f"Failed to update MLflow stage: {e}")
            
            self.logger.info(f"Model {model_id} promoted to {target_status.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to promote model {model_id}: {e}")
            return False
    
    def _validate_production_readiness(self, artifact: ModelArtifact) -> bool:
        """Validate if model is ready for production"""
        checks = []
        
        # Performance thresholds
        checks.append(artifact.metrics.accuracy > 0.8)
        checks.append(artifact.metrics.f1_score > 0.75)
        
        # Model file exists and is accessible
        checks.append(artifact.file_path and Path(artifact.file_path).exists())
        
        # Has required metadata
        checks.append('data_schema' in artifact.metadata)
        checks.append('training_data_hash' in artifact.metadata)
        
        # Recent enough (within last 30 days)
        checks.append((datetime.now() - artifact.created_at).days <= 30)
        
        return all(checks)
    
    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            model_ids: List of model IDs to compare
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_id in model_ids:
            try:
                _, artifact = self.load_model(model_id)
                
                row = {
                    'model_id': artifact.model_id,
                    'name': artifact.name,
                    'version': artifact.version,
                    'status': artifact.status.value,
                    'created_at': artifact.created_at,
                    **artifact.metrics.to_dict()
                }
                
                comparison_data.append(row)
                
            except Exception as e:
                self.logger.warning(f"Failed to load model {model_id} for comparison: {e}")
        
        return pd.DataFrame(comparison_data)
    
    def cleanup_old_models(self, keep_versions: int = 5) -> int:
        """
        Clean up old model versions
        
        Args:
            keep_versions: Number of versions to keep per model name
            
        Returns:
            Number of models cleaned up
        """
        models_by_name = {}
        
        # Group models by name
        for artifact in self.list_models():
            if artifact.name not in models_by_name:
                models_by_name[artifact.name] = []
            models_by_name[artifact.name].append(artifact)
        
        cleaned_count = 0
        
        for name, models in models_by_name.items():
            # Sort by creation time (newest first)
            models.sort(key=lambda x: x.created_at, reverse=True)
            
            # Keep production and staging models
            production_models = [m for m in models if m.status in [ModelStatus.PRODUCTION, ModelStatus.STAGING]]
            other_models = [m for m in models if m.status not in [ModelStatus.PRODUCTION, ModelStatus.STAGING]]
            
            # Remove old non-production models
            models_to_remove = other_models[keep_versions:]
            
            for model in models_to_remove:
                try:
                    # Remove model files
                    if model.file_path and Path(model.file_path).exists():
                        Path(model.file_path).unlink()
                    
                    if model.config_path and Path(model.config_path).exists():
                        Path(model.config_path).unlink()
                    
                    # Remove metadata
                    metadata_path = self.metadata_dir / f"{model.model_id}_metadata.json"
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    cleaned_count += 1
                    self.logger.info(f"Cleaned up model {model.model_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to cleanup model {model.model_id}: {e}")
        
        return cleaned_count

class ModelDeploymentManager:
    """
    Manages model deployment and serving
    """
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.logger = logging.getLogger(__name__)
        self.deployed_models = {}  # model_id -> deployment_info
    
    def deploy_model(self, 
                    model_id: str,
                    deployment_target: str = "local",
                    config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Deploy model to specified target
        
        Args:
            model_id: Model to deploy
            deployment_target: Where to deploy (local, kubernetes, sagemaker, etc.)
            config: Deployment configuration
            
        Returns:
            Deployment success status
        """
        try:
            model, artifact = self.registry.load_model(model_id)
            
            if deployment_target == "local":
                return self._deploy_local(model, artifact, config or {})
            elif deployment_target == "kubernetes":
                return self._deploy_kubernetes(model, artifact, config or {})
            elif deployment_target == "sagemaker":
                return self._deploy_sagemaker(model, artifact, config or {})
            else:
                self.logger.error(f"Unsupported deployment target: {deployment_target}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to deploy model {model_id}: {e}")
            return False
    
    def _deploy_local(self, model: Any, artifact: ModelArtifact, config: Dict) -> bool:
        """Deploy model locally"""
        # For local deployment, we just keep the model in memory
        self.deployed_models[artifact.model_id] = {
            'model': model,
            'artifact': artifact,
            'deployment_time': datetime.now(),
            'target': 'local'
        }
        
        self.logger.info(f"Model {artifact.model_id} deployed locally")
        return True
    
    def _deploy_kubernetes(self, model: Any, artifact: ModelArtifact, config: Dict) -> bool:
        """Deploy model to Kubernetes (placeholder)"""
        # Implementation would create Kubernetes deployment
        self.logger.warning("Kubernetes deployment not implemented")
        return False
    
    def _deploy_sagemaker(self, model: Any, artifact: ModelArtifact, config: Dict) -> bool:
        """Deploy model to AWS SageMaker (placeholder)"""
        # Implementation would create SageMaker endpoint
        self.logger.warning("SageMaker deployment not implemented")
        return False
    
    def predict(self, model_id: str, input_data: Union[pd.DataFrame, np.ndarray]) -> Any:
        """
        Make prediction using deployed model
        
        Args:
            model_id: Model identifier
            input_data: Input data for prediction
            
        Returns:
            Model predictions
        """
        if model_id not in self.deployed_models:
            raise ValueError(f"Model {model_id} not deployed")
        
        deployment_info = self.deployed_models[model_id]
        model = deployment_info['model']
        
        try:
            predictions = model.predict(input_data)
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed for model {model_id}: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize model registry
    registry = ModelRegistry(
        registry_path="models/registry",
        enable_mlflow=True
    )
    
    # Example model registration (placeholder)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create dummy model and data
    X, y = make_classification(n_samples=1000, n_features=20)
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Calculate metrics
    predictions = model.predict(X)
    metrics = ModelMetrics(
        accuracy=accuracy_score(y, predictions),
        precision=precision_score(y, predictions, average='weighted'),
        recall=recall_score(y, predictions, average='weighted'),
        f1_score=f1_score(y, predictions, average='weighted')
    )
    
    # Register model
    artifact = registry.register_model(
        model=model,
        name="example_classifier",
        version="1.0.0",
        model_type=ModelType.SKLEARN,
        metrics=metrics,
        tags=["classification", "example"]
    )
    
    print(f"Model registered with ID: {artifact.model_id}")
    
    # List models
    models = registry.list_models()
    print(f"Found {len(models)} models in registry")
