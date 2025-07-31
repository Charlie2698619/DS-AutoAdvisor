"""
Industrial-Grade Pipeline Integration
====================================

Integration layer that connects all industrial-grade components with the main pipeline.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime

# Import all industrial-grade components
try:
    from .health_checker import PipelineHealthChecker
    from .config_manager import AdvancedConfigManager, Environment
    from .model_lifecycle import ModelRegistry, ModelDeploymentManager, ModelMetrics, ModelType, ModelStatus
    from .security_compliance import SecurityManager, SecurityPolicy, UserRole
    from .scalability_manager import ScalabilityManager, ResourceLimits
except ImportError:
    # Fallback imports for when running as script
    import sys
    sys.path.append(str(Path(__file__).parent))
    from health_checker import PipelineHealthChecker
    from config_manager import AdvancedConfigManager, Environment
    from model_lifecycle import ModelRegistry, ModelDeploymentManager, ModelMetrics, ModelType, ModelStatus
    from security_compliance import SecurityManager, SecurityPolicy, UserRole
    from scalability_manager import ScalabilityManager, ResourceLimits

@dataclass
class IndustrialPipelineConfig:
    """Configuration for industrial-grade pipeline features"""
    
    # Environment settings
    environment: Environment = Environment.DEVELOPMENT
    
    # Component enablement flags
    enable_health_monitoring: bool = True
    enable_advanced_config: bool = True
    enable_model_lifecycle: bool = True
    enable_security: bool = True
    enable_scalability: bool = True
    
    # Resource limits
    max_memory_gb: float = 8.0
    max_cpu_percent: float = 80.0
    max_concurrent_jobs: int = 4
    
    # Security settings
    require_authentication: bool = False  # Disabled by default for development
    require_encryption: bool = False
    audit_log_path: str = "logs/audit.log"
    
    # Paths
    model_registry_path: str = "models/registry"
    cache_dir: str = "cache"
    config_dir: str = "config"

class IndustrialPipelineManager:
    """
    Main manager that integrates all industrial-grade components
    """
    
    def __init__(self, config: IndustrialPipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components based on configuration
        self.health_checker = None
        self.config_manager = None
        self.model_registry = None
        self.deployment_manager = None
        self.security_manager = None
        self.scalability_manager = None
        
        self._initialize_components()
        
        # Pipeline state
        self.current_stage = None
        self.pipeline_session_id = None
        self.active_user_session = None
    
    def _initialize_components(self):
        """Initialize industrial-grade components"""
        
        try:
            # Health monitoring
            if self.config.enable_health_monitoring:
                # Create health checker config
                health_config = {
                    "memory_threshold_gb": self.config.max_memory_gb * 0.8,  # 80% of max
                    "cpu_threshold_percent": self.config.max_cpu_percent * 0.8,  # 80% of max
                    "disk_threshold_percent": 85.0,
                    "check_interval_seconds": 30,
                    "enable_alerts": True,
                    "alert_cooldown_minutes": 5
                }
                self.health_checker = PipelineHealthChecker(config=health_config)
                self.logger.info("Health monitoring initialized")
            
            # Advanced configuration management
            if self.config.enable_advanced_config:
                base_config_path = Path(self.config.config_dir) / "unified_config.yaml"
                self.config_manager = AdvancedConfigManager(
                    base_config_path=str(base_config_path),
                    environment=self.config.environment,
                    secrets_backend="env"
                )
                self.logger.info("Advanced configuration management initialized")
            
            # Model lifecycle management
            if self.config.enable_model_lifecycle:
                self.model_registry = ModelRegistry(
                    registry_path=self.config.model_registry_path,
                    enable_mlflow=True
                )
                self.deployment_manager = ModelDeploymentManager(self.model_registry)
                self.logger.info("Model lifecycle management initialized")
            
            # Security and compliance
            if self.config.enable_security:
                security_policy = SecurityPolicy(
                    require_authentication=self.config.require_authentication,
                    require_encryption=self.config.require_encryption,
                    require_audit_logging=True
                )
                
                self.security_manager = SecurityManager(
                    policy=security_policy,
                    audit_log_path=self.config.audit_log_path
                )
                self.logger.info("Security and compliance initialized")
            
            # Scalability and performance
            if self.config.enable_scalability:
                resource_limits = ResourceLimits(
                    max_memory_gb=self.config.max_memory_gb,
                    max_cpu_percent=self.config.max_cpu_percent,
                    max_concurrent_jobs=self.config.max_concurrent_jobs
                )
                
                self.scalability_manager = ScalabilityManager(
                    resource_limits=resource_limits,
                    cache_dir=self.config.cache_dir
                )
                self.logger.info("Scalability and performance management initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize industrial components: {e}")
            raise
    
    def start_pipeline_session(self, user_id: str = "system", ip_address: str = "127.0.0.1") -> str:
        """
        Start a new pipeline session with security and monitoring
        
        Args:
            user_id: User identifier
            ip_address: Client IP address
            
        Returns:
            Pipeline session ID
        """
        
        self.pipeline_session_id = f"pipeline_{int(datetime.now().timestamp())}"
        
        # Authenticate user if security is enabled
        if self.security_manager and self.config.require_authentication:
            session_token = self.security_manager.access_control.authenticate_user(
                user_id=user_id,
                password="default_password",  # In production, use proper authentication
                ip_address=ip_address
            )
            
            if not session_token:
                raise PermissionError("Authentication failed")
            
            self.active_user_session = session_token
            
            # Set default role for development
            self.security_manager.access_control.set_user_role(
                user_id, UserRole.DATA_SCIENTIST
            )
        
        # Log session start
        if self.security_manager:
            self.security_manager.audit_logger.log_action(
                user_id=user_id,
                action="start_pipeline",
                resource="pipeline",
                result="success",
                ip_address=ip_address
            )
        
        # Initialize health monitoring
        if self.health_checker:
            self.health_checker.start_monitoring()
        
        self.logger.info(f"Pipeline session started: {self.pipeline_session_id}")
        return self.pipeline_session_id
    
    def execute_stage(self, 
                     stage_name: str,
                     data: pd.DataFrame,
                     processing_func: callable,
                     user_id: str = "system") -> Optional[pd.DataFrame]:
        """
        Execute a pipeline stage with all industrial-grade features
        
        Args:
            stage_name: Name of the pipeline stage
            data: Input data
            processing_func: Processing function
            user_id: User identifier
            
        Returns:
            Processed data or None if access denied
        """
        
        self.current_stage = stage_name
        self.logger.info(f"Executing stage: {stage_name}")
        
        try:
            # Security check
            if self.security_manager and self.active_user_session:
                if not self.security_manager.access_control.authorize_action(
                    self.active_user_session, "run_analysis"
                ):
                    self.logger.warning(f"Access denied for stage {stage_name}")
                    return None
                
                # Secure data processing
                data = self.security_manager.secure_data_processing(
                    data=data,
                    dataset_name=f"{stage_name}_data",
                    user_session=self.active_user_session,
                    operation=f"execute_{stage_name}"
                )
                
                if data is None:
                    return None
            
            # Health check before execution
            if self.health_checker:
                health_metrics = self.health_checker.check_system_health()
                # Check if any metric is critical
                critical_metrics = [m for m in health_metrics if m.status == 'critical']
                if critical_metrics:
                    self.logger.error(f"Critical health issues detected: {[m.name for m in critical_metrics]}")
                    raise RuntimeError(f"System health critical: {critical_metrics[0].name}")
            
            # Execute with scalability optimizations
            if self.scalability_manager:
                result = self.scalability_manager.optimize_pipeline_stage(
                    stage_name=stage_name,
                    data=data,
                    processing_func=processing_func,
                    enable_caching=True
                )
            else:
                result = processing_func(data)
            
            # Log successful execution
            if self.security_manager:
                # Safely get data dimensions
                try:
                    input_rows = len(data) if hasattr(data, '__len__') else 0
                    output_rows = len(result) if result is not None and hasattr(result, '__len__') else 0
                except:
                    input_rows, output_rows = 0, 0
                
                self.security_manager.audit_logger.log_action(
                    user_id=user_id,
                    action=f"execute_{stage_name}",
                    resource=f"stage_{stage_name}",
                    result="success",
                    additional_data={
                        'input_rows': input_rows,
                        'output_rows': output_rows
                    }
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Stage {stage_name} execution failed: {e}")
            
            # Log failure
            if self.security_manager:
                self.security_manager.audit_logger.log_action(
                    user_id=user_id,
                    action=f"execute_{stage_name}",
                    resource=f"stage_{stage_name}",
                    result="failure",
                    additional_data={'error': str(e)}
                )
            
            raise
    
    def register_model(self, 
                      model: Any,
                      model_name: str,
                      version: str,
                      metrics: Dict[str, float],
                      config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Register a trained model with lifecycle management
        
        Args:
            model: Trained model object
            model_name: Model name
            version: Model version
            metrics: Performance metrics
            config: Model configuration
            
        Returns:
            Model ID if successful, None otherwise
        """
        
        if not self.model_registry:
            self.logger.warning("Model registry not available")
            return None
        
        try:
            # Convert metrics dict to ModelMetrics object
            model_metrics = ModelMetrics(
                accuracy=metrics.get('accuracy', 0.0),
                precision=metrics.get('precision', 0.0),
                recall=metrics.get('recall', 0.0),
                f1_score=metrics.get('f1_score', 0.0)
            )
            
            # Register model
            artifact = self.model_registry.register_model(
                model=model,
                name=model_name,
                version=version,
                model_type=ModelType.SKLEARN,  # Default, should be detected
                metrics=model_metrics,
                config=config,
                tags=['pipeline_generated']
            )
            
            self.logger.info(f"Model registered: {model_name} v{version} (ID: {artifact.model_id})")
            return artifact.model_id
            
        except Exception as e:
            self.logger.error(f"Model registration failed: {e}")
            return None
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        
        status = {
            'session_id': self.pipeline_session_id,
            'current_stage': self.current_stage,
            'timestamp': datetime.now().isoformat(),
            'environment': self.config.environment.value
        }
        
        # Add component status
        if self.health_checker:
            try:
                health_status = self.health_checker.check_system_health()
                status['system_health'] = health_status
            except Exception as e:
                status['system_health'] = {'error': str(e)}
        
        if self.scalability_manager:
            try:
                performance_report = self.scalability_manager.get_performance_report()
                status['performance'] = performance_report
            except Exception as e:
                status['performance'] = {'error': str(e)}
        
        if self.model_registry:
            try:
                models = self.model_registry.list_models()
                status['model_registry'] = {
                    'total_models': len(models),
                    'production_models': len([m for m in models if m.status == ModelStatus.PRODUCTION])
                }
            except Exception as e:
                status['model_registry'] = {'error': str(e)}
        
        return status
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline report"""
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'pipeline_session_id': self.pipeline_session_id,
            'configuration': {
                'environment': self.config.environment.value,
                'enabled_components': {
                    'health_monitoring': self.config.enable_health_monitoring,
                    'advanced_config': self.config.enable_advanced_config,
                    'model_lifecycle': self.config.enable_model_lifecycle,
                    'security': self.config.enable_security,
                    'scalability': self.config.enable_scalability
                }
            }
        }
        
        # Add detailed component reports
        try:
            if self.health_checker:
                report['health_report'] = self.health_checker.generate_health_report()
            
            if self.scalability_manager:
                report['performance_report'] = self.scalability_manager.get_performance_report()
            
            if self.model_registry:
                models = self.model_registry.list_models()
                if models:
                    model_comparison = self.model_registry.compare_models([m.model_id for m in models[-5:]])
                    report['model_report'] = {
                        'total_models': len(models),
                        'recent_models': model_comparison.to_dict('records') if not model_comparison.empty else []
                    }
            
            if self.security_manager:
                # Get audit log summary (last 100 entries)
                audit_logs = self.security_manager.audit_logger.get_audit_logs()
                report['security_report'] = {
                    'total_audit_entries': len(audit_logs),
                    'recent_actions': [
                        {
                            'timestamp': entry.timestamp.isoformat(),
                            'user_id': entry.user_id,
                            'action': entry.action,
                            'result': entry.result
                        }
                        for entry in audit_logs[-10:]  # Last 10 entries
                    ]
                }
        
        except Exception as e:
            report['report_generation_error'] = str(e)
        
        return report
    
    def cleanup_session(self):
        """Clean up pipeline session"""
        
        try:
            # Stop health monitoring
            if self.health_checker:
                # Health checker runs in background, no explicit stop needed
                pass
            
            # Logout user session
            if self.security_manager and self.active_user_session:
                self.security_manager.access_control.logout_user(self.active_user_session)
            
            # Force garbage collection
            if self.scalability_manager:
                self.scalability_manager.memory_manager.force_garbage_collection()
            
            self.logger.info(f"Pipeline session cleaned up: {self.pipeline_session_id}")
            
        except Exception as e:
            self.logger.error(f"Session cleanup error: {e}")

# Factory function to create industrial pipeline manager
def create_industrial_pipeline(environment: Environment = Environment.DEVELOPMENT,
                             enable_all_features: bool = True) -> IndustrialPipelineManager:
    """
    Factory function to create industrial pipeline manager
    
    Args:
        environment: Environment type
        enable_all_features: Whether to enable all industrial features
        
    Returns:
        Configured IndustrialPipelineManager
    """
    
    # Environment-specific configurations
    config_overrides = {
        Environment.DEVELOPMENT: {
            'require_authentication': False,
            'require_encryption': False,
            'max_memory_gb': 4.0
        },
        Environment.STAGING: {
            'require_authentication': True,
            'require_encryption': False,
            'max_memory_gb': 8.0
        },
        Environment.PRODUCTION: {
            'require_authentication': True,
            'require_encryption': True,
            'max_memory_gb': 16.0
        }
    }
    
    # Create configuration
    config = IndustrialPipelineConfig(
        environment=environment,
        enable_health_monitoring=enable_all_features,
        enable_advanced_config=enable_all_features,
        enable_model_lifecycle=enable_all_features,
        enable_security=enable_all_features,
        enable_scalability=enable_all_features,
        **config_overrides.get(environment, {})
    )
    
    return IndustrialPipelineManager(config)

# Example usage
if __name__ == "__main__":
    # Create industrial pipeline for development
    pipeline_manager = create_industrial_pipeline(
        environment=Environment.DEVELOPMENT,
        enable_all_features=True
    )
    
    # Start pipeline session
    session_id = pipeline_manager.start_pipeline_session(
        user_id="developer_1",
        ip_address="127.0.0.1"
    )
    
    print(f"Pipeline session started: {session_id}")
    
    # Example data processing
    sample_data = pd.DataFrame({
        'id': range(1000),
        'value': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    def sample_processing(df):
        return df.copy()
    
    # Execute stage
    result = pipeline_manager.execute_stage(
        stage_name="sample_processing",
        data=sample_data,
        processing_func=sample_processing
    )
    
    # Get pipeline status
    status = pipeline_manager.get_pipeline_status()
    print(f"Pipeline status: {status['system_health']['overall_health']}")
    
    # Generate comprehensive report
    report = pipeline_manager.generate_comprehensive_report()
    print(f"Report generated with {len(report)} sections")
    
    # Cleanup
    pipeline_manager.cleanup_session()
