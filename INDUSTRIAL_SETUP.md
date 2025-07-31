# Industrial-Grade DS-AutoAdvisor Setup Guide

## Overview

Your DS-AutoAdvisor pipeline has been enhanced with industrial-grade features for production deployment. This guide explains how to set up and use these advanced capabilities.

## Industrial Features Added

### 1. **Health Monitoring & Observability** 🏥
- Real-time system health monitoring
- Performance metrics tracking
- Resource usage alerts
- Automated health checks

### 2. **Advanced Configuration Management** ⚙️
- Environment-specific configurations (dev/staging/prod)
- Secret management integration
- Dynamic configuration updates
- Configuration validation

### 3. **Model Lifecycle Management** 🔄
- Centralized model registry
- Model versioning and metadata
- Automated deployment pipeline
- Performance tracking and comparison

### 4. **Security & Compliance** 🔒
- Role-based access control
- Data encryption and anonymization
- Comprehensive audit logging
- Data privacy compliance (GDPR ready)

### 5. **Scalability & Performance** 🚀
- Intelligent parallel processing
- Memory optimization
- Caching system
- Resource management

## Quick Start - Industrial Mode

### Basic Industrial Mode
```bash
# Run with all industrial features enabled
python complete_pipeline.py --industrial --config config/unified_config.yaml
```

### Development vs Production
```bash
# Development (default - limited security)
python complete_pipeline.py --industrial

# Production (full security and monitoring)
# Set environment in config: environment: "production"
python complete_pipeline.py --industrial --automated
```

## Installation Requirements

### Additional Dependencies
```bash
# Install additional packages for industrial features
pip install cryptography psutil mlflow evidently

# Optional: For advanced monitoring
pip install prometheus-client grafana-api
```

### Environment Setup
```bash
# Create directories for industrial features
mkdir -p logs cache models/registry

# Set environment variables for secrets (optional)
export DS_AUTOADVISOR_SECURITY__ENCRYPTION_KEY="your_secret_key"
export DS_AUTOADVISOR_MONITORING__ENABLE_MLFLOW="true"
```

## Configuration

### Environment-Specific Configs

Create environment-specific configuration files:

**config/unified_config.development.yaml**
```yaml
global:
  environment: "development"
  human_intervention:
    mode: "interactive"
    enabled: true

security:
  require_authentication: false
  require_encryption: false

monitoring:
  enabled: true
  detailed_logging: true
```

**config/unified_config.production.yaml** 
```yaml
global:
  environment: "production"
  human_intervention:
    mode: "fully_automated"
    enabled: false

security:
  require_authentication: true
  require_encryption: true
  audit_logging: true

monitoring:
  enabled: true
  alerting: true
  retention_days: 365
```

## Industrial Pipeline Usage

### 1. Health Monitoring
```python
from src.monitoring.industrial_integration import create_industrial_pipeline, Environment

# Create industrial pipeline
pipeline = create_industrial_pipeline(Environment.PRODUCTION)

# Start session and get health status
session_id = pipeline.start_pipeline_session()
health_status = pipeline.get_pipeline_status()
print(f"System Health: {health_status['system_health']['overall_health']}")
```

### 2. Model Registry
```python
# Register a model
model_id = pipeline.register_model(
    model=trained_model,
    model_name="customer_churn_predictor",
    version="v1.2.0",
    metrics={"accuracy": 0.85, "f1_score": 0.82}
)

# Deploy model
pipeline.deployment_manager.deploy_model(model_id, "production")
```

### 3. Security Features
```python
# The security features are automatically enabled in industrial mode
# User authentication, data encryption, and audit logging happen transparently
```

### 4. Performance Optimization
```python
# Automatic parallel processing and memory optimization
# Caching is enabled by default for expensive operations
```

## Monitoring & Alerts

### System Health Dashboard
```python
# Get comprehensive status report
status = pipeline.generate_comprehensive_report()

# Key metrics to monitor:
# - system_health.overall_health
# - performance.memory_usage
# - performance.cpu_usage
# - model_registry.production_models
```

### Log Analysis
```bash
# Check system logs
tail -f logs/pipeline.log

# Check audit logs
tail -f logs/audit.log

# Check health monitoring
tail -f logs/health_monitor.log
```

## Production Deployment Checklist

### ✅ Pre-Production
- [ ] Environment-specific config created
- [ ] Security credentials configured
- [ ] Resource limits set appropriately
- [ ] Monitoring alerts configured
- [ ] Backup strategy in place

### ✅ Production Ready
- [ ] Authentication enabled
- [ ] Data encryption enabled
- [ ] Audit logging enabled
- [ ] Health monitoring active
- [ ] Model registry configured
- [ ] Performance tuning complete

### ✅ Post-Deployment
- [ ] Monitor system health dashboard
- [ ] Review audit logs regularly
- [ ] Track model performance
- [ ] Set up alerting rules
- [ ] Plan capacity scaling

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install cryptography psutil mlflow
   ```

2. **Permission Errors**
   ```bash
   # Create required directories
   mkdir -p logs cache models/registry
   chmod 755 logs cache models/registry
   ```

3. **Memory Issues**
   ```yaml
   # Adjust resource limits in config
   scalability:
     max_memory_gb: 4.0  # Reduce if needed
     max_concurrent_jobs: 2  # Reduce for smaller systems
   ```

4. **Authentication Issues**
   ```yaml
   # Disable authentication for development
   security:
     require_authentication: false
   ```

## Performance Tuning

### Memory Optimization
- Industrial mode automatically optimizes DataFrame memory usage
- Implements intelligent garbage collection
- Provides memory usage monitoring

### Parallel Processing
- Automatically detects optimal worker count
- Supports thread-based and process-based parallelism
- Includes smart chunking for large datasets

### Caching
- Automatic caching of expensive operations
- Configurable TTL (time-to-live)
- Intelligent cache cleanup

## Support & Documentation

### Key Files
- `src/monitoring/health_checker.py` - System health monitoring
- `src/monitoring/config_manager.py` - Advanced configuration
- `src/monitoring/model_lifecycle.py` - Model management
- `src/monitoring/security_compliance.py` - Security features
- `src/monitoring/scalability_manager.py` - Performance optimization
- `src/monitoring/industrial_integration.py` - Main integration layer

### Getting Help
1. Check the logs in `logs/` directory
2. Review configuration files
3. Use health monitoring dashboard
4. Check system resource usage

## Example: Full Industrial Workflow

```bash
# 1. Set up environment
export DS_AUTOADVISOR_ENV="production"

# 2. Run with industrial features
python complete_pipeline.py \
  --industrial \
  --automated \
  --config config/unified_config.production.yaml

# 3. Monitor progress
tail -f logs/pipeline.log

# 4. Check final report
cat pipeline_outputs/reports/comprehensive_analysis_report.json
```

Your pipeline is now ready for industrial-grade deployment with enterprise-level monitoring, security, and scalability features! 🏭✨
