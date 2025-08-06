# 🏭 DS-AutoAdvisor v2.0 - Production Setup Guide

## Overview

DS-AutoAdvisor v2.0 comes with enterprise-ready features for production deployment. This guide covers how to deploy and monitor your ML pipeline in production environments.

## 🚀 **Production-Ready Features**

### ✅ **MLflow Integration**
- Experiment tracking and versioning
- Model registry and artifact management
- Performance comparison and history
- Automated model deployment

### ✅ **Enhanced Data Quality System**
- Advanced v2.0 quality assessment
- Automated data validation rules
- Quality scoring and recommendations
- Data lineage tracking

### ✅ **Metadata & Lineage Tracking**
- Complete data transformation tracking
- Pipeline execution history
- Performance audit trails
- Configuration versioning

### ✅ **Plugin Architecture**
- Extensible feature selection plugins
- Custom data processing components
- Modular system design
- Easy integration of new algorithms

## 📋 **Production Deployment Steps**

### **1. Environment Setup**
```bash
# Install production dependencies
pip install mlflow psutil evidently

# Setup MLflow tracking
python setup/setup_mlflow.py --start-ui

# Create production directories
mkdir -p logs/production models/production evaluation_results/production
```

### **2. Configuration for Production**
Edit `config/unified_config_v2.yaml`:

```yaml
global:
  environment: "production"
  data_input_path: "data/production_data.csv"
  
infrastructure:
  metadata:
    enabled: true
    connection_string: "sqlite:///metadata/production.db"
    
  mlflow:
    enabled: true
    tracking_uri: "http://localhost:5000"
    experiment_name: "production-ml-pipeline"
    
workflow:
  feature_flags:
    mlflow_tracking: true
    data_quality_v2: true
    plugin_system: true
    metadata_tracking: true
```

### **3. Data Quality Validation**
```bash
# Step 1: Validate your production data
python 1_inspect_data.py --data data/production_data.csv --detailed

# Check data quality scores and recommendations
# Adjust configuration based on quality report
```

### **4. Pipeline Testing**
```bash
# Step 2: Test complete pipeline with production data
python 2_test_pipeline.py --data data/production_data.csv

# Validate each stage interactively
# Ensure all stages complete successfully
```

### **5. Production Execution**
```bash
# Step 3: Run production pipeline with full tracking
python 3_run_pipeline.py --environment production

# Monitor execution in MLflow UI: http://localhost:5000
```

## 🔧 **Configuration Management**

### **Environment-Specific Configs**

Create separate configs for different environments:

**config/unified_config_development.yaml**
```yaml
global:
  environment: "development"
infrastructure:
  metadata:
    enabled: true
  mlflow:
    enabled: false  # Disable for faster dev cycles
workflow:
  feature_flags:
    data_quality_v2: true
    plugin_system: false  # Disable plugins in dev
```

**config/unified_config_production.yaml**
```yaml
global:
  environment: "production"
infrastructure:
  metadata:
    enabled: true
    connection_string: "sqlite:///metadata/production.db"
  mlflow:
    enabled: true
    tracking_uri: "http://production-mlflow:5000"
workflow:
  feature_flags:
    mlflow_tracking: true
    data_quality_v2: true
    plugin_system: true
    metadata_tracking: true
```

### **Run with Specific Config**
```bash
# Development
python 3_run_pipeline.py --config config/unified_config_development.yaml

# Production
python 3_run_pipeline.py --config config/unified_config_production.yaml
```

## 📊 **Monitoring & Observability**

### **MLflow Dashboard**
```bash
# Start MLflow UI for monitoring
python setup/setup_mlflow.py --start-ui

# Access at: http://localhost:5000
# Monitor:
# - Experiment runs and metrics
# - Model performance comparison
# - Artifact storage and versioning
# - Model registry and deployment status
```

### **Data Quality Monitoring**
```python
# Check data quality programmatically
from src.data_quality_system.enhanced_quality_system import DataQualityAssessor

assessor = DataQualityAssessor()
quality_report = assessor.assess_quality(production_data, target_column)

# Monitor quality score over time
print(f"Data Quality Score: {quality_report.overall_score}/100")
print(f"Critical Issues: {quality_report.issues_by_severity.get('critical', 0)}")
```

### **Pipeline Execution Logs**
```bash
# Monitor execution logs
tail -f logs/audit.log

# Check pipeline performance
tail -f pipeline.log

# Monitor model training progress
tail -f logs/training.log
```

## 🔒 **Security & Compliance**

### **Data Protection**
- Use environment variables for sensitive configuration
- Implement data masking for PII data
- Regular backup of model artifacts and metadata

```bash
# Set sensitive variables
export MLFLOW_TRACKING_PASSWORD="your_secure_password"
export DATABASE_CONNECTION_STRING="your_secure_db_string"
```

### **Access Control**
- Configure MLflow authentication
- Set up role-based access to model registry
- Implement audit logging for all operations

### **Data Lineage & Compliance**
```python
# Track data lineage for compliance
from src.infrastructure.metadata_manager import DataLineageTracker

tracker = DataLineageTracker(metadata_store)
lineage = tracker.get_lineage_report(dataset_id)

# Generate compliance reports
compliance_report = tracker.generate_compliance_report()
```

## 🚀 **Scalability & Performance**

### **Resource Optimization**
```yaml
# Adjust resources in config
workflow:
  execution:
    max_workers: 4  # Adjust based on available CPU cores
    memory_limit_gb: 8.0  # Adjust based on available RAM
    chunk_size: 10000  # Optimize for your data size
```

### **Caching Strategy**
```yaml
# Enable caching for expensive operations
infrastructure:
  caching:
    enabled: true
    cache_dir: "cache/"
    ttl_hours: 24
```

### **Parallel Processing**
- Automatic parallelization of model training
- Concurrent data quality assessment
- Parallel feature selection with plugins

## 📈 **Production Workflow Example**

### **Daily Production Run**
```bash
#!/bin/bash
# production_pipeline.sh

set -e  # Exit on any error

echo "Starting daily ML pipeline..."

# Step 1: Data quality check
python 1_inspect_data.py --data data/daily_data.csv
if [ $? -ne 0 ]; then
    echo "Data quality check failed!"
    exit 1
fi

# Step 2: Run production pipeline
python 3_run_pipeline.py --config config/unified_config_production.yaml

# Step 3: Check results
if [ -f "evaluation_results/comprehensive_analysis_report.json" ]; then
    echo "Pipeline completed successfully!"
    
    # Deploy best model (optional)
    # python deploy_model.py --model-registry-uri http://localhost:5000
else
    echo "Pipeline failed - no results generated"
    exit 1
fi
```

### **Model Deployment**
```python
# Example model deployment script
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get best model from current run
experiment = client.get_experiment_by_name("production-ml-pipeline")
runs = client.search_runs(experiment.experiment_id)
best_run = max(runs, key=lambda x: x.data.metrics.get('accuracy', 0))

# Register model for production
model_version = mlflow.register_model(
    model_uri=f"runs:/{best_run.info.run_id}/model",
    name="production-classifier"
)

# Transition to production stage
client.transition_model_version_stage(
    name="production-classifier",
    version=model_version.version,
    stage="Production"
)
```

## 🆘 **Troubleshooting**

### **Common Production Issues**

1. **MLflow Connection Issues**
   ```bash
   # Check MLflow server status
   curl http://localhost:5000/health
   
   # Restart MLflow server
   python setup/setup_mlflow.py --start-ui
   ```

2. **Memory Issues with Large Datasets**
   ```yaml
   # Reduce memory usage in config
   workflow:
     execution:
       chunk_size: 5000  # Smaller chunks
       max_workers: 2    # Fewer parallel processes
   ```

3. **Data Quality Failures**
   ```bash
   # Check detailed quality report
   python 1_inspect_data.py --data problematic_data.csv --detailed
   
   # Adjust quality thresholds in config if needed
   ```

4. **Plugin Loading Issues**
   ```yaml
   # Disable plugins temporarily
   workflow:
     feature_flags:
       plugin_system: false
   ```

## 📚 **Production Checklist**

### ✅ **Pre-Production**
- [ ] MLflow server configured and running
- [ ] Production configuration file created
- [ ] Data quality thresholds validated
- [ ] Pipeline tested with production-like data
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery procedures in place

### ✅ **Production Deployment**
- [ ] Environment variables configured
- [ ] Resource limits set appropriately
- [ ] Logging configured for production
- [ ] Access controls implemented
- [ ] Performance monitoring active

### ✅ **Post-Deployment**
- [ ] Monitor MLflow experiments regularly
- [ ] Review data quality reports
- [ ] Track model performance degradation
- [ ] Maintain model registry
- [ ] Regular backup of artifacts and metadata

## 🔗 **Key Production Files**

- `3_run_pipeline.py` - Main production pipeline
- `config/unified_config_v2.yaml` - Production configuration
- `setup/setup_mlflow.py` - MLflow setup and management
- `src/infrastructure/mlflow_integration.py` - MLflow integration
- `src/data_quality_system/` - Advanced data quality system
- `src/infrastructure/metadata_manager.py` - Metadata and lineage tracking

Your DS-AutoAdvisor v2.0 is now ready for enterprise production deployment! 🏭✨
