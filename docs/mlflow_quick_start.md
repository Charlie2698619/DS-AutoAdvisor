# MLflow Integration Quick Start Guide

## 🚀 Getting Started with MLflow in DS-AutoAdvisor v2.0

This guide will help you set up and use MLflow experiment tracking and model versioning in your enhanced pipeline.

### 1. Initial Setup

Run the setup script to install and configure MLflow:

```bash
python setup_mlflow.py
```

This will:
- Install MLflow if not already present
- Create necessary directories (`mlruns/`, `artifacts/`)
- Enable MLflow features in configuration
- Test the integration with a sample experiment

### 2. Manual Installation (Alternative)

If you prefer manual setup:

```bash
# Install MLflow
pip install mlflow

# Enable MLflow in feature flags
# Edit config/feature_flags.yaml:
# mlflow_tracking_enabled: true
# model_versioning_enabled: true
```

### 3. Running with MLflow Tracking

Run the enhanced pipeline with MLflow tracking:

```bash
python complete_pipeline_v2.py
```

The pipeline will automatically:
- Create MLflow experiments for each run
- Log metrics, parameters, and artifacts
- Register successful models in the model registry
- Track data quality scores and stage results

### 4. Viewing Results in MLflow UI

Start the MLflow tracking server:

```bash
mlflow ui --port 5000
```

Then open your browser to: http://localhost:5000

### 5. Enhanced Human Checkpoints

During pipeline execution, you'll see enhanced checkpoints with:

- **Historical Performance**: View past model performance automatically
- **Model Comparison**: Access detailed model comparison from MLflow registry
- **Plugin Results**: See results from feature selection and other plugins
- **Data Quality Insights**: View automated quality assessments

Available checkpoint actions:
- `continue` - Proceed to next stage
- `retry` - Retry current stage
- `config` - Adjust configuration interactively
- `insights` - View detailed v2.0 insights
- `models` - View MLflow model comparison
- `abort` - Stop pipeline execution

### 6. MLflow Features Available

#### Experiment Tracking
- Automatic experiment creation per pipeline run
- Metric logging (accuracy, F1, precision, recall)
- Parameter logging (model hyperparameters, config settings)
- Artifact logging (models, plots, reports)

#### Model Registry
- Automatic model registration for successful runs
- Version management for models
- Model metadata and lineage tracking
- Performance comparison across versions

#### Data Quality Integration
- Quality scores logged as metrics
- Outlier detection results tracked
- Data profiling artifacts stored

### 7. Configuration Options

In `config/unified_config_v2.yaml`:

```yaml
mlflow:
  tracking_uri: "file:./mlruns"
  experiment_name: "ds-autoadvisor-v2"
  artifact_location: "./artifacts"
  log_artifacts: true
  auto_register_models: true
  model_registry:
    stage: "Staging"  # or "Production"
    description: "Auto-registered model from DS-AutoAdvisor v2.0"
```

### 8. Advanced Usage

#### Comparing Historical Models
```python
# In checkpoint, select 'models' to see:
# - Top 5 models by accuracy
# - Performance metrics comparison
# - Creation dates and versions
# - Best performing model details
```

#### Meta-Learning Ready
The MLflow integration prepares for Phase 3 meta-learning by:
- Storing model performance by data schema
- Tracking hyperparameter effectiveness
- Building historical performance database

### 9. Troubleshooting

#### Common Issues:

**MLflow not found:**
```bash
pip install mlflow
```

**Permission errors:**
```bash
chmod +x setup_mlflow.py
```

**Port already in use:**
```bash
mlflow ui --port 5001  # Use different port
```

**Configuration not loading:**
- Check `config/feature_flags.yaml` has `mlflow_tracking_enabled: true`
- Verify `config/unified_config_v2.yaml` has MLflow section

### 10. Next Steps

After successful MLflow integration:

1. **Run multiple experiments** to build model history
2. **Compare model performance** using the MLflow UI
3. **Explore Phase 3 meta-learning** using accumulated model data
4. **Set up production monitoring** using MLflow model serving

### Example Output

When running with MLflow enabled, you'll see:

```
🤖➡️👨‍💻 ENHANCED CHECKPOINT: TRAINING
================================================================================
Stage: training
Status: ✅ Success
Execution Time: 45.23s
Outputs Created: 3

📊 Data Quality Score: 87.5/100
🔍 Quality Issues: 2

🔌 Plugin Results:
   advanced_feature_selector: 15 features selected

📈 Historical Performance:
   Found 3 historical models
   Best model: RandomForestClassifier (v2)
   Best accuracy: 0.945

🔧 Available Actions:
  continue  - Proceed to next stage
  retry     - Retry this stage
  config    - Adjust configuration
  insights  - View detailed insights (v2.0)
  models    - View model comparison (MLflow)
  abort     - Stop pipeline execution
```

This integration provides the foundation for advanced model versioning and experiment tracking in your v2.0 pipeline!
