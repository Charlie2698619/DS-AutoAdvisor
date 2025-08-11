# DS-AutoAdvisor 🚀

**Smart ML Pipeline: From Raw CSV to Production-Ready Models in 3 Steps**

> Turn your messy data into clean, trained models with deployment-ready artifacts. Transparent, cacheable, and human-guided every step of the way.

## 🎯 What This Does For You

**THE PROBLEM:** Traditional ML pipelines are slow, opaque, and repetitive. You waste hours on data profiling, mysterious cleaning steps, and packaging models for deployment.

**THE SOLUTION:** DS-AutoAdvisor gives you:
- ⚡ **One-time deep data discovery** (no more re-profiling the same data)
- 🔍 **Transparent column-by-column cleaning** (you control every transformation)
- 🧪 **Fast iterative testing** (test cleaning/training separately without full reruns)
- 📦 **Production packaging** (deployment scripts, docs, APIs automatically generated)
- 📊 **Full traceability** (MLflow tracking + structured JSON outputs)

## 🏗️ Simple Architecture

```
📊 Raw Data
    ↓
🔍 Step 1: Deep Discovery (profiles + generates cleaning template)
    ↓ 
✏️  HUMAN STEP: Edit cleaning_config_template.yaml 
    ↓
🧪 Step 2: Test Pipeline Stages (cleaning → training → evaluation)
    ↓
📦 Step 3: Package Best Models (deployment scripts + docs + APIs)
```

## 📋 Table of Contents
1. [Quick Start (5 Minutes)](#quick-start)
2. [The 3-Step Workflow](#workflow)
3. [Important Manual Steps](#manual-steps)
4. [What You Get From Each Step](#outputs)
5. [Common Use Cases](#use-cases)
6. [Configuration Guide](#configuration)
7. [MLflow Tracking](#mlflow)
8. [Design Choices & Limitations](#limitations)
9. [Troubleshooting](#troubleshooting)
10. [Future Features](#roadmap)

---
## Quick Start

**🎯 Goal:** Get from CSV to trained models in 15 minutes

```bash
# Step 1: Analyze your data (one-time, 3-5 minutes)
python 01_data_discovery.py --data data/your_data.csv --target your_target_column

# 🛑 STOP: Edit the generated template 
# Open: pipeline_outputs/01_discovery_*/configs/cleaning_config_template.yaml
# Customize column cleaning rules (imputation, outliers, encoding, etc.)

# Step 2: Test your pipeline (iterative, 2-10 minutes)
python 02_stage_testing.py --stage all

# Step 3: Package for production (1-3 minutes)  
python 03_full_pipeline.py --enable-mlflow

# Optional: View experiment tracking
./mlflow_ui.sh
```

**That's it!** Your models, deployment scripts, and documentation are ready.

---
## The 3-Step Workflow

### 🔍 Step 1: Data Discovery & Configuration
**Script:** `01_data_discovery.py`  
**Purpose:** Deep one-time analysis that creates reusable configurations  
**Time:** 3-5 minutes per dataset

```bash
# Basic usage
python 01_data_discovery.py

# With your data
python 01_data_discovery.py --data data/bank.csv --target deposit

# Skip human review (for automation)
python 01_data_discovery.py --auto-approve

# Enhanced profiling (more statistical details)
python 01_data_discovery.py --enhanced-profiling
```

**What it does:**
- 📊 Comprehensive data profiling (missing values, distributions, correlations)
- 🎯 Automatic target column detection
- 📝 Data quality assessment with scores
- ⚙️ **Generates cleaning template** (the key output!)
- 💾 Caches expensive computations for reuse

**Key Output:** `cleaning_config_template.yaml` - THIS IS WHAT YOU EDIT!

### ✏️ CRUCIAL MANUAL STEP: Edit Your Cleaning Template

**Location:** `pipeline_outputs/01_discovery_*/configs/cleaning_config_template.yaml`

This YAML file controls every data transformation. Example:
```yaml
column_transformations:
  age:
    imputation: 'median'           # How to handle missing values
    outlier_treatment: 'cap'       # How to handle outliers  
    scaling: 'standard'            # How to scale values
    
  salary:
    imputation: 'median'
    outlier_treatment: 'remove'    # Remove extreme salary outliers
    transformation: 'log'          # Log transform skewed salary
    scaling: 'robust'              # Robust scaling for financial data
    
  job_category:
    encoding: 'onehot'             # One-hot encode categories
    text_cleaning: true            # Clean text first
```

**💡 Why This Matters:**
- You have full control over every transformation
- Different datasets need different approaches
- Decisions are documented and reproducible
- You can version control your cleaning logic

### 🧪 Step 2: Stage Testing (Iterative Development)
**Script:** `02_stage_testing.py`  
**Purpose:** Fast testing of individual pipeline components  
**Time:** 30 seconds - 10 minutes depending on scope

```bash
# Test individual stages
python 02_stage_testing.py --stage cleaning      # Test data cleaning only
python 02_stage_testing.py --stage training      # Test model training only
python 02_stage_testing.py --stage evaluation    # Test model evaluation only

# Test multiple stages
python 02_stage_testing.py --stage cleaning,training
python 02_stage_testing.py --stage all

# Use specific discovery results  
python 02_stage_testing.py --stage all --discovery-dir pipeline_outputs/01_discovery_bank_20250811_140000

# Choose mode (fast = fewer models, custom = full control)
python 02_stage_testing.py --mode fast --stage training
```

**Available Stages:**
- **cleaning:** Apply your YAML cleaning rules and see results
- **advisory:** Get ML model recommendations based on your data
- **training:** Train multiple models (limited set for speed)
- **evaluation:** Generate performance metrics and comparisons

**Key Output:** `training_report.json` with model rankings and performance

### 📦 Step 3: Production Packaging (No Retraining)
**Script:** `03_full_pipeline.py`  
**Purpose:** Package your best models with deployment assets  
**Time:** 1-3 minutes

```bash
# Auto-detect latest Stage 2 results
python 03_full_pipeline.py

# Use specific Stage 2 results
python 03_full_pipeline.py --stage2-dir pipeline_outputs/02_stage_testing_custom_20250811_150000

# Enable MLflow experiment tracking
python 03_full_pipeline.py --enable-mlflow
```

**What it does:**
- 📦 Packages top 3 models from Stage 2
- 🐍 Creates model serving script (`model_server.py`)
- 🌐 Generates API template (`api_template.py`)
- 🐳 Creates Dockerfile for deployment
- 📖 Writes executive summary and deployment docs
- 📊 Generates model comparison reports

**Key Design Choice:** No retraining! This step packages existing models for speed.

---
## Important Manual Steps

### 🛑 Step 1.5: Edit cleaning_config_template.yaml (REQUIRED)

After Step 1, you MUST review and customize the cleaning template:

1. **Find the file:**
   ```bash
   ls pipeline_outputs/01_discovery_*/configs/cleaning_config_template.yaml
   ```

2. **Key decisions to make for each column:**
   - **Missing values:** median, mean, mode, or custom value?
   - **Outliers:** remove, cap, or keep?
   - **Encoding:** one-hot, label, target, or ordinal?
   - **Scaling:** standard, minmax, or robust?
   - **Transformations:** log, sqrt, or none?

3. **Common patterns:**
   ```yaml
   # Numeric columns with outliers
   salary:
     imputation: 'median'
     outlier_treatment: 'cap'     # Cap at 95th percentile
     transformation: 'log'        # Log transform for skewness
     scaling: 'robust'
   
   # Categorical with few categories  
   gender:
     imputation: 'most_frequent'
     encoding: 'onehot'
   
   # High cardinality categorical
   job_title:
     encoding: 'target'           # Target encoding
     max_categories: 20           # Limit categories
   ```

4. **Test your changes:**
   ```bash
   python 02_stage_testing.py --stage cleaning
   ```

---
## What You Get From Each Step
| Step | Folder | Key Files | What They Contain |
|------|--------|-----------|-------------------|
| **1** | `01_discovery_*` | `configs/cleaning_config_template.yaml`<br>`reports/profiling_report.html`<br>`cache/` | **The cleaning template to edit**<br>Visual data profiling report<br>Cached computations for reuse |
| **2** | `02_stage_testing_*` | `training_report.json`<br>`models/*.pkl`<br>`cleaned_data/` | Model rankings and metrics<br>Trained model files<br>Cleaned datasets (using your rules) |
| **3** | `03_production_artifacts_*` | `models/` (packaged)<br>`deployment/model_server.py`<br>`deployment/api_template.py`<br>`documentation/` | Production-ready models<br>Flask serving script<br>REST API template<br>Deployment guides |

### 🎯 The Most Important Files

1. **`cleaning_config_template.yaml`** (Step 1) - YOU MUST EDIT THIS
2. **`training_report.json`** (Step 2) - Contains your model rankings
3. **`models_index.json`** (Step 3) - Lists packaged models and metadata
4. **`model_server.py`** (Step 3) - Ready-to-run model serving script

---
## Common Use Cases

### 🆕 New Dataset Workflow
```bash
# First time with new data
python 01_data_discovery.py --data data/new_dataset.csv --target outcome
# Edit: pipeline_outputs/01_discovery_*/configs/cleaning_config_template.yaml
python 02_stage_testing.py --stage all
python 03_full_pipeline.py --enable-mlflow
```

### 🔄 Iterative Development  
```bash
# You have discovery results, testing cleaning changes
python 02_stage_testing.py --stage cleaning  # Test cleaning changes
# Edit cleaning_config_template.yaml again if needed
python 02_stage_testing.py --stage training  # Test training with new cleaning
python 03_full_pipeline.py                   # Package when satisfied
```

### 🚀 Quick Experimentation
```bash
# Test just training with different models
python 02_stage_testing.py --mode fast --stage training
```

### 🏭 Production Pipeline
```bash
# Full pipeline with tracking
python 01_data_discovery.py --auto-approve  # Skip reviews
python 02_stage_testing.py --stage all
python 03_full_pipeline.py --enable-mlflow
./mlflow_ui.sh  # View results
```

---
## Configuration Guide

### 🎛️ Configuration Priority (High → Low)
1. **Command line arguments** (`--data`, `--target`, etc.)
2. **Discovery-generated configs** (from Step 1)
3. **Main config file** (`config/unified_config_v3.yaml`)
4. **Built-in defaults**

### 🔧 Key Configuration Files

**Main Config:** `config/unified_config_v3.yaml`
```yaml
global:
  data_input_path: "data/your_data.csv"
  target_column: "your_target"

fast_mode:         # Fewer models, faster execution
  training:
    max_models: 5
    
custom_mode:       # Full control, more models
  training:
    max_models: 15
    enable_ensemble: true
```

**Cleaning Config:** `cleaning_config_template.yaml` (generated, then edited by you)
```yaml
column_transformations:
  your_column:
    imputation: 'median'
    outlier_treatment: 'cap'
    encoding: 'onehot'
    scaling: 'standard'
```

---
## MLflow Tracking

MLflow tracks your experiments and models for reproducibility.

### 🚀 Enable Tracking
```bash
# Enable for Step 3 (recommended)
python 03_full_pipeline.py --enable-mlflow

# View in browser
./mlflow_ui.sh 5000
# Visit: http://localhost:5000
```

### 📊 What Gets Tracked
- **Experiments:** Each pipeline run
- **Metrics:** Model accuracy, training time, rankings
- **Parameters:** Model types, cleaning settings
- **Artifacts:** Model files, reports, configurations

### 🎯 Why Use MLflow
- Compare multiple pipeline runs
- Track which cleaning settings work best
- Version your models automatically
- Share results with team

**Tradeoff:** Adds ~10% overhead but provides valuable experiment tracking.

---
## Design Choices & Limitations

### 🎯 Key Design Decisions

| Decision | Why We Did It | Benefit | Limitation | When to Revisit |
|----------|---------------|---------|------------|-----------------|
| **Cache discovery results** | Avoid re-profiling same data | 5x faster iteration | Stale if data changes | Add drift detection |
| **Manual cleaning config** | Full transparency & control | You see every transformation | Extra manual step | Auto-suggest improvements |
| **Limited models in Stage 2** | Fast feedback loops | Test in minutes not hours | Might miss best model | Re-run with more models |
| **No retraining in Step 3** | Ultra-fast packaging | 1-3 min deployment prep | Metrics may be outdated | Add optional retrain flag |
| **Pickle model format** | Simple & fast | Works everywhere | Security/portability risk | Move to MLflow format |
| **Flask API template** | Zero dependencies | Easy to understand | No auth or scaling | Upgrade to FastAPI |

### ⚠️ Current Limitations

**Data & Processing:**
- No automatic data drift detection
- No streaming/incremental updates
- Single-process execution (no parallelism)
- Basic hyperparameter optimization

**Production & Deployment:**
- No authentication in API template
- No built-in monitoring or alerting
- No automatic model versioning/promotion
- No A/B testing framework

**MLOps & Infrastructure:**
- No job scheduler or orchestration
- No feature store integration
- No automated retraining triggers
- No fairness/bias detection built-in

### 💡 When to Use Alternative Tools

**Use DS-AutoAdvisor when:**
- Exploring new datasets quickly
- Need transparent, controllable cleaning
- Want fast iteration cycles
- Building proof-of-concepts

**Consider other tools when:**
- Need production MLOps platform (Kubeflow, MLflow + Airflow)
- Require real-time serving (TensorFlow Serving, Seldon)
- Need advanced AutoML (H2O.ai, DataRobot)
- Have streaming data requirements (Apache Kafka + Flink)

---
## Troubleshooting

### 🚨 Common Issues & Quick Fixes

| Problem | Quick Check | Solution |
|---------|-------------|----------|
| **"No discovery directory found"** | `ls pipeline_outputs/01_discovery_*` | Run Step 1 first |
| **"cleaning_config_template.yaml not found"** | Check if Step 1 completed | Look in `pipeline_outputs/01_discovery_*/configs/` |
| **Models not training** | Check target column name | Verify target exists in your data |
| **MLflow UI empty** | Check `--enable-mlflow` flag | Ensure MLflow enabled in Step 3 |
| **"No models packaged"** | Check `training_report.json` | Verify Stage 2 completed successfully |
| **API script fails** | Check model files exist | Verify models/ directory has .pkl files |

### 🔍 Debugging Steps

1. **Check outputs of previous step:**
   ```bash
   ls pipeline_outputs/*/
   cat pipeline_outputs/*/summary.json
   ```

2. **Test individual stages:**
   ```bash
   python 02_stage_testing.py --stage cleaning  # Test just cleaning
   ```

3. **Check logs:**
   ```bash
   ls logs/
   tail -f logs/audit.log
   ```

### 💾 File Locations Cheat Sheet

```
📁 Project Structure:
├── pipeline_outputs/           # All your results
│   ├── 01_discovery_*/        # Step 1 outputs
│   ├── 02_stage_testing_*/    # Step 2 outputs  
│   └── 03_production_*/       # Step 3 outputs
├── config/                    # Configuration files
├── logs/                      # Execution logs
├── mlruns/                    # MLflow tracking
└── data/                      # Your input data
```

---
## Roadmap (Future Features)

### 🎯 Short Term (Next Release)
- **Optional retraining in Step 3** (for up-to-date metrics)
- **Advanced hyperparameter optimization** (Bayesian, Optuna)
- **Data drift detection** (automatic discovery refresh)
- **FastAPI deployment template** (replaces Flask)

### 🚀 Medium Term (6 months)
- **Model registry integration** (automatic versioning/promotion)
- **Fairness & bias detection** (ethical AI metrics)
- **Feature importance caching** (SHAP explanations)
- **CI/CD templates** (GitHub Actions workflows)

### 🏭 Long Term (1 year)
- **Streaming data support** (Kafka integration)
- **Feature store abstraction** (Feast integration)
- **Production monitoring** (data/model drift alerts)
- **ONNX export support** (cross-platform deployment)
- **Security hardening** (auth, input validation, audit)
- **Parallel processing** (multi-core/distributed execution)

### 🤝 Community Features
- **Plugin system** (custom transformations)
- **Model marketplace** (pre-trained models)
- **Template library** (industry-specific configs)
- **Integration gallery** (cloud platforms, databases)

---
## Summary: Why This Design Works

✅ **Transparent:** You see and control every transformation  
✅ **Fast:** Cache expensive operations, test components separately  
✅ **Reproducible:** Version-controlled configs, MLflow tracking  
✅ **Production-ready:** Deployment scripts, docs, APIs included  
✅ **Extensible:** Easy to add stages, models, or deployment targets  

**Perfect for:** Data scientists who want fast iteration with full control over their ML pipeline.

---

**🎉 Ready to start?** Jump to [Quick Start](#quick-start) and turn your data into models in 15 minutes!