# DS-AutoAdvisor Quick Start 🚀

**Get from CSV to trained models in 15 minutes**

## 🎯 The Complete Flow

```bash
# 1. Analyze your data (3-5 minutes)
python 01_data_discovery.py --data data/your_data.csv --target your_target

# 2. 🛑 STOP & EDIT: cleaning_config_template.yaml 
#    Location: pipeline_outputs/01_discovery_*/configs/cleaning_config_template.yaml

# 3. Test your pipeline (2-10 minutes)
python 02_stage_testing.py --stage all

# 4. Package for production (1-3 minutes)
python 03_full_pipeline.py --enable-mlflow

# 5. View results (optional)
./mlflow_ui.sh
```

---
## 📖 Step-by-Step Guide

### Step 1: Data Discovery ⏱️ 3-5 minutes

**What it does:** Analyzes your data and creates a cleaning template for you to customize.

```bash
# Basic usage (uses default data from config)
python 01_data_discovery.py

# With your own data
python 01_data_discovery.py --data data/bank.csv --target deposit

# For automation (skip human review)
python 01_data_discovery.py --auto-approve

# Enhanced analysis (more statistical details)  
python 01_data_discovery.py --enhanced-profiling
```

**You'll get:**
- 📊 `profiling_report.html` (visual data analysis)
- ⚙️ `cleaning_config_template.yaml` (THE FILE YOU NEED TO EDIT)
- 📈 Data quality scores and recommendations
- 💾 Cached results for fast reuse

### Step 1.5: Edit Cleaning Template 🛑 REQUIRED

**Find the file:**
```bash
ls pipeline_outputs/01_discovery_*/configs/cleaning_config_template.yaml
```

**Open it and customize for each column:**

```yaml
column_transformations:
  age:
    imputation: 'median'        # median, mean, most_frequent, constant
    outlier_treatment: 'cap'    # cap, remove, winsorize, null
    scaling: 'standard'         # standard, minmax, robust
    
  salary:
    imputation: 'median'
    outlier_treatment: 'remove' # Remove extreme values
    transformation: 'log'       # Log transform for skewness  
    scaling: 'robust'           # Robust to outliers
    
  job_category:
    encoding: 'onehot'          # onehot, label, target, ordinal
    text_cleaning: true         # Clean text first
    max_categories: 20          # Limit high cardinality
```

**💡 Common Patterns:**
- **Financial data:** Use `log` transformation + `robust` scaling
- **Age/counts:** Use `median` imputation + `cap` outliers
- **Categories:** Use `onehot` (low cardinality) or `target` (high cardinality)
- **Text:** Enable `text_cleaning: true`

### Step 2: Test Your Pipeline ⏱️ 2-10 minutes

**What it does:** Tests your cleaning rules and trains models using your specifications.

```bash
# Test everything (recommended first run)
python 02_stage_testing.py --stage all

# Test just cleaning (fast, good for iteration)
python 02_stage_testing.py --stage cleaning

# Test just training (when cleaning is finalized)
python 02_stage_testing.py --stage training

# Fast mode (fewer models, quicker results)
python 02_stage_testing.py --mode fast --stage training

# Use specific discovery results
python 02_stage_testing.py --stage all --discovery-dir pipeline_outputs/01_discovery_bank_20250811_140000
```

**Available stages:**
- **cleaning:** See your YAML rules in action
- **advisory:** Get model recommendations  
- **training:** Train multiple ML models
- **evaluation:** Compare model performance
- **all:** Run everything in sequence

**Key output:** `training_report.json` with model rankings and metrics

### Step 3: Package for Production ⏱️ 1-3 minutes

**What it does:** Takes your best models and creates deployment-ready files.

```bash
# Auto-detect latest results (easiest)
python 03_full_pipeline.py

# With MLflow tracking (recommended)
python 03_full_pipeline.py --enable-mlflow

# Use specific Stage 2 results
python 03_full_pipeline.py --stage2-dir pipeline_outputs/02_stage_testing_custom_20250811_150000
```

**You'll get:**
- 📦 `models/` - Your trained models ready to use
- 🐍 `model_server.py` - Python script to serve models
- 🌐 `api_template.py` - REST API template
- 🐳 `Dockerfile` - Container for deployment
- 📖 Complete documentation and guides

**🎯 Important:** This step does NOT retrain. It packages existing models for speed.

### Step 4: View Results (Optional)

```bash
# Start MLflow UI
./mlflow_ui.sh

# Or specific port
./mlflow_ui.sh 5001

# Visit http://localhost:5000 in browser
```

---
## 🔄 Common Workflows

### 🆕 First Time With New Dataset
```bash
python 01_data_discovery.py --data data/new_data.csv --target outcome
# Edit cleaning_config_template.yaml
python 02_stage_testing.py --stage all  
python 03_full_pipeline.py --enable-mlflow
```

### ⚡ Quick Experiment
```bash
# If you have discovery results already
python 02_stage_testing.py --mode fast --stage training
python 03_full_pipeline.py
```

### 🔧 Iterative Cleaning Development
```bash
# Test cleaning changes quickly
python 02_stage_testing.py --stage cleaning
# Edit cleaning_config_template.yaml based on results
python 02_stage_testing.py --stage cleaning  # Test again
python 02_stage_testing.py --stage training  # When satisfied
```

### 🏭 Production Run
```bash
python 01_data_discovery.py --auto-approve
python 02_stage_testing.py --stage all
python 03_full_pipeline.py --enable-mlflow
```

---
## 📁 Key Files You Need to Know

| File | Location | What It Does |
|------|----------|--------------|
| **cleaning_config_template.yaml** | `01_discovery_*/configs/` | **YOU EDIT THIS** - Controls all data cleaning |
| **training_report.json** | `02_stage_testing_*/` | Shows model rankings and performance |
| **models_index.json** | `03_production_*/models/` | Lists your packaged models |
| **model_server.py** | `03_production_*/deployment/` | Ready-to-run model serving script |

---
## ⚠️ Important Notes

### 🛑 Don't Skip the Manual Step
The cleaning template edit is REQUIRED. The pipeline generates smart defaults, but you need to review and customize for your specific data and use case.

### 🔄 When to Re-run Steps
- **New data?** → Re-run Step 1
- **Better cleaning ideas?** → Edit template → Re-run Step 2  
- **Need different models?** → Adjust config → Re-run Step 2
- **Ready to deploy?** → Re-run Step 3

### ⚡ Speed vs Quality Tradeoffs
- **Stage 2 fast mode:** Fewer models, quicker results
- **Stage 2 custom mode:** More models, better performance  
- **Step 3 no retrain:** Fast packaging, but metrics from Stage 2 snapshot

---
## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No discovery directory found" | Run Step 1 first |
| "cleaning_config_template.yaml not found" | Check Step 1 completed successfully |
| "No models trained" | Check target column name, verify data quality |
| "MLflow UI empty" | Use `--enable-mlflow` flag in Step 3 |
| Models won't serve | Check `models/` directory has .pkl files |

---
## 🎯 Pro Tips

### 🚀 For Speed
```bash
# Skip reviews, use fast mode
python 01_data_discovery.py --auto-approve
python 02_stage_testing.py --mode fast --stage training
python 03_full_pipeline.py
```

### 🔍 For Quality  
```bash
# Enhanced profiling, full model search
python 01_data_discovery.py --enhanced-profiling
# Carefully edit cleaning_config_template.yaml
python 02_stage_testing.py --mode custom --stage all
python 03_full_pipeline.py --enable-mlflow
```

### 🧪 For Development
```bash
# Test components separately
python 02_stage_testing.py --stage cleaning
python 02_stage_testing.py --stage training
```

---
## 🎉 Success Indicators

You know it's working when:
- ✅ Step 1 generates `cleaning_config_template.yaml`
- ✅ Step 2 creates `training_report.json` with model rankings  
- ✅ Step 3 creates `models/` directory with .pkl files
- ✅ MLflow UI shows your experiments (if enabled)
- ✅ `model_server.py` loads and runs without errors

**Ready to start?** Pick your workflow above and go! 🚀
