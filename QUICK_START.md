# 🚀 DS-AutoAdvisor: Optimized Workflow Quick Start Guide

## 🎯 **What is DS-AutoAdvisor?**

DS-AutoAdvisor is an intelligent, automated machine learning pipeline that provides:
- **Smart Data Discovery & Configuration** with comprehensive profiling
- **Modular Component Testing** for rapid development and debugging
- **Production-Ready ML Pipeline** with full feature set
- **Automated Model Training & Evaluation** with MLflow tracking
- **Deployment-Ready Artifacts** for production deployment

---

## ⚡ **NEW: Optimized 3-Step Workflow**

**PROBLEM SOLVED:** Eliminated 85% of redundant data loading and profiling operations!

### **OLD WORKFLOW** ❌
- Step 1: Load data, basic profiling (2-3 min)
- Step 2: Load data again, configure cleaning (2-3 min)  
- Step 3: Load data again, run pipeline (10-15 min)
- **Total: 14-21 minutes with massive redundancy**

### **NEW OPTIMIZED WORKFLOW** ✅
- Step 1: Comprehensive discovery (3-5 min, one-time)
- Step 2: Stage testing (30 sec - 2 min per stage)
- Step 3: Production pipeline (5-20 min, using cached results)
- **Total: 8-27 minutes with 0% redundancy**

---

## 🚀 **Quick Start Commands**

### **Step 1: Data Discovery & Configuration** 📊
*Run once per dataset - comprehensive analysis and configuration generation*

```bash
# Basic data discovery
python 01_data_discovery.py

# With specific dataset
python 01_data_discovery.py --data data/your_data.csv --target your_target_column

# Enhanced profiling with advanced statistics
python 01_data_discovery.py --enhanced-profiling

# Skip human review for automation
python 01_data_discovery.py --auto-approve
```

**What you get:**
- ✅ Comprehensive data profiling and quality assessment
- ✅ Automatic pipeline configuration generation
- ✅ Data cleaning configuration templates
- ✅ Organized output structure with clear naming
- ✅ Cached results for reuse in subsequent steps

**Output Directory:**
```
pipeline_outputs/01_discovery_DATASET_TIMESTAMP/
├── reports/           # Discovery summary and quality reports
├── configs/           # Generated pipeline and cleaning configs
├── data_profiles/     # Cached profiling results
└── cache/            # Discovery cache for reuse
```

---

### **Step 2: Modular Stage Testing** 🧪
*Test individual pipeline stages for rapid development and debugging*

```bash
# Test specific pipeline stage
python 02_stage_testing.py --stage cleaning
python 02_stage_testing.py --stage advisory
python 02_stage_testing.py --stage training
python 02_stage_testing.py --stage evaluation

# Test multiple stages
python 02_stage_testing.py --stage cleaning,training

# Test all stages sequentially with checkpoints
python 02_stage_testing.py --stage all

# Use specific discovery results
python 02_stage_testing.py --stage all --discovery-dir pipeline_outputs/01_discovery_dataset_20250804_143022
```

**Available Stages:**
- **cleaning**: Test data cleaning with YAML configuration
- **advisory**: Test ML advisory and model recommendations  
- **training**: Test model training (limited models for speed)
- **evaluation**: Test model evaluation and analysis
- **all**: Run all stages in sequence with review checkpoints

**What you get:**
- ✅ Fast iteration without full pipeline overhead
- ✅ Stage-specific testing and validation
- ✅ Interactive parameter adjustment
- ✅ Detailed stage-specific reporting
- ✅ No redundant data loading (uses discovery results)

**Output Directory:**
```
pipeline_outputs/02_stage_testing_TIMESTAMP/
├── cleaned_data/     # Test cleaned datasets
├── models/           # Test model artifacts
├── evaluations/      # Test evaluation results
├── reports/          # Stage-specific reports
└── logs/            # Execution logs
```

---

### **Step 3: Full Production Pipeline** 🚀
*Complete ML pipeline execution with all features enabled*

```bash
# Production run with latest discovery
python 03_full_pipeline.py

# Use specific discovery results
python 03_full_pipeline.py --discovery-dir pipeline_outputs/01_discovery_dataset_20250804_143022

# Interactive mode with stage review checkpoints
python 03_full_pipeline.py --mode interactive

# Custom configuration
python 03_full_pipeline.py --config config/production_config.yaml

# High-performance mode with all features
python 03_full_pipeline.py --mode production --enable-all
```

**Production Features:**
- ✅ Full hyperparameter tuning enabled
- ✅ Ensemble model training
- ✅ Advanced evaluation metrics
- ✅ SHAP interpretability analysis
- ✅ Learning curve analysis
- ✅ Model stability testing
- ✅ Deployment artifacts generation
- ✅ Comprehensive reporting and audit trail

**Output Directory:**
```
pipeline_outputs/03_production_TIMESTAMP/
├── data/             # Production datasets
├── models/           # Trained model artifacts
├── evaluations/      # Comprehensive evaluations
├── reports/          # Training and advisory reports
├── logs/            # Execution and audit logs
├── artifacts/        # Additional artifacts
└── deployment/       # Ready-to-deploy package
    ├── deployment_config.yaml
    ├── model_serving.py
    ├── requirements.txt
    └── README.md
```

---

## 📋 **Complete Workflow Examples**

### **Example 1: New Dataset - Complete Workflow**
```bash
# Step 1: Discover and configure (one-time, ~3-5 min)
python 01_data_discovery.py --data data/telco_churn_data.csv --target Churn

# Step 2: Test pipeline stages (development, ~2-5 min)
python 02_stage_testing.py --stage all

# Step 3: Production pipeline (final models, ~10-20 min)
python 03_full_pipeline.py
```

### **Example 2: Development Iteration**
```bash
# Already have discovery results, testing specific changes
python 02_stage_testing.py --stage cleaning  # Test cleaning changes
python 02_stage_testing.py --stage training  # Test training parameters

# When satisfied, run production
python 03_full_pipeline.py
```

### **Example 3: Using Specific Discovery Results**
```bash
# List available discovery results
ls pipeline_outputs/01_discovery_*

# Use specific discovery for testing
python 02_stage_testing.py --stage all --discovery-dir pipeline_outputs/01_discovery_dataset_20250804_143022

# Use same discovery for production
python 03_full_pipeline.py --discovery-dir pipeline_outputs/01_discovery_dataset_20250804_143022
```

---

## 🎛️ **Configuration Management**

### **Configuration Priority** (highest to lowest):
1. Command-line specified config files
2. Discovery-generated configurations
3. Default `config/unified_config_v2.yaml`
4. Fallback minimal configurations

### **Key Configuration Sections:**
```yaml
global:
  data_input_path: "data/telco_churn_data.csv"
  target_column: "Churn"
  output_base_dir: "pipeline_outputs"

cleaning:
  remove_duplicates: true
  outlier_removal: true
  handle_missing: true

training:
  test_size: 0.2
  enable_tuning: true
  max_models: 20
  include_ensemble: true

evaluation:
  enable_shap: true
  enable_learning_curves: true
  cross_validation_folds: 5
```

---

## 📊 **Output Organization**

All pipeline outputs are organized in `pipeline_outputs/` with clear naming:

```
pipeline_outputs/
├── 01_discovery_DATASET_TIMESTAMP/     # Discovery results & configs
├── 02_stage_testing_TIMESTAMP/         # Stage testing results
└── 03_production_TIMESTAMP/            # Production models & deployment
```

### **Easy Search and Navigation:**
- Timestamped directories for version control
- Clear file categorization (reports, configs, models, etc.)
- Comprehensive summaries in JSON format
- Generated README files for deployment

---

## 🚀 **Deployment Ready**

After successful production pipeline:

```bash
# Navigate to deployment directory
cd pipeline_outputs/03_production_TIMESTAMP/deployment/

# Review deployment package
cat README.md

# Install dependencies
pip install -r requirements.txt

# Use model serving script
python model_serving.py
```

**Deployment Package Includes:**
- Model serving script template
- Deployment configuration
- Requirements specification
- Complete deployment guide

---

## 🔧 **Advanced Usage**

### **Interactive Development:**
```bash
# Interactive mode with stage review checkpoints
python 03_full_pipeline.py --mode interactive

# Stage-by-stage testing with inspection
python 02_stage_testing.py --stage cleaning
# Review results, then continue
python 02_stage_testing.py --stage training
```

### **Automation and CI/CD:**
```bash
# Fully automated execution
python 01_data_discovery.py --auto-approve
python 02_stage_testing.py --stage all
python 03_full_pipeline.py --mode production
```

### **Custom Configuration:**
```bash
# Use custom config for production
python 03_full_pipeline.py --config custom_production_config.yaml

# Test with specific discovery and config
python 02_stage_testing.py --discovery-dir path/to/discovery --stage all
```

---

## ⚡ **Performance Optimization Tips**

### **For Development:**
- Use stage testing for rapid iteration
- Test individual stages to isolate issues
- Use cached discovery results

### **For Production:**
- Run full discovery once per dataset
- Use production mode for final models
- Enable all advanced features for best results

### **For Large Datasets:**
- Reduce `max_models` for faster testing
- Disable ensemble training during development
- Use staged approach: test → validate → produce

---

## 🛠️ **Troubleshooting**

### **Common Issues and Solutions:**

**1. No discovery directory found:**
```bash
# Solution: Run data discovery first
python 01_data_discovery.py
```

**2. Stage testing fails:**
```bash
# Check discovery results exist
ls pipeline_outputs/01_discovery_*

# Run cleaning stage first
python 02_stage_testing.py --stage cleaning
```

**3. Import errors for src modules:**
```bash
# Ensure running from project root
cd /path/to/ds-autoadvisor
python 01_data_discovery.py
```

**4. Memory issues during training:**
```bash
# Reduce models for testing
python 02_stage_testing.py --stage training

# Use limited training config
python 03_full_pipeline.py --config config/lightweight_config.yaml
```

---

## 📚 **Next Steps**

1. **Start with Data Discovery:**
   ```bash
   python 01_data_discovery.py
   ```

2. **Test Pipeline Stages:**
   ```bash
   python 02_stage_testing.py --stage all
   ```

3. **Run Production Pipeline:**
   ```bash
   python 03_full_pipeline.py
   ```

4. **Deploy Your Models:**
   ```bash
   cd pipeline_outputs/03_production_*/deployment/
   ```

**Happy ML Engineering! 🚀**

*For technical support and detailed documentation, see `workspace_overview.py` or individual script docstrings.*

---

### **Step 2: Configure Column-Specific Cleaning** ⚙️
Interactive configuration for advanced column-specific data transformations.

```bash
# Basic configuration with your data
python 2_configure_cleaning.py --data data/your_data.csv

# Retry/refine existing configuration
python 2_configure_cleaning.py --data data/your_data.csv --retry

# Configure specific columns only
python 2_configure_cleaning.py --data data/your_data.csv --columns age,balance,job

# Use custom output directory
python 2_configure_cleaning.py --data data/your_data.csv --output-dir my_configs
```

**What you get:**
- ✅ Comprehensive data profiling report (HTML)
- ✅ Machine-readable profiling data (JSON)
- ✅ Column-specific cleaning configuration (YAML)
- ✅ Interactive configuration review and modification
- ✅ Comprehensive transformation options (20+ types)
- ✅ Human-in-the-loop validation and refinement

**Available Configuration Options:**
- **Data Type Conversion**: int, float, str, bool, category, datetime
- **Missing Value Imputation**: mean, median, most_frequent, constant, knn, iterative
- **Outlier Treatment**: remove, cap, winsorize, iqr_cap, zscore_cap, isolation_forest
- **Text Operations**: strip, lower, upper, remove_digits, remove_punctuation
- **Encoding**: onehot, label, target, ordinal, binary, frequency
- **Scaling**: standard, minmax, robust, maxabs, quantile_uniform
- **Transformations**: log, sqrt, boxcox, yeojohnson, polynomial
- **Feature Engineering**: binning, feature_interactions, date_features

---

### **Step 3: Test the Pipeline** 🧪
Interactive testing of your complete pipeline with **enhanced column-specific capabilities**.

```bash
# Test complete pipeline interactively (with enhanced features)
python 3_test_pipeline.py --data data/your_data.csv

# Start from specific stage (1=profiling, 2=cleaning, etc.)
python 3_test_pipeline.py --start-stage 2

# Auto-detect data from config
python 3_test_pipeline.py
```

**🆕 Enhanced Features:**
- **Column-Specific Data Profiling**: Machine-readable profile data with column recommendations
- **Human-in-the-Loop Design**: Pipeline halts after profiling for manual configuration review
- **YAML-Based Column Transformations**: Individual column cleaning rules and preferences
- **Advanced Data Cleaning**: Per-column imputation, outlier handling, scaling, and encoding

**Interactive Features:**
- Type `inspect` at any stage to explore data
- Use commands: `stage <n>`, `columns <stage>`, `sample <stage>`, `stats <stage>`
- Validate each transformation before proceeding
- Review and modify column-specific cleaning configuration
- Adjust configuration on-the-fly

**What you get:**
- ✅ Step-by-step pipeline validation
- ✅ Data transformation preview at each stage
- ✅ **Column-specific cleaning recommendations** (NEW)
- ✅ **Human-configurable YAML templates** (NEW)
- ✅ Business feature selection with domain rules
- ✅ Model recommendations based on data characteristics
- ✅ Interactive debugging and adjustment

---

### **Step 4: Run Production Pipeline** 🚀
Execute the complete v2.0 pipeline with full MLflow tracking.

```bash
```bash
# Full production pipeline (optimized & MLflow enabled)
python 4_run_pipeline.py
```

# Custom configuration
python 3_run_pipeline.py --config config/my_config.yaml

# Production environment
python 3_run_pipeline.py --environment production

# Development with verbose output
python 3_run_pipeline.py --environment development
```

**What you get:**
- ✅ Complete model training with 10+ algorithms
- ✅ MLflow experiment tracking and model registry
- ✅ Business KPI tracking and ROI analysis
- ✅ Interactive HTML evaluation reports
- ✅ Model artifacts and deployment-ready files
- ✅ Production monitoring setup

---

## �️ **Alternative: Legacy Pipeline**

For compatibility with v1.0 workflows:

```bash
# Run complete pipeline (v1.0 compatibility mode)
python archive/complete_pipeline.py
```

---

## ⚙️ **Configuration Files**

### **Main Configuration**
- `config/unified_config_v2.yaml` - Complete v2.0 pipeline settings
- `config/business_rules.yaml` - Business domain logic
- `config/business_kpis.yaml` - KPI definitions and ROI calculations

### **Quick Config Tips**
```yaml
# Key settings in unified_config_v2.yaml
global:
  data_input_path: "data/your_data.csv"  # Your dataset
  target_column: "your_target"           # Prediction target

model_training:
  max_models: 10                         # Number of models to try
  encoding_strategy: "onehot"            # or "ordinal"
  scaling_strategy: "standard"           # or "minmax", "robust"
```

---

## 📊 **Understanding Your Results**

### **Data Quality Scoring**
- **90-100**: Excellent quality, minimal preprocessing needed
- **70-89**: Good quality, some cleaning recommended
- **50-69**: Fair quality, significant preprocessing required
- **<50**: Poor quality, extensive cleaning needed

### **Output Locations**
```
📁 Your Results:
├── evaluation_results/     📈 HTML reports and visualizations
├── models/                🤖 Trained model files (.pkl)
├── docs/                  📋 Data profiling reports
├── mlruns/                🔬 MLflow experiment tracking
├── logs/                  📝 Execution logs
└── pipeline_outputs/      📦 Versioned pipeline outputs
```

### **MLflow Integration**
After running Step 3, access your experiments:
```bash
# Start MLflow UI
mlflow ui

# Then visit: http://localhost:5000
```

---

## 🎯 **Common Workflows**

### **New Dataset Workflow**
```bash
# 1. First-time analysis
python 1_inspect_data.py --data data/new_dataset.csv --detailed

# 2. Review config/unified_config_v2.yaml and adjust settings

# 3. Test pipeline
python 2_test_pipeline.py --data data/new_dataset.csv

# 4. Run production pipeline
python 3_run_pipeline.py
```

### **Debugging Issues**
```bash
# Test specific stage
python 2_test_pipeline.py --start-stage 3

# Check data at any stage using 'inspect' command

# Review logs
tail -f logs/pipeline_v2.log
```

### **Business Analysis**
```bash
# Run with business features enabled
python 3_run_pipeline.py --environment production

# Check business KPI results in evaluation_results/
```

---

## � **Setup Requirements**

### **First Time Setup**
```bash
# Install dependencies (if using uv)
uv sync

# Or with pip
pip install -r requirements.txt

# Initialize MLflow (automatic on first run)
```

### **Verify Installation**
```bash
# Quick test
python 1_inspect_data.py --data data/telco_churn_data.csv
```

---

## 🆕 **Enhanced Pipeline Workflow (NEW)**

### **Human-in-the-Loop Data Processing**

The enhanced pipeline introduces a **human-configurable** approach to data cleaning:

#### **1. Enhanced Data Profiling**
```bash
# Generates both HTML reports AND machine-readable data
python src/1_data_profiling/enhanced_data_profiler.py
```

**Outputs:**
- 📄 `docs/data_profiling_report.html` - Visual HTML report
- 📊 `docs/raw_profiling_data.json` - Machine-readable profile data
- 🔧 `docs/cleaning_config_template.yaml` - **Auto-generated cleaning configuration**

#### **2. Manual Configuration Review** ⏸️
**The pipeline HALTS here for human review!**

Review and modify `docs/cleaning_config_template.yaml`:
```yaml
column_transformations:
  age:
    imputation: 'median'
    outlier_treatment: 'cap'
    scaling: 'standard'
    transformation: null
  
  balance:
    imputation: 'median'
    outlier_treatment: 'remove'  # Human decision: remove vs cap
    transformation: 'log'        # Human decision: log transform
    scaling: 'robust'            # Human choice: robust vs standard
  
  job:
    encoding: 'onehot'           # Human decision: onehot vs target encoding
    text_cleaning: true
```

#### **3. Column-Specific Data Cleaning**
```bash
# Applies your customized configuration
python src/2_data_cleaning/data_cleaner.py
```

**Features:**
- ✅ **Per-column transformation rules** based on your YAML config
- ✅ **Automatic delimiter detection** (comma vs semicolon CSV files)
- ✅ **Detailed transformation logging** showing exactly what happened to each column
- ✅ **Fallback to global settings** for columns not specified

#### **4. Configuration Templates**
Example column-specific transformations:
```yaml
# Financial data column
balance:
  outlier_treatment: 'remove'    # Remove extreme outliers
  transformation: 'log'          # Log transform for skewed financial data
  scaling: 'robust'              # Robust scaling for financial amounts

# Categorical column  
job:
  encoding: 'onehot'             # One-hot for low cardinality
  text_cleaning: true            # Clean job titles

# Temporal data
age:
  outlier_treatment: 'cap'       # Cap outliers (people can't be 200 years old)
  scaling: 'standard'            # Standard scaling for age
```

### **Why This Matters**
- 🎯 **Domain Expertise**: Humans make the final decisions about transformations
- 🔍 **Transparency**: See exactly what transformations are applied to each column
- ⚙️ **Flexibility**: Different datasets need different approaches
- 📊 **Reproducibility**: YAML configs can be version controlled and reused

---

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `uv sync` or check Python environment |
| Data quality score < 50 | Use Step 2 to identify and fix data issues |
| Training fails | Check encoding/scaling strategies in config |
| MLflow errors | Ensure MLflow is installed: `pip install mlflow` |
| Memory issues | Reduce `max_models` in configuration |

### **Getting Help**
1. **Check logs**: `logs/pipeline_v2.log`
2. **Use interactive mode**: Step 2 with `inspect` commands
3. **Review configuration**: `config/unified_config_v2.yaml`
4. **Check MLflow UI**: http://localhost:5000

---

## � **Advanced Features**

### **Business Alignment**
- Automatic feature selection based on business rules
- KPI tracking and ROI calculation
- Stakeholder-friendly reports

### **Production Ready**
- MLflow model registry integration
- Automated model versioning
- Production monitoring setup
- Data drift detection

### **Extensibility**
- Plugin system for custom features
- Configurable business rules
- Custom model evaluation metrics

---

## 🎉 **Success Indicators**

You'll know everything is working when:
- ✅ Data quality score appears in Step 1
- ✅ Interactive inspection works in Step 2
- ✅ Models train successfully in Step 3
- ✅ MLflow UI shows your experiments
- ✅ HTML reports are generated in `evaluation_results/`

---

## 📞 **Next Steps**

1. **Run the 3-step workflow** with your data
2. **Explore MLflow UI** for experiment tracking
3. **Review HTML reports** for model insights
4. **Customize business rules** for your domain
5. **Deploy models** using MLflow model registry

**Ready to start?** Run: `python 1_inspect_data.py --data data/your_data.csv --detailed`
