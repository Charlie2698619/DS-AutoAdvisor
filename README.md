DS-AutoAdvisor: Automated & Industrial-Grade ML Pipeline v2.0
==============================================================

Overview
--------
DS-AutoAdvisor v2.0 is a modular, production-ready machine learning pipeline orchestrator  
with **enhanced column-specific data processing capabilities**.  
It automates the full ML workflow from data profiling to model evaluation,  
with support for human-in-the-loop checkpoints and optional industrial-grade features  
(security, monitoring, audit logging).

🆕 **New in v2.0:**
- **Enhanced Data Profiling**: Machine-readable profile data with column-specific recommendations
- **Column-Specific Data Cleaning**: Individual column transformation rules via YAML configuration  
- **Human-in-the-Loop Design**: Pipeline halts for manual configuration review
- **YAML-Based Configuration**: Per-column settings for imputation, outlier handling, scaling, encoding
- **Automatic Delimiter Detection**: Handles various CSV formats (comma, semicolon, etc.)

Project Structure & Main Scripts
-------------------------------

### **Core Pipeline Scripts**
- `1_inspect_data.py`  
    Interactive data inspection with quality assessment and configuration recommendations.

- `2_test_pipeline.py`  
    Interactive pipeline testing with **enhanced profiling and column-specific cleaning**.

- `3_run_pipeline.py`  
    Production pipeline execution with full MLflow tracking.

### **Enhanced Data Processing**
- `src/1_data_profiling/enhanced_data_profiler.py` 🆕  
    Advanced data profiling with machine-readable output and YAML template generation.

- `src/2_data_cleaning/data_cleaner.py` 🆕  
    Column-specific data cleaning with YAML configuration support.

### **Legacy Components**
- `complete_pipeline.py` (archived)  
    Legacy v1.0 pipeline orchestrator for backward compatibility.

- `src/profiling_1/data_profiler.py`  
    Traditional data profiling script.

- `src/correction_2/data_cleaner.py` (legacy)  
    Traditional data cleaning script.

- `src/advisor_3/assumption_checker.py`  
    Checks statistical assumptions on the dataset.

- `src/advisor_3/model_recommender.py`  
    Recommends suitable ML models based on data and assumptions.

- `src/pipeline_4/trainer.py`  
    Trains multiple ML models, performs hyperparameter tuning, and saves results.

- `src/pipeline_4/evaluator.py`  
    Evaluates trained models, generates metrics, and creates analysis reports.

### **Configuration Files**
- `config/unified_config_v2.yaml`  
    Main v2.0 pipeline configuration with business rules integration.

- `docs/cleaning_config_template.yaml` 🆕  
    Auto-generated column-specific cleaning configuration template.

- `config/unified_config.yaml`  
    Legacy v1.0 configuration file.

### **Output Directories**
- `pipeline_outputs/`  
    Stores all generated reports, models, and logs.

- `docs/`  
    Contains profiling reports, configuration templates, and documentation.

- `mlruns/`  
    MLflow experiment tracking and model registry.

Enhanced Workflow v2.0
----------------------

1. **Enhanced Profiling**: Analyzes data with machine-readable output and column recommendations.
2. **🛑 Human Review**: Pipeline halts for configuration review and modification.
3. **Column-Specific Cleaning**: Applies individual transformations per column based on YAML config.
4. **ML Advisory**: Checks assumptions and recommends models.
5. **Training & Evaluation**: MLflow-tracked model training with comprehensive evaluation.

### **Workflow Example**
```bash
# Step 1: Enhanced data profiling (generates YAML config)
python 2_test_pipeline.py --data data/your_data.csv

# Step 2: Review and modify generated configuration
# Edit: docs/cleaning_config_template.yaml

# Step 3: Continue with column-specific cleaning
# Pipeline automatically uses your YAML configuration

# Step 4: Production run
python 3_run_pipeline.py
```
4. **Training**: Trains models, tunes hyperparameters, saves best models.
5. **Evaluation**: Evaluates models, generates metrics and visualizations.

Each stage can include a human checkpoint for review, retry, or configuration adjustment.

How to Use
----------

1. **Install dependencies** (if not already):

    ```sh
    pip install -r requirements.txt
    ```

2. **Prepare your configuration**:  
    Edit `config/unified_config.yaml` to set paths, parameters, and workflow stages.

3. **Run the pipeline**:

    ```sh
    python complete_pipeline.py --config config/unified_config.yaml
    ```

    **Optional modes:**

    - Interactive (human checkpoints):  
      ```sh
      python complete_pipeline.py --interactive
      ```
    - Fully automated:  
      ```sh
      python complete_pipeline.py --automated
      ```
    - Industrial-grade (security, monitoring):  
      ```sh
      python complete_pipeline.py --industrial
      ```
    - Run specific stages:  
      ```sh
      python complete_pipeline.py --stages profiling cleaning
      ```

4. **Check outputs**:
    - Reports, models, and logs are saved in [pipeline_outputs](http://_vscodecontentref_/1)
    - A summary report is generated at [pipeline_summary.json](http://_vscodecontentref_/2)

Extending & Customizing
-----------------------
- Add new stages by creating scripts in [src](http://_vscodecontentref_/3) and updating the config.
- Adjust parameters and workflow in [unified_config.yaml](http://_vscodecontentref_/4).
- Use the interactive mode to adjust configuration on the fly.

Support & Issues
----------------
For questions or issues, please open an issue on the project’s GitHub page.