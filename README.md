DS-AutoAdvisor: Automated & Industrial-Grade ML Pipeline
========================================================

Overview
--------
DS-AutoAdvisor is a modular, production-ready machine learning pipeline orchestrator. 
It automates the full ML workflow from data profiling to model evaluation, 
with support for human-in-the-loop checkpoints and optional industrial-grade features 
(security, monitoring, audit logging).

Project Structure & Main Scripts
-------------------------------

- complete_pipeline.py
    The main orchestrator. Runs the entire pipeline using a unified configuration.
    Handles all stages, logging, human checkpoints, and industrial features.

- src/profiling_1/data_profiler.py
    Data profiling script. Generates data reports, schema, and visualizations.

- src/correction_2/data_cleaner.py
    Data cleaning script. Handles missing values, outliers, encoding, and scaling.

- src/advisor_3/assumption_checker.py
    Checks statistical assumptions on the dataset.

- src/advisor_3/model_recommender.py
    Recommends suitable ML models based on data and assumptions.

- src/pipeline_4/trainer.py
    Trains multiple ML models, performs hyperparameter tuning, and saves results.

- src/pipeline_4/evaluator.py
    Evaluates trained models, generates metrics, and creates analysis reports.

- src/monitoring/industrial_integration.py
    (Optional) Adds industrial-grade features: security, monitoring, audit logging.

- config/unified_config.yaml
    Central configuration file for all pipeline stages and settings.

- pipeline_outputs/
    Stores all generated reports, models, and logs.

Workflow
--------

1. **Profiling**: Analyzes the input data, generates reports and schema.
2. **Cleaning**: Cleans the data (missing values, outliers, encoding, scaling).
3. **ML Advisory**: Checks assumptions and recommends models.
4. **Training**: Trains models, tunes hyperparameters, saves best models.
5. **Evaluation**: Evaluates models, generates metrics and visualizations.

Each stage can include a human checkpoint for review, retry, or configuration adjustment.

How to Use
----------

1. **Install dependencies** (if not already):
    pip install -r requirements.txt

2. **Prepare your configuration**:
    Edit `config/unified_config.yaml` to set paths, parameters, and workflow stages.

3. **Run the pipeline**:
    python complete_pipeline.py --config config/unified_config.yaml

    Optional modes:
    - Interactive (human checkpoints): 
        python complete_pipeline.py --interactive
    - Fully automated: 
        python complete_pipeline.py --automated
    - Industrial-grade (security, monitoring): 
        python complete_pipeline.py --industrial
    - Run specific stages: 
        python complete_pipeline.py --stages profiling cleaning

4. **Check outputs**:
    - Reports, models, and logs are saved in `pipeline_outputs/`
    - A summary report is generated at `pipeline_outputs/reports/pipeline_summary.json`

Extending & Customizing
-----------------------
- Add new stages by creating scripts in `src/` and updating the config.
- Adjust parameters and workflow in `config/unified_config.yaml`.
- Use the interactive mode to adjust configuration on the fly.

Support & Issues
----------------
For questions or issues, please open an issue on the project’s GitHub page.
