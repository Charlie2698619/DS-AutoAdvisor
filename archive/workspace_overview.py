#!/usr/bin/env python3
"""
🗂️ DS-AutoAdvisor v2.0 - Workspace Overview
==========================================

ORGANIZED WORKSPACE STRUCTURE:

📋 MAIN WORKFLOW (Run in order):
├── 1_inspect_data.py      🔍 Step 1: Analyze your data quality & get config suggestions
├── 2_test_pipeline.py     🧪 Step 2: Test pipeline stages interactively  
└── 3_run_pipeline.py      🚀 Step 3: Run production pipeline with MLflow

🏗️ CORE INFRASTRUCTURE:
├── src/                   📦 Pipeline components organized by workflow:
│   ├── 1_data_profiling/  📊 Data analysis and quality assessment
│   ├── 2_data_cleaning/   🧹 Data preprocessing and validation
│   ├── 3_advisory/        🤖 Model recommendations and insights
│   ├── 4_model_training/  🏋️ Machine learning model training
│   ├── 5_evaluation/      📈 Model evaluation and reporting
│   ├── data_quality_system/ 🔍 Advanced data quality framework
│   ├── infrastructure/    ⚙️ Core infrastructure (config, metadata, MLflow)
│   ├── monitoring/        📡 Health checks and compliance
│   └── utils/             🛠️ Utility scripts and helpers
├── config/                ⚙️ Configuration files (unified_config_v2.yaml)
├── plugins/               🔌 Custom feature selection & processing plugins
└── setup/                 🔧 Setup utilities (MLflow, dependencies)

📊 DATA & RESULTS:
├── data/                  📁 Your datasets (CSV files)
├── models/                🤖 Trained model files (.pkl)
├── evaluation_results/    📈 HTML reports & evaluation plots
├── mlruns/                🔬 MLflow experiment tracking
├── docs/                  📋 Data profiling & documentation
└── logs/                  📝 Execution logs & audit trails

🧪 TESTING & ARCHIVE:
├── tests/                 ✅ System tests & validation
├── archive/               📦 Old versions & backups
└── utils/                 🛠️  Future utility scripts

📖 DOCUMENTATION:
├── QUICK_START.md         🚀 3-step workflow guide  
├── README.md              📖 Project overview
├── PRODUCTION_SETUP.md    🏭 Enterprise production deployment
├── STATISTICAL_METHODS_ANALYSIS.md 📊 Complete statistical methods & assumptions analysis
├── BUSINESS_FEATURES_GUIDE.md 💼 Business alignment features implementation guide
└── SRC_ORGANIZATION_SUMMARY.md 📁 Source code organization guide

🔌 BUSINESS FEATURES (NEW IN V2.0):
├── plugins/feature_selection/ 🎯 Business-aligned feature selection
│   ├── business_feature_selector.py  # Main feature selection plugin
│   ├── statistical_selectors.py      # Statistical selection methods
│   ├── ml_selectors.py               # ML-based selection methods
│   └── human_interface.py            # Human approval interface
├── plugins/business_metrics/  💰 KPI tracking and ROI analysis
│   └── kpi_tracker.py               # Business KPI and ROI calculator
├── config/business_rules.yaml   📋 Feature selection business rules
├── config/business_kpis.yaml    📊 Custom KPI definitions and ROI parameters
└── evaluation_results/business_metrics/ 📈 Business performance reports

✨ ENHANCED FEATURES:
• Business-aligned feature selection with human oversight
• Custom KPI tracking with ROI analysis  
• Multi-model feature consistency handling
• Business-ML correlation monitoring
• Stakeholder-friendly reporting
• Compliance and governance controls

QUICK START:
1. python 1_inspect_data.py --data data/your_data.csv
2. python 2_test_pipeline.py --data data/your_data.csv  
3. python 3_run_pipeline.py

For detailed help, check QUICK_START.md
"""

def show_workspace_status():
    """Show current workspace organization status"""
    from pathlib import Path
    import os
    
    print("🗂️  DS-AutoAdvisor v2.0 Workspace Status")
    print("=" * 50)
    
    # Check main scripts
    main_scripts = ["1_inspect_data.py", "2_test_pipeline.py", "3_run_pipeline.py"]
    print("\n📋 Main Workflow Scripts:")
    for i, script in enumerate(main_scripts, 1):
        if Path(script).exists():
            print(f"   ✅ Step {i}: {script}")
        else:
            print(f"   ❌ Step {i}: {script} (missing)")
    
    # Check key directories
    key_dirs = ["src", "config", "data", "models", "setup", "tests"]
    print("\n🏗️  Key Directories:")
    for dir_name in key_dirs:
        if Path(dir_name).exists():
            file_count = len(list(Path(dir_name).rglob("*")))
            print(f"   ✅ {dir_name}/ ({file_count} files)")
        else:
            print(f"   ❌ {dir_name}/ (missing)")
    
    # Check organized src structure
    src_modules = [
        "1_data_profiling", "2_data_cleaning", "3_advisory", 
        "4_model_training", "5_evaluation", "data_quality_system",
        "infrastructure", "monitoring", "utils"
    ]
    print("\n📦 Organized Source Modules:")
    for module in src_modules:
        module_path = Path("src") / module
        if module_path.exists():
            py_files = len(list(module_path.glob("*.py")))
            print(f"   ✅ {module:<20} ({py_files} Python files)")
        else:
            print(f"   ❌ {module:<20} (missing)")
    
    # Check configuration
    config_files = ["config/unified_config_v2.yaml", "config/business_rules.yaml", "config/business_kpis.yaml"]
    print("\n⚙️  Configuration Files:")
    for config in config_files:
        if Path(config).exists():
            print(f"   ✅ {config}")
        else:
            print(f"   ❌ {config} (missing)")
    
    # Check business features
    business_dirs = ["plugins/feature_selection", "plugins/business_metrics"]
    print("\n💼 Business Features:")
    for dir_name in business_dirs:
        if Path(dir_name).exists():
            py_files = len(list(Path(dir_name).glob("*.py")))
            print(f"   ✅ {dir_name}/ ({py_files} Python files)")
        else:
            print(f"   ❌ {dir_name}/ (missing)")
    
    # Check documentation
    docs = ["QUICK_START.md", "README.md", "PRODUCTION_SETUP.md", 
            "STATISTICAL_METHODS_ANALYSIS.md", "docs/BUSINESS_FEATURES_GUIDE.md"]
    print("\n📖 Documentation:")
    for doc in docs:
        if Path(doc).exists():
            print(f"   ✅ {doc}")
        else:
            print(f"   ❌ {doc} (missing)")
    
    print("\n🎯 Ready to use! Start with: python 1_inspect_data.py --data your_data.csv")
    print("💼 Business features available: Feature selection, KPI tracking, ROI analysis")

if __name__ == "__main__":
    show_workspace_status()
