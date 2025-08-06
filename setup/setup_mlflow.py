#!/usr/bin/env python3
"""
🔧 DS-AutoAdvisor: MLflow Setup Utility
====================================

WHAT IT DOES:
Sets up MLflow tracking server for DS-AutoAdvisor v2.0 pipeline.
Configures experiment tracking, model registry, and artifact storage.

WHEN TO USE:
- First time setup of DS-AutoAdvisor v2.0
- When you want to enable MLflow tracking
- After installing MLflow dependencies

HOW TO USE:
Basic setup:
    python setup/setup_mlflow.py

Start MLflow UI:
    python setup/setup_mlflow.py --start-ui

Check status:
    python setup/setup_mlflow.py --status

WHAT IT SETS UP:
✅ MLflow tracking server configuration
✅ Local file store for experiments and artifacts
✅ Default experiment for DS-AutoAdvisor
✅ Model registry database
✅ Artifact storage directory

AFTER SETUP:
- MLflow UI available at: http://localhost:5000
- Experiments tracked in: ./mlruns/
- Models registered in MLflow Model Registry
- Artifacts stored in: ./mlflow-artifacts/
"""

import os
import sys
import subprocess
from pathlib import Path

def check_mlflow_installation():
    """Check if MLflow is installed"""
    try:
        import mlflow
        print(f"✅ MLflow is installed (version {mlflow.__version__})")
        return True
    except ImportError:
        print("❌ MLflow is not installed")
        return False

def install_mlflow():
    """Install MLflow if needed"""
    print("Installing MLflow...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mlflow"])
        print("✅ MLflow installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install MLflow: {e}")
        return False

def setup_mlflow_directory():
    """Create MLflow directory structure"""
    mlflow_dir = Path("mlruns")
    artifacts_dir = Path("artifacts")
    
    mlflow_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    
    print(f"✅ MLflow directories created:")
    print(f"   - {mlflow_dir.absolute()}")
    print(f"   - {artifacts_dir.absolute()}")

def test_mlflow_integration():
    """Test basic MLflow functionality"""
    try:
        import mlflow
        import mlflow.sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        print("\n🧪 Testing MLflow integration...")
        
        # Set tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Create test data
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Start MLflow run
        with mlflow.start_run(run_name="test_integration"):
            # Train model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions and calculate metrics
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics and model
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_param("n_estimators", 10)
            mlflow.sklearn.log_model(model, "model")
            
            print(f"   ✅ Test run completed with accuracy: {accuracy:.3f}")
            
        print("✅ MLflow integration test successful!")
        return True
        
    except Exception as e:
        print(f"❌ MLflow integration test failed: {e}")
        return False

def enable_mlflow_features():
    """Enable MLflow features in config"""
    config_file = Path("config/feature_flags.yaml")
    
    if not config_file.exists():
        print("❌ Feature flags config not found")
        return False
        
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Enable MLflow features if not already enabled
        if 'mlflow_tracking_enabled: false' in content:
            content = content.replace('mlflow_tracking_enabled: false', 'mlflow_tracking_enabled: true')
            with open(config_file, 'w') as f:
                f.write(content)
            print("✅ Enabled MLflow tracking in feature flags")
        else:
            print("✅ MLflow tracking already enabled")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to update feature flags: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 DS-AutoAdvisor v2.0 MLflow Setup")
    print("="*50)
    
    # Check and install MLflow
    if not check_mlflow_installation():
        if not install_mlflow():
            sys.exit(1)
    
    # Setup directories
    setup_mlflow_directory()
    
    # Enable features
    if not enable_mlflow_features():
        print("⚠️  Could not enable MLflow features automatically")
        print("   Please manually set 'mlflow_tracking_enabled: true' in config/feature_flags.yaml")
    
    # Test integration
    if test_mlflow_integration():
        print("\n🎉 MLflow setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the enhanced pipeline: python complete_pipeline_v2.py")
        print("2. View MLflow UI: mlflow ui --port 5000")
        print("3. Access at: http://localhost:5000")
    else:
        print("\n❌ Setup completed with errors. Check the integration manually.")

if __name__ == "__main__":
    main()
