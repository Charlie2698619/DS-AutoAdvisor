#!/usr/bin/env python3
"""
🚀 DS-AutoAdvisor: Step 3 - Simplified Production Pipeline
========================================================

SIMPLIFIED PURPOSE:
Takes validated results from Step 2 and generates production-ready artifacts
without re-running the entire ML pipeline.

WHAT IT DOES:
✅ Loads best models from Step 2
✅ Generates deployment packages
✅ Creates serving scripts and documentation
✅ Packages everything for handoff

WHEN TO USE:
- After completing Stage 2 testing
- To package models for deployment
- To create handoff documentation
- For final deliverables

HOW TO USE:
Auto-detect latest Stage 2:
    python 03_simplified_production.py

Use specific Stage 2 results:
    python 03_simplified_production.py --stage2-dir pipeline_outputs/02_stage_testing_custom_20250811_113710

With MLflow tracking:
    python 03_simplified_production.py --enable-mlflow

STAGES (Simplified):
1. Results Validation & Loading
2. Model Packaging & Optimization  
3. Deployment Artifacts Generation

OUTPUTS:
✅ Production model packages
✅ Serving scripts and API templates
✅ Deployment configurations
✅ Executive summary reports
✅ Maintenance documentation
"""

import sys, os, argparse, time, warnings, json, yaml, traceback, shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# Add config manager import
from utils.simplified_config_manager import SimplifiedConfigManager

# Optional MLflow integration
try:
    from src.infrastructure.mlflow_integration import MLflowManager, MLflowConfig
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))


class SimplifiedProductionPipeline:
    """Simplified production pipeline focused on artifact generation"""
    
    def __init__(self, stage2_results_dir: str = None, run_mode: str = "custom", 
                 force_enable_mlflow: bool = False, output_base: str = "pipeline_outputs"):
        self.output_base = Path(output_base)
        self.stage2_dir = self._find_latest_stage2() if not stage2_results_dir else Path(stage2_results_dir)
        self.run_mode = run_mode.lower()
        
        if self.run_mode not in ["fast", "custom"]:
            print(f"⚠️ Invalid run_mode '{self.run_mode}' defaulting to 'custom'")
            self.run_mode = "custom"
        
        # Load configuration
        self.config_manager = SimplifiedConfigManager("config/unified_config_v3.yaml")
        self.config = self.config_manager.config
        
        # Simplified output structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.outputs = {
            'base': self.output_base / f"03_production_artifacts_{timestamp}",
            'models': self.output_base / f"03_production_artifacts_{timestamp}/models",
            'deployment': self.output_base / f"03_production_artifacts_{timestamp}/deployment", 
            'docs': self.output_base / f"03_production_artifacts_{timestamp}/documentation",
            'reports': self.output_base / f"03_production_artifacts_{timestamp}/reports"
        }
        
        for output_dir in self.outputs.values():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow if enabled
        infra_cfg = self.config_manager.get_infrastructure_config() or {}
        feature_flags = (infra_cfg.get('feature_flags') or {})
        print(f"🔍 Debug: MLflow feature flag = {feature_flags.get('mlflow_tracking', False)}")
        print(f"🔍 Debug: Force enable MLflow = {force_enable_mlflow}")
        print(f"🔍 Debug: Has MLflow = {HAS_MLFLOW}")
        
        self.mlflow_enabled = (feature_flags.get('mlflow_tracking', False) or force_enable_mlflow) and HAS_MLFLOW
        self.mlflow_manager = None
        
        print(f"🔍 Debug: MLflow enabled = {self.mlflow_enabled}")
        
        if self.mlflow_enabled:
            mlflow_section = infra_cfg.get('mlflow', {})
            print(f"🔍 Debug: MLflow config section = {mlflow_section}")
            
            try:
                # Use your updated config values
                tracking_uri = mlflow_section.get('tracking_uri', 'file:///mnt/c/Users/tony3/Desktop/tidytuesday/ds-autoadvisor/mlruns')
                experiment_name = mlflow_section.get('experiment_name', 'ds-autoadvisor-v3')
                artifact_location = mlflow_section.get('artifact_location', './mlruns')
                auto_log = mlflow_section.get('auto_log', True)
                log_system_metrics = mlflow_section.get('log_system_metrics', True)
                log_artifacts = mlflow_section.get('log_artifacts', True)
                
                ml_cfg = MLflowConfig(
                    tracking_uri=tracking_uri,
                    experiment_name=experiment_name,
                    artifact_location=artifact_location,
                    auto_log=auto_log,
                    log_system_metrics=log_system_metrics,
                    log_artifacts=log_artifacts
                )
                self.mlflow_manager = MLflowManager(ml_cfg)
                print(f"🧪 MLflow tracking enabled for production artifacts")
                print(f"   📊 Tracking URI: {tracking_uri}")
                print(f"   🎯 Experiment: {experiment_name}")
                print(f"   📂 Artifacts: {artifact_location}")
            except Exception as e:
                print(f"⚠️ Failed to initialize MLflow: {e}")
                import traceback
                traceback.print_exc()
                self.mlflow_enabled = False
        
        # Pipeline state
        self.execution_state = {
            'start_time': datetime.now(),
            'completed_stages': [],
            'failed_stages': [],
            'stage_results': {}
        }
        
        print(f"🚀 Simplified Production Pipeline Initialized")
        print(f"📁 Output directory: {self.outputs['base']}")
        print(f"🔍 Using Stage 2 results: {self.stage2_dir}")
        print(f"⚙️ Run mode: {self.run_mode}")
    
    def _find_latest_stage2(self) -> Optional[Path]:
        """Find the latest Stage 2 results directory"""
        stage2_dirs = list(self.output_base.glob("02_stage_testing_*"))
        if stage2_dirs:
            latest = max(stage2_dirs, key=lambda p: p.stat().st_mtime)
            print(f"🔍 Auto-detected latest Stage 2: {latest}")
            return latest
        else:
            print("❌ No Stage 2 directory found. Run 02_stage_testing.py first.")
            return None
    
    def run_simplified_pipeline(self) -> bool:
        """Execute simplified production pipeline"""
        try:
            print("\n" + "="*80)
            print("🚀 STARTING SIMPLIFIED PRODUCTION PIPELINE")
            print("="*80)
            
            stages = [
                ("validate_stage2", "📊 Validate Stage 2 Results"),
                ("package_models", "📦 Package Best Models"),
                ("generate_deployment", "🚀 Generate Deployment Assets"),
                ("create_documentation", "📖 Create Documentation Package")
            ]
            
            if self.mlflow_enabled and self.mlflow_manager:
                try:
                    self.current_run_id = self.mlflow_manager.start_pipeline_run(
                        self.config, 
                        run_name=f"production-artifacts-{self.run_mode}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    print(f"✅ MLflow run started: {self.current_run_id}")
                except Exception as e:
                    print(f"⚠️ MLflow start failed: {e}")
                    self.mlflow_enabled = False
            
            for i, (stage_id, stage_name) in enumerate(stages, 1):
                print(f"\n{'='*20} STAGE {i}/{len(stages)}: {stage_name} {'='*20}")
                
                stage_start = datetime.now()
                
                if stage_id == "validate_stage2":
                    success = self._validate_stage2_results()
                elif stage_id == "package_models":
                    success = self._package_best_models()
                elif stage_id == "generate_deployment":
                    success = self._generate_deployment_assets()
                elif stage_id == "create_documentation":
                    success = self._create_documentation_package()
                else:
                    print(f"❌ Unknown stage: {stage_id}")
                    success = False
                
                stage_time = (datetime.now() - stage_start).total_seconds()
                
                # Log stage metrics to MLflow
                if self.mlflow_enabled and self.mlflow_manager:
                    try:
                        import mlflow
                        mlflow.log_metric(f"stage_{stage_id}_duration_seconds", stage_time)
                        mlflow.log_metric(f"stage_{stage_id}_success", 1 if success else 0)
                    except Exception as e:
                        print(f"⚠️ MLflow stage logging failed: {e}")
                
                if success:
                    self.execution_state['completed_stages'].append(stage_id)
                    print(f"✅ Stage completed successfully ({stage_time:.2f}s)")
                else:
                    self.execution_state['failed_stages'].append(stage_id)
                    print(f"❌ Stage failed ({stage_time:.2f}s)")
                    return False
            
            return self._finalize_artifacts()
            
        except KeyboardInterrupt:
            print("\n🛑 Pipeline interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Pipeline crashed: {e}")
            traceback.print_exc()
            return False
    
    def _validate_stage2_results(self) -> bool:
        """Validate that Stage 2 completed successfully"""
        try:
            print("📊 Validating Stage 2 results...")
            
            if not self.stage2_dir or not self.stage2_dir.exists():
                print("❌ Stage 2 directory not found")
                return False
            
            # Look for training report
            training_reports = list(self.stage2_dir.glob("**/training_report.json"))
            if not training_reports:
                print("❌ No training report found from Stage 2")
                return False
            
            with open(training_reports[0], 'r') as f:
                self.training_results = json.load(f)
            
            # Look for cleaned data
            cleaned_data_files = list(self.stage2_dir.glob("**/cleaned_data.csv")) or \
                               list(self.stage2_dir.glob("**/cleaned_data/*.csv"))
            
            if cleaned_data_files:
                self.cleaned_data_path = cleaned_data_files[0]
                print(f"✅ Found cleaned data: {self.cleaned_data_path}")
            else:
                print("⚠️ No cleaned data found, will use original data")
                self.cleaned_data_path = None
            
            # Look for model files
            model_files = list(self.stage2_dir.glob("**/*.pkl"))
            self.available_models = model_files
            
            models_count = len(self.training_results.get('model_rankings', []))
            files_count = len(model_files)
            
            print(f"✅ Stage 2 validation successful")
            print(f"   📊 Models in report: {models_count}")
            print(f"   💾 Model files found: {files_count}")
            print(f"   📁 Cleaned data available: {'Yes' if self.cleaned_data_path else 'No'}")
            
            # Show recommended model
            best_model = self.training_results.get('summary', {}).get('best_model', 'Unknown')
            print(f"   🏆 Recommended model: {best_model}")
            
            self.execution_state['stage_results']['validate_stage2'] = {
                'training_report_path': str(training_reports[0]),
                'models_count': models_count,
                'model_files_count': files_count,
                'cleaned_data_available': self.cleaned_data_path is not None,
                'success': True
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Stage 2 validation failed: {e}")
            return False
    
    def _package_best_models(self) -> bool:
        """Package top 3 models for deployment"""
        try:
            print("📦 Packaging best models for deployment...")
            
            # Get top 3 models from training results (using model_rankings)
            model_rankings = self.training_results.get('model_rankings', [])[:3]
            packaged_models = []
            
            print(f"📦 Packaging top {len(model_rankings)} models...")
            
            for i, model_ranking in enumerate(model_rankings, 1):
                model_name = model_ranking.get('model', f'model_{i}')
                
                # Find corresponding model file
                model_file = None
                for file_path in self.available_models:
                    if model_name in str(file_path):
                        model_file = file_path
                        break
                
                if model_file and model_file.exists():
                    # Copy model to production directory
                    production_model_path = self.outputs['models'] / f"{model_name}.pkl"
                    shutil.copy2(model_file, production_model_path)
                    
                    # Get detailed results for this model
                    detailed_results = self.training_results.get('detailed_results', {}).get(model_name, {})
                    
                    # Create model metadata
                    model_metadata = {
                        'name': model_name,
                        'original_file': str(model_file),
                        'production_file': str(production_model_path),
                        'ranking': model_ranking,
                        'detailed_performance': detailed_results,
                        'packaging_timestamp': datetime.now().isoformat()
                    }
                    
                    metadata_path = self.outputs['models'] / f"{model_name}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(model_metadata, f, indent=2, default=str)
                    
                    packaged_models.append({
                        'name': model_name,
                        'model_file': str(production_model_path),
                        'metadata_file': str(metadata_path),
                        'ranking': model_ranking,
                        'performance': detailed_results
                    })
                    
                    print(f"   ✅ Packaged: {model_name}")
                else:
                    print(f"   ⚠️ Model file not found for: {model_name}")
            
            # Create models index
            models_index = {
                'timestamp': datetime.now().isoformat(),
                'total_models': len(packaged_models),
                'models': packaged_models,
                'recommended_model': packaged_models[0]['name'] if packaged_models else None
            }
            
            index_path = self.outputs['models'] / "models_index.json"
            with open(index_path, 'w') as f:
                json.dump(models_index, f, indent=2, default=str)
            
            print(f"✅ Model packaging completed")
            print(f"   📦 Packaged models: {len(packaged_models)}")
            print(f"   🏆 Recommended model: {models_index['recommended_model']}")
            print(f"   📋 Models index: {index_path}")
            
            # Log model metrics to MLflow
            if self.mlflow_enabled and self.mlflow_manager:
                try:
                    import mlflow
                    mlflow.log_metric("packaged_models_count", len(packaged_models))
                    mlflow.log_param("recommended_model", models_index['recommended_model'])
                    
                    # Log performance metrics for each model
                    for model in packaged_models:
                        model_name = model['name']
                        ranking = model.get('ranking', {})
                        if 'primary_score' in ranking:
                            mlflow.log_metric(f"{model_name}_accuracy", ranking['primary_score'])
                        if 'training_time' in ranking:
                            mlflow.log_metric(f"{model_name}_training_time", ranking['training_time'])
                        if 'rank' in ranking:
                            mlflow.log_metric(f"{model_name}_rank", ranking['rank'])
                    
                    print(f"   📊 MLflow: Logged {len(packaged_models)} model metrics")
                except Exception as e:
                    print(f"   ⚠️ MLflow model logging failed: {e}")
            
            self.execution_state['stage_results']['package_models'] = {
                'packaged_count': len(packaged_models),
                'recommended_model': models_index['recommended_model'],
                'models_index_path': str(index_path),
                'packaged_models': packaged_models,
                'success': True
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Model packaging failed: {e}")
            return False
    
    def _generate_deployment_assets(self) -> bool:
        """Generate minimal deployment assets"""
        try:
            print("🚀 Generating deployment assets...")
            
            # Get packaged models info
            models_info = self.execution_state['stage_results']['package_models']
            recommended_model = models_info['recommended_model']
            
            # Create deployment configuration
            deployment_config = {
                'deployment_info': {
                    'created_at': datetime.now().isoformat(),
                    'recommended_model': recommended_model,
                    'total_models': models_info['packaged_count'],
                    'source_stage2_dir': str(self.stage2_dir),
                    'run_mode': self.run_mode
                },
                'model_info': {
                    'target_column': self.config['global']['target_column'],
                    'models_directory': str(self.outputs['models']),
                    'available_models': [m['name'] for m in models_info['packaged_models']]
                },
                'serving_config': {
                    'default_model': recommended_model,
                    'model_format': 'pickle',
                    'prediction_type': 'auto_detect',
                    'preprocessing_required': True
                }
            }
            
            # Save deployment configuration
            config_path = self.outputs['deployment'] / "deployment_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(deployment_config, f, indent=2, default_flow_style=False)
            
            # Create simple serving script
            serving_script = self._create_simple_serving_script(deployment_config)
            serving_path = self.outputs['deployment'] / "model_server.py"
            with open(serving_path, 'w') as f:
                f.write(serving_script)
            
            # Create API template
            api_template = self._create_api_template(deployment_config)
            api_path = self.outputs['deployment'] / "api_template.py"
            with open(api_path, 'w') as f:
                f.write(api_template)
            
            # Create requirements file
            requirements = self._create_requirements()
            req_path = self.outputs['deployment'] / "requirements.txt"
            with open(req_path, 'w') as f:
                f.write(requirements)
            
            # Create Docker file
            dockerfile = self._create_dockerfile()
            docker_path = self.outputs['deployment'] / "Dockerfile"
            with open(docker_path, 'w') as f:
                f.write(dockerfile)
            
            print(f"✅ Deployment assets generated")
            print(f"   ⚙️  Configuration: {config_path}")
            print(f"   🐍 Serving script: {serving_path}")
            print(f"   🌐 API template: {api_path}")
            print(f"   📦 Requirements: {req_path}")
            print(f"   🐳 Dockerfile: {docker_path}")
            
            self.execution_state['stage_results']['generate_deployment'] = {
                'config_path': str(config_path),
                'serving_script_path': str(serving_path),
                'api_template_path': str(api_path),
                'requirements_path': str(req_path),
                'dockerfile_path': str(docker_path),
                'success': True
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Deployment asset generation failed: {e}")
            return False
    
    def _create_documentation_package(self) -> bool:
        """Create comprehensive documentation package"""
        try:
            print("📖 Creating documentation package...")
            
            # Create executive summary
            print("   Creating executive summary...")
            executive_summary = self._create_executive_summary()
            summary_path = self.outputs['docs'] / "executive_summary.md"
            with open(summary_path, 'w') as f:
                f.write(executive_summary)
            
            # Create deployment guide
            print("   Creating deployment guide...")
            deployment_guide = self._create_deployment_guide()
            guide_path = self.outputs['docs'] / "deployment_guide.md"
            with open(guide_path, 'w') as f:
                f.write(deployment_guide)
            
            # Create model comparison report
            print("   Creating model comparison...")
            try:
                model_comparison = self._create_model_comparison()
                comparison_path = self.outputs['reports'] / "model_comparison.html"
                with open(comparison_path, 'w') as f:
                    f.write(model_comparison)
            except Exception as e:
                print(f"     Error in model comparison: {e}")
                raise
            
            # Create maintenance guide
            print("   Creating maintenance guide...")
            maintenance_guide = self._create_maintenance_guide()
            maintenance_path = self.outputs['docs'] / "maintenance_guide.md"
            with open(maintenance_path, 'w') as f:
                f.write(maintenance_guide)
            
            print(f"✅ Documentation package created")
            print(f"   📋 Executive summary: {summary_path}")
            print(f"   📖 Deployment guide: {guide_path}")
            print(f"   📊 Model comparison: {comparison_path}")
            print(f"   🔧 Maintenance guide: {maintenance_path}")
            
            self.execution_state['stage_results']['create_documentation'] = {
                'executive_summary_path': str(summary_path),
                'deployment_guide_path': str(guide_path),
                'model_comparison_path': str(comparison_path),
                'maintenance_guide_path': str(maintenance_path),
                'success': True
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Documentation creation failed: {e}")
            return False
    
    def _create_simple_serving_script(self, config: Dict[str, Any]) -> str:
        """Create a simple model serving script"""
        return f'''#!/usr/bin/env python3
"""
Simple Model Server
Generated by DS-AutoAdvisor Simplified Production Pipeline
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class SimpleModelServer:
    """Simple production model server"""
    
    def __init__(self, models_directory: str = "models"):
        """Initialize model server"""
        self.models_dir = Path(models_directory)
        self.models = {{}}
        self.default_model = "{config['serving_config']['default_model']}"
        self.target_column = "{config['model_info']['target_column']}"
        self.load_models()
    
    def load_models(self):
        """Load all available models"""
        model_files = list(self.models_dir.glob("*.pkl"))
        
        for model_file in model_files:
            model_name = model_file.stem
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                self.models[model_name] = model
                print(f"✅ Loaded model: {{model_name}}")
            except Exception as e:
                print(f"❌ Failed to load {{model_name}}: {{e}}")
        
        if not self.models:
            raise ValueError("No models loaded successfully")
        
        print(f"📊 Total models loaded: {{len(self.models)}}")
        print(f"🎯 Default model: {{self.default_model}}")
    
    def predict(self, data, model_name: str = None):
        """Make predictions"""
        if model_name is None:
            model_name = self.default_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{{model_name}}' not found")
        
        model = self.models[model_name]
        
        # Handle different input types
        if isinstance(data, pd.DataFrame):
            X = data
        elif isinstance(data, dict):
            X = pd.DataFrame([data])
        else:
            raise ValueError("Data must be DataFrame or dict")
        
        # Remove target column if present
        if self.target_column in X.columns:
            X = X.drop(columns=[self.target_column])
        
        predictions = model.predict(X)
        return predictions.tolist() if hasattr(predictions, 'tolist') else predictions
    
    def predict_proba(self, data, model_name: str = None):
        """Make probability predictions (if supported)"""
        if model_name is None:
            model_name = self.default_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{{model_name}}' not found")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model '{{model_name}}' does not support probability predictions")
        
        # Handle different input types
        if isinstance(data, pd.DataFrame):
            X = data
        elif isinstance(data, dict):
            X = pd.DataFrame([data])
        else:
            raise ValueError("Data must be DataFrame or dict")
        
        # Remove target column if present
        if self.target_column in X.columns:
            X = X.drop(columns=[self.target_column])
        
        probabilities = model.predict_proba(X)
        return probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
    
    def get_model_info(self):
        """Get information about loaded models"""
        return {{
            'available_models': list(self.models.keys()),
            'default_model': self.default_model,
            'target_column': self.target_column,
            'total_models': len(self.models)
        }}


def main():
    """Example usage"""
    try:
        # Initialize server
        server = SimpleModelServer("models")
        
        # Show model info
        info = server.get_model_info()
        print(f"\\nModel Server Info:")
        for key, value in info.items():
            print(f"  {{key}}: {{value}}")
        
        # Example prediction (uncomment and modify as needed)
        # sample_data = {{'feature1': 1.0, 'feature2': 2.0}}  # Replace with actual features
        # prediction = server.predict(sample_data)
        # print(f"\\nPrediction: {{prediction}}")
        
    except Exception as e:
        print(f"❌ Error: {{e}}")

if __name__ == "__main__":
    main()
'''
    
    def _create_api_template(self, config: Dict[str, Any]) -> str:
        """Create a simple API template"""
        return f'''#!/usr/bin/env python3
"""
Flask API Template
Generated by DS-AutoAdvisor Simplified Production Pipeline
"""

from flask import Flask, request, jsonify
import pandas as pd
import json
from model_server import SimpleModelServer

app = Flask(__name__)

# Initialize model server
try:
    model_server = SimpleModelServer("models")
    print("✅ Model server initialized")
except Exception as e:
    print(f"❌ Failed to initialize model server: {{e}}")
    model_server = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model_server is None:
        return jsonify({{'status': 'error', 'message': 'Model server not initialized'}}), 500
    
    return jsonify({{'status': 'healthy', 'models': model_server.get_model_info()}})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if model_server is None:
        return jsonify({{'error': 'Model server not initialized'}}), 500
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({{'error': 'No data provided'}}), 400
        
        # Extract model name if provided
        model_name = data.get('model_name', None)
        input_data = data.get('data', data)
        
        # Make prediction
        prediction = model_server.predict(input_data, model_name)
        
        return jsonify({{
            'prediction': prediction,
            'model_used': model_name or model_server.default_model,
            'status': 'success'
        }})
        
    except Exception as e:
        return jsonify({{'error': str(e)}}), 400

@app.route('/predict_proba', methods=['POST'])
def predict_proba():
    """Probability prediction endpoint"""
    if model_server is None:
        return jsonify({{'error': 'Model server not initialized'}}), 500
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({{'error': 'No data provided'}}), 400
        
        # Extract model name if provided
        model_name = data.get('model_name', None)
        input_data = data.get('data', data)
        
        # Make probability prediction
        probabilities = model_server.predict_proba(input_data, model_name)
        
        return jsonify({{
            'probabilities': probabilities,
            'model_used': model_name or model_server.default_model,
            'status': 'success'
        }})
        
    except Exception as e:
        return jsonify({{'error': str(e)}}), 400

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models endpoint"""
    if model_server is None:
        return jsonify({{'error': 'Model server not initialized'}}), 500
    
    return jsonify(model_server.get_model_info())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
    
    def _create_requirements(self) -> str:
        """Create requirements.txt"""
        return '''# DS-AutoAdvisor Production Requirements
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
pyyaml>=6.0
joblib>=1.1.0
flask>=2.0.0
gunicorn>=20.1.0
'''
    
    def _create_dockerfile(self) -> str:
        """Create Dockerfile"""
        return '''FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY models/ models/
COPY *.py .
COPY *.yaml .

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api_template:app"]
'''
    
    def _create_executive_summary(self) -> str:
        """Create executive summary"""
        models_info = self.execution_state['stage_results']['package_models']
        training_results = self.training_results
        
        # Get best model performance
        best_model = models_info['packaged_models'][0] if models_info['packaged_models'] else {}
        
        return f'''# DS-AutoAdvisor Production Summary

## Executive Summary

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Project:** DS-AutoAdvisor Production Deployment  
**Target:** {self.config['global']['target_column']}  

## Key Results

### Model Performance
- **Best Model:** {models_info['recommended_model']}
- **Total Models Trained:** {len(training_results.get('model_rankings', []))}
- **Production-Ready Models:** {models_info['packaged_count']}

### Data Summary
- **Source:** Stage 2 Testing Results
- **Data Quality:** Validated and cleaned
- **Feature Engineering:** Completed in Stage 2

### Deployment Status
- **Production Package:** ✅ Ready
- **Serving Infrastructure:** ✅ Generated
- **API Template:** ✅ Available
- **Documentation:** ✅ Complete

## Next Steps

1. **Test Deployment**
   - Review deployment assets in `deployment/` directory
   - Test model serving script with sample data
   - Validate API endpoints

2. **Production Setup**
   - Deploy to target environment
   - Configure monitoring and logging
   - Set up model versioning

3. **Maintenance**
   - Monitor model performance
   - Plan retraining schedule
   - Update documentation as needed

## File Structure

```
production_artifacts/
├── models/              # Production model files
├── deployment/          # Deployment configuration and scripts
├── documentation/       # Complete documentation
└── reports/            # Performance reports
```

## Contact & Support

For questions about this deployment package, refer to the maintenance guide 
or contact the DS-AutoAdvisor development team.
'''
    
    def _create_deployment_guide(self) -> str:
        """Create deployment guide"""
        return f'''# Deployment Guide

## Quick Start

### 1. Local Testing

```bash
# Install dependencies
pip install -r deployment/requirements.txt

# Test model server
cd deployment
python model_server.py

# Test API (in another terminal)
python api_template.py
```

### 2. API Testing

```bash
# Health check
curl http://localhost:5000/health

# Make a prediction
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"data": {{"feature1": 1.0, "feature2": 2.0}}}}'
```

### 3. Docker Deployment

```bash
# Build image
docker build -t ds-autoadvisor .

# Run container
docker run -p 5000:5000 ds-autoadvisor
```

## Production Deployment

### Environment Setup

1. **Server Requirements**
   - Python 3.9+
   - 2GB+ RAM
   - Docker (optional)

2. **Installation**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration**
   - Review `deployment_config.yaml`
   - Update paths if necessary
   - Configure logging

### Monitoring

1. **Health Checks**
   - `/health` endpoint for status
   - Monitor response times
   - Track prediction accuracy

2. **Logging**
   - Enable application logging
   - Monitor error rates
   - Track usage patterns

### Security

1. **API Security**
   - Add authentication
   - Rate limiting
   - Input validation

2. **Model Security**
   - Secure model files
   - Version control
   - Access controls

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check file paths
   - Verify model format
   - Check dependencies

2. **Prediction Errors**
   - Validate input format
   - Check feature names
   - Verify data types

3. **Performance Issues**
   - Monitor memory usage
   - Check CPU utilization
   - Consider model optimization

### Support

Refer to maintenance guide for detailed troubleshooting steps.
'''
    
    def _create_model_comparison(self) -> str:
        """Create model comparison HTML report"""
        model_rankings = self.training_results.get('model_rankings', [])[:5]  # Top 5 models
        
        html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .best { background-color: #d4edda; }
    </style>
</head>
<body>
    <h1>Model Comparison Report</h1>
    <p>Generated: ''' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '''</p>
    
    <h2>Top Models Performance</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Model Name</th>
            <th>Accuracy</th>
            <th>Training Time (s)</th>
            <th>Status</th>
        </tr>
'''
        
        for i, model_ranking in enumerate(model_rankings, 1):
            name = model_ranking.get('model', 'Unknown')
            accuracy = model_ranking.get('primary_score', 'N/A')
            training_time = model_ranking.get('training_time', 'N/A')
            row_class = 'best' if i == 1 else ''
            
            # Format accuracy and training time safely
            accuracy_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else str(accuracy)
            time_str = f"{training_time:.2f}" if isinstance(training_time, (int, float)) else str(training_time)
            status = 'Recommended' if i == 1 else 'Available'
            
            html_content += f'''
        <tr class="{row_class}">
            <td>{i}</td>
            <td>{name}</td>
            <td>{accuracy_str}</td>
            <td>{time_str}</td>
            <td>{status}</td>
        </tr>
'''
        
        html_content += '''
    </table>
    
    <h2>Deployment Recommendation</h2>
    <p>The top-ranked model has been selected as the default for deployment based on performance metrics.</p>
    
</body>
</html>
'''
        return html_content
    
    def _create_maintenance_guide(self) -> str:
        """Create maintenance guide"""
        return f'''# Maintenance Guide

## Overview

This guide covers ongoing maintenance of the DS-AutoAdvisor production deployment.

## Regular Maintenance Tasks

### Daily Checks
- [ ] Monitor API health endpoint
- [ ] Check error logs
- [ ] Verify prediction accuracy

### Weekly Reviews
- [ ] Analyze usage patterns
- [ ] Review performance metrics
- [ ] Update documentation if needed

### Monthly Tasks
- [ ] Model performance evaluation
- [ ] Security updates
- [ ] Backup model artifacts

## Model Updates

### When to Retrain
- Performance degradation detected
- New data available
- Business requirements change

### Retraining Process
1. Run updated Stage 1 & 2 pipelines
2. Generate new production artifacts
3. A/B test new models
4. Deploy if performance improves

### Version Control
- Tag model versions
- Keep deployment history
- Document changes

## Monitoring

### Key Metrics
- **Prediction Accuracy:** Target vs. actual
- **Response Time:** API endpoint performance
- **Error Rate:** Failed predictions
- **Usage Volume:** Request patterns

### Alerting
Set up alerts for:
- High error rates (>5%)
- Slow response times (>2s)
- Low accuracy (degradation >10%)

## Troubleshooting

### Model Issues
```bash
# Check model loading
python -c "import pickle; print(pickle.load(open('models/best_model.pkl', 'rb')))"

# Verify predictions
python model_server.py
```

### API Issues
```bash
# Check API health
curl http://localhost:5000/health

# View logs
tail -f api.log
```

### Performance Issues
```bash
# Monitor resources
top -p $(pgrep -f api_template.py)

# Check disk space
df -h
```

## Backup and Recovery

### Backup Schedule
- **Daily:** Configuration files
- **Weekly:** Model artifacts
- **Monthly:** Complete deployment package

### Recovery Procedures
1. Restore from backup
2. Verify model integrity
3. Test predictions
4. Update monitoring

## Security

### Regular Security Tasks
- Update dependencies
- Review access logs
- Scan for vulnerabilities
- Update authentication

### Incident Response
1. Isolate affected systems
2. Analyze impact
3. Apply fixes
4. Document lessons learned

## Support Contacts

- **Technical Issues:** Development Team
- **Business Questions:** Stakeholders
- **Infrastructure:** DevOps Team

## Documentation Updates

Keep this guide updated with:
- New procedures
- Lessons learned
- Configuration changes
- Performance optimizations
'''
    
    def _finalize_artifacts(self) -> bool:
        """Finalize production artifacts"""
        try:
            end_time = datetime.now()
            total_time = (end_time - self.execution_state['start_time']).total_seconds()
            
            # Create final summary
            final_summary = {
                'execution_metadata': {
                    'start_time': self.execution_state['start_time'].isoformat(),
                    'end_time': end_time.isoformat(),
                    'total_execution_time': total_time,
                    'run_mode': self.run_mode
                },
                'stage_summary': {
                    'completed_stages': self.execution_state['completed_stages'],
                    'failed_stages': self.execution_state['failed_stages'],
                    'success_rate': len(self.execution_state['completed_stages']) / 
                                  (len(self.execution_state['completed_stages']) + len(self.execution_state['failed_stages']))
                },
                'artifacts_summary': {
                    'models_packaged': self.execution_state['stage_results']['package_models']['packaged_count'],
                    'recommended_model': self.execution_state['stage_results']['package_models']['recommended_model'],
                    'deployment_assets': len([f for f in self.outputs['deployment'].glob('*') if f.is_file()]),
                    'documentation_files': len([f for f in self.outputs['docs'].glob('*') if f.is_file()])
                },
                'output_directories': {k: str(v) for k, v in self.outputs.items()},
                'stage_results': self.execution_state['stage_results'],
                'success': len(self.execution_state['failed_stages']) == 0
            }
            
            # Save final summary
            summary_path = self.outputs['base'] / "production_artifacts_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(final_summary, f, indent=2, default=str)
            
            # MLflow logging
            if self.mlflow_enabled and self.mlflow_manager:
                try:
                    self.mlflow_manager.log_production_artifacts(final_summary)
                    self.mlflow_manager.end_pipeline_run(success=final_summary['success'])
                except Exception as e:
                    print(f"⚠️ MLflow logging failed: {e}")
            
            # Print final results
            print("\n" + "="*80)
            print("🎉 PRODUCTION ARTIFACTS GENERATION COMPLETED")
            print("="*80)
            print(f"⏱️  Total execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
            print(f"✅ Completed stages: {len(self.execution_state['completed_stages'])}")
            print(f"❌ Failed stages: {len(self.execution_state['failed_stages'])}")
            
            if final_summary['success']:
                print(f"\n🚀 Production Artifacts Ready!")
                print(f"📦 Models packaged: {final_summary['artifacts_summary']['models_packaged']}")
                print(f"🏆 Recommended model: {final_summary['artifacts_summary']['recommended_model']}")
                print(f"🚀 Deployment assets: {final_summary['artifacts_summary']['deployment_assets']} files")
                print(f"📖 Documentation: {final_summary['artifacts_summary']['documentation_files']} files")
            else:
                print(f"\n⚠️  Artifacts generation completed with issues")
            
            print(f"\n📁 All artifacts saved to: {self.outputs['base']}")
            print(f"📄 Complete summary: {summary_path}")
            
            print(f"\n🚀 Ready for Deployment:")
            print(f"   📁 Production directory: {self.outputs['base']}")
            print(f"   🏆 Best model: {final_summary['artifacts_summary']['recommended_model']}")
            print(f"   🐍 Serving script: {self.outputs['deployment']}/model_server.py")
            print(f"   🌐 API template: {self.outputs['deployment']}/api_template.py")
            print(f"   📖 Documentation: {self.outputs['docs']}/")
            
            return final_summary['success']
            
        except Exception as e:
            print(f"❌ Artifact finalization failed: {e}")
            traceback.print_exc()
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='DS-AutoAdvisor Simplified Production Pipeline')
    parser.add_argument('--stage2-dir', type=str,
                       help='Path to Stage 2 results directory')
    parser.add_argument('--output', type=str, default='pipeline_outputs',
                       help='Base output directory')
    parser.add_argument('--run-mode', type=str, default='custom', 
                       choices=['fast', 'custom'],
                       help='Configuration mode (v3 unified config)')
    parser.add_argument('--enable-mlflow', action='store_true',
                       help='Force enable MLflow tracking')
    
    args = parser.parse_args()
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Initialize simplified pipeline
    pipeline = SimplifiedProductionPipeline(
        stage2_results_dir=args.stage2_dir,
        run_mode=args.run_mode,
        force_enable_mlflow=args.enable_mlflow,
        output_base=args.output
    )
    
    if not pipeline.stage2_dir:
        print("❌ No Stage 2 directory found. Run 02_stage_testing.py first.")
        return 1
    
    # Run simplified production pipeline
    try:
        success = pipeline.run_simplified_pipeline()
        if success:
            print(f"\n🎉 Production Artifacts Generated Successfully!")
            print(f"📁 All artifacts: {pipeline.outputs['base']}")
            print(f"\n🚀 Ready for Deployment!")
            return 0
        else:
            print(f"\n❌ Production Artifacts Generation Failed")
            return 1
    except KeyboardInterrupt:
        print(f"\n🛑 Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Pipeline crashed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())