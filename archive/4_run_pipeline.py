"""
🚀 DS-AutoAdvisor: Step 3 - Production Pipeline
==============================================

WHAT IT DOES:
Runs the complete DS-AutoAdvisor v2.0 pipeline with full MLflow tracking,
model registration, and production-grade features. This is your main
execution script after inspection and testing.

WHEN TO USE:
- After data inspection (Step 1) and testing (Step 2)
- For production model training and evaluation
- When you want complete MLflow tracking and versioning
- For generating final reports and model artifacts

HOW TO USE:
Basic execution:
    python 3_run_pipeline.py

With custom config:
    python 3_run_pipeline.py --config config/my_config.yaml

Development mode:
    python 3_run_pipeline.py --environment development

Production mode:
    python 3_run_pipeline.py --environment production

FEATURES INCLUDED:
✅ Complete v2.0 pipeline with all 6 stages
✅ MLflow experiment tracking and model registry
✅ Advanced data quality assessment with v2.0 features
✅ Plugin system for feature selection and custom processing
✅ Metadata tracking and data lineage
✅ Enhanced human checkpoints with model comparison
✅ Unified HTML evaluation dashboard
✅ Model versioning and performance history
✅ Production-grade error handling and logging

WHAT YOU GET:
📊 Trained models registered in MLflow
📈 Complete evaluation reports and HTML dashboard
📋 Data quality assessment and recommendations
🔗 Data lineage and transformation tracking
📄 Comprehensive execution logs and metadata
🏆 Model performance comparison with historical data

NEXT STEPS:
- Check MLflow UI for experiment tracking
- Review evaluation results in HTML dashboard
- Deploy best performing model from MLflow registry
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# Import v2.0 infrastructure
from src.infrastructure.enhanced_config_manager import get_config_manager, Environment
from src.infrastructure.metadata_manager import get_metadata_store, DataLineageTracker
from src.infrastructure.plugin_system import get_plugin_manager, PluginType
from src.data_quality_system.enhanced_quality_system import DataQualityAssessor, TypeEnforcer

# Import business features
try:
    from plugins.feature_selection.business_feature_selector import BusinessFeatureSelector
    from plugins.business_metrics.kpi_tracker import BusinessKPITracker
    BUSINESS_FEATURES_AVAILABLE = True
except ImportError:
    BUSINESS_FEATURES_AVAILABLE = False

# Try to import MLflow integration (optional dependency)
try:
    from src.infrastructure.mlflow_integration import get_mlflow_integration
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Import v1.0 components (maintained for compatibility)
import importlib.util
v1_spec = importlib.util.spec_from_file_location("complete_pipeline", project_root / "archive" / "complete_pipeline.py")
v1_module = importlib.util.module_from_spec(v1_spec)
v1_spec.loader.exec_module(v1_module)
DSAutoAdvisorPipeline = v1_module.DSAutoAdvisorPipeline
StageResult = v1_module.StageResult
PipelineState = v1_module.PipelineState

import logging
import pandas as pd
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
import json

class DSAutoAdvisorPipelineV2(DSAutoAdvisorPipeline):
    """
    Enhanced DS-AutoAdvisor Pipeline v2.0
    
    Extends the v1.0 pipeline with advanced features while maintaining
    full backward compatibility.
    """
    
    def __init__(self, config_path: str = None, environment: Environment = None, 
                 enable_v2_features: bool = True):
        """
        Initialize enhanced pipeline
        
        Args:
            config_path: Path to configuration file
            environment: Target environment
            enable_v2_features: Enable v2.0 features (default: True)
        """
        # Initialize v2.0 infrastructure
        self.enable_v2_features = enable_v2_features
        self.enable_industrial = False  # Add compatibility attribute for v1.0 compatibility
        self.industrial_manager = None  # Add compatibility attribute for v1.0 compatibility
        
        if enable_v2_features:
            # Initialize enhanced configuration manager
            self.config_manager = get_config_manager(config_path, environment)
            self.config = self.config_manager.config
            
            # Initialize metadata store and lineage tracker
            metadata_config = self.config_manager.get_metadata_config()
            if metadata_config.enabled:
                self.metadata_store = get_metadata_store(metadata_config.connection_string.split('///')[-1])
                self.lineage_tracker = DataLineageTracker(self.metadata_store)
            else:
                self.metadata_store = None
                self.lineage_tracker = None
            
            # Initialize plugin manager
            plugin_config = self.config_manager.get_plugin_config()
            if plugin_config.enabled:
                self.plugin_manager = get_plugin_manager(
                    plugin_config.plugin_dir, 
                    plugin_config.auto_discovery
                )
            else:
                self.plugin_manager = None
            
            # Initialize enhanced data quality assessor
            self.quality_assessor = DataQualityAssessor()
            self.type_enforcer = TypeEnforcer()
            
            # Initialize business features
            if BUSINESS_FEATURES_AVAILABLE and self.config_manager.is_feature_enabled('business_alignment'):
                try:
                    self.business_feature_selector = BusinessFeatureSelector()
                    self.business_kpi_tracker = BusinessKPITracker()
                    self.logger.info("✅ Business features initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Business features initialization failed: {e}")
                    self.business_feature_selector = None
                    self.business_kpi_tracker = None
            else:
                self.business_feature_selector = None
                self.business_kpi_tracker = None
            
            # Generate pipeline run ID for this execution
            self.pipeline_run_id = str(uuid.uuid4())
            
            # Setup enhanced logging early
            self.logger = self._setup_enhanced_logging()
            
            # Initialize MLflow integration (after logger is available)
            if self.config_manager.is_feature_enabled('mlflow_tracking') and MLFLOW_AVAILABLE:
                try:
                    self.mlflow_integration = get_mlflow_integration(self.config_manager)
                except Exception as e:
                    self.logger.warning(f"MLflow integration failed: {e}")
                    self.mlflow_integration = None
            else:
                self.mlflow_integration = None
                if self.config_manager.is_feature_enabled('mlflow_tracking') and not MLFLOW_AVAILABLE:
                    self.logger.warning("MLflow tracking enabled but MLflow not available")
            
        else:
            # Fallback to v1.0 initialization
            super().__init__(config_path or "config/unified_config.yaml")
            self.config_manager = None
            self.metadata_store = None
            self.lineage_tracker = None
            self.plugin_manager = None
            self.quality_assessor = None
            self.type_enforcer = None
            self.business_feature_selector = None
            self.business_kpi_tracker = None
            self.pipeline_run_id = None
            self.mlflow_integration = None
            
            # Setup basic logging for v1.0 mode
            self.logger = self._setup_enhanced_logging()
        
        # Log initialization completion
        self.logger.info(f"🚀 DS-AutoAdvisor Pipeline v2.0 initialized (v2 features: {enable_v2_features})")
        if enable_v2_features:
            self.logger.info(f"🔄 Pipeline Run ID: {self.pipeline_run_id}")
            if self.business_feature_selector:
                self.logger.info("🏢 Business Feature Selector: Ready")
            if self.business_kpi_tracker:
                self.logger.info("📊 Business KPI Tracker: Ready")
    
    def _setup_enhanced_logging(self) -> logging.Logger:
        """Setup enhanced logging with correlation IDs"""
        if self.enable_v2_features:
            logging_config = self.config_manager.get_config('infrastructure', 'logging', {})
            
            # Create structured logger
            logger = logging.getLogger('DSAutoAdvisorPipelineV2')
            
            # Add correlation ID to log format
            formatter = logging.Formatter(
                f'%(asctime)s - {self.pipeline_run_id or "v1-mode"} - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Configure handlers
            for handler in logger.handlers[:]:
                handler.setFormatter(formatter)
            
            return logger
        else:
            return super()._setup_logging()
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete enhanced pipeline"""
        start_time = datetime.now()
        
        if self.enable_v2_features:
            self.logger.info("🏁 Starting DS-AutoAdvisor Pipeline v2.0")
            
            # Track pipeline run in metadata store
            if self.metadata_store:
                self._track_pipeline_start(start_time)
            
            # Enhanced pipeline execution
            return self._run_enhanced_pipeline(start_time)
        else:
            # Fallback to v1.0 execution
            self.logger.info("🏁 Running in v1.0 compatibility mode")
            return super().run_complete_pipeline()
    
    def _track_pipeline_start(self, start_time: datetime):
        """Track pipeline start in metadata store"""
        from src.infrastructure.metadata_manager import PipelineRunMetadata
        
        pipeline_run = PipelineRunMetadata(
            id=self.pipeline_run_id,
            pipeline_version="2.0",
            start_time=start_time,
            end_time=None,
            status="running",
            input_dataset_id=None,  # Will be set after data loading
            configuration=self.config,
            stages_completed=[],
            error_message=None,
            user_decisions=[]
        )
        
        self.metadata_store.store_pipeline_run_metadata(pipeline_run)
    
    def _run_enhanced_pipeline(self, start_time: datetime) -> Dict[str, Any]:
        """Run the enhanced v2.0 pipeline"""
        
        # Start MLflow tracking
        if self.mlflow_integration:
            mlflow_run_id = self.mlflow_integration.start_pipeline_tracking(
                self.config, 
                f"ds-autoadvisor-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            self.logger.info(f"🔬 MLflow tracking started: {mlflow_run_id}")
        
        try:
            # Initialize pipeline state
            data_path = self.config_manager.get_config('global', 'data_input_path')
            self.pipeline_state = PipelineState(
                current_stage="initialization",
                completed_stages=[],
                stage_results=[],
                data_path=data_path,
                config=self.config
            )
            
            # Validate input data exists
            if not Path(data_path).exists():
                error_msg = f"Input data file not found: {data_path}"
                self.logger.error(error_msg)
                return self._create_enhanced_pipeline_summary(success=False, error=error_msg)
            
            # Load and track input dataset
            try:
                df = pd.read_csv(data_path, delimiter=self.config_manager.get_config('global', 'csv_delimiter'))
                
                if self.lineage_tracker:
                    input_dataset_id = self.lineage_tracker.track_dataset(data_path, df)
                    self.pipeline_state.input_dataset_id = input_dataset_id
            except Exception as e:
                error_msg = f"Failed to load input data: {str(e)}"
                self.logger.error(error_msg)
                return self._create_enhanced_pipeline_summary(success=False, error=error_msg)
            
            # Run enhanced data quality assessment
            if self.config_manager.is_feature_enabled('data_quality_v2'):
                quality_report = self._run_enhanced_data_quality_assessment(df)
                
                # Log data quality to MLflow
                if self.mlflow_integration and quality_report:
                    self.mlflow_integration.log_data_quality(quality_report)
            
            # Execute pipeline stages with v2.0 enhancements
            stages = self.config_manager.get_config('workflow', 'stages', [])
            
            for stage_name in stages:
                self.logger.info(f"🔄 Executing Enhanced Stage: {stage_name}")
                self.pipeline_state.current_stage = stage_name
                
                try:
                    # Execute stage with v2.0 enhancements
                    stage_start_time = datetime.now()
                    stage_result = self._execute_enhanced_stage(stage_name)
                    stage_execution_time = (datetime.now() - stage_start_time).total_seconds()
                    
                    self.pipeline_state.stage_results.append(stage_result)
                    
                    # Log stage result to MLflow
                    if self.mlflow_integration:
                        self.mlflow_integration.log_stage_result(stage_name, stage_result, stage_execution_time)
                    
                    if stage_result.success:
                        self.pipeline_state.completed_stages.append(stage_name)
                        self.logger.info(f"✅ Enhanced stage {stage_name} completed successfully")
                        
                        # Apply plugins if available
                        if self.plugin_manager and self.config_manager.is_feature_enabled('plugin_system'):
                            self._apply_stage_plugins(stage_name, stage_result)
                        
                        # Log training results to MLflow
                        if stage_name == "model_training" and self.mlflow_integration:
                            if hasattr(stage_result, 'artifacts') and 'training_results' in stage_result.artifacts:
                                training_results = stage_result.artifacts['training_results']
                                registered_models = self.mlflow_integration.log_training_results(training_results)
                                self.logger.info(f"🔬 Registered {len(registered_models)} models in MLflow")
                        
                        # Log evaluation results to MLflow
                        if stage_name == "model_evaluation" and self.mlflow_integration:
                            if hasattr(stage_result, 'artifacts') and 'evaluation_results' in stage_result.artifacts:
                                evaluation_results = stage_result.artifacts['evaluation_results']
                                self.mlflow_integration.log_evaluation_results(evaluation_results)
                        
                        # Human checkpoint handling (enhanced)
                        if self._requires_human_checkpoint(stage_name, stage_result):
                            decision = self._enhanced_human_checkpoint(stage_name, stage_result)
                            if decision == "abort":
                                return self._create_enhanced_pipeline_summary(success=False, error="Pipeline aborted by user")
                    
                    else:
                        self.logger.error(f"❌ Enhanced stage {stage_name} failed: {stage_result.error_message}")
                        # Enhanced error handling could go here
                        
                except Exception as e:
                    self.logger.error(f"❌ Unexpected error in enhanced stage {stage_name}: {str(e)}")
                    return self._create_enhanced_pipeline_summary(success=False, error=str(e))
            
            # Generate enhanced final summary
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"🎉 Enhanced pipeline completed in {execution_time:.2f} seconds")
            
            # End MLflow tracking with success
            if self.mlflow_integration:
                self.mlflow_integration.end_pipeline_tracking(success=True)
            
            return self._create_enhanced_pipeline_summary(success=True, execution_time=execution_time)
        
        except Exception as e:
            # End MLflow tracking with error
            if self.mlflow_integration:
                self.mlflow_integration.end_pipeline_tracking(success=False, error_message=str(e))
            raise
    
    def _run_enhanced_data_quality_assessment(self, df: pd.DataFrame):
        """Run enhanced data quality assessment"""
        self.logger.info("📊 Running Enhanced Data Quality Assessment")
        
        try:
            target_column = self.config_manager.get_config('global', 'target_column')
            quality_report = self.quality_assessor.assess_quality(df, target_column)
            
            self.logger.info(f"📈 Data Quality Score: {quality_report.overall_score:.1f}/100")
            self.logger.info(f"🔍 Found {quality_report.total_issues} quality issues")
            
            # Log quality issues by severity
            for severity, count in quality_report.issues_by_severity.items():
                if count > 0:
                    self.logger.info(f"   {severity.upper()}: {count} issues")
            
            # Store quality assessment in pipeline state
            if not hasattr(self.pipeline_state, 'quality_report'):
                self.pipeline_state.quality_report = quality_report
            
            return quality_report
            
        except Exception as e:
            self.logger.warning(f"Enhanced data quality assessment failed: {e}")
            return None
    
    def _execute_enhanced_stage(self, stage_name: str) -> StageResult:
        """Execute a pipeline stage with v2.0 enhancements"""
        
        # Track stage execution start
        stage_start_time = datetime.now()
        stage_execution_id = str(uuid.uuid4())
        
        if self.metadata_store:
            self._track_stage_start(stage_execution_id, stage_name, stage_start_time)
        
        # Execute the stage (use v1.0 implementation but with enhanced tracking)
        try:
            stage_result = self._execute_stage(stage_name)
            
            # Enhanced stage result processing
            if stage_result.success and self.lineage_tracker:
                self._track_stage_lineage(stage_name, stage_result)
            
            # Track stage completion
            if self.metadata_store:
                self._track_stage_completion(stage_execution_id, stage_result)
            
            return stage_result
            
        except Exception as e:
            # Enhanced error tracking
            stage_result = StageResult(
                stage_name=stage_name,
                success=False,
                execution_time=(datetime.now() - stage_start_time).total_seconds(),
                outputs_created=[],
                human_interventions=[],
                error_message=str(e)
            )
            
            if self.metadata_store:
                self._track_stage_completion(stage_execution_id, stage_result)
            
            return stage_result
    
    def _track_stage_start(self, stage_execution_id: str, stage_name: str, start_time: datetime):
        """Track stage execution start"""
        from src.infrastructure.metadata_manager import StageExecutionMetadata
        
        stage_execution = StageExecutionMetadata(
            id=stage_execution_id,
            pipeline_run_id=self.pipeline_run_id,
            stage_name=stage_name,
            start_time=start_time,
            end_time=None,
            status="running",
            configuration=self.config_manager.get_stage_config(stage_name)
        )
        
        self.metadata_store.store_stage_execution_metadata(stage_execution)
    
    def _track_stage_completion(self, stage_execution_id: str, stage_result: StageResult):
        """Track stage execution completion"""
        # Update stage execution metadata
        # This would typically involve updating the existing record
        # For simplicity, we'll log the completion
        self.logger.debug(f"Stage {stage_result.stage_name} completed with status: {'success' if stage_result.success else 'failed'}")
    
    def _track_stage_lineage(self, stage_name: str, stage_result: StageResult):
        """Track data lineage for stage outputs"""
        if not stage_result.outputs_created:
            return
        
        # Track lineage relationships between stage inputs and outputs
        for output_path in stage_result.outputs_created:
            if Path(output_path).exists() and output_path.endswith('.csv'):
                try:
                    output_df = pd.read_csv(output_path)
                    output_dataset_id = self.lineage_tracker.track_dataset(output_path, output_df)
                    
                    # Create lineage relationship if we have input dataset ID
                    if hasattr(self.pipeline_state, 'input_dataset_id') and self.pipeline_state.input_dataset_id:
                        self.lineage_tracker.track_transformation(
                            self.pipeline_state.input_dataset_id,
                            output_dataset_id,
                            stage_name,
                            metadata=stage_result.artifacts or {}
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to track lineage for {output_path}: {e}")
    
    def _apply_stage_plugins(self, stage_name: str, stage_result: StageResult):
        """Apply relevant plugins to stage results"""
        
        # Apply business feature selection after data cleaning
        if stage_name == "data_cleaning" and stage_result.success and self.business_feature_selector:
            self.logger.info("🏢 Applying Business Feature Selection")
            
            try:
                # Load cleaned data
                cleaned_data_path = self.config_manager.get_config('data_cleaning', 'output_path')
                if Path(cleaned_data_path).exists():
                    df = pd.read_csv(cleaned_data_path)
                    target_column = self.config_manager.get_config('global', 'target_column')
                    
                    X = df.drop(columns=[target_column])
                    y = df[target_column]
                    
                    # Apply business feature selection
                    selected_features = self.business_feature_selector.select_features(X, y)
                    self.logger.info(f"🎯 Business Feature Selector selected {len(selected_features)} features")
                    
                    # Store business feature selection results
                    if not stage_result.artifacts:
                        stage_result.artifacts = {}
                    stage_result.artifacts['business_selected_features'] = selected_features
                    
                    # Save business-selected features to file
                    business_features_df = X[selected_features]
                    business_features_df[target_column] = y
                    business_output_path = cleaned_data_path.replace('.csv', '_business_features.csv')
                    business_features_df.to_csv(business_output_path, index=False)
                    stage_result.outputs_created.append(business_output_path)
                    self.logger.info(f"💾 Business features saved to: {business_output_path}")
                    
            except Exception as e:
                self.logger.warning(f"Business feature selection failed: {e}")
        
        # Apply traditional feature selection plugins 
        if stage_name == "data_cleaning" and stage_result.success:
            if self.plugin_manager:
                feature_selection_plugins = self.plugin_manager.get_plugins_by_type(PluginType.FEATURE_SELECTION)
                
                if feature_selection_plugins:
                    self.logger.info(f"🔌 Applying {len(feature_selection_plugins)} traditional feature selection plugins")
                    
                    try:
                        # Load cleaned data
                        cleaned_data_path = self.config_manager.get_config('data_cleaning', 'output_path')
                        if Path(cleaned_data_path).exists():
                            df = pd.read_csv(cleaned_data_path)
                            target_column = self.config_manager.get_config('global', 'target_column')
                            
                            X = df.drop(columns=[target_column])
                            y = df[target_column]
                            
                            # Apply each plugin
                            for plugin in feature_selection_plugins:
                                selected_features = plugin.execute((X, y))
                                self.logger.info(f"🎯 Plugin '{plugin.plugin_info.name}' selected {len(selected_features)} features")
                                
                                # Store plugin results in stage result
                                if not stage_result.artifacts:
                                    stage_result.artifacts = {}
                                stage_result.artifacts[f'plugin_{plugin.plugin_info.name}_features'] = selected_features
                    
                    except Exception as e:
                        self.logger.warning(f"Plugin application failed: {e}")
        
        # Apply business KPI tracking after model evaluation
        if stage_name == "model_evaluation" and stage_result.success and self.business_kpi_tracker:
            self.logger.info("📊 Calculating Business KPIs")
            
            try:
                # Load evaluation results
                if hasattr(stage_result, 'artifacts') and 'evaluation_results' in stage_result.artifacts:
                    evaluation_results = stage_result.artifacts['evaluation_results']
                    
                    # Calculate business KPIs
                    kpi_results = self.business_kpi_tracker.calculate_kpis(evaluation_results)
                    roi_analysis = self.business_kpi_tracker.analyze_roi(evaluation_results)
                    
                    # Store business metrics
                    stage_result.artifacts['business_kpis'] = kpi_results
                    stage_result.artifacts['roi_analysis'] = roi_analysis
                    
                    self.logger.info(f"📈 Calculated {len(kpi_results)} business KPIs")
                    self.logger.info(f"💰 ROI Analysis: {roi_analysis.projected_roi:.2%}")
                    
                    # Log business metrics to MLflow if available
                    if self.mlflow_integration:
                        self.mlflow_integration.log_business_metrics(kpi_results, roi_analysis)
                    
            except Exception as e:
                self.logger.warning(f"Business KPI calculation failed: {e}")
    
    def _enhanced_human_checkpoint(self, stage_name: str, stage_result: StageResult) -> str:
        """Enhanced human checkpoint with v2.0 features"""
        
        print("\n" + "="*80)
        print(f"🤖➡️👨‍💻 ENHANCED CHECKPOINT: {stage_name.upper()}")
        print("="*80)
        
        # Show basic stage information
        print(f"Stage: {stage_name}")
        print(f"Status: {'✅ Success' if stage_result.success else '❌ Failed'}")
        print(f"Execution Time: {stage_result.execution_time:.2f}s")
        print(f"Outputs Created: {len(stage_result.outputs_created)}")
        
        # Show v2.0 enhancements if available
        if hasattr(self.pipeline_state, 'quality_report') and stage_name == "profiling":
            quality_report = self.pipeline_state.quality_report
            print(f"\n📊 Data Quality Score: {quality_report.overall_score:.1f}/100")
            print(f"🔍 Quality Issues: {quality_report.total_issues}")
        
        # Show plugin results if available
        if stage_result.artifacts:
            plugin_results = {k: v for k, v in stage_result.artifacts.items() if k.startswith('plugin_')}
            if plugin_results:
                print("\n🔌 Plugin Results:")
                for plugin_key, result in plugin_results.items():
                    plugin_name = plugin_key.replace('plugin_', '').replace('_features', '')
                    if isinstance(result, list):
                        print(f"   {plugin_name}: {len(result)} features selected")
        
        # Show metadata insights if available
        if self.metadata_store and self.config_manager.is_feature_enabled('data_quality_v2'):
            print("\n📈 Historical Performance:")
            if self.mlflow_integration:
                try:
                    historical_models = self.mlflow_integration.get_historical_performance("accuracy")
                    if historical_models:
                        print(f"   Found {len(historical_models)} historical models")
                        best_model = historical_models[0]
                        print(f"   Best model: {best_model.name} (v{best_model.version})")
                        print(f"   Best accuracy: {best_model.metrics.get('accuracy', 'N/A')}")
                    else:
                        print("   No historical models found")
                except Exception as e:
                    print(f"   Could not retrieve historical data: {e}")
            else:
                print("   (MLflow integration not available)")
        
        print("\n🔧 Available Actions:")
        print("  continue  - Proceed to next stage")
        print("  retry     - Retry this stage")
        print("  config    - Adjust configuration")
        print("  insights  - View detailed insights (v2.0)")
        print("  models    - View model comparison (MLflow)")
        print("  abort     - Stop pipeline execution")
        
        while True:
            choice = input("\nYour decision (continue/retry/config/insights/models/abort): ").lower().strip()
            
            if choice == 'continue':
                break
            elif choice == 'retry':
                break
            elif choice == 'abort':
                break
            elif choice == 'config':
                self._adjust_configuration_interactive(stage_name, stage_result)
                continue
            elif choice == 'insights':
                self._show_enhanced_insights(stage_name, stage_result)
                continue
            elif choice == 'models' and self.mlflow_integration:
                self._show_model_comparison()
                continue
            else:
                print("❌ Invalid choice. Please enter 'continue', 'retry', 'config', 'insights', 'models', or 'abort'")
                continue
        
        return choice
    
    def _show_model_comparison(self):
        """Show MLflow model comparison"""
        if not self.mlflow_integration:
            print("❌ MLflow integration not available")
            return
            
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON (MLflow)")
        print("="*60)
        
        try:
            # Get top models by accuracy
            historical_models = self.mlflow_integration.get_historical_performance("accuracy", limit=5)
            
            if not historical_models:
                print("No models found in MLflow registry")
                return
                
            print(f"Found {len(historical_models)} models:")
            print(f"{'Rank':<5} {'Model':<20} {'Version':<8} {'Accuracy':<10} {'F1':<10} {'Date':<12}")
            print("-" * 65)
            
            for i, model in enumerate(historical_models, 1):
                accuracy = model.metrics.get('accuracy', 'N/A')
                f1_score = model.metrics.get('f1_score', 'N/A')
                creation_time = model.creation_timestamp.strftime('%Y-%m-%d') if hasattr(model, 'creation_timestamp') else 'Unknown'
                
                print(f"{i:<5} {model.name[:18]:<20} {model.version:<8} {accuracy:<10} {f1_score:<10} {creation_time:<12}")
            
            # Show best performing model details
            best_model = historical_models[0]
            print(f"\n🏆 Best Model: {best_model.name} (v{best_model.version})")
            print("   Metrics:")
            for metric_name, metric_value in best_model.metrics.items():
                print(f"   - {metric_name}: {metric_value}")
                
        except Exception as e:
            print(f"❌ Error retrieving model comparison: {e}")
    
    def _show_enhanced_insights(self, stage_name: str, stage_result: StageResult):
        """Show enhanced v2.0 insights"""
        print("\n" + "="*60)
        print("🔍 ENHANCED INSIGHTS (v2.0)")
        print("="*60)
        
        # Data lineage insights
        if self.lineage_tracker and hasattr(self.pipeline_state, 'input_dataset_id'):
            print("📊 Data Lineage:")
            lineage = self.metadata_store.get_data_lineage(self.pipeline_state.input_dataset_id)
            print(f"   Connected entities: {len(lineage['connected_entities'])}")
            print(f"   Relationships: {len(lineage['relationships'])}")
        
        # Quality insights
        if hasattr(self.pipeline_state, 'quality_report'):
            quality_report = self.pipeline_state.quality_report
            print("\n📈 Data Quality Insights:")
            print(f"   Overall Score: {quality_report.overall_score:.1f}/100")
            
            if quality_report.recommendations:
                print("   Top Recommendations:")
                for i, rec in enumerate(quality_report.recommendations[:3], 1):
                    print(f"   {i}. {rec}")
        
        # Plugin insights
        if self.plugin_manager:
            plugin_status = self.plugin_manager.get_plugin_status()
            active_plugins = [name for name, info in plugin_status.items() if info.status.value == "active"]
            print(f"\n🔌 Active Plugins: {len(active_plugins)}")
            for plugin_name in active_plugins[:5]:  # Show first 5
                print(f"   • {plugin_name}")
        
        input("\nPress Enter to continue...")
    
    def _create_enhanced_pipeline_summary(self, success: bool, execution_time: float = 0.0, 
                                        error: str = None) -> Dict[str, Any]:
        """Create enhanced pipeline execution summary"""
        
        summary = {
            "pipeline_version": "2.0",
            "success": success,
            "execution_time": execution_time,
            "completed_stages": len(self.pipeline_state.completed_stages) if self.pipeline_state else 0,
            "total_stages": len(self.config.get('workflow', {}).get('stages', [])),
            "error_message": error
        }
        
        # Add v2.0 specific information
        if self.enable_v2_features:
            summary["pipeline_run_id"] = self.pipeline_run_id
            summary["v2_features_enabled"] = True
            
            # Data quality summary
            if hasattr(self.pipeline_state, 'quality_report'):
                quality_report = self.pipeline_state.quality_report
                summary["data_quality"] = {
                    "overall_score": quality_report.overall_score,
                    "total_issues": quality_report.total_issues,
                    "issues_by_severity": quality_report.issues_by_severity
                }
            
            # Plugin summary
            if self.plugin_manager:
                plugin_status = self.plugin_manager.get_plugin_status()
                summary["plugins"] = {
                    "total_plugins": len(plugin_status),
                    "active_plugins": len([p for p in plugin_status.values() if p.status.value == "active"])
                }
            
            # Metadata summary
            if self.metadata_store:
                summary["metadata_tracking"] = True
                summary["lineage_tracking"] = self.lineage_tracker is not None
            
            # Business features summary
            if self.business_feature_selector or self.business_kpi_tracker:
                business_summary = {}
                
                if self.business_feature_selector:
                    business_summary["feature_selection"] = True
                
                if self.business_kpi_tracker:
                    business_summary["kpi_tracking"] = True
                    
                    # Add KPI results if available
                    if hasattr(self.pipeline_state, 'stage_results'):
                        for stage_result in self.pipeline_state.stage_results:
                            if hasattr(stage_result, 'artifacts') and stage_result.artifacts:
                                if 'business_kpis' in stage_result.artifacts:
                                    business_summary["calculated_kpis"] = len(stage_result.artifacts['business_kpis'])
                                if 'roi_analysis' in stage_result.artifacts:
                                    roi = stage_result.artifacts['roi_analysis']
                                    business_summary["projected_roi"] = f"{roi.projected_roi:.2%}"
                
                summary["business_features"] = business_summary
        
        # Update pipeline run metadata if tracking enabled
        if self.metadata_store and self.pipeline_run_id:
            self._update_pipeline_completion(summary)
        
        return summary
    
    def _update_pipeline_completion(self, summary: Dict[str, Any]):
        """Update pipeline run completion in metadata store"""
        # This would update the pipeline run record with completion information
        self.logger.debug(f"Pipeline {self.pipeline_run_id} completed with summary: {summary}")


def main():
    """Enhanced main function with v2.0 features"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DS-AutoAdvisor Pipeline v2.0')
    parser.add_argument('--config', type=str, default='config/unified_config_v2.yaml',
                       help='Configuration file path')
    parser.add_argument('--environment', type=str, choices=['development', 'staging', 'production'],
                       default='development', help='Target environment')
    parser.add_argument('--v1-mode', action='store_true',
                       help='Run in v1.0 compatibility mode')
    parser.add_argument('--disable-plugins', action='store_true',
                       help='Disable plugin system')
    
    args = parser.parse_args()
    
    # Determine environment
    environment = Environment(args.environment) if not args.v1_mode else None
    
    # Initialize and run pipeline
    try:
        pipeline = DSAutoAdvisorPipelineV2(
            config_path=args.config,
            environment=environment,
            enable_v2_features=not args.v1_mode
        )
        
        result = pipeline.run_complete_pipeline()
        
        # Print summary
        print("\n" + "="*80)
        print("🎯 PIPELINE EXECUTION SUMMARY")
        print("="*80)
        print(f"Success: {'✅ Yes' if result['success'] else '❌ No'}")
        print(f"Version: {result.get('pipeline_version', '1.0')}")
        print(f"Execution Time: {result.get('execution_time', 0):.2f}s")
        print(f"Stages Completed: {result.get('completed_stages', 0)}/{result.get('total_stages', 0)}")
        
        if result.get('v2_features_enabled'):
            print(f"Pipeline Run ID: {result.get('pipeline_run_id', 'N/A')}")
            
            if 'data_quality' in result:
                dq = result['data_quality']
                print(f"Data Quality Score: {dq['overall_score']:.1f}/100")
            
            if 'plugins' in result:
                plugins = result['plugins']
                print(f"Active Plugins: {plugins['active_plugins']}/{plugins['total_plugins']}")
            
            if 'business_features' in result:
                business = result['business_features']
                if business.get('feature_selection'):
                    print("🏢 Business Feature Selection: Enabled")
                if business.get('kpi_tracking'):
                    print("📊 Business KPI Tracking: Enabled")
                if business.get('calculated_kpis'):
                    print(f"📈 KPIs Calculated: {business['calculated_kpis']}")
                if business.get('projected_roi'):
                    print(f"💰 Projected ROI: {business['projected_roi']}")
        
        if result.get('error_message'):
            print(f"Error: {result['error_message']}")
        
        print("="*80)
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
