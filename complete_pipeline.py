"""
DS-AutoAdvisor Complete Pipeline Orchestrator
============================================

Executes the complete ML pipeline using your existing tested scripts
with unified configuration and human intervention checkpoints.

Now enhanced with industrial-grade features for production use.

Usage:
    python complete_pipeline.py --config config/unified_config.yaml
    python complete_pipeline.py --interactive  # Interactive mode
    python complete_pipeline.py --automated    # Fully automated mode
    python complete_pipeline.py --industrial   # Industrial-grade mode
    
Features:
- Uses all your existing tested scripts without modification
- Single unified configuration file
- Human intervention checkpoints
- Comprehensive logging and reporting
- Error handling and recovery
- Industrial-grade monitoring, security, and scalability
"""

import os
import sys
import yaml
import json
import logging
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import argparse

# Add src to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# Try to import industrial-grade components
try:
    from src.monitoring.industrial_integration import create_industrial_pipeline, Environment
    INDUSTRIAL_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Industrial features not available: {e}")
    INDUSTRIAL_FEATURES_AVAILABLE = False

@dataclass
class StageResult:
    """Results from a pipeline stage"""
    stage_name: str
    success: bool
    execution_time: float
    outputs_created: List[str]
    human_interventions: List[Dict]
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = None

@dataclass
class PipelineState:
    """Current state of the pipeline execution"""
    current_stage: str
    completed_stages: List[str]
    stage_results: List[StageResult]
    data_path: str
    config: Dict[str, Any]
    human_decisions: List[Dict] = None
    
    def __post_init__(self):
        if self.human_decisions is None:
            self.human_decisions = []

class DSAutoAdvisorPipeline:
    """Complete DS-AutoAdvisor Pipeline Orchestrator with Industrial-Grade Features"""
    
    def __init__(self, config_path: str, enable_industrial: bool = False):
        """Initialize pipeline with unified configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.pipeline_state = None
        self.enable_industrial = enable_industrial and INDUSTRIAL_FEATURES_AVAILABLE
        
        # Initialize industrial pipeline manager if available
        self.industrial_manager = None
        if self.enable_industrial:
            try:
                # Determine environment from config or default to development
                env_name = self.config.get('global', {}).get('environment', 'development')
                environment = Environment(env_name.lower())
                
                self.industrial_manager = create_industrial_pipeline(
                    environment=environment,
                    enable_all_features=True
                )
                
                # Start industrial pipeline session
                self.pipeline_session_id = self.industrial_manager.start_pipeline_session(
                    user_id="pipeline_user",
                    ip_address="127.0.0.1"
                )
                
                self.logger.info("🏭 Industrial-grade features enabled")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize industrial features: {e}")
                self.enable_industrial = False
        
        # Create output directories
        self._create_output_directories()
        
        self.logger.info("🚀 DS-AutoAdvisor Pipeline initialized")
        self.logger.info(f"📋 Config loaded from: {config_path}")
        if self.enable_industrial:
            self.logger.info(f"🏭 Industrial session: {self.pipeline_session_id}")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load unified configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        log_config = self.config.get('workflow', {}).get('logging', {})
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config.get('log_file', 'pipeline.log')),
                logging.StreamHandler() if log_config.get('console_output', True) else logging.NullHandler()
            ]
        )
        
        return logging.getLogger('DSAutoAdvisorPipeline')
    
    def _create_output_directories(self):
        """Create necessary output directories that will actually be used"""
        base_dir = Path(self.config['global']['output_base_dir'])
        
        # Only create directories that will actually be used
        directories = [
            Path(self.config['profiling']['output_dir']),  # docs/
            Path(self.config['data_cleaning']['output_path']).parent,  # data/ (for cleaned files)
            Path(self.config['model_training']['model_dir']),  # models/
            Path(self.config['model_evaluation']['output_dir']),  # evaluation_results/
            base_dir / "reports",  # For final reports
            base_dir / "advisory"   # For ML advisory results
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📁 Created directory: {directory}")
            
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete DS-AutoAdvisor pipeline"""
        
        start_time = datetime.now()
        self.logger.info("🏁 Starting Complete DS-AutoAdvisor Pipeline")
        
        # Initialize pipeline state
        data_path = self.config['global']['data_input_path']
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
            return self._create_pipeline_summary(success=False, error=error_msg)
        
        # Execute pipeline stages
        stages = self.config.get('workflow', {}).get('stages', [])
        
        for stage_name in stages:
            self.logger.info(f"🔄 Executing Stage: {stage_name}")
            self.pipeline_state.current_stage = stage_name
            
            try:
                # Execute stage with human intervention check
                stage_result = self._execute_stage(stage_name)
                self.pipeline_state.stage_results.append(stage_result)
                
                if stage_result.success:
                    self.pipeline_state.completed_stages.append(stage_name)
                    self.logger.info(f"✅ Stage {stage_name} completed successfully")
                    
                    # Human checkpoint if required
                    if self._requires_human_checkpoint(stage_name, stage_result):
                        while True:  # Loop for retry handling
                            decision = self._human_checkpoint(stage_name, stage_result)
                            if decision == "abort":
                                return self._create_pipeline_summary(success=False, error="Pipeline aborted by user")
                            elif decision == "retry":
                                self.logger.info(f"🔄 Retrying stage {stage_name}...")
                                # Remove from completed stages since we're retrying
                                if stage_name in self.pipeline_state.completed_stages:
                                    self.pipeline_state.completed_stages.remove(stage_name)
                                
                                # Retry the stage
                                retry_result = self._execute_stage(stage_name)
                                self.pipeline_state.stage_results[-1] = retry_result
                                
                                if retry_result.success:
                                    self.pipeline_state.completed_stages.append(stage_name)
                                    self.logger.info(f"✅ Stage {stage_name} retry completed successfully")
                                    stage_result = retry_result  # Update for next checkpoint
                                    # Continue loop to check if another checkpoint is needed
                                else:
                                    self.logger.error(f"❌ Stage {stage_name} retry failed: {retry_result.error_message}")
                                    stage_result = retry_result  # Update for next checkpoint
                                    # Continue loop to allow user to decide what to do with failed retry
                            elif decision == "continue":
                                break  # Exit retry loop and continue to next stage
                            elif decision == "config":
                                # Config adjustment is handled within _human_checkpoint
                                continue  # Stay in loop for next decision
                    
                else:
                    self.logger.error(f"❌ Stage {stage_name} failed: {stage_result.error_message}")
                    
                    error_handling = self.config.get('workflow', {}).get('error_handling', {})
                    if error_handling.get('stop_on_error', False):
                        return self._create_pipeline_summary(success=False, error=stage_result.error_message)
                    elif not error_handling.get('skip_failed_stages', True):
                        return self._create_pipeline_summary(success=False, error=stage_result.error_message)
                    
            except Exception as e:
                self.logger.error(f"❌ Unexpected error in stage {stage_name}: {str(e)}")
                return self._create_pipeline_summary(success=False, error=str(e))
        
        # Generate final summary
        execution_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"🎉 Pipeline completed in {execution_time:.2f} seconds")
        
        return self._create_pipeline_summary(success=True, execution_time=execution_time)
    
    def _execute_stage(self, stage_name: str) -> StageResult:
        """Execute a specific pipeline stage with industrial-grade features"""
        
        start_time = datetime.now()
        
        try:
            # Use industrial features if available
            if self.enable_industrial and self.industrial_manager:
                self.logger.info(f"🏭 Executing {stage_name} with industrial-grade features")
                
                # Execute stage with industrial features
                if stage_name == "profiling":
                    return self._execute_profiling_stage_industrial()
                elif stage_name == "data_cleaning":
                    return self._execute_cleaning_stage_industrial()
                elif stage_name == "ml_advisory":
                    return self._execute_advisory_stage_industrial()
                elif stage_name == "model_training":
                    return self._execute_training_stage_industrial()
                elif stage_name == "model_evaluation":
                    return self._execute_evaluation_stage_industrial()
                else:
                    raise ValueError(f"Unknown stage: {stage_name}")
            
            else:
                # Standard execution without industrial features
                if stage_name == "profiling":
                    return self._execute_profiling_stage()
                elif stage_name == "data_cleaning":
                    return self._execute_cleaning_stage()
                elif stage_name == "ml_advisory":
                    return self._execute_advisory_stage()
                elif stage_name == "model_training":
                    return self._execute_training_stage()
                elif stage_name == "model_evaluation":
                    return self._execute_evaluation_stage()
                else:
                    raise ValueError(f"Unknown stage: {stage_name}")
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return StageResult(
                stage_name=stage_name,
                success=False,
                execution_time=execution_time,
                outputs_created=[],
                human_interventions=[],
                error_message=str(e)
            )
    
    def _execute_profiling_stage(self) -> StageResult:
        """Execute data profiling using existing profiling_1/data_profiler.py"""
        return self._execute_profiling_stage_core()
    
    def _execute_profiling_stage_core(self) -> StageResult:
        """Core profiling logic that works with both standard and industrial modes"""
        
        start_time = datetime.now()
        self.logger.info("📊 Executing Data Profiling Stage")
        
        try:
            # Update profiling script configuration
            config = self.config['profiling']
            
            # Import and run profiling
            from src.profiling_1.data_profiler import main as run_profiler
            
            # Temporarily modify global variables in profiler (if needed)
            original_globals = {}
            profiler_module = sys.modules.get('src.profiling_1.data_profiler')
            
            if profiler_module:
                # Store original values
                original_globals['DATA_PATH'] = getattr(profiler_module, 'DATA_PATH', None)
                original_globals['OUTPUT_DIR'] = getattr(profiler_module, 'OUTPUT_DIR', None)
                original_globals['CSV_DELIMITER'] = getattr(profiler_module, 'CSV_DELIMITER', None)
                original_globals['CSV_ENCODING'] = getattr(profiler_module, 'CSV_ENCODING', None)
                
                # Set new values - use global settings
                profiler_module.DATA_PATH = self.config['global']['data_input_path']
                profiler_module.OUTPUT_DIR = config['output_dir']
                profiler_module.CSV_DELIMITER = self.config['global']['csv_delimiter']
                profiler_module.CSV_ENCODING = self.config['global']['csv_encoding']
            
            # Run profiling
            run_profiler()
            
            # Restore original values
            if profiler_module:
                for key, value in original_globals.items():
                    if value is not None:
                        setattr(profiler_module, key, value)
            
            outputs = [
                f"{config['output_dir']}/data_profiling_report.html",
                f"{config['output_dir']}/data_schema.yaml",
                f"{config['output_dir']}/missing_data_matrix.png"
            ]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return StageResult(
                stage_name="profiling",
                success=True,
                execution_time=execution_time,
                outputs_created=outputs,
                human_interventions=[],
                artifacts={"profiling_config": config}
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return StageResult(
                stage_name="profiling",
                success=False,
                execution_time=execution_time,
                outputs_created=[],
                human_interventions=[],
                error_message=f"Profiling failed: {str(e)}"
            )
    
    def _execute_cleaning_stage(self) -> StageResult:
        """Execute data cleaning using existing cleaning/data_cleaner.py"""
        return self._execute_cleaning_stage_core()
    
    def _execute_cleaning_stage_core(self) -> StageResult:
        """Core cleaning logic that works with both standard and industrial modes"""
        
        start_time = datetime.now()
        self.logger.info("🧹 Executing Data Cleaning Stage")
        
        try:
            # Import and run cleaning
            from src.correction_2.data_cleaner import DataCleaner, CleaningConfig
            
            config = self.config['data_cleaning']
            
            # Create cleaning config object
            cleaning_config = CleaningConfig(
                input_path=self.config['global']['data_input_path'],
                output_path=config['output_path'],
                log_path=config['log_file'],  # Map log_file to log_path
                output_delimiter=config.get('output_delimiter', ','),
                output_quoting=config.get('output_quoting', 1),
                output_quotechar=config.get('output_quotechar', '"'),
                output_escapechar=config.get('output_escapechar'),
                output_encoding=config.get('output_encoding', 'utf-8'),
                output_lineterminator=config.get('output_lineterminator', '\n'),
                output_doublequote=config.get('output_doublequote', True),
                remove_duplicates=config.get('remove_duplicates', True),
                remove_low_variance=config.get('remove_low_variance', True),
                low_variance_thresh=config.get('low_variance_thresh', 1)
            )
            
            # Configure cleaner
            cleaner = DataCleaner(config=cleaning_config)
            
            # Execute cleaning
            clean_data = cleaner.clean_data()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            outputs = [
                config['output_path'],
                config['log_file']  # Use log_file from config
            ]
            
            return StageResult(
                stage_name="cleaning",
                success=True,
                execution_time=execution_time,
                outputs_created=outputs,
                human_interventions=[],
                artifacts={"cleaning_config": config, "data_shape": clean_data.shape if clean_data is not None else None}
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return StageResult(
                stage_name="cleaning",
                success=False,
                execution_time=execution_time,
                outputs_created=[],
                human_interventions=[],
                error_message=f"Cleaning failed: {str(e)}"
            )
    
    def _execute_advisory_stage(self) -> StageResult:
        """Execute ML advisory using existing advisor_3/ scripts"""
        return self._execute_advisory_stage_core()
    
    def _execute_advisory_stage_core(self) -> StageResult:
        """Core advisory logic that works with both standard and industrial modes"""
        
        start_time = datetime.now()
        self.logger.info("🎯 Executing ML Advisory Stage")
        
        try:
            from src.advisor_3.assumption_checker import EnhancedAssumptionChecker, AssumptionConfig
            from src.advisor_3.model_recommender import recommend_model, infer_target_type
            
            # Load cleaned data
            data_path = self.pipeline_state.data_path
            delimiter = self.config['global']['csv_delimiter']
            df = pd.read_csv(data_path, delimiter=delimiter)
            
            target_column = self.config['global']['target_column']
            
            # Create assumption config
            assumption_config = AssumptionConfig(
                normality_alpha=self.config['ml_advisory']['assumption_testing']['normality_alpha'],
                vif_threshold=self.config['ml_advisory']['assumption_testing']['vif_threshold'],
                correlation_threshold=self.config['ml_advisory']['assumption_testing']['correlation_threshold'],
                verbose=self.config['ml_advisory']['assumption_testing']['verbose']
            )
            
            # Run assumption checking
            checker = EnhancedAssumptionChecker(assumption_config)
            assumptions = checker.run_all_checks(df, target_column)
            
            # Infer target type and get recommendations
            target_type = infer_target_type(df, target_column)
            recommendations = recommend_model(assumptions, target_type)
            
            # Save results
            advisory_output = {
                "assumptions": assumptions,
                "target_type": target_type,
                "model_recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
            output_path = f"{self.config['global']['output_base_dir']}/advisory/advisory_results.json"
            with open(output_path, 'w') as f:
                json.dump(advisory_output, f, indent=2, default=str)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return StageResult(
                stage_name="ml_advisory",
                success=True,
                execution_time=execution_time,
                outputs_created=[output_path],
                human_interventions=[],
                artifacts=advisory_output
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return StageResult(
                stage_name="ml_advisory",
                success=False,
                execution_time=execution_time,
                outputs_created=[],
                human_interventions=[],
                error_message=f"ML advisory failed: {str(e)}"
            )
    
    def _execute_training_stage(self) -> StageResult:
        """Execute model training using existing pipeline_4/trainer.py"""
        return self._execute_training_stage_core()
    
    def _execute_training_stage_core(self) -> StageResult:
        """Core training logic that works with both standard and industrial modes"""
        
        start_time = datetime.now()
        self.logger.info("🚀 Executing Model Training Stage")
        
        try:
            from src.pipeline_4.trainer import EnhancedModelTrainer, TrainerConfig
            
            # Load cleaned data
            data_path = self.pipeline_state.data_path
            delimiter = self.config['global']['csv_delimiter']
            df = pd.read_csv(data_path, delimiter=delimiter)
            
            target_column = self.config['global']['target_column']
            
            # Create trainer config from unified config
            training_config = self.config['model_training']
            
            config = TrainerConfig(
                test_size=training_config['test_size'],
                validation_size=training_config['validation_size'],
                random_state=self.config['global']['random_state'],  # Use global random_state
                max_models=training_config['max_models'],
                include_ensemble=training_config['include_ensemble'],
                include_advanced=training_config['include_advanced'],
                enable_tuning=training_config['enable_tuning'],
                tuning_method=training_config['tuning_method'],
                encoding_strategy=training_config['encoding_strategy'],
                scaling_strategy=training_config['scaling_strategy'],
                save_models=training_config['save_models'],
                model_dir=training_config['model_dir'],
                verbose=training_config['verbose']
            )
            
            # Initialize trainer and train models
            trainer = EnhancedModelTrainer(config)
            results = trainer.train_all_models(df, target_column)
            
            # Generate comprehensive report
            target_type = trainer.infer_target_type(df[target_column])
            report = trainer.generate_report(results, target_type, "comprehensive_training_report.json")
            
            outputs = [
                "comprehensive_training_report.json"
            ]
            
            # Add model files to outputs
            for result in results:
                if result.model_path:
                    outputs.append(result.model_path)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return StageResult(
                stage_name="model_training",
                success=True,
                execution_time=execution_time,
                outputs_created=outputs,
                human_interventions=[],
                artifacts={
                    "models_trained": len(results),
                    "best_model": report['summary']['best_model'],
                    "training_report": report
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return StageResult(
                stage_name="model_training",
                success=False,
                execution_time=execution_time,
                outputs_created=[],
                human_interventions=[],
                error_message=f"Model training failed: {str(e)}"
            )
    
    def _execute_evaluation_stage(self) -> StageResult:
        """Execute model evaluation using existing pipeline_4/evaluator.py"""
        return self._execute_evaluation_stage_core()
    
    def _execute_evaluation_stage_core(self) -> StageResult:
        """Core evaluation logic that works with both standard and industrial modes"""
        
        start_time = datetime.now()
        self.logger.info("📈 Executing Model Evaluation Stage")
        
        try:
            from src.pipeline_4.evaluator import ModelAnalyzer, AnalysisConfig
            
            # Create evaluation config
            eval_config = self.config['model_evaluation']
            
            config = AnalysisConfig(
                training_report_path=Path(eval_config['training_report_path']),
                output_dir=Path(eval_config['output_dir']),
                enable_shap=eval_config['enable_shap'],
                enable_learning_curves=eval_config['enable_learning_curves'],
                enable_residual_analysis=eval_config['enable_residual_analysis'],
                enable_stability_analysis=eval_config['enable_stability_analysis'],
                enable_interpretability=eval_config['enable_interpretability'],
                verbose=eval_config['verbose']
            )
            
            # Initialize analyzer
            analyzer = ModelAnalyzer(config)
            
            # Load cleaned data for evaluation
            data_path = self.pipeline_state.data_path
            delimiter = self.config['global']['csv_delimiter']
            df = pd.read_csv(data_path, delimiter=delimiter)
            
            target_column = self.config['global']['target_column']
            X_test = df.drop(columns=[target_column])
            y_test = df[target_column]
            
            # Run evaluation
            results = analyzer.run_analysis(X_test, y_test)
            
            # List output files
            output_dir = Path(eval_config['output_dir'])
            outputs = [str(f) for f in output_dir.glob("*") if f.is_file()]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return StageResult(
                stage_name="model_evaluation",
                success=True,
                execution_time=execution_time,
                outputs_created=outputs,
                human_interventions=[],
                artifacts={
                    "models_evaluated": len(results),
                    "evaluation_results": results
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return StageResult(
                stage_name="model_evaluation",
                success=False,
                execution_time=execution_time,
                outputs_created=[],
                human_interventions=[],
                error_message=f"Model evaluation failed: {str(e)}"
            )
    
    # Industrial-grade stage execution methods
    def _execute_profiling_stage_industrial(self) -> StageResult:
        """Execute profiling with industrial-grade features"""
        start_time = datetime.now()
        
        try:
            # Check security access first
            if self.industrial_manager.security_manager and self.industrial_manager.active_user_session:
                if not self.industrial_manager.security_manager.access_control.authorize_action(
                    self.industrial_manager.active_user_session, "run_analysis"
                ):
                    self.logger.warning(f"Access denied for stage profiling")
                    return StageResult(
                        stage_name="profiling",
                        success=False,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        outputs_created=[],
                        human_interventions=[],
                        error_message="Industrial security check failed or access denied"
                    )
            
            # Execute the actual profiling logic with industrial monitoring
            core_result = self._execute_profiling_stage_core()
            
            # Log successful execution with audit trail
            if self.industrial_manager.security_manager:
                self.industrial_manager.security_manager.audit_logger.log_action(
                    user_id="pipeline_user",
                    action="execute_profiling",
                    resource="stage_profiling",
                    result="success" if core_result.success else "failure",
                    additional_data={
                        'execution_time': core_result.execution_time,
                        'outputs_created': len(core_result.outputs_created)
                    }
                )
            
            return core_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log failure
            if self.industrial_manager.security_manager:
                self.industrial_manager.security_manager.audit_logger.log_action(
                    user_id="pipeline_user",
                    action="execute_profiling",
                    resource="stage_profiling",
                    result="failure",
                    additional_data={'error': str(e)}
                )
            
            return StageResult(
                stage_name="profiling",
                success=False,
                execution_time=execution_time,
                outputs_created=[],
                human_interventions=[],
                error_message=f"Industrial profiling failed: {str(e)}"
            )
    
    def _execute_cleaning_stage_industrial(self) -> StageResult:
        """Execute cleaning with industrial-grade features"""
        start_time = datetime.now()
        
        try:
            # Check security access
            if self.industrial_manager.security_manager and self.industrial_manager.active_user_session:
                if not self.industrial_manager.security_manager.access_control.authorize_action(
                    self.industrial_manager.active_user_session, "run_analysis"
                ):
                    return StageResult(
                        stage_name="data_cleaning",
                        success=False,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        outputs_created=[],
                        human_interventions=[],
                        error_message="Industrial security check failed or access denied"
                    )
            
            # Execute the actual cleaning logic with industrial monitoring
            core_result = self._execute_cleaning_stage_core()
            
            # Log execution with audit trail
            if self.industrial_manager.security_manager:
                self.industrial_manager.security_manager.audit_logger.log_action(
                    user_id="pipeline_user",
                    action="execute_data_cleaning",
                    resource="stage_data_cleaning",
                    result="success" if core_result.success else "failure",
                    additional_data={
                        'execution_time': core_result.execution_time,
                        'outputs_created': len(core_result.outputs_created)
                    }
                )
            
            return core_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if self.industrial_manager.security_manager:
                self.industrial_manager.security_manager.audit_logger.log_action(
                    user_id="pipeline_user",
                    action="execute_data_cleaning",
                    resource="stage_data_cleaning",
                    result="failure",
                    additional_data={'error': str(e)}
                )
            
            return StageResult(
                stage_name="data_cleaning",
                success=False,
                execution_time=execution_time,
                outputs_created=[],
                human_interventions=[],
                error_message=f"Industrial cleaning failed: {str(e)}"
            )
    
    def _execute_advisory_stage_industrial(self) -> StageResult:
        """Execute advisory with industrial-grade features"""
        start_time = datetime.now()
        
        try:
            # Check security access
            if self.industrial_manager.security_manager and self.industrial_manager.active_user_session:
                if not self.industrial_manager.security_manager.access_control.authorize_action(
                    self.industrial_manager.active_user_session, "run_analysis"
                ):
                    return StageResult(
                        stage_name="ml_advisory",
                        success=False,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        outputs_created=[],
                        human_interventions=[],
                        error_message="Industrial security check failed or access denied"
                    )
            
            # Execute the actual advisory logic with industrial monitoring
            core_result = self._execute_advisory_stage_core()
            
            # Log execution with audit trail
            if self.industrial_manager.security_manager:
                self.industrial_manager.security_manager.audit_logger.log_action(
                    user_id="pipeline_user",
                    action="execute_ml_advisory",
                    resource="stage_ml_advisory",
                    result="success" if core_result.success else "failure",
                    additional_data={
                        'execution_time': core_result.execution_time,
                        'outputs_created': len(core_result.outputs_created)
                    }
                )
            
            return core_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if self.industrial_manager.security_manager:
                self.industrial_manager.security_manager.audit_logger.log_action(
                    user_id="pipeline_user",
                    action="execute_ml_advisory",
                    resource="stage_ml_advisory",
                    result="failure",
                    additional_data={'error': str(e)}
                )
            
            return StageResult(
                stage_name="ml_advisory",
                success=False,
                execution_time=execution_time,
                outputs_created=[],
                human_interventions=[],
                error_message=f"Industrial advisory failed: {str(e)}"
            )
    
    def _execute_training_stage_industrial(self) -> StageResult:
        """Execute training with industrial-grade features"""
        start_time = datetime.now()
        
        try:
            # Check security access
            if self.industrial_manager.security_manager and self.industrial_manager.active_user_session:
                if not self.industrial_manager.security_manager.access_control.authorize_action(
                    self.industrial_manager.active_user_session, "run_analysis"
                ):
                    return StageResult(
                        stage_name="model_training",
                        success=False,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        outputs_created=[],
                        human_interventions=[],
                        error_message="Industrial security check failed or access denied"
                    )
            
            # Execute the actual training logic with industrial monitoring
            core_result = self._execute_training_stage_core()
            
            # Log execution with audit trail
            if self.industrial_manager.security_manager:
                self.industrial_manager.security_manager.audit_logger.log_action(
                    user_id="pipeline_user",
                    action="execute_model_training",
                    resource="stage_model_training",
                    result="success" if core_result.success else "failure",
                    additional_data={
                        'execution_time': core_result.execution_time,
                        'outputs_created': len(core_result.outputs_created)
                    }
                )
            
            return core_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if self.industrial_manager.security_manager:
                self.industrial_manager.security_manager.audit_logger.log_action(
                    user_id="pipeline_user",
                    action="execute_model_training",
                    resource="stage_model_training",
                    result="failure",
                    additional_data={'error': str(e)}
                )
            
            return StageResult(
                stage_name="model_training",
                success=False,
                execution_time=execution_time,
                outputs_created=[],
                human_interventions=[],
                error_message=f"Industrial training failed: {str(e)}"
            )
    
    def _execute_evaluation_stage_industrial(self) -> StageResult:
        """Execute evaluation with industrial-grade features"""
        start_time = datetime.now()
        
        try:
            # Check security access
            if self.industrial_manager.security_manager and self.industrial_manager.active_user_session:
                if not self.industrial_manager.security_manager.access_control.authorize_action(
                    self.industrial_manager.active_user_session, "run_analysis"
                ):
                    return StageResult(
                        stage_name="model_evaluation",
                        success=False,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        outputs_created=[],
                        human_interventions=[],
                        error_message="Industrial security check failed or access denied"
                    )
            
            # Execute the actual evaluation logic with industrial monitoring
            core_result = self._execute_evaluation_stage_core()
            
            # Log execution with audit trail
            if self.industrial_manager.security_manager:
                self.industrial_manager.security_manager.audit_logger.log_action(
                    user_id="pipeline_user",
                    action="execute_model_evaluation",
                    resource="stage_model_evaluation",
                    result="success" if core_result.success else "failure",
                    additional_data={
                        'execution_time': core_result.execution_time,
                        'outputs_created': len(core_result.outputs_created)
                    }
                )
            
            return core_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if self.industrial_manager.security_manager:
                self.industrial_manager.security_manager.audit_logger.log_action(
                    user_id="pipeline_user",
                    action="execute_model_evaluation",
                    resource="stage_model_evaluation",
                    result="failure",
                    additional_data={'error': str(e)}
                )
            
            return StageResult(
                stage_name="model_evaluation",
                success=False,
                execution_time=execution_time,
                outputs_created=[],
                human_interventions=[],
                error_message=f"Industrial evaluation failed: {str(e)}"
            )
    
    def _requires_human_checkpoint(self, stage_name: str, stage_result: StageResult) -> bool:
        """Determine if human checkpoint is required"""
        
        human_config = self.config.get('global', {}).get('human_intervention', {})
        
        if not human_config.get('enabled', True):
            self.logger.info(f"🤖 Human intervention disabled for {stage_name}")
            return False
        
        mode = human_config.get('mode', 'interactive')
        
        if mode == 'fully_automated':
            self.logger.info(f"🤖 Fully automated mode - skipping checkpoint for {stage_name}")
            return False
        
        if mode == 'interactive':
            # Always require checkpoint in interactive mode
            self.logger.info(f"👥 Interactive mode enabled - requiring checkpoint for {stage_name}")
            return True
        
        if mode == 'semi_automated':
            # Check if stage is in approval_required list
            approval_required = human_config.get('approval_required', [])
            if stage_name not in approval_required:
                self.logger.info(f"🎯 Stage {stage_name} not in approval_required list - skipping checkpoint")
                return False
            
            # Check confidence thresholds for required stages
            confidence_thresholds = human_config.get('confidence_thresholds', {})
            stage_confidence = self._calculate_stage_confidence(stage_name, stage_result)
            
            threshold = confidence_thresholds.get(stage_name, 0.8)
            requires_checkpoint = stage_confidence < threshold
            
            self.logger.info(f"🎯 {stage_name} confidence: {stage_confidence:.2f}, threshold: {threshold}, requires checkpoint: {requires_checkpoint}")
            return requires_checkpoint
        
        # Default: require checkpoint
        return True
    
    def _calculate_stage_confidence(self, stage_name: str, stage_result: StageResult) -> float:
        """Calculate confidence score for a stage result"""
        
        # Simplified confidence calculation
        # In practice, this would use more sophisticated metrics
        
        if not stage_result.success:
            return 0.0
        
        base_confidence = 0.8
        
        # Adjust based on execution time (faster = more confident)
        if stage_result.execution_time < 10:
            base_confidence += 0.1
        elif stage_result.execution_time > 60:
            base_confidence -= 0.1
        
        # Adjust based on outputs created
        if len(stage_result.outputs_created) > 0:
            base_confidence += 0.05
        
        return min(1.0, max(0.0, base_confidence))
    
    def _human_checkpoint(self, stage_name: str, stage_result: StageResult) -> str:
        """Interactive human checkpoint with configuration adjustment capability"""
        
        print("\\n" + "="*80)
        print(f"🤖➡️👨‍💻 HUMAN CHECKPOINT: {stage_name.upper()}")
        print("="*80)
        
        print(f"Stage: {stage_name}")
        print(f"Status: {'✅ Success' if stage_result.success else '❌ Failed'}")
        print(f"Execution Time: {stage_result.execution_time:.2f}s")
        print(f"Outputs Created: {len(stage_result.outputs_created)}")
        
        if stage_result.outputs_created:
            print("\\nFiles Created:")
            for output in stage_result.outputs_created:
                print(f"  📄 {output}")
        
        if stage_result.artifacts:
            print("\\n📊 Stage Summary:")
            for key, value in stage_result.artifacts.items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"   {key}: {value}")
                elif isinstance(value, dict) and 'total_issues' in value:
                    print(f"   {key}: {value.get('total_issues', 0)} issues found")
        
        print("\\n🔧 Available Actions:")
        print("  continue  - Proceed to next stage")
        print("  retry     - Retry this stage")
        print("  config    - Adjust configuration for upcoming stages")
        print("  abort     - Stop pipeline execution")
        
        while True:
            choice = input("\\nYour decision (continue/retry/config/abort): ").lower().strip()
            
            if choice == 'continue':
                break
            elif choice == 'retry':
                break
            elif choice == 'abort':
                break
            elif choice == 'config':
                self._adjust_configuration_interactive(stage_name, stage_result)
                continue
            else:
                print("❌ Invalid choice. Please enter 'continue', 'retry', 'config', or 'abort'")
                continue
        
        # Log human decision
        decision_log = {
            'stage': stage_name,
            'decision': choice,
            'timestamp': datetime.now().isoformat(),
            'stage_success': stage_result.success,
            'execution_time': stage_result.execution_time
        }
        self.pipeline_state.human_decisions.append(decision_log)
        
        return choice
    
    def _adjust_configuration_interactive(self, current_stage: str, stage_result: StageResult):
        """Allow interactive adjustment of configuration for upcoming stages"""
        
        print("\\n" + "="*60)
        print("⚙️  CONFIGURATION ADJUSTMENT")
        print("="*60)
        
        # Show upcoming stages
        stages = self.config.get('workflow', {}).get('stages', [])
        current_index = stages.index(current_stage) if current_stage in stages else -1
        upcoming_stages = stages[current_index + 1:] if current_index >= 0 else stages
        
        if not upcoming_stages:
            print("ℹ️  No upcoming stages to configure.")
            return
        
        print(f"Upcoming stages: {', '.join(upcoming_stages)}")
        print("\\nWhich stage would you like to configure?")
        
        for i, stage in enumerate(upcoming_stages, 1):
            print(f"  {i}. {stage}")
        print("  0. Go back")
        
        while True:
            try:
                choice = int(input("\\nEnter stage number: "))
                if choice == 0:
                    return
                elif 1 <= choice <= len(upcoming_stages):
                    selected_stage = upcoming_stages[choice - 1]
                    self._configure_stage_interactive(selected_stage, stage_result)
                    break
                else:
                    print(f"❌ Please enter a number between 0 and {len(upcoming_stages)}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def _configure_stage_interactive(self, stage_name: str, stage_result: StageResult):
        """Configure a specific stage interactively"""
        
        print(f"\\n🔧 Configuring {stage_name.upper()} Stage")
        print("-" * 40)
        
        if stage_name == 'data_cleaning':
            self._configure_cleaning_stage(stage_result)
        elif stage_name == 'ml_advisory':
            self._configure_advisory_stage(stage_result)
        elif stage_name == 'model_training':
            self._configure_training_stage(stage_result)
        elif stage_name == 'model_evaluation':
            self._configure_evaluation_stage(stage_result)
        else:
            print(f"ℹ️  Configuration for {stage_name} is not yet implemented.")
    
    def _configure_cleaning_stage(self, stage_result: StageResult):
        """Configure data cleaning parameters based on profiling results"""
        
        print("Current cleaning configuration:")
        cleaning_config = self.config['data_cleaning']
        
        # Show key parameters
        key_params = ['remove_duplicates', 'missing_col_thresh', 'impute_num', 'impute_cat', 
                     'outlier_removal', 'outlier_method', 'encoding', 'scaling']
        
        for param in key_params:
            if param in cleaning_config:
                print(f"  {param}: {cleaning_config[param]}")
        
        print("\\nWhat would you like to adjust?")
        print("1. Missing data handling")
        print("2. Outlier removal")
        print("3. Encoding strategy")
        print("4. Scaling method")
        print("0. Go back")
        
        while True:
            try:
                choice = int(input("Enter option: "))
                if choice == 0:
                    break
                elif choice == 1:
                    self._adjust_missing_data_config()
                elif choice == 2:
                    self._adjust_outlier_config()
                elif choice == 3:
                    self._adjust_encoding_config()
                elif choice == 4:
                    self._adjust_scaling_config()
                else:
                    print("❌ Invalid option")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def _adjust_missing_data_config(self):
        """Adjust missing data handling configuration"""
        current_thresh = self.config['data_cleaning']['missing_col_thresh']
        current_num_impute = self.config['data_cleaning']['impute_num']
        current_cat_impute = self.config['data_cleaning']['impute_cat']
        
        print(f"\\nCurrent missing data configuration:")
        print(f"  Column threshold: {current_thresh}")
        print(f"  Numeric imputation: {current_num_impute}")
        print(f"  Categorical imputation: {current_cat_impute}")
        
        # Column threshold
        new_thresh = input(f"New column threshold (current: {current_thresh}): ").strip()
        if new_thresh:
            try:
                self.config['data_cleaning']['missing_col_thresh'] = float(new_thresh)
                print(f"✅ Updated column threshold to {new_thresh}")
            except ValueError:
                print("❌ Invalid threshold value")
        
        # Numeric imputation
        print("\\nNumeric imputation methods: mean, median, mode, drop")
        new_num_impute = input(f"New numeric method (current: {current_num_impute}): ").strip()
        if new_num_impute and new_num_impute in ['mean', 'median', 'mode', 'drop']:
            self.config['data_cleaning']['impute_num'] = new_num_impute
            print(f"✅ Updated numeric imputation to {new_num_impute}")
        
        # Categorical imputation  
        print("\\nCategorical imputation methods: mode, constant, drop")
        new_cat_impute = input(f"New categorical method (current: {current_cat_impute}): ").strip()
        if new_cat_impute and new_cat_impute in ['mode', 'constant', 'drop']:
            self.config['data_cleaning']['impute_cat'] = new_cat_impute
            print(f"✅ Updated categorical imputation to {new_cat_impute}")
    
    def _adjust_outlier_config(self):
        """Adjust outlier removal configuration"""
        current_removal = self.config['data_cleaning']['outlier_removal']
        current_method = self.config['data_cleaning']['outlier_method']
        
        print(f"\\nCurrent outlier configuration:")
        print(f"  Removal enabled: {current_removal}")
        print(f"  Method: {current_method}")
        
        # Enable/disable outlier removal
        enable_str = input(f"Enable outlier removal? (y/n, current: {'y' if current_removal else 'n'}): ").strip().lower()
        if enable_str in ['y', 'yes']:
            self.config['data_cleaning']['outlier_removal'] = True
        elif enable_str in ['n', 'no']:
            self.config['data_cleaning']['outlier_removal'] = False
        
        # Method selection
        if self.config['data_cleaning']['outlier_removal']:
            print("\\nOutlier detection methods: iqr, isoforest, zscore")
            new_method = input(f"New method (current: {current_method}): ").strip()
            if new_method and new_method in ['iqr', 'isoforest', 'zscore']:
                self.config['data_cleaning']['outlier_method'] = new_method
                print(f"✅ Updated outlier method to {new_method}")
    
    def _adjust_encoding_config(self):
        """Adjust encoding strategy configuration"""
        current_encoding = self.config['data_cleaning']['encoding']
        
        print(f"\\nCurrent encoding: {current_encoding}")
        print("Available methods: onehot, label, ordinal, target")
        
        new_encoding = input(f"New encoding method (current: {current_encoding}): ").strip()
        if new_encoding and new_encoding in ['onehot', 'label', 'ordinal', 'target']:
            self.config['data_cleaning']['encoding'] = new_encoding
            print(f"✅ Updated encoding to {new_encoding}")
    
    def _adjust_scaling_config(self):
        """Adjust scaling method configuration"""
        current_scaling = self.config['data_cleaning']['scaling']
        
        print(f"\\nCurrent scaling: {current_scaling}")
        print("Available methods: standard, minmax, robust, none")
        
        new_scaling = input(f"New scaling method (current: {current_scaling}): ").strip()
        if new_scaling and new_scaling in ['standard', 'minmax', 'robust', 'none']:
            self.config['data_cleaning']['scaling'] = new_scaling
            print(f"✅ Updated scaling to {new_scaling}")
    
    def _configure_advisory_stage(self, stage_result: StageResult):
        """Configure ML advisory stage"""
        print("ML advisory configuration options:")
        print("  (Assumption testing thresholds, model recommendation criteria)")
        print("  This is a placeholder - specific configuration options would go here")  
    
    def _configure_training_stage(self, stage_result: StageResult):
        """Configure model training stage"""
        print("Model training configuration options:")
        print("  (Model selection, hyperparameter tuning, cross-validation)")
        print("  This is a placeholder - specific configuration options would go here")
    
    def _configure_evaluation_stage(self, stage_result: StageResult):
        """Configure model evaluation stage"""
        print("Model evaluation configuration options:")
        print("  (Evaluation metrics, analysis methods, visualization)")
        print("  This is a placeholder - specific configuration options would go here")
    
    def _inspect_stage_details(self, stage_name: str, stage_result: StageResult):
        """Inspect detailed stage results and current configuration"""
        
        print("\\n" + "="*60)
        print(f"🔍 DETAILED INSPECTION: {stage_name.upper()}")
        print("="*60)
        
        # Stage results
        print("📊 Stage Results:")
        print(f"  Success: {stage_result.success}")
        print(f"  Execution Time: {stage_result.execution_time:.2f}s")
        print(f"  Outputs: {len(stage_result.outputs_created)}")
        print(f"  Error: {stage_result.error_message or 'None'}")
        
        # Detailed artifacts
        if stage_result.artifacts:
            print("\\n📋 Detailed Results:")
            print(json.dumps(stage_result.artifacts, indent=2, default=str))
        
        # Current configuration for next stages
        print("\\n⚙️  Current Configuration (next stages):")
        stages = self.config.get('workflow', {}).get('stages', [])
        current_index = stages.index(stage_name) if stage_name in stages else -1
        upcoming_stages = stages[current_index + 1:] if current_index >= 0 else []
        
        for next_stage in upcoming_stages[:2]:  # Show next 2 stages
            if next_stage in self.config:
                print(f"\\n{next_stage.upper()}:")
                stage_config = self.config[next_stage]
                for key, value in stage_config.items():
                    if not isinstance(value, dict):
                        print(f"  {key}: {value}")
        
        input("\\nPress Enter to continue...")
    
    def _create_pipeline_summary(self, success: bool, execution_time: float = 0.0, 
                                error: str = None) -> Dict[str, Any]:
        """Create comprehensive pipeline execution summary"""
        
        summary = {
            "pipeline_execution": {
                "success": success,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "config_used": self.config_path,
                "error": error
            },
            "stages_executed": len(self.pipeline_state.completed_stages) if self.pipeline_state else 0,
            "completed_stages": self.pipeline_state.completed_stages if self.pipeline_state else [],
            "stage_results": [
                {
                    "stage": result.stage_name,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "outputs": len(result.outputs_created),
                    "error": result.error_message
                }
                for result in (self.pipeline_state.stage_results if self.pipeline_state else [])
            ],
            "human_interventions": len(self.pipeline_state.human_decisions) if self.pipeline_state else 0,
            "human_decisions": self.pipeline_state.human_decisions if self.pipeline_state else [],
            "final_outputs": []
        }
        
        # Collect all outputs
        if self.pipeline_state:
            for stage_result in self.pipeline_state.stage_results:
                summary["final_outputs"].extend(stage_result.outputs_created)
        
        # Save summary report
        summary_path = f"{self.config['global']['output_base_dir']}/reports/pipeline_summary.json"
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        summary["summary_report_path"] = summary_path
        
        return summary

def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(description='DS-AutoAdvisor Complete Pipeline')
    parser.add_argument('--config', default='config/unified_config.yaml', 
                      help='Path to unified configuration file')
    parser.add_argument('--interactive', action='store_true',
                      help='Run in interactive mode with human checkpoints')
    parser.add_argument('--automated', action='store_true', 
                      help='Run in fully automated mode')
    parser.add_argument('--industrial', action='store_true',
                      help='Enable industrial-grade features (monitoring, security, scalability)')
    parser.add_argument('--stages', nargs='+', 
                      help='Specific stages to run (e.g., --stages profiling cleaning training)',
                      choices=['profiling', 'cleaning', 'ml_advisory', 'model_training', 'model_evaluation'])
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        print("Please create the configuration file or specify a different path.")
        return 1
    
    try:
        # Initialize pipeline with industrial features if requested
        pipeline = DSAutoAdvisorPipeline(args.config, enable_industrial=args.industrial)
        
        # Override mode if specified
        if args.interactive:
            pipeline.config['global']['human_intervention']['mode'] = 'interactive'
        elif args.automated:
            pipeline.config['global']['human_intervention']['mode'] = 'fully_automated'
        
        # Override stages if specified
        if args.stages:
            # Update the workflow configuration to run only specified stages
            pipeline.config['workflow']['stages'] = args.stages
            print(f"🎯 Running specific stages: {', '.join(args.stages)}")
        
        # Run complete pipeline
        print("🚀 Starting DS-AutoAdvisor Complete Pipeline")
        print(f"📋 Configuration: {args.config}")
        print(f"🤖 Mode: {pipeline.config['global']['human_intervention']['mode']}")
        if pipeline.enable_industrial:
            print("🏭 Industrial-grade features: ENABLED")
        else:
            print("🏭 Industrial-grade features: DISABLED")
        print()
        
        result = pipeline.run_complete_pipeline()
        
        # Display results
        if result['pipeline_execution']['success']:
            print("\\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Execution Time: {result['pipeline_execution']['execution_time']:.2f}s")
            print(f"📊 Stages Completed: {result['stages_executed']}")
            print(f"👥 Human Interventions: {result['human_interventions']}")
            print(f"📄 Summary Report: {result.get('summary_report_path', 'N/A')}")
            
            print("\\n✅ Completed Stages:")
            for stage in result['completed_stages']:
                print(f"   • {stage}")
            
            if result['final_outputs']:
                print(f"\\n📁 Output Files Created ({len(result['final_outputs'])}):")
                for output in result['final_outputs'][:10]:  # Show first 10
                    print(f"   📄 {output}")
                if len(result['final_outputs']) > 10:
                    print(f"   ... and {len(result['final_outputs']) - 10} more files")
            
            return 0
        else:
            print("\\n❌ PIPELINE FAILED")
            print(f"Error: {result['pipeline_execution']['error']}")
            print(f"📊 Stages Completed: {result['stages_executed']}")
            return 1
            
    except Exception as e:
        print(f"\\n💥 Pipeline crashed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
