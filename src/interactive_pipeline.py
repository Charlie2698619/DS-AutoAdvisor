"""
Interactive DS-AutoAdvisor Pipeline
==================================

Production-ready ML pipeline with human-in-the-loop decision points.
Demonstrates industry-standard patterns for handling human intervention
in automated ML workflows.

Author: DS-AutoAdvisor Team
"""

import pandas as pd
import numpy as np
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import pipeline components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.profiling_1.data_profiler import DataProfiler
from src.correction_2.data_cleaner import DataCleaner, CleaningConfig
from src.advisor_3.assumption_checker import EnhancedAssumptionChecker, AssumptionConfig
from src.advisor_3.model_recommender import recommend_model
from src.pipeline_4.trainer import ModelTrainer, TrainerConfig
from src.pipeline_4.evaluator import ModelAnalyzer, AnalysisConfig

@dataclass
class InterventionPoint:
    """Represents a point where human intervention may be required"""
    stage: str
    task: str
    confidence: float
    data_summary: Dict[str, Any]
    recommended_action: str
    business_impact: str
    requires_approval: bool
    approvers: List[str]
    timestamp: datetime

@dataclass 
class PipelineResult:
    """Comprehensive pipeline execution results"""
    success: bool
    stages_completed: List[str]
    human_interventions: List[InterventionPoint]
    final_models: Dict[str, Any]
    performance_metrics: Dict[str, float]
    business_metrics: Dict[str, float]
    execution_time: float
    artifacts_created: List[str]
    recommendations: List[str]

class InteractiveMLPipeline:
    """
    Production ML Pipeline with Human-in-the-Loop Decision Points
    
    Features:
    - Confidence-based automation
    - Interactive checkpoints  
    - Business rule integration
    - Progressive learning
    - Stakeholder approval workflows
    """
    
    def __init__(self, config_path: str, business_rules_path: str):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.business_rules = self._load_business_rules(business_rules_path)
        self.intervention_log = []
        self.confidence_history = {}
        self.human_feedback = {}
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize automation levels
        self.automation_mode = self.config.get('human_intervention', {}).get('mode', 'interactive')
        self.auto_approve_threshold = self.config.get('human_intervention', {}).get('auto_approve_threshold', 0.95)
        
        self.logger.info(f"🚀 Initialized Interactive ML Pipeline (mode: {self.automation_mode})")
    
    def run_complete_pipeline(self, data_path: str, target_column: str) -> PipelineResult:
        """
        Execute complete DS-AutoAdvisor pipeline with human intervention points
        
        Args:
            data_path: Path to input data
            target_column: Name of target variable
            
        Returns:
            PipelineResult with comprehensive execution details
        """
        
        start_time = datetime.now()
        self.logger.info("🏁 Starting Complete DS-AutoAdvisor Pipeline")
        
        try:
            # Stage 1: Data Profiling
            self.logger.info("📊 Stage 1: Data Profiling & Quality Assessment")
            profile_results = self._stage1_profiling(data_path)
            
            # CHECKPOINT 1: Data Quality Review
            if self._requires_intervention('data_quality', profile_results):
                action = self._handle_intervention_point(
                    stage="data_quality",
                    task="review_data_quality_issues", 
                    data=profile_results,
                    confidence=self._calculate_data_quality_confidence(profile_results),
                    options=["continue", "request_data_fixes", "abort"],
                    context="Data quality issues detected that may impact model performance"
                )
                
                if action == "abort":
                    return self._create_pipeline_result(success=False, reason="Aborted at data quality check")
                elif action == "request_data_fixes":
                    return self._create_pipeline_result(success=False, reason="Data fixes requested by user")
            
            # Stage 2: Data Correction & Preprocessing
            self.logger.info("🧹 Stage 2: Data Correction & Preprocessing")
            cleaning_plan = self._generate_cleaning_plan(profile_results)
            
            # CHECKPOINT 2: Data Cleaning Strategy Review
            if self._requires_intervention('cleaning_strategy', cleaning_plan):
                approved_plan = self._review_cleaning_strategy(cleaning_plan)
                cleaned_data = self._execute_cleaning(data_path, approved_plan)
            else:
                cleaned_data = self._execute_cleaning(data_path, cleaning_plan)
            
            # CHECKPOINT 3: Feature Engineering Review
            self.logger.info("🔧 Stage 3: Feature Engineering & Selection")  
            feature_engineering_plan = self._suggest_feature_engineering(cleaned_data, target_column)
            
            if self._requires_intervention('feature_engineering', feature_engineering_plan):
                approved_features = self._review_feature_engineering(feature_engineering_plan)
                engineered_data = self._execute_feature_engineering(cleaned_data, approved_features)
            else:
                engineered_data = self._execute_feature_engineering(cleaned_data, feature_engineering_plan)
            
            # Stage 4: ML Advisory & Model Recommendation
            self.logger.info("🎯 Stage 4: ML Advisory & Model Recommendation")
            assumptions_results = self._check_ml_assumptions(engineered_data, target_column)
            model_recommendations = self._get_model_recommendations(assumptions_results)
            
            # CHECKPOINT 4: Model Selection Review
            if self._requires_intervention('model_selection', model_recommendations):
                final_model_list = self._review_model_selection(model_recommendations, assumptions_results)
            else:
                final_model_list = self._auto_select_models(model_recommendations)
            
            # Stage 5: Model Training & Evaluation
            self.logger.info("🚀 Stage 5: Model Training & Evaluation")
            training_results = self._train_models(engineered_data, target_column, final_model_list)
            evaluation_results = self._evaluate_models(training_results)
            
            # CHECKPOINT 5: Final Model Approval
            if self._requires_intervention('deployment_approval', evaluation_results):
                deployment_decision = self._review_deployment_approval(evaluation_results)
                if deployment_decision != "approve":
                    return self._create_pipeline_result(success=False, reason=f"Deployment not approved: {deployment_decision}")
            
            # Generate final results
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return PipelineResult(
                success=True,
                stages_completed=["profiling", "cleaning", "feature_engineering", "advisory", "training", "evaluation"],
                human_interventions=self.intervention_log,
                final_models=training_results,
                performance_metrics=evaluation_results.get('performance_metrics', {}),
                business_metrics=self._calculate_business_metrics(evaluation_results),
                execution_time=execution_time,
                artifacts_created=self._list_artifacts_created(),
                recommendations=self._generate_final_recommendations(evaluation_results)
            )
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}")
            return self._create_pipeline_result(success=False, reason=str(e))
    
    def _requires_intervention(self, stage: str, data: Dict[str, Any]) -> bool:
        """Determine if human intervention is required for this stage"""
        
        if self.automation_mode == "fully_automated":
            return False
        
        if self.automation_mode == "interactive":
            return True
        
        # Semi-automated: Use confidence thresholds
        confidence = self._calculate_stage_confidence(stage, data)
        return confidence < self.auto_approve_threshold
    
    def _handle_intervention_point(self, stage: str, task: str, data: Dict, 
                                 confidence: float, options: List[str], 
                                 context: str) -> str:
        """Handle a human intervention point"""
        
        intervention = InterventionPoint(
            stage=stage,
            task=task,
            confidence=confidence,
            data_summary=self._summarize_data_for_human(data),
            recommended_action=self._get_recommended_action(stage, data, confidence),
            business_impact=self._assess_business_impact(stage, data),
            requires_approval=stage in self.config.get('human_intervention', {}).get('approval_required', []),
            approvers=self._get_required_approvers(stage),
            timestamp=datetime.now()
        )
        
        self.intervention_log.append(intervention)
        
        return self._prompt_human_decision(intervention, options, context)
    
    def _prompt_human_decision(self, intervention: InterventionPoint, 
                             options: List[str], context: str) -> str:
        """Interactive prompt for human decision"""
        
        print("\n" + "="*80)
        print(f"🤖➡️👨‍💻 HUMAN INTERVENTION REQUIRED")
        print("="*80)
        print(f"Stage: {intervention.stage}")
        print(f"Task: {intervention.task}")
        print(f"Confidence: {intervention.confidence:.2%}")
        print(f"Context: {context}")
        print(f"Business Impact: {intervention.business_impact}")
        print(f"Recommended Action: {intervention.recommended_action}")
        
        if intervention.requires_approval:
            print(f"Required Approvers: {', '.join(intervention.approvers)}")
        
        print("\n📊 Data Summary:")
        for key, value in intervention.data_summary.items():
            print(f"   {key}: {value}")
        
        print(f"\n🔧 Available Options: {', '.join(options)}")
        print("-"*80)
        
        # Get human input
        while True:
            choice = input(f"Your decision ({'/'.join(options)}): ").lower().strip()
            if choice in [opt.lower() for opt in options]:
                # Log the decision
                self._log_human_decision(intervention, choice, context)
                return choice
            print(f"❌ Invalid choice. Please select from: {', '.join(options)}")
    
    def _calculate_stage_confidence(self, stage: str, data: Dict[str, Any]) -> float:
        """Calculate confidence score for automated decision making"""
        
        confidence_factors = {
            'data_quality': self._calculate_data_quality_confidence(data),
            'cleaning_strategy': self._calculate_cleaning_confidence(data),
            'feature_engineering': self._calculate_feature_confidence(data),
            'model_selection': self._calculate_model_confidence(data),
            'deployment_approval': self._calculate_deployment_confidence(data)
        }
        
        return confidence_factors.get(stage, 0.5)  # Default to medium confidence
    
    def _calculate_data_quality_confidence(self, profile_results: Dict) -> float:
        """Calculate confidence in data quality assessment"""
        
        factors = []
        
        # Missing data factor
        missing_percentage = profile_results.get('missing_percentage', 0)
        if missing_percentage < 0.05:
            factors.append(0.9)  # Very low missing data
        elif missing_percentage < 0.15:
            factors.append(0.7)  # Acceptable missing data
        else:
            factors.append(0.3)  # High missing data - low confidence
        
        # Outlier factor
        outlier_percentage = profile_results.get('outlier_percentage', 0)
        if outlier_percentage < 0.02:
            factors.append(0.9)
        elif outlier_percentage < 0.05:
            factors.append(0.7)
        else:
            factors.append(0.4)
        
        # Data type consistency
        type_issues = profile_results.get('type_issues', 0)
        if type_issues == 0:
            factors.append(0.95)
        elif type_issues < 3:
            factors.append(0.8)
        else:
            factors.append(0.5)
        
        return np.mean(factors)
    
    def _log_human_decision(self, intervention: InterventionPoint, 
                           decision: str, context: str):
        """Log human decision for learning"""
        
        decision_log = {
            'intervention_id': len(self.intervention_log),
            'stage': intervention.stage,
            'task': intervention.task,
            'confidence': intervention.confidence,
            'decision': decision,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'business_impact': intervention.business_impact
        }
        
        # Store for learning
        if intervention.stage not in self.human_feedback:
            self.human_feedback[intervention.stage] = []
        self.human_feedback[intervention.stage].append(decision_log)
        
        # Update confidence model (simplified learning)
        self._update_confidence_model(intervention.stage, decision, intervention.confidence)
        
        self.logger.info(f"📝 Human decision logged: {intervention.stage} -> {decision}")
    
    def _update_confidence_model(self, stage: str, decision: str, confidence: float):
        """Update confidence model based on human feedback"""
        
        if stage not in self.confidence_history:
            self.confidence_history[stage] = []
        
        # Simple learning: adjust confidence based on decision patterns
        learning_rate = 0.1
        
        if decision in ['continue', 'approve']:
            # Human approved automated suggestion - increase confidence
            new_confidence = min(1.0, confidence + learning_rate)
        else:
            # Human overrode automation - decrease confidence
            new_confidence = max(0.0, confidence - learning_rate * 2)
        
        self.confidence_history[stage].append({
            'timestamp': datetime.now(),
            'old_confidence': confidence,
            'new_confidence': new_confidence,
            'decision': decision
        })
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pipeline_interactive.log'),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger('InteractiveMLPipeline')
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load pipeline configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_business_rules(self, rules_path: str) -> Dict[str, Any]:
        """Load business rules and constraints"""
        with open(rules_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Placeholder methods for pipeline stages (to be implemented)
    def _stage1_profiling(self, data_path: str) -> Dict[str, Any]:
        """Execute data profiling stage"""
        # Implementation would call actual data profiler
        return {"status": "completed", "missing_percentage": 0.15, "outlier_percentage": 0.03}
    
    def _generate_cleaning_plan(self, profile_results: Dict) -> Dict[str, Any]:
        """Generate data cleaning strategy"""
        return {"outlier_removal": True, "imputation_strategy": "knn"}
    
    def _create_pipeline_result(self, success: bool, reason: str = "") -> PipelineResult:
        """Create pipeline result object"""
        return PipelineResult(
            success=success,
            stages_completed=[],
            human_interventions=self.intervention_log,
            final_models={},
            performance_metrics={},
            business_metrics={},
            execution_time=0.0,
            artifacts_created=[],
            recommendations=[reason] if reason else []
        )

def main():
    """Demonstrate interactive pipeline"""
    
    pipeline = InteractiveMLPipeline(
        config_path="config/pipeline.yaml",
        business_rules_path="config/business_rules.yaml"
    )
    
    # Example usage
    result = pipeline.run_complete_pipeline(
        data_path="data/bank.csv",
        target_column="y"
    )
    
    print(f"\n🎯 Pipeline Result: {'✅ Success' if result.success else '❌ Failed'}")
    print(f"📊 Human Interventions: {len(result.human_interventions)}")
    print(f"⏱️  Execution Time: {result.execution_time:.2f}s")

if __name__ == "__main__":
    main()
