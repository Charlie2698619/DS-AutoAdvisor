#!/usr/bin/env python3
"""
Test script to verify human checkpoints are working in the DS-AutoAdvisor pipeline
"""

import sys
from pathlib import Path
from complete_pipeline import DSAutoAdvisorPipeline

def test_checkpoints():
    """Test human checkpoints functionality"""
    
    print("🧪 Testing DS-AutoAdvisor Human Checkpoints")
    print("=" * 60)
    
    # Initialize pipeline with interactive mode
    config_path = "config/unified_config.yaml"
    pipeline = DSAutoAdvisorPipeline(config_path)
    
    # Check configuration
    workflow_config = pipeline.config.get('workflow', {})
    human_intervention = pipeline.config.get('human_intervention', {})  # Fixed: human_intervention is at root level
    interactive_mode = human_intervention.get('mode', 'unknown') == 'interactive'
    enabled = human_intervention.get('enabled', False)
    
    print(f"📋 Configuration Check:")
    print(f"   • Human Intervention Enabled: {enabled}")
    print(f"   • Interactive Mode: {interactive_mode}")
    print(f"   • Config Path: {config_path}")
    
    # Show which stages would trigger checkpoints
    stages = workflow_config.get('stages', [])
    print(f"\n🔍 Pipeline Stages ({len(stages)} total):")
    
    for stage in stages:
        # Simulate stage result to check if checkpoint would trigger
        from dataclasses import dataclass
        from datetime import datetime
        
        @dataclass
        class MockStageResult:
            success: bool = True
            execution_time: float = 1.0
            error_message: str = None
            outputs_created: list = None
            artifacts: dict = None
            human_interventions: list = None
            stage_name: str = ""
        
        mock_result = MockStageResult(
            success=True,
            execution_time=1.5,
            outputs_created=[f"mock_output_{stage}.csv"],
            artifacts={"test": "value"},
            human_interventions=[],
            stage_name=stage
        )
        
        would_checkpoint = pipeline._requires_human_checkpoint(stage, mock_result)
        print(f"   • {stage}: {'✅ Checkpoint' if would_checkpoint else '⏭️  Auto-continue'}")
    
    print(f"\n{'='*60}")
    print("💡 To test interactive mode, run:")
    print("   python complete_pipeline.py")
    print("\n   The pipeline should pause at each stage marked with ✅ Checkpoint")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_checkpoints()
