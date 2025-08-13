#!/usr/bin/env python3
"""
📝 Report Generator
==================

Generates HTML and JSON reports for monitoring results.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

class ReportGenerator:
    """Generate monitoring reports in various formats"""
    
    def __init__(self, output_dir: str = "pipeline_outputs"):
        """Initialize report generator"""
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger('report_generator')
    
    def generate_slack_summary(self, monitoring_results: Dict[str, Any]) -> str:
        """Generate Slack-friendly summary"""
        
        alerts = monitoring_results.get('alerts', {})
        summary = monitoring_results.get('summary', {})
        
        # Emoji for alert levels
        alert_emoji = {
            'INFO': '✅',
            'WARNING': '⚠️',
            'CRITICAL': '🚨'
        }
        
        alert_level = alerts.get('alert_level', 'INFO')
        emoji = alert_emoji.get(alert_level, '📊')
        
        message = f"""
{emoji} *Monitoring Report - {alert_level}*

📊 *Data Summary:*
• Reference samples: {summary.get('data_samples', {}).get('reference', 'N/A')}
• Current samples: {summary.get('data_samples', {}).get('current', 'N/A')}
• Columns analyzed: {summary.get('columns_analyzed', 'N/A')}

🔍 *Alert Summary:*
• Alert level: {alert_level}
• Total alerts: {summary.get('total_alerts', 0)}

⏰ *Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return message.strip()
    
    def save_summary_json(self, results: Dict[str, Any], filename: str = None) -> Path:
        """Save monitoring summary as JSON"""
        
        if filename is None:
            filename = f"monitoring_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Summary saved: {output_path}")
        return output_path
