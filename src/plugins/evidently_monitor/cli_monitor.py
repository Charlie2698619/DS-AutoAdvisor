#!/usr/bin/env python3
"""
⚡ CLI Monitor for Evidently
===========================

Command-line interface for running monitoring tasks.
Supports cronable operations and Slack integration.
"""

import argparse
import sys
import os
import pandas as pd
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import requests
import smtplib

# Email imports with fallback
try:
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
except ImportError:
    # Fallback for older Python versions
    try:
        from email.MIMEText import MIMEText
        from email.MIMEMultipart import MIMEMultipart
    except ImportError:
        # Create dummy classes if email is not available
        class MIMEText:
            def __init__(self, *args, **kwargs):
                pass
        class MIMEMultipart:
            def __init__(self, *args, **kwargs):
                pass

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class CLIMonitor:
    """
    Command-line interface for monitoring operations.
    
    Supports:
    - Scheduled monitoring runs
    - Slack notifications
    - Email alerts
    - Report generation
    """
    
    def __init__(self, config_file: str = None):
        """Initialize CLI monitor"""
        self.config_file = config_file
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        print("⚡ CLI Monitor initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for CLI operations"""
        logger = logging.getLogger('cli_monitor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # File handler (if logs directory exists)
            log_dir = Path("logs")
            if log_dir.exists():
                file_handler = logging.FileHandler(log_dir / "cli_monitor.log")
                file_handler.setLevel(logging.INFO)
                
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            
            # Console formatter
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'monitoring': {
                'output_dir': 'pipeline_outputs',
                'reference_data_path': None,
                'current_data_path': None,
                'target_column': None,
                'prediction_column': None
            },
            'alerts': {
                'enabled': True,
                'slack': {
                    'enabled': False,
                    'webhook_url': None,
                    'channel': '#alerts',
                    'username': 'DS-AutoAdvisor Monitor'
                },
                'email': {
                    'enabled': False,
                    'smtp_server': None,
                    'smtp_port': 587,
                    'username': None,
                    'password': None,
                    'recipients': []
                }
            },
            'drift_thresholds': {
                'data_drift': 0.5,
                'column_drift': 0.5,
                'missing_values': 0.1
            }
        }
        
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                
                # Merge configurations
                default_config.update(user_config)
                self.logger.info(f"Loaded configuration from {self.config_file}")
                
            except Exception as e:
                self.logger.warning(f"Error loading config file: {e}, using defaults")
        
        return default_config
    
    def send_slack_notification(self, message: str, alert_level: str = "INFO", 
                                attachments: List[Dict] = None) -> bool:
        """Send notification to Slack"""
        
        slack_config = self.config.get('alerts', {}).get('slack', {})
        
        if not slack_config.get('enabled', False) or not slack_config.get('webhook_url'):
            self.logger.info("Slack notifications not configured")
            return False
        
        try:
            # Color coding for different alert levels
            colors = {
                'INFO': '#36a64f',      # Green
                'WARNING': '#ff9500',   # Orange
                'CRITICAL': '#ff0000'   # Red
            }
            
            payload = {
                'channel': slack_config.get('channel', '#alerts'),
                'username': slack_config.get('username', 'DS-AutoAdvisor Monitor'),
                'text': f"🔔 *{alert_level}* Alert from DS-AutoAdvisor",
                'attachments': [
                    {
                        'color': colors.get(alert_level, '#36a64f'),
                        'fields': [
                            {
                                'title': 'Alert Level',
                                'value': alert_level,
                                'short': True
                            },
                            {
                                'title': 'Timestamp',
                                'value': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'short': True
                            },
                            {
                                'title': 'Message',
                                'value': message,
                                'short': False
                            }
                        ]
                    }
                ]
            }
            
            # Add additional attachments if provided
            if attachments:
                payload['attachments'].extend(attachments)
            
            response = requests.post(slack_config['webhook_url'], json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info("Slack notification sent successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending Slack notification: {e}")
            return False
    
    def send_email_notification(self, subject: str, message: str, 
                                alert_level: str = "INFO") -> bool:
        """Send email notification"""
        
        email_config = self.config.get('alerts', {}).get('email', {})
        
        if not email_config.get('enabled', False):
            self.logger.info("Email notifications not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ", ".join(email_config['recipients'])
            msg['Subject'] = f"[{alert_level}] DS-AutoAdvisor: {subject}"
            
            # HTML body
            html_body = f"""
            <html>
            <body>
                <h2>DS-AutoAdvisor Monitoring Alert</h2>
                <p><strong>Alert Level:</strong> {alert_level}</p>
                <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Message:</strong></p>
                <div style="background-color: #f5f5f5; padding: 10px; border-left: 4px solid #007cba;">
                    {message}
                </div>
                <hr>
                <p><em>This is an automated message from DS-AutoAdvisor monitoring system.</em></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            
            text = msg.as_string()
            server.sendmail(email_config['username'], email_config['recipients'], text)
            server.quit()
            
            self.logger.info("Email notification sent successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email notification: {e}")
            return False
    
    def run_monitoring_check(self, reference_data_path: str, current_data_path: str,
                             target_column: str = None, prediction_column: str = None) -> Dict[str, Any]:
        """Run a complete monitoring check"""
        
        try:
            # Import here to avoid circular imports
            from .monitor_manager import EvidentiallyMonitor
            
            # Load data
            self.logger.info("Loading data files...")
            reference_data = pd.read_csv(reference_data_path)
            current_data = pd.read_csv(current_data_path)
            
            # Initialize monitor
            monitor = EvidentiallyMonitor(self.config['monitoring']['output_dir'])
            monitor.set_drift_thresholds(self.config['drift_thresholds'])
            
            # Run monitoring workflow
            results = monitor.run_monitoring_workflow(
                reference_data, current_data, target_column, prediction_column
            )
            
            # Process alerts
            if results['alerts']['alerts']:
                alert_level = results['alerts']['alert_level']
                alert_message = f"Detected {len(results['alerts']['alerts'])} drift issues"
                
                # Send notifications
                self.send_slack_notification(alert_message, alert_level)
                self.send_email_notification("Drift Detection Alert", alert_message, alert_level)
            
            return results
            
        except Exception as e:
            error_msg = f"Error in monitoring check: {e}"
            self.logger.error(error_msg)
            
            # Send critical alert
            self.send_slack_notification(error_msg, "CRITICAL")
            self.send_email_notification("Monitoring Error", error_msg, "CRITICAL")
            
            raise e
    
    def generate_html_report_for_slack(self, report_path: Path) -> str:
        """Generate Slack-friendly summary from HTML report"""
        
        try:
            # For now, return a simple summary
            # In the future, we could parse the HTML and extract key metrics
            
            summary = f"""
📊 *Monitoring Report Generated*

📁 Report Location: `{report_path.name}`
🕐 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

To view the full report, access: `{report_path}`
            """
            
            return summary.strip()
            
        except Exception as e:
            self.logger.warning(f"Error generating Slack report summary: {e}")
            return f"Report generated: {report_path.name}"
    
    def create_cron_command(self, reference_data_path: str, current_data_path: str,
                            config_file_path: str = None, target_column: str = None) -> str:
        """Generate a cron command for scheduled monitoring"""
        
        # Build command
        python_path = sys.executable
        script_path = Path(__file__).absolute()
        
        cmd_parts = [
            python_path,
            str(script_path),
            '--reference-data', reference_data_path,
            '--current-data', current_data_path
        ]
        
        if config_file_path:
            cmd_parts.extend(['--config', config_file_path])
        
        if target_column:
            cmd_parts.extend(['--target-column', target_column])
        
        command = ' '.join(cmd_parts)
        
        # Example cron schedule (daily at 2 AM)
        cron_example = f"0 2 * * * {command}"
        
        print(f"📅 Example cron command (daily at 2 AM):")
        print(f"   {cron_example}")
        print(f"\n💡 To install:")
        print(f"   crontab -e")
        print(f"   # Add the line above")
        
        return cron_example


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='DS-AutoAdvisor Monitoring CLI')
    
    parser.add_argument('--reference-data', required=True,
                        help='Path to reference (training) data CSV')
    parser.add_argument('--current-data', required=True,
                        help='Path to current (production) data CSV')
    parser.add_argument('--target-column', 
                        help='Name of the target column')
    parser.add_argument('--prediction-column',
                        help='Name of the prediction column')
    parser.add_argument('--config', 
                        help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--generate-cron', action='store_true',
                        help='Generate cron command example')
    parser.add_argument('--demo-drift', choices=['low', 'medium', 'high'],
                        help='Generate synthetic drift demo with specified severity')
    
    args = parser.parse_args()
    
    try:
        # Initialize CLI monitor
        cli_monitor = CLIMonitor(args.config)
        
        if args.generate_cron:
            cli_monitor.create_cron_command(
                args.reference_data, args.current_data, 
                args.config, args.target_column
            )
            return
        
        # Handle synthetic drift demo
        if args.demo_drift:
            print(f"🎭 Generating synthetic drift demo with {args.demo_drift} severity...")
            
            # Load reference data
            reference_data = pd.read_csv(args.reference_data)
            
            # Create synthetic drift data
            from .monitor_manager import EvidentiallyMonitor
            monitor = EvidentiallyMonitor()
            drift_data = monitor.create_synthetic_drift_demo(
                reference_data, args.target_column, args.demo_drift
            )
            
            # Save synthetic data
            drift_file = Path(args.current_data).parent / f"synthetic_drift_{args.demo_drift}.csv"
            drift_data.to_csv(drift_file, index=False)
            print(f"💾 Synthetic drift data saved: {drift_file}")
            
            # Update current data path to use synthetic data
            args.current_data = str(drift_file)
        
        # Run monitoring check
        print("🚀 Starting monitoring check...")
        results = cli_monitor.run_monitoring_check(
            args.reference_data, args.current_data,
            args.target_column, args.prediction_column
        )
        
        print("✅ Monitoring check completed successfully")
        print(f"📊 Summary: {results['summary']}")
        
        if results['alerts']['alerts']:
            print(f"⚠️  {len(results['alerts']['alerts'])} alerts generated")
        else:
            print("✅ No alerts - data looks good!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
