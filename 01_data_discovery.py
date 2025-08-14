#!/usr/bin/env python3
"""
🔍 DS-AutoAdvisor: Step 1 - Comprehensive Data Discovery & Configuration
======================================================================

WHAT IT DOES:
Combines data inspection + profiling + configuration generation into one comprehensive step.
This replaces both 1_inspect_data.py and 2_configure_cleaning.py with a single, efficient workflow.

WHEN TO USE:
- First time working with new dataset
- When data changes significantly  
- When you need fresh configuration templates

HOW TO USE:
Comprehensive analysis with your data:
    python 01_data_discovery.py --data data/your_data.csv

Quick analysis with existing config:
    python 01_data_discovery.py --data data/your_data.csv --use-existing-config

Force refresh (ignore cache):
    python 01_data_discovery.py --data data/your_data.csv --force-refresh

INTERACTIVE FEATURES:
- Automated data quality assessment
- Column-specific cleaning recommendations
- Human review checkpoints for configuration
- Configuration template generation and validation

OUTPUTS:
✅ Comprehensive profiling report (HTML + JSON)
✅ Column-specific cleaning configuration (YAML)
✅ Data quality assessment report
✅ Configuration validation and recommendations

NEXT STEP:
After configuration review, test components with: python 02_stage_testing.py
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

class ComprehensiveDataDiscovery:
    """Comprehensive data discovery and configuration generation"""
    
    def __init__(self, data_path: str, output_base: str = "pipeline_outputs"):
        """Initialize discovery process"""
        self.data_path = Path(data_path)
        self.output_base = Path(output_base)
        self.dataset_name = self.data_path.stem
        
        # Create organized output structure
        self.discovery_outputs = self._create_output_structure()
        
        # Load or create configuration
        self.config = self._load_unified_config()
        
    def _create_output_structure(self) -> Dict[str, Path]:
        """Create organized output directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        outputs = {
            'base': self.output_base,
            'discovery': self.output_base / f"01_discovery_{self.dataset_name}_{timestamp}",
            'reports': self.output_base / f"01_discovery_{self.dataset_name}_{timestamp}" / "reports",
            'configs': self.output_base / f"01_discovery_{self.dataset_name}_{timestamp}" / "configs",
            'data_profiles': self.output_base / f"01_discovery_{self.dataset_name}_{timestamp}" / "data_profiles",
            'cache': self.output_base / "cache"
        }
        
        # Create all directories
        for output_dir in outputs.values():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        return outputs
    
    def _load_unified_config(self) -> Dict[str, Any]:
        """Load unified configuration"""
        # Try v3 first, then v2, then default
        config_paths = [
            project_root / "config" / "unified_config_v3.yaml",
            project_root / "config" / "unified_config_v2.yaml",
            project_root / "config" / "unified_config.yaml"
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        print(f"📋 Loaded configuration: {config_path.name}")
                        return config
                except Exception as e:
                    print(f"⚠️ Error loading {config_path}: {e}")
                    continue
        
        print(f"⚠️ No config files found, using defaults")
        return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            'global': {
                'data_input_path': str(self.data_path),
                'csv_delimiter': ',',
                'csv_encoding': 'utf-8',
                'target_column': 'Churn',  # Will be updated based on data analysis
                'output_base_dir': str(self.output_base)
            }
        }
    
    def run_comprehensive_discovery(self, force_refresh: bool = False) -> bool:
        """Run complete data discovery workflow"""
        print("🔍 DS-AutoAdvisor: Comprehensive Data Discovery")
        print("=" * 80)
        
        try:
            # Step 1: Basic Data Inspection
            if not self._basic_data_inspection():
                return False
            
            # Step 2: Enhanced Data Profiling
            if not self._enhanced_data_profiling(force_refresh):
                return False
            
            # Step 3: Data Quality Assessment
            if not self._data_quality_assessment():
                return False
            
            # Step 4: Configuration Generation
            if not self._generate_configurations():
                return False
            
            # Step 5: Human Review Checkpoint
            if not self._human_review_checkpoint():
                return False
            
            # Step 6: Save Discovery Summary
            self._save_discovery_summary()
            
            print("\n🎉 Comprehensive Data Discovery Completed Successfully!")
            print(f"📁 All outputs saved to: {self.discovery_outputs['discovery']}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Data discovery failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _basic_data_inspection(self) -> bool:
        """Step 1: Basic data inspection and validation"""
        print("\n" + "="*60)
        print("📊 STEP 1: BASIC DATA INSPECTION")
        print("="*60)
        
        try:
            # Load data
            print(f"📁 Loading data from: {self.data_path}")
            
            if not self.data_path.exists():
                print(f"❌ Data file not found: {self.data_path}")
                return False
            
            # Try to determine delimiter
            delimiter = self.config['global']['csv_delimiter']
            encoding = self.config['global']['csv_encoding']
            
            # Load with error handling
            try:
                df = pd.read_csv(self.data_path, delimiter=delimiter, encoding=encoding, nrows=5)
                if len(df.columns) == 1:
                    # Try semicolon
                    print("🔄 Trying semicolon delimiter...")
                    df = pd.read_csv(self.data_path, delimiter=';', encoding=encoding)
                    delimiter = ';'
                    self.config['global']['csv_delimiter'] = ';'
                else:
                    df = pd.read_csv(self.data_path, delimiter=delimiter, encoding=encoding)
            except UnicodeDecodeError:
                print("🔄 Trying latin-1 encoding...")
                df = pd.read_csv(self.data_path, delimiter=delimiter, encoding='latin-1')
                encoding = 'latin-1'
                self.config['global']['csv_encoding'] = 'latin-1'
            
            print(f"✅ Data loaded successfully")
            print(f"   Shape: {df.shape}")
            print(f"   Delimiter: '{delimiter}'")
            print(f"   Encoding: {encoding}")
            print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Basic column analysis
            print(f"\n📋 Column Overview ({len(df.columns)} columns):")
            for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                unique_count = df[col].nunique()
                
                print(f"   {i:2d}. {col:<20} | {str(dtype):<12} | Missing: {missing_count:4d} ({missing_pct:5.1f}%) | Unique: {unique_count:6d}")
            
            # Sample data preview
            print(f"\n📝 Data Sample (first 3 rows):")
            print(df.head(3).to_string(index=False, max_cols=8))
            if len(df.columns) > 8:
                print(f"   ... and {len(df.columns) - 8} more columns")
            
            # Save basic inspection results
            inspection_results = {
                'data_path': str(self.data_path),
                'shape': df.shape,
                'delimiter': delimiter,
                'encoding': encoding,
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'missing_summary': df.isnull().sum().to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.discovery_outputs['reports'] / "basic_inspection.json", 'w') as f:
                json.dump(inspection_results, f, indent=2, default=str)
            
            # Update config with detected settings
            self.config['global'].update({
                'data_input_path': str(self.data_path),
                'csv_delimiter': delimiter,
                'csv_encoding': encoding
            })
            
            return True
            
        except Exception as e:
            print(f"❌ Basic inspection failed: {e}")
            return False
    
    def _enhanced_data_profiling(self, force_refresh: bool = False) -> bool:
        """Step 2: Enhanced data profiling with machine-readable output"""
        print("\n" + "="*60)
        print("🔍 STEP 2: ENHANCED DATA PROFILING")
        print("="*60)
        
        try:
            # Check cache first
            cache_file = self.discovery_outputs['cache'] / f"profiling_{self.dataset_name}.json"
            
            if not force_refresh and cache_file.exists():
                print("📊 Using cached profiling results...")
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Validate cache
                cache_time = datetime.fromisoformat(cached_data['timestamp'])
                if (datetime.now() - cache_time).days < 1:
                    print(f"✅ Cache valid (age: {datetime.now() - cache_time})")
                    return True
            
            # Run enhanced profiling
            print("🔄 Running enhanced data profiling...")
            
            try:
                # Import enhanced profiler
                sys.path.append(str(project_root / "src" / "1_data_profiling"))
                from enhanced_data_profiler import EnhancedDataProfiler
                
                profiler = EnhancedDataProfiler(
                    data_path=str(self.data_path),
                    output_dir=str(self.discovery_outputs['data_profiles']),
                    generate_html=True,
                    save_raw_profile=True
                )
                
                # Run profiling with detected settings
                delimiter = self.config['global']['csv_delimiter']
                encoding = self.config['global']['csv_encoding']
                enhanced_profile, json_path, yaml_path = profiler.run_complete_profiling(
                    delimiter=delimiter, 
                    encoding=encoding
                )
                
                # Determine HTML report path (standard location from profiler)
                html_path = profiler.output_dir / 'data_profiling_report.html'
                
                print("✅ Enhanced profiling completed")
                print(f"   📄 HTML Report: {html_path}")
                print(f"   📊 Raw Profile Data: {json_path}")
                print(f"   🔧 Config Template: {yaml_path}")
                
                # Cache results
                cache_data = {
                    'timestamp': datetime.now().isoformat(),
                    'html_report': str(html_path),
                    'raw_data': str(json_path),
                    'config_template': str(yaml_path)
                }
                
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2, default=str)
                
                # Store paths for later use
                self.profiling_outputs = {
                    'html_report': html_path,
                    'raw_data': json_path,
                    'config_template': yaml_path
                }
                
                return True
                
            except ImportError:
                print("⚠️ Enhanced profiler not available, running basic profiling...")
                return self._basic_profiling_fallback()
            
        except Exception as e:
            print(f"❌ Enhanced profiling failed: {e}")
            return False
    
    def _basic_profiling_fallback(self) -> bool:
        """Fallback basic profiling if enhanced version not available"""
        print("🔄 Running basic profiling...")
        
        try:
            # Load data
            delimiter = self.config['global']['csv_delimiter']
            encoding = self.config['global']['csv_encoding']
            df = pd.read_csv(self.data_path, delimiter=delimiter, encoding=encoding)
            
            # Generate basic profile
            profile_data = {
                'dataset_info': {
                    'name': self.dataset_name,
                    'shape': df.shape,
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
                },
                'columns': {},
                'missing_data': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Column-specific analysis
            for col in df.columns:
                profile_data['columns'][col] = {
                    'dtype': str(df[col].dtype),
                    'missing_count': int(df[col].isnull().sum()),
                    'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                    'unique_count': int(df[col].nunique()),
                    'unique_percentage': float(df[col].nunique() / len(df) * 100)
                }
                
                if df[col].dtype in ['object', 'category']:
                    # Categorical analysis
                    value_counts = df[col].value_counts().head(10)
                    profile_data['columns'][col]['top_values'] = value_counts.to_dict()
                else:
                    # Numeric analysis
                    profile_data['columns'][col]['statistics'] = {
                        'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                        'std': float(df[col].std()) if not df[col].isnull().all() else None,
                        'min': float(df[col].min()) if not df[col].isnull().all() else None,
                        'max': float(df[col].max()) if not df[col].isnull().all() else None
                    }
            
            # Save profile data
            profile_path = self.discovery_outputs['data_profiles'] / f"{self.dataset_name}_profile.json"
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2, default=str)
            
            print(f"✅ Basic profiling completed")
            print(f"   📊 Profile data: {profile_path}")
            
            # Store for later use
            self.profiling_outputs = {
                'raw_data': str(profile_path),
                'html_report': None,
                'config_template': None
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Basic profiling failed: {e}")
            return False
    
    def _data_quality_assessment(self) -> bool:
        """Step 3: Data quality assessment using configuration-controlled system"""
        print("\n" + "="*60)
        print("📈 STEP 3: DATA QUALITY ASSESSMENT")
        print("="*60)
        
        try:
            # Load data for quality assessment
            delimiter = self.config['global']['csv_delimiter']
            encoding = self.config['global']['csv_encoding']
            df = pd.read_csv(self.data_path, delimiter=delimiter, encoding=encoding)
            
            # Determine which quality system to use based on configuration
            use_enhanced = False
            
            # Check configuration for enhanced system preference
            if 'custom_mode' in self.config:
                discovery_config = self.config.get('custom_mode', {}).get('data_discovery', {})
                quality_config = discovery_config.get('quality_assessment', {})
                use_enhanced = quality_config.get('use_enhanced_system', False)
            elif 'fast_mode' in self.config:
                discovery_config = self.config.get('fast_mode', {}).get('data_discovery', {})
                quality_config = discovery_config.get('quality_assessment', {})
                use_enhanced = quality_config.get('use_enhanced_system', False)
            
            print(f"🔄 Running {'enhanced' if use_enhanced else 'basic'} data quality assessment...")
            
            if use_enhanced:
                try:
                    # Import the enhanced data quality system
                    sys.path.append(str(project_root / "src" / "data_quality_system"))
                    from enhanced_quality_system import DataQualityAssessor
                    
                    # Initialize the quality assessor with config
                    config_path = project_root / "config" / "unified_config_v3.yaml"
                    assessor = DataQualityAssessor(config_path=str(config_path))
                    
                    # Determine target column (try common target column names)
                    target_column = None
                    common_targets = ['Churn', 'target', 'label', 'y']
                    for target in common_targets:
                        if target in df.columns:
                            target_column = target
                            break
                    
                    # Run comprehensive quality assessment
                    quality_report = assessor.assess_quality(df, target_column=target_column)
                    
                    # Convert enhanced report to compatible format for saving
                    quality_metrics = {
                        'overall_score': quality_report.overall_score,
                        'total_issues': quality_report.total_issues,
                        'issues_by_severity': quality_report.issues_by_severity,
                        'column_scores': quality_report.column_scores,
                        'recommendations': quality_report.recommendations,
                        'metadata': quality_report.metadata,
                        'target_column': target_column,
                        'enhanced_system_used': True,
                        'component_results': {
                            'type_inference': quality_report.type_inference_results,
                            'pattern_detection': quality_report.pattern_detection_results,
                            'quality_metrics': quality_report.quality_metrics_results
                        },
                        'issues': [
                            {
                                'severity': issue.severity,
                                'column': issue.column,
                                'issue_type': issue.issue_type,
                                'description': issue.description,
                                'suggested_action': issue.suggested_action,
                                'affected_rows_count': len(issue.affected_rows),
                                'metadata': issue.metadata
                            }
                            for issue in quality_report.issues
                        ]
                    }
                    
                    print(f"✅ Enhanced quality assessment completed")
                    print(f"   📊 Overall Quality Score: {quality_report.overall_score:.1f}/100")
                    print(f"   ⚠️ Total Issues: {quality_report.total_issues}")
                    print(f"   🎯 Target Column: {target_column or 'Auto-detected'}")
                    
                    # Show issues by severity
                    if quality_report.total_issues > 0:
                        print(f"\n⚠️ Issues by Severity:")
                        for severity, count in quality_report.issues_by_severity.items():
                            if count > 0:
                                print(f"   {severity.upper()}: {count}")
                    
                    # Show top recommendations
                    if quality_report.recommendations:
                        print(f"\n💡 Key Recommendations:")
                        for i, rec in enumerate(quality_report.recommendations[:3], 1):
                            print(f"   {i}. {rec}")
                    
                    # Save detailed enhanced report
                    enhanced_report_path = self.discovery_outputs['reports'] / "enhanced_quality_assessment.json"
                    assessor.save_assessment_report(quality_report, str(enhanced_report_path))
                    
                except ImportError as e:
                    print(f"⚠️ Enhanced quality system not available ({e}), falling back to basic assessment...")
                    return self._basic_quality_assessment_fallback(df)
                except Exception as e:
                    print(f"⚠️ Enhanced quality assessment failed ({e}), falling back to basic assessment...")
                    return self._basic_quality_assessment_fallback(df)
            else:
                # Use basic quality assessment
                print("⚠️ Using basic quality assessment (enhanced system disabled in config)")
                return self._basic_quality_assessment_fallback(df)
            
            # Save quality assessment
            quality_path = self.discovery_outputs['reports'] / "data_quality_assessment.json"
            with open(quality_path, 'w') as f:
                json.dump(quality_metrics, f, indent=2, default=str)
            
            print(f"   📄 Report: {quality_path}")
            
            self.quality_assessment = quality_metrics
            return True
            
        except Exception as e:
            print(f"❌ Data quality assessment failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _basic_quality_assessment_fallback(self, df: pd.DataFrame) -> bool:
        """Fallback basic quality assessment if enhanced system not available"""
        print("🔄 Running basic quality assessment...")
        
        # Calculate basic quality metrics
        quality_metrics = {
            'overall_score': 0.0,
            'completeness': {},
            'consistency': {},
            'uniqueness': {},
            'validity': {},
            'issues': [],
            'recommendations': []
        }
        
        # Completeness assessment
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness_score = (1 - missing_cells / total_cells) * 100
        
        quality_metrics['completeness'] = {
            'score': completeness_score,
            'total_cells': total_cells,
            'missing_cells': int(missing_cells),
            'missing_percentage': (missing_cells / total_cells) * 100
        }
        
        # Column-specific quality assessment
        for col in df.columns:
            col_missing = df[col].isnull().sum()
            col_total = len(df)
            
            if col_missing / col_total > 0.5:
                quality_metrics['issues'].append({
                    'severity': 'high',
                    'column': col,
                    'issue': f'High missing data: {col_missing}/{col_total} ({col_missing/col_total*100:.1f}%)',
                    'recommendation': 'Consider dropping column or advanced imputation'
                })
            elif col_missing / col_total > 0.1:
                quality_metrics['issues'].append({
                    'severity': 'medium',
                    'column': col,
                    'issue': f'Moderate missing data: {col_missing}/{col_total} ({col_missing/col_total*100:.1f}%)',
                    'recommendation': 'Use appropriate imputation strategy'
                })
        
        # Uniqueness assessment (potential identifiers)
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95 and df[col].dtype == 'object':
                quality_metrics['issues'].append({
                    'severity': 'info',
                    'column': col,
                    'issue': f'Potential unique identifier: {unique_ratio*100:.1f}% unique values',
                    'recommendation': 'Consider dropping for modeling'
                })
        
        # Calculate overall score
        severity_weights = {'high': 10, 'medium': 5, 'low': 2, 'info': 0}
        total_penalty = sum(severity_weights.get(issue['severity'], 0) for issue in quality_metrics['issues'])
        quality_metrics['overall_score'] = max(0, 100 - total_penalty)
        
        # Generate recommendations
        if quality_metrics['overall_score'] > 80:
            quality_metrics['recommendations'].append("Data quality is good. Proceed with standard preprocessing.")
        elif quality_metrics['overall_score'] > 60:
            quality_metrics['recommendations'].append("Data quality is acceptable but needs attention to missing values.")
        else:
            quality_metrics['recommendations'].append("Data quality issues detected. Comprehensive cleaning required.")
        
        return True
    
    def _generate_configurations(self) -> bool:
        """Step 4: Generate configuration templates"""
        print("\n" + "="*60)
        print("🔧 STEP 4: CONFIGURATION GENERATION")
        print("="*60)
        
        try:
            # Generate cleaning configuration template
            if not self._generate_cleaning_config():
                return False
            
            # Generate pipeline configuration  
            if not self._generate_pipeline_config():
                return False
            
            print("✅ Configuration generation completed")
            return True
            
        except Exception as e:
            print(f"❌ Configuration generation failed: {e}")
            return False
    
    def _generate_cleaning_config(self) -> bool:
        """Generate column-specific cleaning configuration"""
        try:
            print("🔄 Generating cleaning configuration template...")
            
            # Try to use enhanced profiler config if available
            if hasattr(self, 'profiling_outputs') and self.profiling_outputs.get('config_template'):
                template_path = self.profiling_outputs['config_template']
                if Path(template_path).exists():
                    # Copy to our outputs and update the path reference
                    import shutil
                    target_path = self.discovery_outputs['configs'] / "cleaning_config_template.yaml"
                    shutil.copy2(template_path, target_path)
                    
                    # Update profiling outputs to point to the correct location
                    self.profiling_outputs['config_template'] = str(target_path)
                    
                    print(f"   📋 Enhanced config template: {target_path}")
                    return True
            
            # Generate basic cleaning config
            delimiter = self.config['global']['csv_delimiter']
            encoding = self.config['global']['csv_encoding']
            df = pd.read_csv(self.data_path, delimiter=delimiter, encoding=encoding)
            
            cleaning_config = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'dataset': self.dataset_name,
                    'total_columns': len(df.columns),
                    'data_shape': list(df.shape)
                },
                'global_settings': {
                    'remove_duplicates': True,
                    'outlier_removal': True,
                    'outlier_method': 'iqr'
                },
                'columns': {}
            }
            
            # Generate column-specific configurations
            for col in df.columns:
                col_info = {
                    'dtype': str(df[col].dtype),
                    'missing_count': int(df[col].isnull().sum()),
                    'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                    'unique_count': int(df[col].nunique()),
                    'enabled': True
                }
                
                # Suggest actions based on column characteristics
                missing_pct = col_info['missing_percentage']
                unique_ratio = col_info['unique_count'] / len(df)
                
                # Drop column suggestions
                if unique_ratio > 0.95 and df[col].dtype == 'object':
                    col_info['drop_column'] = True
                    col_info['drop_reason'] = 'Likely unique identifier'
                else:
                    col_info['drop_column'] = False
                
                # Missing value handling
                if missing_pct > 50:
                    col_info['imputation_method'] = 'drop_column'
                elif missing_pct > 10:
                    if df[col].dtype in ['int64', 'float64']:
                        col_info['imputation_method'] = 'median'
                    else:
                        col_info['imputation_method'] = 'most_frequent'
                else:
                    col_info['imputation_method'] = 'none'
                
                # Data type conversion
                if df[col].dtype == 'object':
                    # Try to infer better type
                    sample_values = df[col].dropna().head(100)
                    if len(sample_values) > 0:
                        # Check if it could be numeric
                        try:
                            pd.to_numeric(sample_values)
                            col_info['convert_dtype'] = 'float'
                        except:
                            # Check if it could be datetime
                            try:
                                pd.to_datetime(sample_values)
                                col_info['convert_dtype'] = 'datetime'
                            except:
                                col_info['convert_dtype'] = 'string'
                
                cleaning_config['columns'][col] = col_info
            
            # Save cleaning configuration
            config_path = self.discovery_outputs['configs'] / "cleaning_config_template.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(cleaning_config, f, default_flow_style=False, indent=2)
            
            print(f"   📋 Basic config template: {config_path}")
            return True
            
        except Exception as e:
            print(f"❌ Cleaning config generation failed: {e}")
            return False
    
    def _generate_pipeline_config(self) -> bool:
        """Generate updated pipeline configuration"""
        try:
            print("🔄 Generating pipeline configuration...")
            
            # Update unified config with discovered settings
            self.config['global'].update({
                'data_input_path': str(self.data_path),
                'output_base_dir': str(self.discovery_outputs['base']),
                'csv_delimiter': self.config['global']['csv_delimiter'],
                'csv_encoding': self.config['global']['csv_encoding']
            })
            
            # Add discovery-specific paths
            self.config['discovery'] = {
                'discovery_timestamp': datetime.now().isoformat(),
                'discovery_output_dir': str(self.discovery_outputs['discovery']),
                'data_quality_score': getattr(self, 'quality_assessment', {}).get('overall_score', 0),
                'cleaning_config_template': str(self.discovery_outputs['configs'] / "cleaning_config_template.yaml")
            }
            
            # Save updated pipeline config
            config_path = self.discovery_outputs['configs'] / "pipeline_config_updated.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            print(f"   ⚙️ Pipeline config: {config_path}")
            return True
            
        except Exception as e:
            print(f"❌ Pipeline config generation failed: {e}")
            return False
    
    def _human_review_checkpoint(self) -> bool:
        """Step 5: Human review checkpoint"""
        print("\n" + "="*60)
        print("👨‍💻 STEP 5: HUMAN REVIEW CHECKPOINT")
        print("="*60)
        
        print("📋 Discovery Summary:")
        if hasattr(self, 'quality_assessment'):
            print(f"   Data Quality Score: {self.quality_assessment['overall_score']:.1f}/100")
            print(f"   Issues Found: {len(self.quality_assessment['issues'])}")
        
        print(f"   Output Directory: {self.discovery_outputs['discovery']}")
        print(f"   Configuration Templates Generated: ✅")
        
        print("\n🔍 Please Review:")
        print("   1. Data quality assessment report")
        print("   2. Cleaning configuration template")
        print("   3. Modify settings as needed")
        
        # Interactive confirmation
        while True:
            response = input("\nConfiguration review complete? (y/n/i for inspect): ").lower().strip()
            if response in ['y', 'yes']:
                print("✅ Configuration approved for testing")
                return True
            elif response in ['n', 'no']:
                print("❌ Discovery stopped for manual review")
                return False
            elif response in ['i', 'inspect']:
                self._interactive_inspection()
            else:
                print("Please enter 'y' (yes), 'n' (no), or 'i' (inspect)")
    
    def _interactive_inspection(self):
        """Interactive inspection of discovery results"""
        print("\n🔍 Interactive Discovery Inspection")
        print("Commands: 'quality', 'configs', 'files', 'data', 'quit'")
        
        while True:
            command = input("🔍 > ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'quality':
                if hasattr(self, 'quality_assessment'):
                    print(f"\n📊 Data Quality Assessment:")
                    print(f"   Overall Score: {self.quality_assessment['overall_score']:.1f}/100")
                    print(f"   Total Issues: {len(self.quality_assessment['issues'])}")
                    for issue in self.quality_assessment['issues'][:5]:
                        print(f"   [{issue['severity'].upper()}] {issue['column']}: {issue['issue']}")
                else:
                    print("No quality assessment available")
            elif command == 'configs':
                print(f"\n📋 Configuration Files:")
                config_dir = self.discovery_outputs['configs']
                for config_file in config_dir.glob("*.yaml"):
                    print(f"   📄 {config_file}")
            elif command == 'files':
                print(f"\n📁 Generated Files:")
                for category, path in self.discovery_outputs.items():
                    if path.is_dir():
                        file_count = len(list(path.glob("*")))
                        print(f"   {category}: {file_count} files in {path}")
            elif command == 'data':
                print(f"\n📊 Dataset Information:")
                print(f"   Path: {self.data_path}")
                print(f"   Name: {self.dataset_name}")
            else:
                print("   Unknown command. Available: 'quality', 'configs', 'files', 'data', 'quit'")
    
    def _save_discovery_summary(self):
        """Step 6: Save comprehensive discovery summary"""
        try:
            summary = {
                'discovery_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'dataset_name': self.dataset_name,
                    'dataset_path': str(self.data_path),
                    'output_directory': str(self.discovery_outputs['discovery'])
                },
                'data_summary': {
                    'delimiter': self.config['global']['csv_delimiter'],
                    'encoding': self.config['global']['csv_encoding']
                },
                'quality_assessment': getattr(self, 'quality_assessment', {}),
                'generated_files': {
                    'cleaning_config': str(self.discovery_outputs['configs'] / "cleaning_config_template.yaml"),
                    'pipeline_config': str(self.discovery_outputs['configs'] / "pipeline_config_updated.yaml"),
                    'quality_report': str(self.discovery_outputs['reports'] / "data_quality_assessment.json"),
                    'basic_inspection': str(self.discovery_outputs['reports'] / "basic_inspection.json")
                },
                'next_steps': [
                    "Review and modify cleaning configuration as needed",
                    "Test individual pipeline components with: python 02_stage_testing.py",
                    "Run full pipeline with: python 03_full_pipeline.py"
                ]
            }
            
            # Add profiling outputs if available
            if hasattr(self, 'profiling_outputs'):
                summary['profiling_outputs'] = self.profiling_outputs
            
            # Save summary
            summary_path = self.discovery_outputs['discovery'] / "discovery_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"📄 Discovery summary saved: {summary_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save discovery summary: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='DS-AutoAdvisor Comprehensive Data Discovery')
    parser.add_argument('--data', type=str, default='data/telco_churn_data.csv',
                       help='Path to input data file')
    parser.add_argument('--output', type=str, default='pipeline_outputs',
                       help='Base output directory')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh, ignore cache')
    parser.add_argument('--use-existing-config', action='store_true',
                       help='Use existing configuration if available')
    
    args = parser.parse_args()
    
    # Initialize and run discovery
    discovery = ComprehensiveDataDiscovery(
        data_path=args.data,
        output_base=args.output
    )
    
    success = discovery.run_comprehensive_discovery(force_refresh=args.force_refresh)
    
    if success:
        print(f"\n🎉 Data Discovery Completed Successfully!")
        print(f"📁 All outputs in: {discovery.discovery_outputs['discovery']}")
        print(f"\n🚀 Next Steps:")
        print(f"   1. Review configuration: {discovery.discovery_outputs['configs']}")
        print(f"   2. Test components: python 02_stage_testing.py --discovery-dir {discovery.discovery_outputs['discovery']}")
        print(f"   3. Run full pipeline: python 03_full_pipeline.py")
        return 0
    else:
        print(f"\n❌ Data Discovery Failed")
        return 1


if __name__ == "__main__":
    exit(main())
