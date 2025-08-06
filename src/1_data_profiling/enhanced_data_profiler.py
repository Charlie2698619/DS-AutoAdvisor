import pandas as pd
import numpy as np
import json
from ydata_profiling import ProfileReport
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class ColumnProfile:
    """Detailed profile for a single column"""
    name: str
    dtype: str
    count: int
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    
    # Numeric stats (None for non-numeric)
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    q25: Optional[float] = None
    q50: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    outliers_iqr: Optional[int] = None
    outliers_zscore: Optional[int] = None
    
    # Categorical stats (None for numeric)
    most_frequent_value: Optional[str] = None
    most_frequent_count: Optional[int] = None
    cardinality: Optional[int] = None
    top_values: Optional[Dict[str, int]] = None
    
    # Text analysis
    has_text: bool = False
    avg_length: Optional[float] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    
    # Data quality issues
    has_duplicates: bool = False
    has_whitespace_issues: bool = False
    has_mixed_types: bool = False
    suspected_date_column: bool = False
    
    # Recommendations
    recommended_actions: List[str] = None
    transformation_suggestions: List[str] = None

@dataclass 
class DatasetProfile:
    """Complete dataset profile"""
    name: str
    shape: tuple
    total_missing: int
    total_duplicates: int
    memory_usage_mb: float
    columns: Dict[str, ColumnProfile]
    correlations: Optional[Dict[str, Dict[str, float]]] = None
    profiling_timestamp: str = None
    
    def __post_init__(self):
        if self.profiling_timestamp is None:
            self.profiling_timestamp = datetime.now().isoformat()

class EnhancedDataProfiler:
    """Enhanced data profiler that extracts machine-readable insights"""
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = "docs/",
                 generate_html: bool = True,
                 save_raw_profile: bool = True):
        
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.generate_html = generate_html
        self.save_raw_profile_enabled = save_raw_profile  # Renamed to avoid conflict
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data
        self.df = None
        self.ydata_profile = None
        self.enhanced_profile = None
    
    def load_data(self, **read_csv_kwargs) -> pd.DataFrame:
        """Load data with flexible CSV options"""
        print(f"📊 Loading data from {self.data_path}")
        
        # Default CSV settings
        default_kwargs = {
            'delimiter': ';',
            'encoding': 'utf-8',
            'header': 0
        }
        default_kwargs.update(read_csv_kwargs)
        
        try:
            self.df = pd.read_csv(self.data_path, **default_kwargs)
            print(f"✅ Data loaded successfully: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            raise
    
    def generate_ydata_profile(self) -> ProfileReport:
        """Generate YData profiling report"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("📈 Generating YData profiling report...")
        
        self.ydata_profile = ProfileReport(
            self.df,
            title=f"Profile Report - {self.data_path.stem}",
            explorative=True,
            minimal=False,
            samples={"head": 5, "tail": 5},
            correlations={"auto": {"calculate": True}},
            missing_diagrams={"heatmap": True, "bar": True}
        )
        
        if self.generate_html:
            html_path = self.output_dir / "data_profiling_report.html"
            self.ydata_profile.to_file(html_path)
            print(f"📄 HTML report saved: {html_path}")
        
        return self.ydata_profile
    
    def extract_raw_profile_data(self) -> DatasetProfile:
        """Extract machine-readable data from YData profile"""
        if self.ydata_profile is None:
            raise ValueError("YData profile not generated. Call generate_ydata_profile() first.")
        
        print("🔍 Extracting raw profiling data...")
        
        # Get the report description (ydata-profiling v4+ compatible)
        try:
            # Try new API first (v4+)
            description = self.ydata_profile.get_description()
            # Convert to dict if it's a BaseDescription object
            if hasattr(description, 'variables'):
                variables = description.variables
            else:
                # Fallback: try to access as dict
                variables = description.get("variables", {})
        except Exception as e:
            print(f"⚠️ Could not extract description: {e}")
            # Fallback: manual extraction from DataFrame
            variables = self._extract_manual_profile()
        
        # Extract column profiles
        columns = {}
        
        for col_name in self.df.columns:
            try:
                # Try to get column data from profiling output
                if hasattr(variables, col_name):
                    col_data = getattr(variables, col_name)
                elif isinstance(variables, dict) and col_name in variables:
                    col_data = variables[col_name]
                else:
                    # Manual extraction if profiling data not available
                    col_data = self._extract_manual_column_profile(col_name)
                
                # Build column profile
                col_profile = self._build_column_profile(col_name, col_data)
                columns[col_name] = col_profile
                
            except Exception as e:
                print(f"⚠️ Could not process column '{col_name}': {e}")
                # Create basic profile from manual extraction
                col_profile = self._extract_manual_column_profile(col_name)
                columns[col_name] = col_profile
        
        # Extract correlations if available
        correlations = None
        try:
            if hasattr(description, 'correlations'):
                correlations = description.correlations
            elif isinstance(description, dict) and "correlations" in description:
                correlations = description["correlations"]
        except:
            pass
        
        # Create dataset profile
        self.enhanced_profile = DatasetProfile(
            name=self.data_path.stem,
            shape=self.df.shape,
            total_missing=self.df.isnull().sum().sum(),
            total_duplicates=self.df.duplicated().sum(),
            memory_usage_mb=self.df.memory_usage(deep=True).sum() / 1024**2,
            columns=columns,
            correlations=correlations
        )
        
        return self.enhanced_profile
    
    def _extract_manual_profile(self) -> dict:
        """Manual profile extraction when ydata-profiling API fails"""
        print("📊 Falling back to manual profiling extraction...")
        variables = {}
        
        for col_name in self.df.columns:
            col_data = self._extract_manual_column_profile(col_name)
            variables[col_name] = col_data
            
        return variables
    
    def _extract_manual_column_profile(self, col_name: str) -> ColumnProfile:
        """Extract column profile manually from DataFrame"""
        col_series = self.df[col_name]
        
        # Basic statistics
        count = len(col_series)
        missing_count = col_series.isnull().sum()
        missing_percentage = (missing_count / count) * 100 if count > 0 else 0
        unique_count = col_series.nunique()
        unique_percentage = (unique_count / count) * 100 if count > 0 else 0
        
        # Create base profile
        col_profile = ColumnProfile(
            name=col_name,
            dtype=str(col_series.dtype),
            count=count,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage
        )
        
        # Numeric statistics
        if pd.api.types.is_numeric_dtype(col_series):
            try:
                col_profile.mean = float(col_series.mean()) if not col_series.empty else None
                col_profile.std = float(col_series.std()) if not col_series.empty else None
                col_profile.min_value = float(col_series.min()) if not col_series.empty else None
                col_profile.max_value = float(col_series.max()) if not col_series.empty else None
                
                # Quartiles
                quartiles = col_series.quantile([0.25, 0.5, 0.75])
                col_profile.q25 = float(quartiles[0.25]) if len(quartiles) > 0 else None
                col_profile.q50 = float(quartiles[0.5]) if len(quartiles) > 0 else None
                col_profile.q75 = float(quartiles[0.75]) if len(quartiles) > 0 else None
                
                # Skewness and kurtosis
                from scipy import stats
                col_profile.skewness = float(stats.skew(col_series.dropna())) if len(col_series.dropna()) > 0 else None
                col_profile.kurtosis = float(stats.kurtosis(col_series.dropna())) if len(col_series.dropna()) > 0 else None
                
                # Outliers (IQR method)
                if col_profile.q25 and col_profile.q75:
                    iqr = col_profile.q75 - col_profile.q25
                    lower_bound = col_profile.q25 - 1.5 * iqr
                    upper_bound = col_profile.q75 + 1.5 * iqr
                    col_profile.outliers_iqr = int(((col_series < lower_bound) | (col_series > upper_bound)).sum())
                
                # Z-score outliers
                if col_profile.mean and col_profile.std and col_profile.std > 0:
                    z_scores = np.abs((col_series - col_profile.mean) / col_profile.std)
                    col_profile.outliers_zscore = int((z_scores > 3).sum())
                    
            except Exception as e:
                print(f"⚠️ Error calculating numeric stats for {col_name}: {e}")
        
        # Categorical statistics
        elif col_series.dtype == 'object' or pd.api.types.is_categorical_dtype(col_series):
            try:
                col_profile.cardinality = unique_count
                
                # Get top values
                value_counts = col_series.value_counts().head(10)
                col_profile.top_values = value_counts.to_dict()
                
                # Most frequent
                if len(value_counts) > 0:
                    col_profile.most_frequent_value = str(value_counts.index[0])
                    col_profile.most_frequent_count = int(value_counts.iloc[0])
                    
            except Exception as e:
                print(f"⚠️ Error calculating categorical stats for {col_name}: {e}")
        
        # Text analysis for string columns
        if col_series.dtype == 'object':
            try:
                col_profile.has_text = True
                text_lengths = col_series.astype(str).str.len()
                col_profile.avg_length = float(text_lengths.mean()) if not text_lengths.empty else None
                col_profile.max_length = int(text_lengths.max()) if not text_lengths.empty else None
                col_profile.min_length = int(text_lengths.min()) if not text_lengths.empty else None
                
                # Check for whitespace issues
                col_profile.has_whitespace_issues = bool((
                    col_series.astype(str).str.strip() != col_series.astype(str)
                ).any())
                
            except Exception as e:
                print(f"⚠️ Error calculating text stats for {col_name}: {e}")
        
        # Check for duplicates
        col_profile.has_duplicates = bool(col_series.duplicated().any())
        
        # Date detection (simple heuristics)
        if col_profile.has_text:
            try:
                sample_values = col_series.dropna().astype(str).head(100)
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                    r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                ]
                import re
                for pattern in date_patterns:
                    if sample_values.str.contains(pattern, regex=True).any():
                        col_profile.suspected_date_column = True
                        break
            except Exception as e:
                print(f"⚠️ Error in date detection for {col_name}: {e}")
        
        # Generate recommendations
        col_profile.recommended_actions = self._generate_column_recommendations(col_profile)
        col_profile.transformation_suggestions = self._generate_transformation_suggestions(col_profile)
        
        return col_profile
    
    def _build_column_profile(self, col_name: str, col_data) -> ColumnProfile:
        """Build ColumnProfile from ydata-profiling output (when available)"""
        try:
            # Try to extract from ydata-profiling format
            if hasattr(col_data, 'count'):
                # New API format
                count = getattr(col_data, 'count', 0)
                missing_count = getattr(col_data, 'n_missing', 0)
                unique_count = getattr(col_data, 'n_unique', 0)
            elif isinstance(col_data, dict):
                # Dict format
                count = col_data.get("count", 0)
                missing_count = col_data.get("n_missing", 0)
                unique_count = col_data.get("n_unique", 0)
            else:
                # Fallback to manual extraction
                return self._extract_manual_column_profile(col_name)
            
            # Base statistics
            col_profile = ColumnProfile(
                name=col_name,
                dtype=str(self.df[col_name].dtype),
                count=count,
                missing_count=missing_count,
                missing_percentage=(missing_count / count) * 100 if count > 0 else 0,
                unique_count=unique_count,
                unique_percentage=(unique_count / count) * 100 if count > 0 else 0
            )
            
            # Try to extract additional statistics
            try:
                if hasattr(col_data, 'type') or (isinstance(col_data, dict) and 'type' in col_data):
                    col_type = getattr(col_data, 'type', None) or col_data.get('type')
                    
                    if col_type in ["Numeric", "Real"]:
                        # Extract numeric statistics
                        stats_attr = getattr(col_data, 'statistics', None) or col_data.get('statistics', {})
                        if stats_attr:
                            col_profile.mean = stats_attr.get("mean")
                            col_profile.std = stats_attr.get("std")
                            col_profile.min_value = stats_attr.get("min")
                            col_profile.max_value = stats_attr.get("max")
                            col_profile.q25 = stats_attr.get("25%")
                            col_profile.q50 = stats_attr.get("50%")
                            col_profile.q75 = stats_attr.get("75%")
                            col_profile.skewness = stats_attr.get("skewness")
                            col_profile.kurtosis = stats_attr.get("kurtosis")
                    
                    elif col_type in ["Categorical", "Text"]:
                        # Extract categorical statistics
                        col_profile.cardinality = unique_count
                        
                        # Try to get value counts
                        value_counts_attr = getattr(col_data, 'value_counts_with_nan', None) or col_data.get('value_counts_with_nan', {})
                        if value_counts_attr:
                            # Convert to regular dict and get top 10
                            top_values = dict(list(value_counts_attr.items())[:10]) if hasattr(value_counts_attr, 'items') else {}
                            col_profile.top_values = top_values
                            
                            # Most frequent
                            if top_values:
                                most_freq = list(top_values.items())[0]
                                col_profile.most_frequent_value = str(most_freq[0])
                                col_profile.most_frequent_count = most_freq[1]
            except Exception as e:
                print(f"⚠️ Could not extract detailed stats for {col_name}: {e}")
            
            # Fill in remaining fields manually
            col_profile = self._fill_missing_profile_data(col_profile, col_name)
            
            return col_profile
            
        except Exception as e:
            print(f"⚠️ Error building profile for {col_name}: {e}")
            # Fallback to manual extraction
            return self._extract_manual_column_profile(col_name)
    
    def _fill_missing_profile_data(self, col_profile: ColumnProfile, col_name: str) -> ColumnProfile:
        """Fill in missing profile data with manual calculations"""
        col_series = self.df[col_name]
        
        # Text analysis for string columns
        if col_series.dtype == 'object':
            col_profile.has_text = True
            if col_profile.avg_length is None:
                text_lengths = col_series.astype(str).str.len()
                col_profile.avg_length = float(text_lengths.mean()) if not text_lengths.empty else None
                col_profile.max_length = int(text_lengths.max()) if not text_lengths.empty else None
                col_profile.min_length = int(text_lengths.min()) if not text_lengths.empty else None
            
            # Check for whitespace issues
            col_profile.has_whitespace_issues = bool((
                col_series.astype(str).str.strip() != col_series.astype(str)
            ).any())
        
        # Check for duplicates
        col_profile.has_duplicates = bool(col_series.duplicated().any())
        
        # Date detection (simple heuristics)
        if col_profile.has_text:
            sample_values = col_series.dropna().astype(str).head(100)
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            ]
            import re
            for pattern in date_patterns:
                try:
                    if sample_values.str.contains(pattern, regex=True).any():
                        col_profile.suspected_date_column = True
                        break
                except:
                    pass
        
        # Calculate outliers if missing
        if pd.api.types.is_numeric_dtype(col_series) and col_profile.outliers_iqr is None:
            try:
                Q1, Q3 = col_series.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_profile.outliers_iqr = int(((col_series < lower_bound) | (col_series > upper_bound)).sum())
                
                # Z-score outliers
                if col_profile.mean and col_profile.std and col_profile.std > 0:
                    z_scores = np.abs((col_series - col_profile.mean) / col_profile.std)
                    col_profile.outliers_zscore = int((z_scores > 3).sum())
            except:
                pass
        
        # Generate recommendations
        col_profile.recommended_actions = self._generate_column_recommendations(col_profile)
        col_profile.transformation_suggestions = self._generate_transformation_suggestions(col_profile)
        
        return col_profile
    
    def _generate_column_recommendations(self, col_profile: ColumnProfile) -> List[str]:
        """Generate cleaning recommendations for a column"""
        recommendations = []
        
        # Missing data
        if col_profile.missing_percentage > 50:
            recommendations.append("consider_dropping_column")
        elif col_profile.missing_percentage > 10:
            recommendations.append("impute_missing_values")
        
        # High cardinality
        if col_profile.cardinality and col_profile.cardinality > 50:
            recommendations.append("reduce_cardinality")
        
        # Outliers
        if col_profile.outliers_iqr and col_profile.outliers_iqr > 0:
            recommendations.append("handle_outliers")
        
        # Text issues
        if col_profile.has_whitespace_issues:
            recommendations.append("clean_whitespace")
        
        # Date columns
        if col_profile.suspected_date_column:
            recommendations.append("convert_to_datetime")
        
        return recommendations
    
    def _generate_transformation_suggestions(self, col_profile: ColumnProfile) -> List[str]:
        """Generate transformation suggestions for a column"""
        suggestions = []
        
        # Numeric transformations
        if col_profile.skewness:
            if abs(col_profile.skewness) > 1:
                suggestions.append("apply_log_transform")
            elif abs(col_profile.skewness) > 0.5:
                suggestions.append("apply_sqrt_transform")
        
        # Categorical transformations
        if col_profile.cardinality:
            if col_profile.cardinality <= 10:
                suggestions.append("one_hot_encoding")
            elif col_profile.cardinality <= 50:
                suggestions.append("label_encoding")
            else:
                suggestions.append("target_encoding")
        
        # Scaling suggestions
        if col_profile.std and col_profile.mean:
            if col_profile.std / abs(col_profile.mean) > 1:
                suggestions.append("standard_scaling")
            else:
                suggestions.append("minmax_scaling")
        
        return suggestions
    
    def save_raw_profile(self, filename: str = "raw_profiling_data.json"):
        """Save machine-readable profile data to JSON"""
        if self.enhanced_profile is None:
            raise ValueError("Enhanced profile not generated. Call extract_raw_profile_data() first.")
        
        output_path = self.output_dir / filename
        
        # Convert dataclasses to dict for JSON serialization
        profile_dict = asdict(self.enhanced_profile)
        
        # Clean up any None values and convert numpy types
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        clean_dict = clean_for_json(profile_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_dict, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Raw profiling data saved: {output_path}")
        return output_path
    
    def generate_cleaning_config_template(self, filename: str = "cleaning_config_template.yaml"):
        """Generate a YAML template for column-specific cleaning"""
        if self.enhanced_profile is None:
            raise ValueError("Enhanced profile not generated. Call extract_raw_profile_data() first.")
        
        # Save to config directory instead of output directory
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        output_path = config_dir / filename
        
        yaml_content = self._build_cleaning_config_yaml()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"⚙️ Cleaning config template saved: {output_path}")
        return output_path
    
    def _build_cleaning_config_yaml(self) -> str:
        """Build YAML content for cleaning configuration"""
        yaml_lines = [
            "# Auto-generated cleaning configuration template",
            f"# Generated from: {self.data_path.name}",
            f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "# Global settings",
            "global_settings:",
            "  remove_duplicates: true",
            "  outlier_method: 'iqr'  # iqr, isoforest, zscore",
            "  missing_row_threshold: 0.5",
            "  correlation_threshold: 0.95",
            "",
            "# Column-specific transformations",
            "column_transformations:"
        ]
        
        for col_name, col_profile in self.enhanced_profile.columns.items():
            # Determine if column should be dropped
            is_unique_identifier = False
            drop_reason = None
            
            # Check for unique identifier patterns
            dataset_rows = self.enhanced_profile.shape[0]
            if col_profile.unique_count == dataset_rows and col_profile.unique_count > 1:
                is_unique_identifier = True
                drop_reason = "unique_identifier"
            elif col_name.lower() in ['id', 'customer_id', 'customerid', 'user_id', 'userid', 'index', 'row_id']:
                is_unique_identifier = True
                drop_reason = "unique_identifier"
            elif col_profile.cardinality and col_profile.cardinality > 0.95 * dataset_rows:
                is_unique_identifier = True
                drop_reason = "high_cardinality"
            
            yaml_lines.extend([
                f"  {col_name}:",
                f"    # Data type: {col_profile.dtype}",
                f"    # Missing: {col_profile.missing_percentage:.1f}%",
                f"    # Unique: {col_profile.unique_count}",
            ])
            
            # Add specific recommendations
            if col_profile.recommended_actions:
                yaml_lines.append(f"    # Recommendations: {', '.join(col_profile.recommended_actions)}")
            
            # Column-level actions - suggest dropping if it's a unique identifier
            if is_unique_identifier:
                yaml_lines.append(f"    drop_column: true  # Suggested: appears to be a unique identifier")
                yaml_lines.append(f"    drop_reason: '{drop_reason}'  # unique_identifier, irrelevant, redundant, high_cardinality, low_variance")
            else:
                yaml_lines.append(f"    drop_column: false  # Set to true to remove this column entirely")
                yaml_lines.append(f"    drop_reason: null  # unique_identifier, irrelevant, redundant, high_cardinality, low_variance")
            
            # Data type conversion
            yaml_lines.append(f"    dtype_conversion: null  # int, float, str, bool, category, datetime")
            yaml_lines.append(f"    datetime_format: null  # '%Y-%m-%d', '%d/%m/%Y', etc.")
            
            # Missing value handling
            if col_profile.missing_percentage > 0:
                if col_profile.dtype in ['int64', 'float64']:
                    yaml_lines.append(f"    imputation: 'median'  # mean, median, most_frequent, constant, knn, iterative, interpolate, forward_fill, backward_fill")
                else:
                    yaml_lines.append(f"    imputation: 'most_frequent'  # mean, median, most_frequent, constant, knn, iterative")
            else:
                yaml_lines.append(f"    imputation: null  # No missing values")
            yaml_lines.append(f"    imputation_value: null  # for constant imputation")
            
            # Outlier handling for numeric columns
            if col_profile.dtype in ['int64', 'float64'] and col_profile.outliers_iqr and col_profile.outliers_iqr > 0:
                yaml_lines.append(f"    outlier_treatment: 'remove'  # remove, cap, winsorize, null, iqr_cap, zscore_cap, isolation_forest")
                yaml_lines.append(f"    outlier_threshold: 1.5  # IQR factor or Z-score threshold")
            else:
                yaml_lines.append(f"    outlier_treatment: null")
                yaml_lines.append(f"    outlier_threshold: null")
            
            # Text cleaning
            if col_profile.dtype == 'object':
                yaml_lines.append(f"    text_cleaning: false")
                yaml_lines.append(f"    text_operations: null  # [strip, lower, upper, title, remove_digits, remove_punctuation, remove_extra_spaces]")
                yaml_lines.append(f"    regex_replace: null  # {{'pattern': 'replacement'}}")
            else:
                yaml_lines.append(f"    text_cleaning: false")
                yaml_lines.append(f"    text_operations: null")
                yaml_lines.append(f"    regex_replace: null")
            
            # Encoding for categorical columns
            if col_profile.dtype == 'object':
                if col_profile.suspected_date_column:
                    yaml_lines.append(f"    encoding: null")
                    yaml_lines.append(f"    date_conversion: true")
                    yaml_lines.append(f"    extract_date_features: null  # [year, month, day, dayofweek, quarter, is_weekend, hour, minute]")
                elif col_profile.cardinality and col_profile.cardinality <= 10:
                    yaml_lines.append(f"    encoding: 'onehot'  # onehot, label, target, ordinal, binary, frequency, mean_encoding")
                    yaml_lines.append(f"    date_conversion: false")
                    yaml_lines.append(f"    extract_date_features: null")
                elif col_profile.cardinality and col_profile.cardinality <= 50:
                    yaml_lines.append(f"    encoding: 'label'  # onehot, label, target, ordinal, binary, frequency, mean_encoding")
                    yaml_lines.append(f"    date_conversion: false")
                    yaml_lines.append(f"    extract_date_features: null")
                else:
                    yaml_lines.append(f"    encoding: 'target'  # onehot, label, target, ordinal, binary, frequency, mean_encoding")
                    yaml_lines.append(f"    date_conversion: false")
                    yaml_lines.append(f"    extract_date_features: null")
                yaml_lines.append(f"    encoding_categories: null  # explicit order for ordinal encoding")
                yaml_lines.append(f"    max_categories: null  # limit for high cardinality")
            else:
                yaml_lines.append(f"    encoding: null")
                yaml_lines.append(f"    date_conversion: false")
                yaml_lines.append(f"    extract_date_features: null")
                yaml_lines.append(f"    encoding_categories: null")
                yaml_lines.append(f"    max_categories: null")
            
            # Transformations and scaling for numeric columns
            if col_profile.dtype in ['int64', 'float64']:
                if col_profile.skewness and abs(col_profile.skewness) > 1:
                    yaml_lines.append(f"    transformation: 'log'  # log, log1p, sqrt, square, reciprocal, boxcox, yeojohnson, polynomial")
                    yaml_lines.append(f"    scaling: 'robust'  # standard, minmax, robust, maxabs, quantile_uniform, quantile_normal")
                else:
                    yaml_lines.append(f"    transformation: null")
                    yaml_lines.append(f"    scaling: 'standard'  # standard, minmax, robust, maxabs, quantile_uniform, quantile_normal")
                yaml_lines.append(f"    polynomial_degree: null  # for polynomial transformation")
            else:
                yaml_lines.append(f"    transformation: null")
                yaml_lines.append(f"    scaling: null")
                yaml_lines.append(f"    polynomial_degree: null")
            
            # Feature engineering
            yaml_lines.append(f"    binning: null  # equal_width, equal_frequency, kmeans, quantile")
            yaml_lines.append(f"    n_bins: null  # number of bins")
            yaml_lines.append(f"    feature_interactions: null  # [col1, col2] to create interactions with")
            
            # Data validation
            yaml_lines.append(f"    value_constraints: null  # {{min_value: 0, max_value: 100, allowed_values: [a, b, c]}}")
            
            # Custom functions
            yaml_lines.append(f"    custom_function: null  # name of custom function")
            yaml_lines.append(f"    custom_parameters: null  # parameters for custom function")
            
            yaml_lines.append("")
        
        return "\n".join(yaml_lines)
    
    def run_complete_profiling(self, **load_kwargs) -> tuple:
        """Run the complete enhanced profiling pipeline"""
        print("🚀 Starting Enhanced Data Profiling Pipeline")
        print("=" * 50)
        
        # Step 1: Load data
        self.load_data(**load_kwargs)
        
        # Step 2: Generate YData profile  
        self.generate_ydata_profile()
        
        # Step 3: Extract raw data
        self.extract_raw_profile_data()
        
        # Step 4: Save machine-readable data
        if self.save_raw_profile_enabled:
            json_path = self.save_raw_profile()
        else:
            json_path = None
        
        # Step 5: Generate cleaning config template
        yaml_path = self.generate_cleaning_config_template()
        
        print("\n🎉 Enhanced profiling complete!")
        print(f"📊 Dataset shape: {self.df.shape}")
        print(f"📄 HTML report: {self.output_dir / 'data_profiling_report.html'}")
        if json_path:
            print(f"💾 Raw profile data: {json_path}")
        print(f"⚙️ Cleaning template: {yaml_path}")
        
        return self.enhanced_profile, json_path, yaml_path

# Example usage
if __name__ == "__main__":
    # Configuration
    data_path = "/mnt/c/Users/tony3/Desktop/tidytuesday/ds-autoadvisor/data/bank.csv"
    output_dir = "/mnt/c/Users/tony3/Desktop/tidytuesday/ds-autoadvisor/docs"
    
    # Create enhanced profiler
    profiler = EnhancedDataProfiler(
        data_path=data_path,
        output_dir=output_dir,
        generate_html=True,
        save_raw_profile=True
    )
    
    # Run complete profiling with custom CSV settings
    enhanced_profile, json_path, yaml_path = profiler.run_complete_profiling(
        delimiter=';',
        encoding='utf-8'
    )
    
    # Show summary
    print(f"\n📋 PROFILING SUMMARY")
    print(f"Columns analyzed: {len(enhanced_profile.columns)}")
    print(f"Missing data: {enhanced_profile.total_missing}")
    print(f"Duplicate rows: {enhanced_profile.total_duplicates}")
    print(f"Memory usage: {enhanced_profile.memory_usage_mb:.1f} MB")
    
    # Show column recommendations
    print(f"\n🔧 COLUMN RECOMMENDATIONS:")
    for col_name, col_profile in enhanced_profile.columns.items():
        if col_profile.recommended_actions:
            print(f"  {col_name}: {', '.join(col_profile.recommended_actions)}")
