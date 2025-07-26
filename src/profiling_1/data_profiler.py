import pandas as pd
from ydata_profiling import ProfileReport
import missingno as msno
import sweetviz as sv
import pandera as pa
import os
import sys
from pathlib import Path

# =============================================================================
# USER CONFIG - EDIT THESE SETTINGS
# =============================================================================

# 1. Data file path (absolute path recommended)
DATA_PATH = "/mnt/c/Users/tony3/Desktop/tidytuesday/ds-autoadvisor/data/bank.csv"

# 2. Output directory for reports (absolute path recommended)  
OUTPUT_DIR = "/mnt/c/Users/tony3/Desktop/tidytuesday/ds-autoadvisor/docs"

# 3. Data loading settings
CSV_DELIMITER = ";"        # Change to "," if needed
CSV_ENCODING = "utf-8"     # Change if needed (e.g., "latin-1")
CSV_HEADER = 0 # 0 for first row as header, None for no header

# 4. Report settings
REPORT_TITLE = "Bank Data Analysis Report"
GENERATE_HTML_REPORT = True
GENERATE_MISSING_PLOT = True
GENERATE_SCHEMA = True

# =============================================================================
# MAIN PROFILING CODE - NO NEED TO EDIT BELOW
# =============================================================================

def main():
    print("🔍 Starting Data Profiling...")
    print("="*50)
    
    # Convert to Path objects for better handling
    data_path = Path(DATA_PATH)
    output_dir = Path(OUTPUT_DIR)
    
    # Validate input file
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print(f"Please check the DATA_PATH in the config section")
        sys.exit(1)
    
    print(f"📁 Data file: {data_path}")
    print(f"📂 Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\n📊 Loading dataset...")
    try:
        df = pd.read_csv(
            data_path, 
            delimiter=CSV_DELIMITER, 
            encoding=CSV_ENCODING,
            header=CSV_HEADER
        )
        print(f"✅ Data loaded successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        print(f"💡 Try changing CSV_DELIMITER or CSV_ENCODING in config")
        sys.exit(1)
    
    # Generate reports
    print(f"\n🔄 Generating reports...")
    
    # 1. YData Profiling Report
    if GENERATE_HTML_REPORT:
        try:
            print("  📈 Creating comprehensive HTML report...")
            profile = ProfileReport(
                df, 
                title=REPORT_TITLE, 
                explorative=True,
                minimal=False
            )
            html_path = output_dir / "data_profiling_report.html"
            profile.to_file(html_path)
            print(f"  ✅ HTML report saved: {html_path}")
        except Exception as e:
            print(f"  ❌ HTML report failed: {e}")
    
    # 2. Missing Data Visualization
    if GENERATE_MISSING_PLOT:
        try:
            print("  📊 Creating missing data visualization...")
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            msno.matrix(df)
            plt.title("Missing Data Pattern")
            
            missing_path = output_dir / "missing_data_matrix.png"
            plt.savefig(missing_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Missing data plot saved: {missing_path}")
        except Exception as e:
            print(f"  ❌ Missing data plot failed: {e}")
    
    # 3. Data Schema
    if GENERATE_SCHEMA:
        try:
            print("  📋 Inferring data schema...")
            schema = pa.infer_schema(df)
            schema_path = output_dir / "data_schema.yaml"
            
            with open(schema_path, 'w') as f:
                f.write(schema.to_yaml())
            print(f"  ✅ Schema saved: {schema_path}")
        except Exception as e:
            print(f"  ❌ Schema generation failed: {e}")
    
    # Summary statistics
    print(f"\n📈 DATASET SUMMARY")
    print(f"="*30)
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Data types summary
    print(f"\nDATA TYPES:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Show sample data
    print(f"\nSAMPLE DATA (first 3 rows):")
    print(df.head(3).to_string())
    
    print(f"\n🎉 Profiling complete!")
    print(f"📂 All reports saved to: {output_dir}")

if __name__ == "__main__":
    main()