"""
Quick test of trainer with bank data - single model
"""
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline_4.trainer import TrainerConfig, EnhancedModelTrainer

def load_bank_data():
    """Smart loading of bank data with different format handling"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Try different file paths and formats
    file_attempts = [
        # (file_path, read_params)
        (os.path.join(base_dir, "data", "bank_cleaned.csv"), 
         {"delimiter": ';', "quotechar": '"', "encoding": 'utf-8'}),
        (os.path.join(base_dir, "data", "bank.csv"), 
         {"delimiter": ';', "encoding": 'utf-8'}),
        (os.path.join(base_dir, "data", "bank.csv"), 
         {"delimiter": ',', "encoding": 'utf-8'}),
    ]
    
    for file_path, params in file_attempts:
        if os.path.exists(file_path):
            try:
                print(f"🔄 Trying to load: {file_path}")
                print(f"   Parameters: {params}")
                
                df = pd.read_csv(file_path, **params)
                
                # Validate the loading
                if df.shape[1] > 1 and df.shape[0] > 100:
                    print(f"✅ Successfully loaded: {df.shape}")
                    print(f"   Columns: {list(df.columns)}")
                    return df, file_path
                else:
                    print(f"⚠️ Loaded but suspicious shape: {df.shape}")
                    continue
                    
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")
                continue
    
    print("❌ Could not load any bank data file!")
    return None, None

def test_single_model():
    print("🧪 Testing Enhanced Trainer - Single Model")
    print("="*50)
    
    # Load bank data with smart loading
    df, file_path = load_bank_data()
    if df is None:
        return
    
    print(f"\n📊 Dataset loaded from: {file_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show sample data to verify correct parsing
    print(f"\n🔍 Sample data (first 3 rows):")
    print(df.head(3).to_string())
    
    # Find target
    target = 'y' if 'y' in df.columns else df.columns[-1]
    print(f"\n🎯 Target column: {target}")
    
    # Check target values
    target_values = df[target].value_counts()
    print(f"Target distribution: {target_values.to_dict()}")
    print(f"Unique target values: {list(target_values.index)}")
    
    # Validate target has reasonable distribution
    if len(target_values) > 1000:  # Too many unique values = parsing error
        print("❌ Too many unique target values - likely a parsing error!")
        print("First 10 target values:")
        print(list(df[target].head(10)))
        return
    
    min_class_size = target_values.min()
    print(f"Minimum class size: {min_class_size}")
    
    # Continue only if target makes sense
    if min_class_size >= 2:
        print("✅ Target distribution looks good!")
    else:
        print("⚠️ Some classes have < 2 samples")
        
        # Filter out small classes
        valid_classes = target_values[target_values >= 2].index
        if len(valid_classes) == 0:
            print("❌ No valid classes found!")
            return
            
        print(f"Valid classes: {list(valid_classes)}")
        df = df[df[target].isin(valid_classes)]
        print(f"Filtered dataset shape: {df.shape}")
    
    # Configure trainer for quick test
    config = TrainerConfig(
        max_models=1,           # Just one model
        enable_tuning=False,    # No hyperparameter tuning
        save_models=False,      # Don't save
        verbose=True,
        test_size=0.3,          # Larger test set for quick run
        scoring_strategy="fast" # Skip cross-validation
    )
    
    # Initialize trainer
    trainer = EnhancedModelTrainer(config)
    
    # Test data preparation
    print(f"\n🔍 Testing data preparation...")
    try:
        X, y, target_type = trainer.prepare_data(df, target)
        print(f"✅ Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"✅ Target type: {target_type}")
        print(f"✅ Feature types: {X.dtypes.value_counts().to_dict()}")
        
        # Check target distribution after preparation
        print(f"✅ Final target distribution: {y.value_counts().to_dict()}")
        
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return
    
    # Train a simple model
    print(f"\n🤖 Training RandomForest...")
    
    # Import train_test_split
    from sklearn.model_selection import train_test_split
    
    # Safe stratification - only use if all classes have at least 2 samples
    y_counts = y.value_counts()
    use_stratify = target_type == "classification" and y_counts.min() >= 2
    
    print(f"Using stratification: {use_stratify}")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.3, 
            random_state=42,
            stratify=y if use_stratify else None
        )
        
        print(f"✅ Data split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
        
    except Exception as e:
        print(f"❌ Data split failed: {e}")
        return
    
    # Get model config
    if target_type == "classification":
        model_name = "RandomForestClassifier"
        model_config = trainer.classification_models[model_name]
    else:
        model_name = "RandomForestRegressor"
        model_config = trainer.regression_models[model_name]
    
    print(f"Selected model: {model_name}")
    
    # Train single model
    try:
        result = trainer.train_single_model(
            model_name, model_config,
            X_train, X_test, y_train, y_test,
            None, None, target_type
        )
        
        if result:
            print(f"✅ Training successful!")
            print(f"   Model: {result.name}")
            print(f"   Training time: {result.training_time:.2f}s")
            print(f"   Scores: {result.scores}")
            if result.feature_importance:
                # Show top 5 important features
                top_features = sorted(result.feature_importance.items(), 
                                    key=lambda x: abs(x[1]), reverse=True)[:5]
                print(f"   Top features: {[f[0] for f in top_features]}")
        else:
            print("❌ Training failed!")
            
    except Exception as e:
        print(f"❌ Training failed with error: {e}")

if __name__ == "__main__":
    test_single_model()