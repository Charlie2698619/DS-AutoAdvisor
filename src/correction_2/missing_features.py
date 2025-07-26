# Additional ML preprocessing features to add

MISSING_FEATURES = {
    "feature_selection": {
        "mutual_info": "For categorical target variables",
        "chi2": "For categorical features and targets", 
        "f_regression": "For continuous targets",
        "recursive_feature_elimination": "Model-based selection"
    },
    
    "advanced_encoding": {
        "target_encoding": "For high-cardinality categoricals",
        "binary_encoding": "Memory-efficient alternative to one-hot",
        "hash_encoding": "For extremely high cardinality",
        "leave_one_out": "Cross-validated target encoding"
    },
    
    "time_series": {
        "lag_features": "Previous values as features",
        "rolling_statistics": "Moving averages, std, etc.",
        "seasonal_decomposition": "Trend, seasonal, residual",
        "date_cyclical": "Sin/cos encoding for cyclical features"
    },
    
    "text_preprocessing": {
        "tfidf_vectorization": "Convert text to numerical",
        "word2vec_embedding": "Dense text representations",
        "text_length_features": "Character/word counts",
        "sentiment_analysis": "Text sentiment scores"
    },
    
    "advanced_outlier_detection": {
        "local_outlier_factor": "Density-based detection",
        "one_class_svm": "Support vector approach",
        "elliptic_envelope": "Robust covariance estimation",
        "dbscan_outliers": "Clustering-based detection"
    },
    
    "imbalanced_data": {
        "smote": "Synthetic minority oversampling",
        "adasyn": "Adaptive synthetic sampling", 
        "borderline_smote": "Focus on borderline cases",
        "undersampling": "Random/Tomek/EditedNN"
    },
    
    "pipeline_integration": {
        "sklearn_pipeline": "Integrate with sklearn Pipeline",
        "feature_union": "Parallel feature processing",
        "column_transformer": "Different transforms per column type",
        "custom_transformers": "Reusable transform classes"
    }
}
