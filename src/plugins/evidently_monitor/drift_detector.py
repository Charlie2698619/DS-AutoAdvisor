#!/usr/bin/env python3
"""
🎯 Drift Detector
================

Focused drift detection utilities for DS-AutoAdvisor.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

class DriftDetector:
    """Simple drift detection utilities"""
    
    def __init__(self):
        """Initialize drift detector"""
        self.logger = logging.getLogger('drift_detector')
    
    def detect_statistical_drift(self, reference: pd.DataFrame, 
                                  current: pd.DataFrame) -> Dict[str, Any]:
        """Simple statistical drift detection"""
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False,
            'column_drift': {}
        }
        
        for col in reference.columns:
            if col in current.columns:
                if reference[col].dtype in ['int64', 'float64']:
                    # Numerical drift detection using mean and std comparison
                    ref_mean, ref_std = reference[col].mean(), reference[col].std()
                    cur_mean, cur_std = current[col].mean(), current[col].std()
                    
                    mean_drift = abs(ref_mean - cur_mean) / (ref_std + 1e-8)
                    std_drift = abs(ref_std - cur_std) / (ref_std + 1e-8)
                    
                    drift_results['column_drift'][col] = {
                        'type': 'numerical',
                        'mean_drift': mean_drift,
                        'std_drift': std_drift,
                        'drift_detected': mean_drift > 0.5 or std_drift > 0.5
                    }
                    
                    if drift_results['column_drift'][col]['drift_detected']:
                        drift_results['drift_detected'] = True
                
                else:
                    # Categorical drift detection using distribution comparison
                    ref_dist = reference[col].value_counts(normalize=True)
                    cur_dist = current[col].value_counts(normalize=True)
                    
                    # Simple overlap measure
                    overlap = 0
                    for val in ref_dist.index:
                        if val in cur_dist.index:
                            overlap += min(ref_dist[val], cur_dist[val])
                    
                    drift_score = 1 - overlap
                    
                    drift_results['column_drift'][col] = {
                        'type': 'categorical',
                        'distribution_drift': drift_score,
                        'drift_detected': drift_score > 0.3
                    }
                    
                    if drift_results['column_drift'][col]['drift_detected']:
                        drift_results['drift_detected'] = True
        
        return drift_results
