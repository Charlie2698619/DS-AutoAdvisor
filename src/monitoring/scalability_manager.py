"""
Industrial-Grade Scalability and Performance Framework
=====================================================

Advanced scalability features, performance optimization, and distributed processing
capabilities for production ML pipelines.
"""

import os
import time
import psutil
import threading
import multiprocessing as mp
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
import pandas as pd
import numpy as np
from functools import wraps
import joblib
import gc

class ScalingStrategy(Enum):
    """Scaling strategies for different workloads"""
    SINGLE_THREAD = "single_thread"
    MULTI_THREAD = "multi_thread"
    MULTI_PROCESS = "multi_process"
    DISTRIBUTED = "distributed"
    AUTO = "auto"

class ResourceType(Enum):
    """System resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    execution_time: float = 0.0
    memory_peak: float = 0.0
    memory_average: float = 0.0
    cpu_usage: float = 0.0
    disk_io: float = 0.0
    throughput: float = 0.0  # records/second
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ResourceLimits:
    """Resource limits configuration"""
    max_memory_gb: float = 8.0
    max_cpu_percent: float = 80.0
    max_disk_usage_gb: float = 100.0
    max_execution_time_minutes: float = 120.0
    max_concurrent_jobs: int = 4

class PerformanceMonitor:
    """
    Real-time performance monitoring and resource management
    """
    
    def __init__(self, resource_limits: ResourceLimits):
        self.resource_limits = resource_limits
        self.logger = logging.getLogger(__name__)
        
        self.metrics_history: List[PerformanceMetrics] = []
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Performance thresholds
        self.warning_thresholds = {
            'memory': resource_limits.max_memory_gb * 0.8,
            'cpu': resource_limits.max_cpu_percent * 0.8,
            'disk': resource_limits.max_disk_usage_gb * 0.8
        }
    
    def monitor_performance(self, func: Callable) -> Callable:
        """
        Decorator to monitor function performance
        
        Args:
            func: Function to monitor
            
        Returns:
            Wrapped function with performance monitoring
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            job_id = f"{func.__name__}_{int(time.time())}"
            
            # Start monitoring
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            self.active_jobs[job_id] = {
                'function': func.__name__,
                'start_time': start_time,
                'start_memory': start_memory
            }
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate metrics
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                metrics = PerformanceMetrics(
                    execution_time=end_time - start_time,
                    memory_peak=max(start_memory, end_memory),
                    memory_average=(start_memory + end_memory) / 2,
                    cpu_usage=psutil.cpu_percent(),
                    disk_io=self._get_disk_io(),
                    error_rate=0.0
                )
                
                self.metrics_history.append(metrics)
                
                # Check resource usage
                self._check_resource_usage(metrics)
                
                self.logger.info(f"Performance metrics for {func.__name__}: "
                               f"Time={metrics.execution_time:.2f}s, "
                               f"Memory={metrics.memory_peak:.2f}GB")
                
                return result
                
            except Exception as e:
                # Record error metrics
                metrics = PerformanceMetrics(
                    execution_time=time.time() - start_time,
                    memory_peak=self._get_memory_usage(),
                    error_rate=1.0
                )
                self.metrics_history.append(metrics)
                
                self.logger.error(f"Performance monitoring error in {func.__name__}: {e}")
                raise
                
            finally:
                # Cleanup
                self.active_jobs.pop(job_id, None)
        
        return wrapper
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        return psutil.virtual_memory().used / (1024**3)
    
    def _get_disk_io(self) -> float:
        """Get current disk I/O rate"""
        try:
            disk_io = psutil.disk_io_counters()
            return (disk_io.read_bytes + disk_io.write_bytes) / (1024**2)  # MB
        except:
            return 0.0
    
    def _check_resource_usage(self, metrics: PerformanceMetrics):
        """Check if resource usage exceeds thresholds"""
        
        # Memory check
        if metrics.memory_peak > self.warning_thresholds['memory']:
            self.logger.warning(f"High memory usage: {metrics.memory_peak:.2f}GB "
                              f"(threshold: {self.warning_thresholds['memory']:.2f}GB)")
        
        # CPU check
        if metrics.cpu_usage > self.warning_thresholds['cpu']:
            self.logger.warning(f"High CPU usage: {metrics.cpu_usage:.1f}% "
                              f"(threshold: {self.warning_thresholds['cpu']:.1f}%)")
        
        # Execution time check
        max_time_seconds = self.resource_limits.max_execution_time_minutes * 60
        if metrics.execution_time > max_time_seconds:
            self.logger.warning(f"Long execution time: {metrics.execution_time:.2f}s "
                              f"(threshold: {max_time_seconds}s)")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_status = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3),
            'active_jobs': len(self.active_jobs),
            'timestamp': datetime.now().isoformat()
        }
        
        # Determine overall health
        health_issues = []
        if health_status['cpu_percent'] > self.resource_limits.max_cpu_percent:
            health_issues.append('high_cpu')
        if health_status['memory_percent'] > 90:
            health_issues.append('high_memory')
        if health_status['disk_percent'] > 90:
            health_issues.append('high_disk')
        
        health_status['status'] = 'unhealthy' if health_issues else 'healthy'
        health_status['issues'] = health_issues
        
        return health_status

class ParallelProcessor:
    """
    Advanced parallel processing with auto-scaling capabilities
    """
    
    def __init__(self, 
                 resource_limits: ResourceLimits,
                 performance_monitor: PerformanceMonitor):
        
        self.resource_limits = resource_limits
        self.performance_monitor = performance_monitor
        self.logger = logging.getLogger(__name__)
        
        # Dynamic worker management
        self.optimal_workers = self._calculate_optimal_workers()
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on system resources"""
        
        cpu_cores = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Conservative estimation
        cpu_based_workers = max(1, int(cpu_cores * 0.8))
        memory_based_workers = max(1, int(memory_gb / 2))  # 2GB per worker
        
        # Use resource limits
        limit_based_workers = min(
            self.resource_limits.max_concurrent_jobs,
            cpu_based_workers,
            memory_based_workers
        )
        
        self.logger.info(f"Calculated optimal workers: {limit_based_workers} "
                        f"(CPU: {cpu_cores}, Memory: {memory_gb:.1f}GB)")
        
        return limit_based_workers
    
    def process_dataframe_parallel(self, 
                                 df: pd.DataFrame,
                                 processing_func: Callable,
                                 strategy: ScalingStrategy = ScalingStrategy.AUTO,
                                 chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Process DataFrame in parallel chunks
        
        Args:
            df: Input DataFrame
            processing_func: Function to apply to each chunk
            strategy: Scaling strategy to use
            chunk_size: Size of chunks (auto-calculated if None)
            
        Returns:
            Processed DataFrame
        """
        
        if strategy == ScalingStrategy.AUTO:
            strategy = self._determine_optimal_strategy(df)
        
        if strategy == ScalingStrategy.SINGLE_THREAD:
            return processing_func(df)
        
        # Calculate chunk size if not provided
        if chunk_size is None:
            chunk_size = max(1000, len(df) // (self.optimal_workers * 2))
        
        # Split DataFrame into chunks
        chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        self.logger.info(f"Processing {len(chunks)} chunks with {self.optimal_workers} workers "
                        f"using {strategy.value} strategy")
        
        start_time = time.time()
        
        if strategy == ScalingStrategy.MULTI_THREAD:
            results = self._process_with_threads(chunks, processing_func)
        elif strategy == ScalingStrategy.MULTI_PROCESS:
            results = self._process_with_processes(chunks, processing_func)
        else:
            # Fallback to single thread
            results = [processing_func(chunk) for chunk in chunks]
        
        # Combine results
        combined_result = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        
        execution_time = time.time() - start_time
        throughput = len(df) / execution_time if execution_time > 0 else 0
        
        self.logger.info(f"Parallel processing completed in {execution_time:.2f}s "
                        f"({throughput:.0f} records/sec)")
        
        return combined_result
    
    def _determine_optimal_strategy(self, df: pd.DataFrame) -> ScalingStrategy:
        """Determine optimal scaling strategy based on data characteristics"""
        
        data_size_mb = df.memory_usage(deep=True).sum() / (1024**2)
        num_rows = len(df)
        
        # Decision logic
        if data_size_mb < 100 or num_rows < 10000:
            return ScalingStrategy.SINGLE_THREAD
        elif data_size_mb < 1000:
            return ScalingStrategy.MULTI_THREAD
        else:
            return ScalingStrategy.MULTI_PROCESS
    
    def _process_with_threads(self, chunks: List[pd.DataFrame], 
                            processing_func: Callable) -> List[pd.DataFrame]:
        """Process chunks using thread pool"""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.optimal_workers) as executor:
            future_to_chunk = {executor.submit(processing_func, chunk): chunk 
                             for chunk in chunks}
            
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Thread processing error: {e}")
                    # Return empty DataFrame for failed chunks
                    results.append(pd.DataFrame())
        
        return results
    
    def _process_with_processes(self, chunks: List[pd.DataFrame], 
                              processing_func: Callable) -> List[pd.DataFrame]:
        """Process chunks using process pool"""
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.optimal_workers) as executor:
            future_to_chunk = {executor.submit(processing_func, chunk): chunk 
                             for chunk in chunks}
            
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Process processing error: {e}")
                    results.append(pd.DataFrame())
        
        return results

class MemoryManager:
    """
    Advanced memory management and optimization
    """
    
    def __init__(self, resource_limits: ResourceLimits):
        self.resource_limits = resource_limits
        self.logger = logging.getLogger(__name__)
        
        # Memory optimization settings
        self.gc_threshold = resource_limits.max_memory_gb * 0.7
        self.cleanup_interval = 300  # 5 minutes
        
        # Start memory monitoring thread
        self._start_memory_monitor()
    
    def _start_memory_monitor(self):
        """Start background memory monitoring"""
        def monitor():
            while True:
                try:
                    current_memory = psutil.virtual_memory().used / (1024**3)
                    if current_memory > self.gc_threshold:
                        self.force_garbage_collection()
                    time.sleep(self.cleanup_interval)
                except Exception as e:
                    self.logger.error(f"Memory monitor error: {e}")
                    time.sleep(60)  # Wait before retrying
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        self.logger.info("Memory monitoring started")
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage
        
        Args:
            df: Input DataFrame
            
        Returns:
            Memory-optimized DataFrame
        """
        start_memory = df.memory_usage(deep=True).sum() / (1024**2)
        
        optimized_df = df.copy()
        
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            
            if col_min >= -128 and col_max <= 127:
                optimized_df[col] = optimized_df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                optimized_df[col] = optimized_df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                optimized_df[col] = optimized_df[col].astype('int32')
        
        # Optimize float columns
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = optimized_df[col].astype('float32')
        
        # Optimize object columns (strings)
        for col in optimized_df.select_dtypes(include=['object']).columns:
            num_unique_values = len(optimized_df[col].unique())
            num_total_values = len(optimized_df[col])
            
            if num_unique_values / num_total_values < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
        
        end_memory = optimized_df.memory_usage(deep=True).sum() / (1024**2)
        memory_reduction = (start_memory - end_memory) / start_memory * 100
        
        self.logger.info(f"Memory optimization: {start_memory:.1f}MB -> {end_memory:.1f}MB "
                        f"({memory_reduction:.1f}% reduction)")
        
        return optimized_df
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory"""
        before_memory = psutil.virtual_memory().used / (1024**3)
        
        # Force garbage collection
        gc.collect()
        
        after_memory = psutil.virtual_memory().used / (1024**3)
        freed_memory = before_memory - after_memory
        
        if freed_memory > 0.1:  # Only log if significant memory was freed
            self.logger.info(f"Garbage collection freed {freed_memory:.2f}GB of memory")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get detailed memory usage report"""
        
        memory = psutil.virtual_memory()
        
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'used_percent': memory.percent,
            'gc_threshold_gb': self.gc_threshold,
            'timestamp': datetime.now().isoformat()
        }

class CacheManager:
    """
    Intelligent caching system for expensive operations
    """
    
    def __init__(self, cache_dir: str, max_cache_size_gb: float = 5.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size = max_cache_size_gb * (1024**3)  # Convert to bytes
        self.logger = logging.getLogger(__name__)
        
        # Cache metadata
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = self._load_cache_metadata()
    
    def _load_cache_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
        
        return {'entries': {}, 'total_size': 0}
    
    def _save_cache_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def cache_operation(self, cache_key: str, ttl_hours: float = 24):
        """
        Decorator to cache expensive operations
        
        Args:
            cache_key: Unique key for the cached result
            ttl_hours: Time to live in hours
            
        Returns:
            Cached decorator
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key with function arguments
                arg_key = hashlib.md5(str((args, kwargs)).encode()).hexdigest()
                full_cache_key = f"{cache_key}_{arg_key}"
                
                # Check if cached result exists and is valid
                cached_result = self._get_cached_result(full_cache_key, ttl_hours)
                if cached_result is not None:
                    self.logger.info(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Execute function and cache result
                self.logger.info(f"Cache miss for {func.__name__}, executing...")
                result = func(*args, **kwargs)
                self._cache_result(full_cache_key, result)
                
                return result
            
            return wrapper
        return decorator
    
    def _get_cached_result(self, cache_key: str, ttl_hours: float) -> Any:
        """Get cached result if valid"""
        
        cache_entry = self.cache_metadata['entries'].get(cache_key)
        if not cache_entry:
            return None
        
        # Check TTL
        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        if datetime.now() - cache_time > timedelta(hours=ttl_hours):
            self._remove_cache_entry(cache_key)
            return None
        
        # Load cached data
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if not cache_file.exists():
            self._remove_cache_entry(cache_key)
            return None
        
        try:
            return joblib.load(cache_file)
        except Exception as e:
            self.logger.warning(f"Failed to load cached result: {e}")
            self._remove_cache_entry(cache_key)
            return None
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache operation result"""
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            # Save result
            joblib.dump(result, cache_file)
            
            # Update metadata
            file_size = cache_file.stat().st_size
            self.cache_metadata['entries'][cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'size': file_size,
                'file': str(cache_file)
            }
            self.cache_metadata['total_size'] += file_size
            
            # Check cache size and cleanup if needed
            self._cleanup_cache_if_needed()
            
            self._save_cache_metadata()
            
            self.logger.info(f"Cached result for key {cache_key} ({file_size / (1024**2):.1f}MB)")
            
        except Exception as e:
            self.logger.error(f"Failed to cache result: {e}")
    
    def _cleanup_cache_if_needed(self):
        """Clean up old cache entries if size limit exceeded"""
        
        if self.cache_metadata['total_size'] <= self.max_cache_size:
            return
        
        # Sort entries by timestamp (oldest first)
        entries = [(k, v) for k, v in self.cache_metadata['entries'].items()]
        entries.sort(key=lambda x: x[1]['timestamp'])
        
        # Remove oldest entries until under size limit
        target_size = self.max_cache_size * 0.8  # Leave some buffer
        
        for cache_key, entry in entries:
            if self.cache_metadata['total_size'] <= target_size:
                break
            
            self._remove_cache_entry(cache_key)
            self.logger.info(f"Removed old cache entry: {cache_key}")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove cache entry"""
        
        entry = self.cache_metadata['entries'].get(cache_key)
        if not entry:
            return
        
        # Remove file
        cache_file = Path(entry['file'])
        if cache_file.exists():
            cache_file.unlink()
        
        # Update metadata
        self.cache_metadata['total_size'] -= entry['size']
        del self.cache_metadata['entries'][cache_key]
    
    def clear_cache(self):
        """Clear all cached data"""
        
        for cache_key in list(self.cache_metadata['entries'].keys()):
            self._remove_cache_entry(cache_key)
        
        self._save_cache_metadata()
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        return {
            'total_entries': len(self.cache_metadata['entries']),
            'total_size_gb': self.cache_metadata['total_size'] / (1024**3),
            'max_size_gb': self.max_cache_size / (1024**3),
            'utilization_percent': (self.cache_metadata['total_size'] / self.max_cache_size) * 100,
            'cache_dir': str(self.cache_dir)
        }

class ScalabilityManager:
    """
    Main scalability manager coordinating all performance components
    """
    
    def __init__(self, 
                 resource_limits: ResourceLimits,
                 cache_dir: str = "cache",
                 enable_monitoring: bool = True):
        
        self.resource_limits = resource_limits
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor(resource_limits) if enable_monitoring else None
        self.parallel_processor = ParallelProcessor(resource_limits, self.performance_monitor)
        self.memory_manager = MemoryManager(resource_limits)
        self.cache_manager = CacheManager(cache_dir)
        
        self.logger.info("Scalability manager initialized")
    
    def optimize_pipeline_stage(self, 
                              stage_name: str,
                              data: pd.DataFrame,
                              processing_func: Callable,
                              enable_caching: bool = True,
                              cache_ttl_hours: float = 24) -> pd.DataFrame:
        """
        Optimize a pipeline stage with all available optimizations
        
        Args:
            stage_name: Name of the pipeline stage
            data: Input data
            processing_func: Processing function
            enable_caching: Whether to enable caching
            cache_ttl_hours: Cache TTL in hours
            
        Returns:
            Optimized processing result
        """
        
        self.logger.info(f"Optimizing pipeline stage: {stage_name}")
        
        # Memory optimization
        optimized_data = self.memory_manager.optimize_dataframe_memory(data)
        
        # Apply caching if enabled
        if enable_caching:
            processing_func = self.cache_manager.cache_operation(
                cache_key=f"stage_{stage_name}",
                ttl_hours=cache_ttl_hours
            )(processing_func)
        
        # Apply performance monitoring
        if self.performance_monitor:
            processing_func = self.performance_monitor.monitor_performance(processing_func)
        
        # Execute with parallel processing
        result = self.parallel_processor.process_dataframe_parallel(
            df=optimized_data,
            processing_func=processing_func,
            strategy=ScalingStrategy.AUTO
        )
        
        # Final memory optimization
        result = self.memory_manager.optimize_dataframe_memory(result)
        
        return result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_health': self.performance_monitor.get_system_health() if self.performance_monitor else {},
            'memory_report': self.memory_manager.get_memory_report(),
            'cache_stats': self.cache_manager.get_cache_stats(),
            'resource_limits': {
                'max_memory_gb': self.resource_limits.max_memory_gb,
                'max_cpu_percent': self.resource_limits.max_cpu_percent,
                'max_concurrent_jobs': self.resource_limits.max_concurrent_jobs
            }
        }
        
        # Add performance metrics history if available
        if self.performance_monitor and self.performance_monitor.metrics_history:
            recent_metrics = self.performance_monitor.metrics_history[-10:]  # Last 10 metrics
            report['recent_performance'] = [
                {
                    'execution_time': m.execution_time,
                    'memory_peak': m.memory_peak,
                    'cpu_usage': m.cpu_usage,
                    'throughput': m.throughput
                }
                for m in recent_metrics
            ]
        
        return report

# Example usage
if __name__ == "__main__":
    # Initialize resource limits
    resource_limits = ResourceLimits(
        max_memory_gb=8.0,
        max_cpu_percent=80.0,
        max_concurrent_jobs=4
    )
    
    # Initialize scalability manager
    scalability_manager = ScalabilityManager(
        resource_limits=resource_limits,
        cache_dir="cache",
        enable_monitoring=True
    )
    
    # Example processing function
    def sample_processing(df: pd.DataFrame) -> pd.DataFrame:
        # Simulate expensive processing
        time.sleep(0.1)
        return df.copy()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'id': range(10000),
        'value': np.random.randn(10000),
        'category': np.random.choice(['A', 'B', 'C'], 10000)
    })
    
    # Optimize pipeline stage
    result = scalability_manager.optimize_pipeline_stage(
        stage_name="sample_processing",
        data=sample_data,
        processing_func=sample_processing,
        enable_caching=True
    )
    
    # Get performance report
    performance_report = scalability_manager.get_performance_report()
    
    print(f"Processed {len(result)} records")
    print(f"System health: {performance_report['system_health']['status']}")
    print(f"Cache utilization: {performance_report['cache_stats']['utilization_percent']:.1f}%")
