# gpu_performance_manager.py
"""
GPU性能管理和回退机制模块

该模块提供GPU性能监控、自动回退和错误处理功能，
确保在GPU不可用或性能不佳时能自动切换到CPU模式。

主要功能：
- GPU性能监控
- 自动回退机制
- 错误处理和恢复
- 性能基准测试
"""

import time
import warnings
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
import threading

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    gpu_time: float = 0.0
    cpu_time: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_errors: int = 0
    cpu_errors: int = 0
    total_processed: int = 0


class GPUPerformanceManager:
    """GPU性能管理器"""
    
    def __init__(self, fallback_threshold: float = 2.0, error_threshold: int = 3):
        """
        初始化性能管理器
        
        Args:
            fallback_threshold: GPU性能回退阈值（GPU时间/CPU时间）
            error_threshold: 错误次数阈值，超过此值将回退到CPU
        """
        self.fallback_threshold = fallback_threshold
        self.error_threshold = error_threshold
        self.metrics = PerformanceMetrics()
        self.use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
        self.force_cpu = False
        self.lock = threading.Lock()
        
        # 性能历史记录
        self.performance_history = []
        self.max_history_size = 100
        
    def should_use_gpu(self) -> bool:
        """判断是否应该使用GPU"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available() or self.force_cpu:
            return False
        
        # 如果错误次数过多，回退到CPU
        if self.metrics.gpu_errors >= self.error_threshold:
            return False
        
        # 如果有足够的性能数据，比较GPU和CPU性能
        if self.metrics.total_processed > 10:
            if self.metrics.gpu_time > 0 and self.metrics.cpu_time > 0:
                gpu_avg_time = self.metrics.gpu_time / max(1, self.metrics.total_processed)
                cpu_avg_time = self.metrics.cpu_time / max(1, self.metrics.total_processed)
                
                if gpu_avg_time / cpu_avg_time > self.fallback_threshold:
                    warnings.warn(f"GPU性能不佳，回退到CPU模式 (GPU/CPU比率: {gpu_avg_time/cpu_avg_time:.2f})")
                    return False
        
        return True
    
    def execute_with_fallback(self, gpu_func: Callable, cpu_func: Callable, 
                            *args, **kwargs) -> Any:
        """
        执行函数，支持GPU到CPU的自动回退
        
        Args:
            gpu_func: GPU版本的函数
            cpu_func: CPU版本的函数
            *args, **kwargs: 函数参数
            
        Returns:
            函数执行结果
        """
        if self.should_use_gpu():
            try:
                start_time = time.time()
                
                # 监控GPU内存
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    initial_memory = torch.cuda.memory_allocated()
                
                result = gpu_func(*args, **kwargs)
                
                # 记录性能指标
                gpu_time = time.time() - start_time
                with self.lock:
                    self.metrics.gpu_time += gpu_time
                    self.metrics.total_processed += 1
                    
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() - initial_memory
                        self.metrics.gpu_memory_used += memory_used
                
                # 记录性能历史
                self._record_performance('gpu', gpu_time)
                
                return result
                
            except Exception as e:
                with self.lock:
                    self.metrics.gpu_errors += 1
                
                warnings.warn(f"GPU执行失败，回退到CPU: {e}")
                
                # 如果GPU错误过多，强制使用CPU
                if self.metrics.gpu_errors >= self.error_threshold:
                    self.force_cpu = True
                    warnings.warn("GPU错误次数过多，强制切换到CPU模式")
        
        # 执行CPU版本
        try:
            start_time = time.time()
            result = cpu_func(*args, **kwargs)
            
            cpu_time = time.time() - start_time
            with self.lock:
                self.metrics.cpu_time += cpu_time
                if not self.should_use_gpu():  # 只有在CPU模式下才增加处理计数
                    self.metrics.total_processed += 1
            
            # 记录性能历史
            self._record_performance('cpu', cpu_time)
            
            return result
            
        except Exception as e:
            with self.lock:
                self.metrics.cpu_errors += 1
            raise e
    
    def _record_performance(self, mode: str, execution_time: float):
        """记录性能历史"""
        with self.lock:
            self.performance_history.append({
                'mode': mode,
                'time': execution_time,
                'timestamp': time.time()
            })
            
            # 限制历史记录大小
            if len(self.performance_history) > self.max_history_size:
                self.performance_history.pop(0)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        with self.lock:
            total_gpu_time = self.metrics.gpu_time
            total_cpu_time = self.metrics.cpu_time
            total_processed = self.metrics.total_processed
            
            report = {
                'total_processed': total_processed,
                'gpu_errors': self.metrics.gpu_errors,
                'cpu_errors': self.metrics.cpu_errors,
                'gpu_memory_used_mb': self.metrics.gpu_memory_used / 1024**2,
                'current_mode': 'gpu' if self.should_use_gpu() else 'cpu',
                'force_cpu': self.force_cpu
            }
            
            if total_processed > 0:
                report.update({
                    'avg_gpu_time': total_gpu_time / total_processed if total_gpu_time > 0 else 0,
                    'avg_cpu_time': total_cpu_time / total_processed if total_cpu_time > 0 else 0,
                    'total_gpu_time': total_gpu_time,
                    'total_cpu_time': total_cpu_time
                })
                
                if total_gpu_time > 0 and total_cpu_time > 0:
                    report['performance_ratio'] = (total_gpu_time / total_processed) / (total_cpu_time / total_processed)
            
            return report
    
    def reset_metrics(self):
        """重置性能指标"""
        with self.lock:
            self.metrics = PerformanceMetrics()
            self.performance_history.clear()
            self.force_cpu = False
    
    def benchmark_performance(self, test_func: Callable, test_data: List[Any], 
                            iterations: int = 5) -> Dict[str, float]:
        """
        性能基准测试
        
        Args:
            test_func: 测试函数（应该接受gpu_mode参数）
            test_data: 测试数据
            iterations: 测试迭代次数
            
        Returns:
            性能基准结果
        """
        gpu_times = []
        cpu_times = []
        
        # GPU基准测试
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for _ in range(iterations):
                try:
                    start_time = time.time()
                    test_func(test_data, gpu_mode=True)
                    gpu_times.append(time.time() - start_time)
                except Exception as e:
                    warnings.warn(f"GPU基准测试失败: {e}")
                    break
        
        # CPU基准测试
        for _ in range(iterations):
            try:
                start_time = time.time()
                test_func(test_data, gpu_mode=False)
                cpu_times.append(time.time() - start_time)
            except Exception as e:
                warnings.warn(f"CPU基准测试失败: {e}")
                break
        
        result = {}
        if gpu_times:
            result['gpu_avg_time'] = sum(gpu_times) / len(gpu_times)
            result['gpu_min_time'] = min(gpu_times)
            result['gpu_max_time'] = max(gpu_times)
        
        if cpu_times:
            result['cpu_avg_time'] = sum(cpu_times) / len(cpu_times)
            result['cpu_min_time'] = min(cpu_times)
            result['cpu_max_time'] = max(cpu_times)
        
        if gpu_times and cpu_times:
            result['speedup_ratio'] = result['cpu_avg_time'] / result['gpu_avg_time']
        
        return result


# 全局性能管理器实例
_global_performance_manager = None
_performance_manager_lock = threading.Lock()


def get_global_performance_manager(**kwargs) -> GPUPerformanceManager:
    """获取全局性能管理器实例"""
    global _global_performance_manager
    
    if _global_performance_manager is None:
        with _performance_manager_lock:
            if _global_performance_manager is None:
                _global_performance_manager = GPUPerformanceManager(**kwargs)
    
    return _global_performance_manager


def cleanup_global_performance_manager():
    """清理全局性能管理器"""
    global _global_performance_manager
    
    if _global_performance_manager is not None:
        _global_performance_manager.reset_metrics()
        _global_performance_manager = None
