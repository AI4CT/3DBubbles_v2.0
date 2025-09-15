#!/usr/bin/env python3
# gpu_monitoring_patch.py
"""
GPU监控补丁 - 用于增强GPU使用情况的监控和诊断

使用方法：
1. 在需要监控的代码中导入此模块
2. 使用装饰器或上下文管理器监控GPU使用情况
"""

import time
import functools
from typing import Optional, Dict, Any
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GPUMonitor:
    """GPU使用监控器"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.enabled = TORCH_AVAILABLE and torch.cuda.is_available()
    
    def get_memory_info(self) -> Dict[str, float]:
        """获取GPU内存信息"""
        if not self.enabled:
            return {}
        
        try:
            allocated = torch.cuda.memory_allocated(self.device_id) / 1024**2
            cached = torch.cuda.memory_reserved(self.device_id) / 1024**2
            total = torch.cuda.get_device_properties(self.device_id).total_memory / 1024**2
            
            return {
                'allocated_mb': allocated,
                'cached_mb': cached,
                'total_mb': total,
                'free_mb': total - allocated,
                'utilization_percent': (allocated / total) * 100
            }
        except Exception as e:
            warnings.warn(f"获取GPU内存信息失败: {e}")
            return {}
    
    def get_utilization(self) -> float:
        """获取GPU利用率（基于内存使用）"""
        info = self.get_memory_info()
        return info.get('utilization_percent', 0.0)
    
    def print_status(self, label: str = "GPU状态"):
        """打印GPU状态"""
        if not self.enabled:
            print(f"{label}: GPU不可用")
            return
        
        info = self.get_memory_info()
        if info:
            print(f"{label}:")
            print(f"  GPU {self.device_id}: {torch.cuda.get_device_name(self.device_id)}")
            print(f"  已分配: {info['allocated_mb']:.1f}MB")
            print(f"  已缓存: {info['cached_mb']:.1f}MB")
            print(f"  总内存: {info['total_mb']:.1f}MB")
            print(f"  利用率: {info['utilization_percent']:.1f}%")


class GPUUsageTracker:
    """GPU使用跟踪器（上下文管理器）"""
    
    def __init__(self, operation_name: str, device_id: int = 0, verbose: bool = True):
        self.operation_name = operation_name
        self.device_id = device_id
        self.verbose = verbose
        self.monitor = GPUMonitor(device_id)
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        if self.monitor.enabled:
            self.start_time = time.time()
            self.start_memory = self.monitor.get_memory_info()
            
            if self.verbose:
                print(f"\n=== 开始 {self.operation_name} ===")
                self.monitor.print_status("操作前GPU状态")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.monitor.enabled and self.start_time is not None:
            end_time = time.time()
            end_memory = self.monitor.get_memory_info()
            duration = end_time - self.start_time
            
            if self.verbose:
                self.monitor.print_status("操作后GPU状态")
                print(f"操作耗时: {duration:.4f}s")
                
                if self.start_memory and end_memory:
                    memory_change = end_memory['allocated_mb'] - self.start_memory['allocated_mb']
                    print(f"内存变化: {memory_change:+.1f}MB")
                
                print(f"=== 完成 {self.operation_name} ===\n")


def gpu_monitor(operation_name: Optional[str] = None, device_id: int = 0, verbose: bool = True):
    """GPU监控装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__name__}"
            
            with GPUUsageTracker(name, device_id, verbose):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def check_gpu_activity():
    """检查GPU活动状态"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        print("GPU不可用")
        return
    
    print("=== GPU活动检查 ===")
    
    for i in range(torch.cuda.device_count()):
        monitor = GPUMonitor(i)
        info = monitor.get_memory_info()
        
        if info['allocated_mb'] > 0:
            print(f"GPU {i}: 活跃 (已分配 {info['allocated_mb']:.1f}MB)")
        else:
            print(f"GPU {i}: 空闲")
    
    print("=== 检查完成 ===")


def stress_test_gpu(device_id: int = 0, duration: float = 5.0):
    """GPU压力测试"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        print("GPU不可用，无法进行压力测试")
        return
    
    print(f"=== GPU {device_id} 压力测试 ({duration}s) ===")
    
    monitor = GPUMonitor(device_id)
    monitor.print_status("测试前")
    
    # 创建大量GPU张量进行计算
    device = torch.device(f'cuda:{device_id}')
    
    start_time = time.time()
    tensors = []
    
    try:
        while time.time() - start_time < duration:
            # 创建随机张量并进行计算
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            c = torch.matmul(a, b)
            tensors.append(c)
            
            # 每秒检查一次状态
            if len(tensors) % 10 == 0:
                current_time = time.time() - start_time
                info = monitor.get_memory_info()
                print(f"  {current_time:.1f}s: 已分配 {info['allocated_mb']:.1f}MB")
    
    finally:
        # 清理张量
        del tensors
        torch.cuda.empty_cache()
        
        monitor.print_status("测试后")
        print("=== 压力测试完成 ===")


if __name__ == '__main__':
    # 示例用法
    check_gpu_activity()
    
    # 压力测试
    stress_test_gpu(duration=3.0)
    
    # 装饰器示例
    @gpu_monitor("示例GPU操作")
    def example_gpu_operation():
        if TORCH_AVAILABLE and torch.cuda.is_available():
            x = torch.randn(100, 100, device='cuda:0')
            y = torch.matmul(x, x.T)
            return y.cpu().numpy()
        return None
    
    result = example_gpu_operation()
    print(f"操作结果形状: {result.shape if result is not None else 'None'}")
