# gpu_manager.py
"""
GPU管理模块

该模块提供多GPU并行处理功能，主要用于优化筛选阶段的性能瓶颈。
支持自动检测GPU数量、灵活配置GPU使用策略，并确保向后兼容性。

主要功能：
- 自动检测可用GPU数量
- 支持指定使用的GPU卡号
- 多GPU并行推理
- 数据同步和结果合并
- 内存管理优化
"""

import os
import gc
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings

# GPU相关导入
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch未安装，GPU管理器将在CPU模式下运行")


class GPUManager:
    """GPU管理器类"""
    
    def __init__(self, gpu_ids: Optional[List[int]] = None, max_gpus: int = 4):
        """
        初始化GPU管理器
        
        Args:
            gpu_ids: 指定使用的GPU ID列表，None表示自动检测
            max_gpus: 最大使用GPU数量
        """
        self.max_gpus = max_gpus
        self.available_gpus = self._detect_gpus()
        
        if gpu_ids is not None:
            # 验证指定的GPU ID是否可用
            self.gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id in self.available_gpus]
        else:
            # 自动选择可用GPU
            self.gpu_ids = self.available_gpus[:self.max_gpus]
        
        self.num_gpus = len(self.gpu_ids)
        self.use_gpu = self.num_gpus > 0
        
        print(f"GPU管理器初始化完成:")
        print(f"  可用GPU: {self.available_gpus}")
        print(f"  使用GPU: {self.gpu_ids}")
        print(f"  GPU数量: {self.num_gpus}")
        
    def _detect_gpus(self) -> List[int]:
        """检测可用的GPU"""
        if not TORCH_AVAILABLE:
            print("PyTorch未安装，将使用CPU模式")
            return []

        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                available_gpus = []

                for i in range(gpu_count):
                    try:
                        # 测试GPU是否可用
                        torch.cuda.set_device(i)
                        torch.cuda.empty_cache()

                        # 测试GPU内存
                        test_tensor = torch.zeros(100, 100, device=f'cuda:{i}')
                        del test_tensor
                        torch.cuda.empty_cache()

                        available_gpus.append(i)
                        print(f"GPU {i} 可用: {torch.cuda.get_device_name(i)}")
                    except Exception as e:
                        print(f"GPU {i} 不可用: {e}")

                return available_gpus
            else:
                print("CUDA不可用，将使用CPU模式")
                return []
        except Exception as e:
            print(f"GPU检测失败: {e}")
            return []
    
    def get_device(self, gpu_idx: int = 0) -> str:
        """获取设备字符串"""
        if self.use_gpu and gpu_idx < len(self.gpu_ids):
            return f"cuda:{self.gpu_ids[gpu_idx]}"
        return "cpu"
    
    def distribute_tasks(self, tasks: List[Any]) -> List[List[Any]]:
        """将任务分配到不同的GPU"""
        if not self.use_gpu or self.num_gpus == 1:
            return [tasks]
        
        # 平均分配任务
        tasks_per_gpu = len(tasks) // self.num_gpus
        remainder = len(tasks) % self.num_gpus
        
        distributed_tasks = []
        start_idx = 0
        
        for i in range(self.num_gpus):
            # 为前remainder个GPU多分配一个任务
            end_idx = start_idx + tasks_per_gpu + (1 if i < remainder else 0)
            distributed_tasks.append(tasks[start_idx:end_idx])
            start_idx = end_idx
            
        return distributed_tasks
    
    def parallel_process(self, process_func, tasks: List[Any], **kwargs) -> List[Any]:
        """
        并行处理任务
        
        Args:
            process_func: 处理函数
            tasks: 任务列表
            **kwargs: 传递给处理函数的额外参数
            
        Returns:
            处理结果列表
        """
        if not self.use_gpu or self.num_gpus == 1:
            # 单GPU或CPU模式
            device = self.get_device(0)
            return self._process_on_device(process_func, tasks, device, **kwargs)
        
        # 多GPU并行处理
        distributed_tasks = self.distribute_tasks(tasks)
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = []
            
            for gpu_idx, gpu_tasks in enumerate(distributed_tasks):
                if len(gpu_tasks) > 0:
                    device = self.get_device(gpu_idx)
                    future = executor.submit(
                        self._process_on_device, 
                        process_func, 
                        gpu_tasks, 
                        device, 
                        **kwargs
                    )
                    futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"GPU处理任务时发生错误: {e}")
        
        return all_results
    
    def _process_on_device(self, process_func, tasks: List[Any], device: str, **kwargs) -> List[Any]:
        """在指定设备上处理任务"""
        try:
            # 设置当前设备
            if device.startswith('cuda') and TORCH_AVAILABLE:
                gpu_id = int(device.split(':')[1])
                torch.cuda.set_device(gpu_id)

                # 检查GPU内存状态
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(gpu_id)
                    memory_cached = torch.cuda.memory_reserved(gpu_id)
                    print(f"GPU {gpu_id} 内存状态 - 已分配: {memory_allocated/1024**2:.1f}MB, 已缓存: {memory_cached/1024**2:.1f}MB")

            results = []
            for task in tasks:
                try:
                    result = process_func(task, device=device, **kwargs)
                    results.append(result)
                except Exception as e:
                    print(f"处理任务时发生错误: {e}")
                    results.append(None)
                finally:
                    # 清理GPU内存
                    if device.startswith('cuda') and TORCH_AVAILABLE:
                        torch.cuda.empty_cache()
                    gc.collect()

            # 最终内存清理
            if device.startswith('cuda') and TORCH_AVAILABLE:
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated(gpu_id) if torch.cuda.is_available() else 0
                print(f"GPU {gpu_id} 处理完成，剩余内存: {final_memory/1024**2:.1f}MB")

            return results

        except Exception as e:
            print(f"设备 {device} 处理失败: {e}")
            return [None] * len(tasks)
    
    def cleanup(self):
        """清理GPU内存"""
        if self.use_gpu and TORCH_AVAILABLE:
            for gpu_id in self.gpu_ids:
                try:
                    torch.cuda.set_device(gpu_id)

                    # 获取清理前的内存状态
                    if torch.cuda.is_available():
                        before_memory = torch.cuda.memory_allocated(gpu_id)
                        before_cached = torch.cuda.memory_reserved(gpu_id)

                    # 清理缓存
                    torch.cuda.empty_cache()

                    # 获取清理后的内存状态
                    if torch.cuda.is_available():
                        after_memory = torch.cuda.memory_allocated(gpu_id)
                        after_cached = torch.cuda.memory_reserved(gpu_id)
                        print(f"GPU {gpu_id} 内存清理: {before_memory/1024**2:.1f}MB -> {after_memory/1024**2:.1f}MB")

                except Exception as e:
                    print(f"清理GPU {gpu_id} 内存失败: {e}")
        gc.collect()

    def get_gpu_memory_info(self) -> Dict[int, Dict[str, float]]:
        """获取GPU内存信息"""
        memory_info = {}

        if self.use_gpu and TORCH_AVAILABLE:
            for gpu_id in self.gpu_ids:
                try:
                    torch.cuda.set_device(gpu_id)
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2  # MB
                        cached = torch.cuda.memory_reserved(gpu_id) / 1024**2  # MB
                        total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2  # MB

                        memory_info[gpu_id] = {
                            'allocated': allocated,
                            'cached': cached,
                            'total': total,
                            'free': total - cached
                        }
                except Exception as e:
                    print(f"获取GPU {gpu_id} 内存信息失败: {e}")

        return memory_info


def create_gpu_manager(gpu_ids: Optional[List[int]] = None, max_gpus: int = 4) -> GPUManager:
    """
    创建GPU管理器的便捷函数
    
    Args:
        gpu_ids: 指定使用的GPU ID列表
        max_gpus: 最大使用GPU数量
        
    Returns:
        GPUManager实例
    """
    return GPUManager(gpu_ids=gpu_ids, max_gpus=max_gpus)


# 全局GPU管理器实例（延迟初始化）
_global_gpu_manager = None
_gpu_manager_lock = threading.Lock()


def get_global_gpu_manager(**kwargs) -> GPUManager:
    """获取全局GPU管理器实例"""
    global _global_gpu_manager
    
    if _global_gpu_manager is None:
        with _gpu_manager_lock:
            if _global_gpu_manager is None:
                _global_gpu_manager = create_gpu_manager(**kwargs)
    
    return _global_gpu_manager


def cleanup_global_gpu_manager():
    """清理全局GPU管理器"""
    global _global_gpu_manager
    
    if _global_gpu_manager is not None:
        _global_gpu_manager.cleanup()
        _global_gpu_manager = None
