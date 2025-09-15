#!/usr/bin/env python3
# gpu_optimization_config.py
"""
GPU优化配置文件

根据不同的GPU型号和内存大小，提供优化的配置参数
"""

import warnings
from typing import Dict, Any, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch未安装，将使用默认CPU配置")


class GPUOptimizationConfig:
    """GPU优化配置类"""
    
    # 不同GPU型号的推荐配置
    GPU_CONFIGS = {
        'RTX 4090': {
            'batch_size': 32,
            'max_gpus': 4,
            'memory_fraction': 0.8,
            'enable_mixed_precision': True
        },
        'RTX 4080': {
            'batch_size': 24,
            'max_gpus': 2,
            'memory_fraction': 0.8,
            'enable_mixed_precision': True
        },
        'RTX 3090': {
            'batch_size': 28,
            'max_gpus': 2,
            'memory_fraction': 0.8,
            'enable_mixed_precision': True
        },
        'RTX 3080': {
            'batch_size': 20,
            'max_gpus': 2,
            'memory_fraction': 0.8,
            'enable_mixed_precision': True
        },
        'RTX 3070': {
            'batch_size': 16,
            'max_gpus': 1,
            'memory_fraction': 0.7,
            'enable_mixed_precision': True
        },
        'default': {
            'batch_size': 16,
            'max_gpus': 1,
            'memory_fraction': 0.7,
            'enable_mixed_precision': False
        }
    }
    
    @classmethod
    def get_optimal_config(cls, gpu_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取最优GPU配置
        
        Args:
            gpu_name: GPU名称，如果为None则自动检测
            
        Returns:
            优化配置字典
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return cls.GPU_CONFIGS['default']
        
        if gpu_name is None:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except:
                gpu_name = 'unknown'
        
        # 查找匹配的配置
        for key, config in cls.GPU_CONFIGS.items():
            if key.lower() in gpu_name.lower():
                return config.copy()
        
        return cls.GPU_CONFIGS['default'].copy()
    
    @classmethod
    def get_memory_info(cls) -> Dict[str, Any]:
        """获取GPU内存信息"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}
        
        info = {}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            total = props.total_memory / 1024**3  # GB
            
            info[f'gpu_{i}'] = {
                'name': props.name,
                'total_memory_gb': total,
                'allocated_memory_gb': allocated,
                'free_memory_gb': total - allocated,
                'compute_capability': f"{props.major}.{props.minor}"
            }
        
        return info
    
    @classmethod
    def recommend_batch_size(cls, image_size: int = 128, available_memory_gb: float = None) -> int:
        """
        根据图像大小和可用内存推荐批处理大小
        
        Args:
            image_size: 图像尺寸（假设为正方形）
            available_memory_gb: 可用GPU内存（GB）
            
        Returns:
            推荐的批处理大小
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 8  # CPU默认批处理大小
        
        if available_memory_gb is None:
            # 获取第一个GPU的可用内存
            props = torch.cuda.get_device_properties(0)
            total_memory_gb = props.total_memory / 1024**3
            allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            available_memory_gb = total_memory_gb - allocated_gb
        
        # 估算单个图像的内存使用（包括中间计算）
        # 假设：输入图像 + 中间张量 + 梯度等 ≈ 图像大小 * 10
        bytes_per_image = image_size * image_size * 4 * 10  # float32 * 估算倍数
        gb_per_image = bytes_per_image / 1024**3
        
        # 保留一些内存余量
        usable_memory = available_memory_gb * 0.8
        
        recommended_batch_size = max(1, int(usable_memory / gb_per_image))
        
        # 限制在合理范围内
        return min(recommended_batch_size, 64)
    
    @classmethod
    def print_optimization_report(cls):
        """打印GPU优化报告"""
        print("=== GPU优化配置报告 ===")
        
        if not TORCH_AVAILABLE:
            print("PyTorch未安装，无法进行GPU优化")
            return
        
        if not torch.cuda.is_available():
            print("CUDA不可用，将使用CPU模式")
            return
        
        print(f"检测到 {torch.cuda.device_count()} 个GPU:")
        
        memory_info = cls.get_memory_info()
        for gpu_id, info in memory_info.items():
            print(f"\n{gpu_id.upper()}:")
            print(f"  名称: {info['name']}")
            print(f"  总内存: {info['total_memory_gb']:.1f} GB")
            print(f"  已用内存: {info['allocated_memory_gb']:.1f} GB")
            print(f"  可用内存: {info['free_memory_gb']:.1f} GB")
            print(f"  计算能力: {info['compute_capability']}")
            
            # 获取推荐配置
            config = cls.get_optimal_config(info['name'])
            print(f"  推荐配置:")
            print(f"    批处理大小: {config['batch_size']}")
            print(f"    内存使用率: {config['memory_fraction']*100:.0f}%")
            print(f"    混合精度: {'启用' if config['enable_mixed_precision'] else '禁用'}")
            
            # 推荐批处理大小
            recommended_batch = cls.recommend_batch_size(
                available_memory_gb=info['free_memory_gb']
            )
            print(f"    动态推荐批处理大小: {recommended_batch}")
        
        print("\n=== 报告完成 ===")


def get_optimized_gpu_args() -> Dict[str, Any]:
    """
    获取优化的GPU参数，可直接用于FlowRenderer.py
    
    Returns:
        优化的GPU参数字典
    """
    config = GPUOptimizationConfig.get_optimal_config()
    
    # 动态调整批处理大小
    if TORCH_AVAILABLE and torch.cuda.is_available():
        memory_info = GPUOptimizationConfig.get_memory_info()
        if memory_info:
            first_gpu = list(memory_info.values())[0]
            recommended_batch = GPUOptimizationConfig.recommend_batch_size(
                available_memory_gb=first_gpu['free_memory_gb']
            )
            config['batch_size'] = max(config['batch_size'], recommended_batch)
    
    return {
        'gpu_batch_size': config['batch_size'],
        'max_gpus': config['max_gpus'],
        'enable_gpu_acceleration': TORCH_AVAILABLE and torch.cuda.is_available(),
        'gpu_memory_fraction': config['memory_fraction']
    }


if __name__ == '__main__':
    # 打印优化报告
    GPUOptimizationConfig.print_optimization_report()
    
    # 获取优化参数
    print("\n=== 推荐的启动参数 ===")
    args = get_optimized_gpu_args()
    for key, value in args.items():
        print(f"--{key.replace('_', '-')}: {value}")
