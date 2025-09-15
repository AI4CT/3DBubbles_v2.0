# bubble_screening.py
"""
气泡筛选模块

该模块提供优化的气泡筛选功能，支持多GPU并行加速，
主要用于提升筛选阶段的性能，解决性能瓶颈问题。

主要功能：
- 多GPU并行筛选
- 内存优化管理
- 批量处理优化
- 结果合并和同步
"""

import os
import gc
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .bubble_analysis import (
    analyze_bubble_image,
    analyze_bubble_image_from_array,
    analyze_bubble_image_gpu,
    analyze_bubble_image_from_array_gpu,
    analyze_bubble_batch_gpu,
    analyze_bubble_image_smart,
    analyze_bubble_image_from_array_smart,
    analyze_bubble_batch_smart,
    get_analysis_performance_report,
    TORCH_AVAILABLE
)
from .gpu_manager import get_global_gpu_manager, GPUManager


class BubbleScreener:
    """气泡筛选器类"""
    
    def __init__(self, gpu_manager: Optional[GPUManager] = None, batch_size: int = 32):
        """
        初始化气泡筛选器
        
        Args:
            gpu_manager: GPU管理器实例
            batch_size: 批处理大小
        """
        self.gpu_manager = gpu_manager or get_global_gpu_manager()
        self.batch_size = batch_size
        self.screening_criteria = self._init_screening_criteria()
        
    def _init_screening_criteria(self) -> Dict[str, Tuple[float, float]]:
        """初始化筛选标准"""
        return {
            'circularity': (0.3, 1.0),      # 圆形度范围
            'solidity': (0.5, 1.0),         # 凸度范围
            'major_axis_length': (0.1, 0.9), # 长轴长度范围（归一化）
            'minor_axis_length': (0.1, 0.9), # 短轴长度范围（归一化）
            'shadow_ratio': (0.0, 0.8),     # 阴影比例范围
            'edge_gradient': (0.1, 1.0)     # 边缘梯度范围
        }
    
    def screen_bubble_images(self, image_paths: List[str], 
                           enable_gpu_acceleration: bool = True) -> Dict[str, Any]:
        """
        筛选气泡图像
        
        Args:
            image_paths: 图像路径列表
            enable_gpu_acceleration: 是否启用GPU加速
            
        Returns:
            筛选结果字典
        """
        if not image_paths:
            return {'passed': [], 'failed': [], 'analysis_data': {}}
        
        print(f"  开始筛选 {len(image_paths)} 个气泡图像...")
        
        if enable_gpu_acceleration and self.gpu_manager.use_gpu:
            return self._screen_with_gpu_acceleration(image_paths)
        else:
            return self._screen_sequential(image_paths)
    
    def _screen_sequential(self, image_paths: List[str]) -> Dict[str, Any]:
        """顺序筛选（CPU模式）"""
        passed_images = []
        failed_images = []
        analysis_data = {}
        
        for image_path in image_paths:
            try:
                # 分析图像
                analysis_result = analyze_bubble_image(image_path)
                
                if analysis_result is not None:
                    analysis_data[image_path] = analysis_result
                    
                    # 应用筛选标准
                    if self._passes_screening_criteria(analysis_result):
                        passed_images.append(image_path)
                    else:
                        failed_images.append(image_path)
                else:
                    failed_images.append(image_path)
                    
            except Exception as e:
                failed_images.append(image_path)
                continue
            finally:
                # 内存清理
                gc.collect()
        
        return {
            'passed': passed_images,
            'failed': failed_images,
            'analysis_data': analysis_data
        }
    
    def _screen_with_gpu_acceleration(self, image_paths: List[str]) -> Dict[str, Any]:
        """GPU加速筛选"""
        # 将图像路径分批处理
        batches = self._create_batches(image_paths, self.batch_size)
        
        all_passed = []
        all_failed = []
        all_analysis_data = {}
        
        # 使用GPU管理器进行并行处理
        batch_results = self.gpu_manager.parallel_process(
            self._process_batch,
            batches,
            screening_criteria=self.screening_criteria
        )
        
        # 合并结果
        for result in batch_results:
            if result is not None:
                all_passed.extend(result['passed'])
                all_failed.extend(result['failed'])
                all_analysis_data.update(result['analysis_data'])
        
        return {
            'passed': all_passed,
            'failed': all_failed,
            'analysis_data': all_analysis_data
        }
    
    def _create_batches(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """创建批次"""
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        return batches
    
    def _process_batch(self, batch: List[str], device: str,
                      screening_criteria: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        处理一个批次的图像（GPU加速版本）

        Args:
            batch: 图像路径批次
            device: 处理设备
            screening_criteria: 筛选标准

        Returns:
            批次处理结果
        """
        passed_images = []
        failed_images = []
        analysis_data = {}

        try:
            # 检查是否使用GPU加速
            use_gpu = TORCH_AVAILABLE and device.startswith('cuda')

            if use_gpu:
                return self._process_batch_gpu(batch, device, screening_criteria)
            else:
                return self._process_batch_cpu(batch, device, screening_criteria)

        except Exception as e:
            print(f"批次处理失败 (设备: {device}): {e}")
            # 回退到CPU处理
            return self._process_batch_cpu(batch, device, screening_criteria)

    def _process_batch_gpu(self, batch: List[str], device: str,
                          screening_criteria: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        GPU加速的批次处理
        """
        passed_images = []
        failed_images = []
        analysis_data = {}

        try:
            # 预加载图像到内存
            images = self._preload_images(batch)
            valid_images = []
            valid_paths = []

            # 筛选有效图像
            for i, (image_path, image) in enumerate(zip(batch, images)):
                if image is not None:
                    valid_images.append(image)
                    valid_paths.append(image_path)
                else:
                    failed_images.append(image_path)

            if not valid_images:
                return {
                    'passed': passed_images,
                    'failed': failed_images,
                    'analysis_data': analysis_data
                }

            # 智能批量分析（自动选择GPU或CPU）
            batch_results = analyze_bubble_batch_smart(
                valid_images,
                device=device,
                batch_size=min(8, len(valid_images))  # 限制GPU批次大小
            )

            # 处理分析结果
            for image_path, analysis_result in zip(valid_paths, batch_results):
                if analysis_result is not None:
                    analysis_data[image_path] = analysis_result

                    # 应用筛选标准
                    if self._passes_screening_criteria(analysis_result):
                        passed_images.append(image_path)
                    else:
                        failed_images.append(image_path)
                else:
                    failed_images.append(image_path)

            return {
                'passed': passed_images,
                'failed': failed_images,
                'analysis_data': analysis_data
            }

        except Exception as e:
            print(f"GPU批次处理失败: {e}")
            # 回退到CPU处理
            return self._process_batch_cpu(batch, device, screening_criteria)
        finally:
            # 清理内存
            if 'images' in locals():
                del images
            if 'valid_images' in locals():
                del valid_images
            gc.collect()

    def _process_batch_cpu(self, batch: List[str], device: str,
                          screening_criteria: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        CPU版本的批次处理（原有逻辑）
        """
        passed_images = []
        failed_images = []
        analysis_data = {}

        try:
            # 预加载图像到内存（批量优化）
            images = self._preload_images(batch)

            for i, image_path in enumerate(batch):
                try:
                    if i < len(images) and images[i] is not None:
                        # 使用预加载的图像进行分析
                        analysis_result = analyze_bubble_image_from_array(images[i])
                    else:
                        # 回退到文件读取
                        analysis_result = analyze_bubble_image(image_path)

                    if analysis_result is not None:
                        analysis_data[image_path] = analysis_result

                        # 应用筛选标准
                        if self._passes_screening_criteria(analysis_result):
                            passed_images.append(image_path)
                        else:
                            failed_images.append(image_path)
                    else:
                        failed_images.append(image_path)

                except Exception as e:
                    failed_images.append(image_path)
                    continue

            return {
                'passed': passed_images,
                'failed': failed_images,
                'analysis_data': analysis_data
            }

        except Exception as e:
            print(f"CPU批次处理失败: {e}")
            return {
                'passed': [],
                'failed': batch,
                'analysis_data': {}
            }
        finally:
            # 清理内存
            if 'images' in locals():
                del images
            gc.collect()
    
    def _preload_images(self, image_paths: List[str]) -> List[Optional[np.ndarray]]:
        """预加载图像到内存"""
        images = []
        for image_path in image_paths:
            try:
                img = cv2.imread(image_path)
                images.append(img)
            except Exception:
                images.append(None)
        return images
    
    def _passes_screening_criteria(self, analysis_result: Dict[str, float]) -> bool:
        """检查是否通过筛选标准"""
        try:
            for param_name, (min_val, max_val) in self.screening_criteria.items():
                if param_name in analysis_result:
                    value = analysis_result[param_name]
                    if not (min_val <= value <= max_val):
                        return False
            return True
        except Exception:
            return False
    
    def update_screening_criteria(self, new_criteria: Dict[str, Tuple[float, float]]):
        """更新筛选标准"""
        self.screening_criteria.update(new_criteria)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        analysis_report = get_analysis_performance_report()
        gpu_memory_info = self.gpu_manager.get_gpu_memory_info() if self.gpu_manager else {}

        return {
            'analysis_performance': analysis_report,
            'gpu_memory_info': gpu_memory_info,
            'gpu_manager_status': {
                'use_gpu': self.gpu_manager.use_gpu if self.gpu_manager else False,
                'num_gpus': self.gpu_manager.num_gpus if self.gpu_manager else 0,
                'gpu_ids': self.gpu_manager.gpu_ids if self.gpu_manager else []
            }
        }

    def cleanup(self):
        """清理资源"""
        if self.gpu_manager:
            self.gpu_manager.cleanup()


def create_bubble_screener(gpu_ids: Optional[List[int]] = None, 
                          max_gpus: int = 4, 
                          batch_size: int = 32) -> BubbleScreener:
    """
    创建气泡筛选器的便捷函数
    
    Args:
        gpu_ids: 指定使用的GPU ID列表
        max_gpus: 最大使用GPU数量
        batch_size: 批处理大小
        
    Returns:
        BubbleScreener实例
    """
    gpu_manager = get_global_gpu_manager(gpu_ids=gpu_ids, max_gpus=max_gpus)
    return BubbleScreener(gpu_manager=gpu_manager, batch_size=batch_size)


def screen_bubble_images_parallel(image_paths: List[str],
                                 gpu_ids: Optional[List[int]] = None,
                                 max_gpus: int = 4,
                                 batch_size: int = 32,
                                 enable_gpu_acceleration: bool = True) -> Dict[str, Any]:
    """
    并行筛选气泡图像的便捷函数
    
    Args:
        image_paths: 图像路径列表
        gpu_ids: 指定使用的GPU ID列表
        max_gpus: 最大使用GPU数量
        batch_size: 批处理大小
        enable_gpu_acceleration: 是否启用GPU加速
        
    Returns:
        筛选结果字典
    """
    screener = create_bubble_screener(gpu_ids=gpu_ids, max_gpus=max_gpus, batch_size=batch_size)
    try:
        return screener.screen_bubble_images(image_paths, enable_gpu_acceleration)
    finally:
        screener.cleanup()
