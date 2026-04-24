# bubble_screening.py
"""
气泡筛选模块

该模块提供气泡筛选功能，支持目录输入、输出文件管理、错误处理和日志记录。
主要用于筛选符合条件的单个气泡图像文件，确保与主程序无缝集成。

主要功能：
- 预处理：创建输出文件夹
- 筛选：基于分析结果应用条件，复制合格文件
- 错误处理：捕获异常并日志
- 集成：filter_bubbles 函数可直接在主程序调用
"""

import os
import shutil
import logging
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入现有模块（假设这些已存在）
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


def filter_bubbles(input_dir: str, output_dir: str, 
                   enable_gpu: bool = True, batch_size: int = 32) -> Dict[str, Any]:
    """
    气泡筛选主函数：处理输入目录中的气泡图像文件，筛选并复制到输出目录。

    关键步骤：
    1. 预处理：检查并创建输出文件夹。
    2. 收集输入文件：遍历 input_dir，获取图像文件列表（支持 .png, .jpg 等）。
    3. 筛选逻辑：逐个或批量分析文件，应用筛选条件。
    4. 处理结果：复制 passed 文件到 output_dir，记录日志和统计。
    5. 错误处理：捕获异常，跳过问题文件。

    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径（合格文件复制至此）
        enable_gpu: 是否启用 GPU 加速
        batch_size: 批处理大小

    Returns:
        结果字典：{'passed': [路径列表], 'failed': [路径列表], 'stats': {统计}, 'log_file': 日志路径}
    """
    # 1. 预处理阶段：检查并创建输出文件夹
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"创建输出目录: {output_dir}")
        else:
            logger.info(f"输出目录已存在: {output_dir}")
        
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"输出目录 {output_dir} 无写入权限")
        
        # 检查输入目录
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")
        
        logger.info(f"开始筛选，输入目录: {input_dir}")
        
    except Exception as e:
        logger.error(f"预处理失败: {e}")
        return {'passed': [], 'failed': [], 'stats': {}, 'error': str(e)}

    # 2. 收集输入文件
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_paths = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(supported_extensions):
            image_paths.append(os.path.join(input_dir, file))
    
    if not image_paths:
        logger.warning("输入目录中无支持的图像文件")
        return {'passed': [], 'failed': [], 'stats': {'total': 0}, 'log_file': None}
    
    total_files = len(image_paths)
    logger.info(f"发现 {total_files} 个图像文件，开始筛选...")

    # 初始化筛选器（复用现有逻辑）
    gpu_manager = get_global_gpu_manager() if enable_gpu else None
    screener = BubbleScreener(gpu_manager=gpu_manager, batch_size=batch_size)
    
    # 筛选标准（示例，放宽以提高通过率）
    screening_criteria = {
        'circularity': (0.1, 1.0),
        'solidity': (0.3, 1.0),
        'major_axis_length': (0.05, 0.95),
        'minor_axis_length': (0.05, 0.95),
        'shadow_ratio': (0.0, 0.9),
        'edge_gradient': (0.05, 1.0)
    }
    screener.update_screening_criteria(screening_criteria)

    # 3. 筛选逻辑
    passed = []
    failed = []
    analysis_data = {}
    
    try:
        # 使用现有筛选方法
        results = screener.screen_bubble_images(image_paths, enable_gpu_acceleration=enable_gpu)
        passed_paths = results['passed']
        failed_paths = results['failed']
        analysis_data = results['analysis_data']
        
        # 4. 处理合格文件：复制到输出目录
        for path in passed_paths:
            try:
                filename = os.path.basename(path)
                dest_path = os.path.join(output_dir, filename)
                shutil.copy2(path, dest_path)  # 复制文件，保留元数据
                passed.append(dest_path)
                logger.info(f"复制合格文件: {filename}")
            except Exception as e:
                logger.error(f"复制文件失败 {path}: {e}")
                failed.append(path)
        
        # 失败文件记录
        failed.extend(failed_paths)
        
    except Exception as e:
        logger.error(f"筛选过程失败: {e}")
        failed.extend(image_paths)
        passed = []
    
    # 统计
    stats = {
        'total': total_files,
        'passed_count': len(passed),
        'failed_count': len(failed),
        'pass_rate': len(passed) / total_files * 100 if total_files > 0 else 0
    }
    logger.info(f"筛选完成: 通过 {stats['passed_count']}/{total_files} ({stats['pass_rate']:.1f}%), "
                f"失败 {stats['failed_count']}")

    # 日志文件
    log_file = os.path.join(output_dir, 'screening_log.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"筛选统计: {stats}\n")
        f.write(f"通过文件: {passed}\n")
        f.write(f"失败文件: {failed}\n")
    
    # 清理
    screener.cleanup()
    gc.collect()
    
    return {
        'passed': passed,
        'failed': failed,
        'stats': stats,
        'analysis_data': analysis_data,
        'log_file': log_file
    }


class BubbleScreener:
    """气泡筛选器类（原有逻辑，保持兼容）"""
    
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
        
    def _init_screening_criteria(self) -> Dict[str, tuple]:
        """初始化筛选标准"""
        return {
            'circularity': (0.1, 1.0),
            'solidity': (0.3, 1.0),
            'major_axis_length': (0.05, 0.95),
            'minor_axis_length': (0.05, 0.95),
            'shadow_ratio': (0.0, 0.9),
            'edge_gradient': (0.05, 1.0)
        }
    
    def screen_bubble_images(self, image_paths: List[str], 
                             enable_gpu_acceleration: bool = True) -> Dict[str, Any]:
        """
        筛选气泡图像（原有方法，简化版）
        
        Args:
            image_paths: 图像路径列表
            enable_gpu_acceleration: 是否启用GPU加速
            
        Returns:
            筛选结果字典
        """
        if not image_paths:
            return {'passed': [], 'failed': [], 'analysis_data': {}}
        
        logger.info(f"  开始筛选 {len(image_paths)} 个气泡图像...")
        
        if enable_gpu_acceleration and self.gpu_manager and self.gpu_manager.use_gpu:
            return self._screen_with_gpu(image_paths)
        else:
            return self._screen_sequential(image_paths)
    
    def _screen_sequential(self, image_paths: List[str]) -> Dict[str, Any]:
        """顺序筛选（CPU）"""
        passed_images = []
        failed_images = []
        analysis_data = {}
        
        for image_path in image_paths:
            try:
                analysis_result = analyze_bubble_image(image_path)
                if analysis_result is not None:
                    analysis_data[image_path] = analysis_result
                    if self._passes_screening_criteria(analysis_result):
                        passed_images.append(image_path)
                    else:
                        failed_images.append(image_path)
                else:
                    failed_images.append(image_path)
            except Exception as e:
                logger.error(f"处理文件 {image_path} 失败: {e}")
                failed_images.append(image_path)
            finally:
                gc.collect()
        
        return {'passed': passed_images, 'failed': failed_images, 'analysis_data': analysis_data}
    
    def _screen_with_gpu(self, image_paths: List[str]) -> Dict[str, Any]:
        """GPU筛选（简化，复用原有批处理逻辑）"""
        # 这里简化调用原有 GPU 逻辑，实际可扩展为批处理
        return self._screen_sequential(image_paths)  # 临时回退，实际可集成 GPU
    
    def _passes_screening_criteria(self, analysis_result: Dict[str, float]) -> bool:
        """检查筛选标准"""
        for param, (min_val, max_val) in self.screening_criteria.items():
            if param in analysis_result:
                value = analysis_result[param]
                if not (min_val <= value <= max_val):
                    return False
        return True
    
    def update_screening_criteria(self, new_criteria: Dict[str, tuple]):
        """更新筛选标准"""
        self.screening_criteria.update(new_criteria)
    
    def cleanup(self):
        """清理资源"""
        if self.gpu_manager:
            self.gpu_manager.cleanup()


# 便捷函数（原有，保持）
def create_bubble_screener(gpu_ids: Optional[List[int]] = None, 
                           max_gpus: int = 4, 
                           batch_size: int = 32) -> 'BubbleScreener':
    gpu_manager = get_global_gpu_manager(gpu_ids=gpu_ids, max_gpus=max_gpus)
    return BubbleScreener(gpu_manager=gpu_manager, batch_size=batch_size)


def screen_bubble_images_parallel(image_paths: List[str],
                                  gpu_ids: Optional[List[int]] = None,
                                  max_gpus: int = 4,
                                  batch_size: int = 32,
                                  enable_gpu_acceleration: bool = True) -> Dict[str, Any]:
    screener = create_bubble_screener(gpu_ids=gpu_ids, max_gpus=max_gpus, batch_size=batch_size)
    try:
        return screener.screen_bubble_images(image_paths, enable_gpu_acceleration)
    finally:
        screener.cleanup()


# 示例主程序集成（可选，在主文件中调用）
# if __name__ == "__main__":
#     result = filter_bubbles("input_bubbles", "output_bubbles")
#     print(f"筛选结果: {result['stats']}")
