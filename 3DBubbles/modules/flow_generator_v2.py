"""
重构后的流场生成器 v2.0

该模块使用统一缩放管理系统重构了原有的流场生成逻辑，解决了尺寸一致性问题。

主要改进：
1. 使用UnifiedScalingManager管理所有缩放操作
2. 保存完整的原始STL信息和缩放历史
3. 建立从STL到最终结果的完整追溯链
4. 优化mesh位置分配逻辑，确保几何信息不丢失
5. 提供向后兼容的接口

作者：3DBubbles_v2.0 优化团队
版本：2.0.0
"""

import os
import csv
import random
import multiprocessing as mp
import numpy as np
import pyvista as pv
from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path

from .unified_scaling_system import (
    UnifiedScalingManager, 
    BubbleGeometryRecord,
    OriginalGeometryInfo,
    ScalingTransformation
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowGeneratorV2:
    """重构后的流场生成器"""
    
    def __init__(self, base_path: str, enable_size_tracking: bool = True):
        """
        初始化流场生成器
        
        Args:
            base_path: 基础路径
            enable_size_tracking: 是否启用尺寸追踪
        """
        self.base_path = Path(base_path)
        self.scaling_manager = UnifiedScalingManager(base_path, enable_size_tracking)
        self.bubble_records: Dict[str, BubbleGeometryRecord] = {}
        
        logger.info(f"FlowGeneratorV2 初始化完成，基础路径: {self.base_path}")
    
    def generate_points_in_cube(self, num_points: int, cube_size: np.ndarray = np.array([100, 100, 100]),
                              num: int = 100, poisson_max_iter: int = 100000) -> np.ndarray:
        """
        在立方体中生成泊松圆盘采样点（从原flow_generator.py移植）
        
        Args:
            num_points: 点的数量
            cube_size: 立方体尺寸
            num: 密度参数
            poisson_max_iter: 最大迭代次数
            
        Returns:
            points: 生成的点坐标
        """
        try:
            rnd_points = []
            look_up_num = 0
            min_distance = cube_size.min() / (num**(1/3)) * 0.5
            
            while len(rnd_points) < num_points:
                if look_up_num >= poisson_max_iter:
                    logger.warning(f"达到最大迭代次数 {poisson_max_iter}，当前生成 {len(rnd_points)} 个点")
                    break
                
                x, y, z = np.random.rand(3) * cube_size
                
                # 检查与现有点的距离
                valid_point = True
                for existing_point in rnd_points:
                    if np.linalg.norm(np.array([x, y, z]) - existing_point) <= min_distance:
                        valid_point = False
                        break
                
                # 检查是否在有效区域内
                if valid_point and np.linalg.norm(np.array([x, y]) - cube_size[:2]/2) < cube_size[0]/2*0.85:
                    rnd_points.append([x, y, z])
                
                look_up_num += 1
            
            if len(rnd_points) < num_points:
                logger.warning(f"只生成了 {len(rnd_points)} 个点，少于目标 {num_points} 个")
            
            return np.array(rnd_points)
            
        except Exception as e:
            logger.error(f"生成立方体采样点失败: {e}")
            raise
    
    def process_and_scale_mesh_batch(self, stl_files: List[str], chosen_volumes: List[float],
                                   num_clusters: int = 20000, sample_spacing: float = 0.1) -> List[Tuple]:
        """
        批量处理和缩放mesh，替代原有的multiprocessing调用
        
        Args:
            stl_files: STL文件列表
            chosen_volumes: 目标体积列表
            num_clusters: 目标点数
            sample_spacing: 采样间距
            
        Returns:
            List[Tuple]: 处理结果列表
        """
        try:
            results = []
            
            # 使用多进程处理
            with mp.Pool(processes=mp.cpu_count()) as pool:
                # 准备参数
                args_list = []
                for i, volume in enumerate(chosen_volumes):
                    stl_file = random.choice(stl_files)
                    args_list.append((stl_file, volume, num_clusters, sample_spacing, i))
                
                # 并行处理
                pool_results = pool.starmap(self._process_single_mesh, args_list)
            
            # 整理结果
            for result in pool_results:
                if result is not None:
                    results.append(result)
            
            logger.info(f"批量处理完成，成功处理 {len(results)} 个mesh")
            return results
            
        except Exception as e:
            logger.error(f"批量处理mesh失败: {e}")
            raise
    
    def _process_single_mesh(self, stl_file: str, target_volume: float, 
                           num_clusters: int, sample_spacing: float, 
                           mesh_index: int) -> Optional[Tuple]:
        """
        处理单个mesh（多进程调用的函数）
        
        Args:
            stl_file: STL文件路径
            target_volume: 目标体积
            num_clusters: 目标点数
            sample_spacing: 采样间距
            mesh_index: mesh索引
            
        Returns:
            Tuple: (stl_file, processed_mesh, original_mesh, original_info, scaling_info, bubble_id)
        """
        try:
            # 创建临时的缩放管理器（用于多进程）
            temp_manager = UnifiedScalingManager(str(self.base_path), enable_size_tracking=False)
            
            # 处理和缩放mesh
            processed_mesh, original_mesh, original_info, scaling_info = temp_manager.process_and_scale_mesh(
                stl_file, target_volume, num_clusters, sample_spacing
            )
            
            # 生成气泡ID
            bubble_id = f"bubble_{mesh_index:04d}_{Path(stl_file).stem}"
            
            return (stl_file, processed_mesh, original_mesh, original_info, scaling_info, bubble_id)
            
        except Exception as e:
            logger.error(f"处理单个mesh失败 {stl_file}: {e}")
            return None
    
    def allocate_bubble_positions(self, mesh_results: List[Tuple], volume_size_x: float,
                                volume_size_y: float, volume_height: float,
                                poisson_max_iter: int = 100000) -> Tuple[List, List, List]:
        """
        分配气泡位置，替代原有的位置分配逻辑
        
        Args:
            mesh_results: mesh处理结果
            volume_size_x: 流场X方向尺寸
            volume_size_y: 流场Y方向尺寸
            volume_height: 流场Z方向尺寸
            poisson_max_iter: 泊松采样最大迭代次数
            
        Returns:
            Tuple: (allocated_meshes, allocated_origin_meshes, bubble_records)
        """
        try:
            # 生成位置点
            cube_size = np.array([volume_size_x, volume_size_y, volume_height]) * 1.2
            position_points = self.generate_points_in_cube(
                len(mesh_results), cube_size, poisson_max_iter=poisson_max_iter
            )
            
            allocated_meshes = []
            allocated_origin_meshes = []
            bubble_records = []
            
            # 分配位置并创建记录
            for i, (stl_file, processed_mesh, original_mesh, original_info, scaling_info, bubble_id) in enumerate(mesh_results):
                if i < len(position_points):
                    position_offset = position_points[i]
                    
                    # 应用位置偏移
                    updated_scaling_info = self.scaling_manager.apply_position_offset(
                        processed_mesh, original_mesh, position_offset, bubble_id, scaling_info
                    )
                    
                    # 创建气泡几何记录
                    bubble_record = self.scaling_manager.create_bubble_geometry_record(
                        bubble_id, original_info, updated_scaling_info, processed_mesh
                    )
                    
                    allocated_meshes.append(processed_mesh)
                    allocated_origin_meshes.append(original_mesh)
                    bubble_records.append(bubble_record)
                    
                    # 保存到实例记录中
                    self.bubble_records[bubble_id] = bubble_record
                    
                    logger.debug(f"位置分配完成: {bubble_id}, 位置: {position_offset}")
                else:
                    logger.warning(f"位置点不足，跳过 {bubble_id}")
            
            logger.info(f"位置分配完成，共分配 {len(allocated_meshes)} 个气泡")
            return allocated_meshes, allocated_origin_meshes, bubble_records
            
        except Exception as e:
            logger.error(f"分配气泡位置失败: {e}")
            raise
    
    def save_bubble_info_csv(self, bubble_records: List[BubbleGeometryRecord], 
                           position_points: np.ndarray, volumes: List[float]) -> str:
        """
        保存气泡信息到CSV文件，替代原有的names_points.csv
        
        Args:
            bubble_records: 气泡记录列表
            position_points: 位置点
            volumes: 体积列表
            
        Returns:
            str: CSV文件路径
        """
        try:
            csv_file_path = self.base_path / 'bubble_info_v2.csv'
            
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                # 写入表头
                writer.writerow([
                    'BubbleID', 'STLFile', 'X', 'Y', 'Z', 'Volume', 
                    'OriginalVolume', 'ScaleFactor', 'OriginalBounds',
                    'FinalBounds', 'SizeConsistencyRatio'
                ])
                
                # 写入数据
                for i, record in enumerate(bubble_records):
                    if i < len(position_points) and i < len(volumes):
                        pos = position_points[i]
                        volume = volumes[i]
                        
                        # 计算尺寸一致性比率
                        consistency_ratio = (record.final_scaled_volume / record.original_info.original_volume 
                                           if record.original_info.original_volume > 0 else 0)
                        
                        writer.writerow([
                            record.bubble_id,
                            Path(record.original_info.stl_file_path).name,
                            pos[0], pos[1], pos[2],
                            volume,
                            record.original_info.original_volume,
                            record.scaling_info.volume_scale_factor,
                            str(record.original_info.original_bounds),
                            str(record.final_scaled_bounds),
                            f"{consistency_ratio:.6f}"
                        ])
            
            logger.info(f"气泡信息已保存到: {csv_file_path}")
            return str(csv_file_path)
            
        except Exception as e:
            logger.error(f"保存气泡信息CSV失败: {e}")
            raise


def generater_v2(stl_files: List[str], base_path: str, volume_size_x: float, volume_size_y: float,
                volume_height: float, gas_holdups: List[float], alpha: float = 8, truncation: float = 0.75,
                poisson_max_iter: int = 100000, sample_spacing: float = 0.1,
                enable_hq_generation: bool = False, enable_bubble_composition: bool = False,
                use_pregenerated_dataset: bool = True, pregenerated_image_dir: Optional[str] = None,
                pregenerated_struct_csv: Optional[str] = None, enable_size_tracking: bool = True,
                **kwargs) -> Dict[str, Any]:
    """
    重构后的主要流场生成函数，使用统一缩放管理系统

    Args:
        stl_files: STL文件列表
        base_path: 基础保存路径
        volume_size_x, volume_size_y, volume_height: 流场尺寸
        gas_holdups: 气含率列表
        alpha: 角度权重指数
        truncation: 截断值
        poisson_max_iter: 泊松采样最大迭代次数
        sample_spacing: 采样间距
        enable_hq_generation: 是否启用高质量生成
        enable_bubble_composition: 是否启用流场合成
        use_pregenerated_dataset: 是否使用预生成数据集筛选
        pregenerated_image_dir: 预生成图像目录
        pregenerated_struct_csv: 预生成结构参数CSV文件
        enable_size_tracking: 是否启用尺寸追踪
        **kwargs: 其他参数（向后兼容）

    Returns:
        Dict: 生成结果信息
    """
    try:
        # 初始化流场生成器
        flow_generator = FlowGeneratorV2(base_path, enable_size_tracking)

        results = {
            'total_gas_holdups': len(gas_holdups),
            'processed_gas_holdups': 0,
            'bubble_records': {},
            'size_mapping_files': [],
            'csv_files': []
        }

        logger.info(f"开始处理 {len(gas_holdups)} 个气含率...")

        for gas_idx, gas_holdup in enumerate(gas_holdups):
            logger.info(f"正在处理气含率 {gas_idx + 1}/{len(gas_holdups)}: {gas_holdup}")

            # 计算期望体积
            expected_volume = volume_size_x * volume_size_y * volume_height * gas_holdup

            # 生成目标体积列表
            chosen_volumes = []
            total_volume = 0

            while total_volume < expected_volume:
                chosen_volume = np.random.lognormal(mean=3.5, sigma=1.0) / 1000
                chosen_volumes.append(chosen_volume)
                total_volume += chosen_volume

            logger.info(f"生成 {len(chosen_volumes)} 个气泡，总体积: {total_volume:.6f}")

            # 批量处理和缩放mesh
            mesh_results = flow_generator.process_and_scale_mesh_batch(
                stl_files, chosen_volumes, num_clusters=20000, sample_spacing=sample_spacing
            )

            # 分配气泡位置
            allocated_meshes, allocated_origin_meshes, bubble_records = flow_generator.allocate_bubble_positions(
                mesh_results, volume_size_x, volume_size_y, volume_height, poisson_max_iter
            )

            # 生成位置点用于CSV保存
            cube_size = np.array([volume_size_x, volume_size_y, volume_height]) * 1.2
            position_points = flow_generator.generate_points_in_cube(
                len(bubble_records), cube_size, poisson_max_iter=poisson_max_iter
            )

            # 保存气泡信息CSV
            csv_file = flow_generator.save_bubble_info_csv(
                bubble_records, position_points, [vol * 10 for vol in chosen_volumes]
            )

            # 合并mesh（保持与原有流程兼容）
            if allocated_meshes:
                merged_mesh = pv.merge(allocated_meshes)
                merged_origin_mesh = pv.merge(allocated_origin_meshes)

                # 保存合并后的STL文件
                merged_stl_path = Path(base_path) / f"merged_gas_holdup_{gas_holdup:.3f}.stl"
                merged_origin_mesh.save(str(merged_stl_path))
                logger.info(f"合并STL已保存: {merged_stl_path}")

            # 保存尺寸映射
            if enable_size_tracking:
                mapping_file = flow_generator.scaling_manager.save_size_mapping()
                results['size_mapping_files'].append(mapping_file)

            results['bubble_records'][f'gas_holdup_{gas_holdup}'] = bubble_records
            results['csv_files'].append(csv_file)
            results['processed_gas_holdups'] += 1

            logger.info(f"气含率 {gas_holdup} 处理完成")

        # 验证尺寸一致性
        if enable_size_tracking:
            consistency_report = flow_generator.scaling_manager.validate_size_consistency()
            results['size_consistency_report'] = consistency_report
            logger.info(f"尺寸一致性验证完成，一致性率: {consistency_report.get('consistency_rate', 0):.2%}")

        logger.info("所有气含率处理完成")
        return results

    except Exception as e:
        logger.error(f"流场生成失败: {e}")
        raise


# 向后兼容的函数别名
def upsample_and_scale_mesh_v2(stl_files: List[str], num_clusters: int, chosen_volume: float,
                              sample_spacing: float, base_path: str = "/tmp") -> Tuple:
    """
    向后兼容的mesh处理函数

    Args:
        stl_files: STL文件列表
        num_clusters: 目标点数
        chosen_volume: 目标体积
        sample_spacing: 采样间距
        base_path: 基础路径

    Returns:
        Tuple: (stl_file, mesh, mesh_origin, chosen_volume, original_info, scaling_info)
    """
    try:
        stl_file = random.choice(stl_files)

        # 使用统一缩放管理器
        scaling_manager = UnifiedScalingManager(base_path, enable_size_tracking=False)
        processed_mesh, original_mesh, original_info, scaling_info = scaling_manager.process_and_scale_mesh(
            stl_file, chosen_volume, num_clusters, sample_spacing
        )

        return stl_file, processed_mesh, original_mesh, chosen_volume, original_info, scaling_info

    except Exception as e:
        logger.error(f"向后兼容mesh处理失败: {e}")
        raise
