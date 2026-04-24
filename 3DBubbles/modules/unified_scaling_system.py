"""
统一缩放管理系统

该模块实现了3DBubbles_v2.0项目的统一缩放管理系统，解决了原有多层缩放导致的尺寸不一致问题。

主要功能：
1. 统一管理STL文件的加载、缩放和位置分配
2. 保存原始几何信息，建立完整的尺寸追溯链
3. 提供统一的缩放接口，替代原有的多层缩放系统
4. 支持向后兼容，确保与现有流程无缝集成

作者：3DBubbles_v2.0 优化团队
版本：2.0.0
"""

import os
import json
import numpy as np
import pyvista as pv
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from scipy.spatial import ConvexHull
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OriginalGeometryInfo:
    """原始几何信息数据类"""
    stl_file_path: str
    original_volume: float
    original_surface_area: float
    original_bounds: List[float]  # [xmin, xmax, ymin, ymax, zmin, zmax]
    original_center: List[float]  # [x, y, z]
    original_ellipsoid_radii: Optional[List[float]] = None  # [a, b, c] 按降序排列
    original_ellipsoid_center: Optional[List[float]] = None
    mesh_processing_params: Optional[Dict[str, Any]] = None


@dataclass
class ScalingTransformation:
    """缩放变换信息数据类"""
    volume_scale_factor: float  # STL体积缩放因子
    target_volume: float  # 目标体积
    position_offset: List[float]  # 位置偏移 [x, y, z]
    canvas_scale_factor: float = 100.0  # 画布坐标缩放因子（固定）
    final_image_size: int = 128  # 最终图像尺寸（固定）
    transformation_timestamp: Optional[str] = None


@dataclass
class BubbleGeometryRecord:
    """气泡几何记录数据类"""
    bubble_id: str
    original_info: OriginalGeometryInfo
    scaling_info: ScalingTransformation
    final_scaled_volume: float
    final_scaled_surface_area: float
    final_scaled_bounds: List[float]
    final_position: List[float]  # 最终3D位置


class UnifiedScalingManager:
    """统一缩放管理器"""
    
    def __init__(self, base_path: str, enable_size_tracking: bool = True):
        """
        初始化统一缩放管理器
        
        Args:
            base_path: 基础路径，用于保存尺寸映射文件
            enable_size_tracking: 是否启用尺寸追踪
        """
        self.base_path = Path(base_path)
        self.enable_size_tracking = enable_size_tracking
        self.geometry_records: Dict[str, BubbleGeometryRecord] = {}
        self.rendering_records: Dict[str, Dict] = {}  # 渲染记录
        self.size_mapping_file = self.base_path / "bubble_size_mapping.json"
        
        # 确保目录存在
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"统一缩放管理器初始化完成，基础路径: {self.base_path}")
    
    def ellipsoid_fit(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        椭球拟合函数（从bubble_rendering.py移植）
        
        Args:
            points: 点云数据
            
        Returns:
            center: 椭球中心
            evecs: 特征向量
            radii: 椭球半径（按降序排列）
        """
        try:
            # 简化的椭球拟合实现
            center = np.mean(points, axis=0)
            centered_points = points - center
            
            # 计算协方差矩阵
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # 按降序排列
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # 计算椭球半径
            radii = np.sqrt(eigenvalues) * 2  # 近似计算
            
            return center, eigenvectors, radii
            
        except Exception as e:
            logger.warning(f"椭球拟合失败: {e}")
            return np.zeros(3), np.eye(3), np.ones(3)
    
    def extract_original_geometry_info(self, stl_file_path: str, 
                                     mesh_processing_params: Optional[Dict] = None) -> OriginalGeometryInfo:
        """
        提取STL文件的原始几何信息
        
        Args:
            stl_file_path: STL文件路径
            mesh_processing_params: 网格处理参数
            
        Returns:
            OriginalGeometryInfo: 原始几何信息
        """
        try:
            # 加载原始STL文件
            mesh_origin = pv.read(stl_file_path)
            
            # 提取基本几何信息
            original_volume = mesh_origin.volume
            original_surface_area = mesh_origin.area
            original_bounds = list(mesh_origin.bounds)
            original_center = list(mesh_origin.center)
            
            # 椭球拟合
            try:
                center, evecs, radii = self.ellipsoid_fit(mesh_origin.points)
                original_ellipsoid_radii = sorted(radii, reverse=True)
                original_ellipsoid_center = list(center)
            except Exception as e:
                logger.warning(f"椭球拟合失败 {stl_file_path}: {e}")
                original_ellipsoid_radii = None
                original_ellipsoid_center = None
            
            return OriginalGeometryInfo(
                stl_file_path=stl_file_path,
                original_volume=original_volume,
                original_surface_area=original_surface_area,
                original_bounds=original_bounds,
                original_center=original_center,
                original_ellipsoid_radii=original_ellipsoid_radii,
                original_ellipsoid_center=original_ellipsoid_center,
                mesh_processing_params=mesh_processing_params
            )
            
        except Exception as e:
            logger.error(f"提取原始几何信息失败 {stl_file_path}: {e}")
            raise
    
    def upsample_point_cloud(self, points: np.ndarray, num_clusters: int, 
                           sample_spacing: float = 0.1) -> pv.PolyData:
        """
        上采样点云（从flow_generator.py移植并优化）
        
        Args:
            points: 原始点云
            num_clusters: 目标点数
            sample_spacing: 采样间距
            
        Returns:
            mesh: 上采样后的mesh
        """
        try:
            cloud = pv.PolyData(points)
            current_spacing = sample_spacing
            
            while True:
                mesh = cloud.reconstruct_surface(nbr_sz=10, sample_spacing=current_spacing)
                new_points = np.asarray(mesh.points)
                
                if new_points.shape[0] < num_clusters * 1.1:
                    current_spacing *= 0.8
                else:
                    break
                    
                # 防止无限循环
                if current_spacing < 0.01:
                    logger.warning("采样间距过小，停止迭代")
                    break
            
            return mesh
            
        except Exception as e:
            logger.error(f"上采样点云失败: {e}")
            raise

    def process_and_scale_mesh(self, stl_file_path: str, target_volume: float,
                             num_clusters: int = 20000, sample_spacing: float = 0.1) -> Tuple[pv.PolyData, pv.PolyData, OriginalGeometryInfo, ScalingTransformation]:
        """
        统一的mesh处理和缩放函数，替代原有的upsample_and_scale_mesh

        Args:
            stl_file_path: STL文件路径
            target_volume: 目标体积
            num_clusters: 目标点数
            sample_spacing: 采样间距

        Returns:
            processed_mesh: 处理后的mesh
            original_mesh: 原始mesh（已缩放）
            original_info: 原始几何信息
            scaling_info: 缩放变换信息
        """
        try:
            # 1. 提取原始几何信息
            original_info = self.extract_original_geometry_info(
                stl_file_path,
                {'num_clusters': num_clusters, 'sample_spacing': sample_spacing}
            )

            # 2. 加载和处理mesh
            mesh_origin = pv.read(stl_file_path)
            processed_mesh = self.upsample_point_cloud(mesh_origin.points, num_clusters, sample_spacing)

            # 3. 应用平滑和填充
            processed_mesh.smooth_taubin(n_iter=10, pass_band=5, inplace=True)
            processed_mesh = processed_mesh.fill_holes(100)

            # 4. 计算缩放因子
            current_volume = processed_mesh.volume
            volume_scale_factor = (target_volume / current_volume) ** (1/3)

            # 5. 应用缩放
            processed_mesh.points *= volume_scale_factor
            mesh_origin.points *= volume_scale_factor

            # 6. 创建缩放变换信息
            scaling_info = ScalingTransformation(
                volume_scale_factor=volume_scale_factor,
                target_volume=target_volume,
                position_offset=[0.0, 0.0, 0.0],  # 初始位置偏移为0
                transformation_timestamp=str(np.datetime64('now'))
            )

            logger.info(f"Mesh处理完成: {stl_file_path}, 缩放因子: {volume_scale_factor:.4f}")

            return processed_mesh, mesh_origin, original_info, scaling_info

        except Exception as e:
            logger.error(f"Mesh处理和缩放失败 {stl_file_path}: {e}")
            raise

    def apply_position_offset(self, mesh: pv.PolyData, mesh_origin: pv.PolyData,
                            position_offset: np.ndarray, bubble_id: str,
                            scaling_info: ScalingTransformation) -> ScalingTransformation:
        """
        应用位置偏移，替代原有的mesh.points += point操作

        Args:
            mesh: 处理后的mesh
            mesh_origin: 原始mesh
            position_offset: 位置偏移向量
            bubble_id: 气泡ID
            scaling_info: 缩放信息

        Returns:
            updated_scaling_info: 更新后的缩放信息
        """
        try:
            # 应用位置偏移
            mesh.points += position_offset
            mesh_origin.points += position_offset

            # 更新缩放信息
            updated_scaling_info = ScalingTransformation(
                volume_scale_factor=scaling_info.volume_scale_factor,
                target_volume=scaling_info.target_volume,
                position_offset=list(position_offset),
                canvas_scale_factor=scaling_info.canvas_scale_factor,
                final_image_size=scaling_info.final_image_size,
                transformation_timestamp=scaling_info.transformation_timestamp
            )

            logger.debug(f"位置偏移应用完成: {bubble_id}, 偏移: {position_offset}")

            return updated_scaling_info

        except Exception as e:
            logger.error(f"应用位置偏移失败 {bubble_id}: {e}")
            raise

    def create_bubble_geometry_record(self, bubble_id: str, original_info: OriginalGeometryInfo,
                                    scaling_info: ScalingTransformation,
                                    final_mesh: pv.PolyData) -> BubbleGeometryRecord:
        """
        创建气泡几何记录

        Args:
            bubble_id: 气泡ID
            original_info: 原始几何信息
            scaling_info: 缩放信息
            final_mesh: 最终mesh

        Returns:
            BubbleGeometryRecord: 气泡几何记录
        """
        try:
            # 计算最终几何参数
            final_scaled_volume = final_mesh.volume
            final_scaled_surface_area = final_mesh.area
            final_scaled_bounds = list(final_mesh.bounds)
            final_position = list(final_mesh.center)

            record = BubbleGeometryRecord(
                bubble_id=bubble_id,
                original_info=original_info,
                scaling_info=scaling_info,
                final_scaled_volume=final_scaled_volume,
                final_scaled_surface_area=final_scaled_surface_area,
                final_scaled_bounds=final_scaled_bounds,
                final_position=final_position
            )

            # 如果启用尺寸追踪，保存记录
            if self.enable_size_tracking:
                self.geometry_records[bubble_id] = record

            return record

        except Exception as e:
            logger.error(f"创建气泡几何记录失败 {bubble_id}: {e}")
            raise

    def save_size_mapping(self) -> str:
        """
        保存尺寸映射到文件

        Returns:
            str: 映射文件路径
        """
        try:
            if not self.enable_size_tracking:
                logger.warning("尺寸追踪未启用，跳过保存")
                return ""

            def convert_to_serializable(obj):
                """递归转换numpy类型为Python原生类型"""
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, float) and np.isnan(obj):
                    return None  # 处理NaN
                else:
                    return obj

            # 转换为可序列化的格式
            mapping_data = {
                'metadata': {
                    'version': '2.0.0',
                    'total_bubbles': len(self.geometry_records),
                    'total_rendered': len(self.rendering_records),
                    'creation_time': str(np.datetime64('now')),
                    'true_size_rendering_enabled': True
                },
                'bubble_records': {},
                'rendering_records': convert_to_serializable(self.rendering_records.copy())  # 包含渲染记录
            }

            for bubble_id, record in self.geometry_records.items():
                mapping_data['bubble_records'][bubble_id] = {
                    'original_info': convert_to_serializable(asdict(record.original_info)),
                    'scaling_info': convert_to_serializable(asdict(record.scaling_info)),
                    'final_scaled_volume': float(record.final_scaled_volume),
                    'final_scaled_surface_area': float(record.final_scaled_surface_area),
                    'final_scaled_bounds': [float(b) for b in record.final_scaled_bounds],
                    'final_position': [float(p) for p in record.final_position]
                }

            # 保存到文件
            with open(self.size_mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, indent=2, ensure_ascii=False)

            logger.info(f"尺寸映射已保存: {self.size_mapping_file}")
            return str(self.size_mapping_file)

        except Exception as e:
            logger.error(f"保存尺寸映射失败: {e}")
            raise

    def load_size_mapping(self) -> bool:
        """
        从文件加载尺寸映射

        Returns:
            bool: 是否加载成功
        """
        try:
            if not self.size_mapping_file.exists():
                logger.info("尺寸映射文件不存在")
                return False

            with open(self.size_mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)

            # 重建几何记录
            self.geometry_records.clear()

            for bubble_id, record_data in mapping_data.get('bubble_records', {}).items():
                original_info = OriginalGeometryInfo(**record_data['original_info'])
                scaling_info = ScalingTransformation(**record_data['scaling_info'])

                record = BubbleGeometryRecord(
                    bubble_id=bubble_id,
                    original_info=original_info,
                    scaling_info=scaling_info,
                    final_scaled_volume=record_data['final_scaled_volume'],
                    final_scaled_surface_area=record_data['final_scaled_surface_area'],
                    final_scaled_bounds=record_data['final_scaled_bounds'],
                    final_position=record_data['final_position']
                )

                self.geometry_records[bubble_id] = record

            logger.info(f"尺寸映射加载完成，共 {len(self.geometry_records)} 条记录")
            return True

        except Exception as e:
            logger.error(f"加载尺寸映射失败: {e}")
            return False

    def get_original_size_info(self, bubble_id: str) -> Optional[Dict[str, Any]]:
        """
        获取气泡的原始尺寸信息

        Args:
            bubble_id: 气泡ID

        Returns:
            Dict: 原始尺寸信息，如果不存在则返回None
        """
        if bubble_id not in self.geometry_records:
            return None

        record = self.geometry_records[bubble_id]
        return {
            'stl_file_path': record.original_info.stl_file_path,
            'original_volume': record.original_info.original_volume,
            'original_surface_area': record.original_info.original_surface_area,
            'original_bounds': record.original_info.original_bounds,
            'original_ellipsoid_radii': record.original_info.original_ellipsoid_radii,
            'volume_scale_factor': record.scaling_info.volume_scale_factor,
            'final_scaled_volume': record.final_scaled_volume,
            'size_consistency_ratio': record.final_scaled_volume / record.original_info.original_volume if record.original_info.original_volume > 0 else 0
        }

    def validate_size_consistency(self, tolerance: float = 0.1) -> Dict[str, Any]:
        """
        验证尺寸一致性

        Args:
            tolerance: 容差范围

        Returns:
            Dict: 验证结果
        """
        if not self.geometry_records:
            return {'status': 'no_data', 'message': '没有可验证的数据'}

        inconsistent_bubbles = []
        total_bubbles = len(self.geometry_records)

        for bubble_id, record in self.geometry_records.items():
            expected_volume = record.original_info.original_volume * (record.scaling_info.volume_scale_factor ** 3)
            actual_volume = record.final_scaled_volume

            if abs(expected_volume - actual_volume) / expected_volume > tolerance:
                inconsistent_bubbles.append({
                    'bubble_id': bubble_id,
                    'expected_volume': expected_volume,
                    'actual_volume': actual_volume,
                    'relative_error': abs(expected_volume - actual_volume) / expected_volume
                })

        return {
            'status': 'completed',
            'total_bubbles': total_bubbles,
            'inconsistent_count': len(inconsistent_bubbles),
            'consistency_rate': (total_bubbles - len(inconsistent_bubbles)) / total_bubbles,
            'inconsistent_bubbles': inconsistent_bubbles
        }

    def add_rendering_record(self, bubble_id: str, rendering_info: Dict[str, Any]) -> None:
        """
        添加渲染记录

        Args:
            bubble_id: 气泡唯一标识
            rendering_info: 渲染信息字典
        """
        if self.enable_size_tracking:
            self.rendering_records[bubble_id] = rendering_info
            logger.debug(f"添加渲染记录: {bubble_id}")

    def get_rendering_record(self, bubble_id: str) -> Optional[Dict[str, Any]]:
        """
        获取渲染记录

        Args:
            bubble_id: 气泡唯一标识

        Returns:
            渲染记录字典或None
        """
        return self.rendering_records.get(bubble_id)

    def get_size_consistency_report(self) -> Dict[str, Any]:
        """
        获取尺寸一致性报告（包含渲染信息）

        Returns:
            包含渲染信息的一致性报告
        """
        base_report = self.validate_size_consistency()

        # 添加渲染统计信息
        if self.rendering_records:
            true_size_count = sum(1 for record in self.rendering_records.values()
                                if record.get('is_true_size_rendering', False))
            small_bubble_count = sum(1 for record in self.rendering_records.values()
                                   if record.get('original_canvas_size', 128) < 64)

            base_report['rendering_stats'] = {
                'total_rendered': len(self.rendering_records),
                'true_size_rendered': true_size_count,
                'small_bubbles_preserved': small_bubble_count,
                'true_size_rate': true_size_count / len(self.rendering_records) if self.rendering_records else 0
            }

        return base_report
