# coordinate_transform.py
"""
坐标变换模块

该模块包含3D到2D坐标变换、图像旋转、深度排序等相关函数，
主要用于处理3D气泡在不同视角下的投影和变换。

主要功能：
- 3D到2D坐标变换
- 图像旋转变换
- 气泡深度排序
- 坐标系变换计算
"""

import cv2
import numpy as np


def rotate_image(image, angle, borderValue=(255, 255, 255)):
    """
    旋转图像

    Args:
        image: 输入图像
        angle: 旋转角度（度数）
        borderValue: 填充颜色，默认白色

    Returns:
        rotated_image: 旋转后的图像
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows), borderValue=borderValue)
    return rotated_image


def transform_3d_to_2d(point_3d, projection_point, v_vector, canvas_size, scale_factor):
    """
    将3D世界坐标转换为2D画布坐标

    Args:
        point_3d: 3D坐标点 [x, y, z]
        projection_point: 投影点（视角方向）
        v_vector: 旋转向量
        canvas_size: 画布大小
        scale_factor: 缩放因子

    Returns:
        tuple: (x_2d, y_2d, depth) 2D坐标和深度值
    """
    try:
        # 应用旋转变换（与render_single_bubble中的旋转一致）
        rotation_axis = np.cross(projection_point, v_vector)
        rotation_angle = np.arccos(np.dot(projection_point, v_vector)) * 180 / np.pi

        # 创建旋转矩阵
        axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)
        angle_rad = np.radians(rotation_angle)

        # 使用罗德里格旋转公式
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # 构建旋转矩阵
        ux, uy, uz = axis_normalized
        rotation_matrix = np.array([
            [cos_angle + ux*ux*(1-cos_angle), ux*uy*(1-cos_angle) - uz*sin_angle, ux*uz*(1-cos_angle) + uy*sin_angle],
            [uy*ux*(1-cos_angle) + uz*sin_angle, cos_angle + uy*uy*(1-cos_angle), uy*uz*(1-cos_angle) - ux*sin_angle],
            [uz*ux*(1-cos_angle) - uy*sin_angle, uz*uy*(1-cos_angle) + ux*sin_angle, cos_angle + uz*uz*(1-cos_angle)]
        ])

        # 应用旋转
        rotated_point = np.dot(rotation_matrix, point_3d)

        # 投影到2D（使用x, y坐标，z作为深度）
        x_2d = rotated_point[0] * scale_factor + canvas_size // 2
        y_2d = rotated_point[1] * scale_factor + canvas_size // 2
        depth = rotated_point[2]  # z坐标作为深度值

        return x_2d, y_2d, depth

    except Exception as e:
        print(f"3D到2D坐标变换失败: {e}")
        return None, None, None


def sort_bubbles_by_depth(bubble_positions, projection_point, v_vector):
    """
    根据深度对气泡进行排序

    Args:
        bubble_positions: 气泡位置列表 [(x, y, z), ...]
        projection_point: 投影点（视角方向）
        v_vector: 旋转向量

    Returns:
        list: 按深度排序的气泡索引列表（从远到近）
    """
    try:
        bubble_depths = []

        for i, pos in enumerate(bubble_positions):
            _, _, depth = transform_3d_to_2d(pos, projection_point, v_vector, 1000, 100)
            if depth is not None:
                bubble_depths.append((i, depth))

        # 按深度排序（从远到近，即z值从小到大）
        bubble_depths.sort(key=lambda x: x[1])

        return [idx for idx, _ in bubble_depths]

    except Exception as e:
        print(f"气泡深度排序失败: {e}")
        return list(range(len(bubble_positions)))


def transform_3d_to_pixel_coloring_coords(point_3d, projection_point, v_vector, min_x, min_y, scale_factor):
    """
    将3D坐标变换为与pixel_coloring函数一致的坐标系统

    该函数实现与预渲染阶段完全一致的坐标变换逻辑：
    1. 根据投影点和旋转向量计算旋转矩阵
    2. 将3D坐标应用旋转变换（将投影点旋转至Z轴正方向）
    3. 取变换后坐标的X和Y分量进行2D投影
    4. 应用与pixel_coloring函数相同的缩放和偏移

    Args:
        point_3d: 3D坐标点 [x, y, z]
        projection_point: 投影点（视角方向）
        v_vector: 旋转向量
        min_x, min_y: 最小坐标值（来自pixel_coloring的坐标范围）
        scale_factor: 缩放因子

    Returns:
        tuple: (mapped_x, mapped_y, depth) 映射后的坐标和深度值
               如果变换失败则返回 (None, None, None)
    """
    try:
        # 输入验证
        if point_3d is None or projection_point is None or v_vector is None:
            return None, None, None

        point_3d = np.array(point_3d)
        projection_point = np.array(projection_point)
        v_vector = np.array(v_vector)

        # 计算旋转轴和角度（与render_single_bubble中的逻辑完全一致）
        rotation_axis = np.cross(projection_point, v_vector)

        # 检查旋转轴是否为零向量（投影点和旋转向量平行的情况）
        axis_norm = np.linalg.norm(rotation_axis)
        if axis_norm < 1e-10:
            # 如果投影点和旋转向量平行，不需要旋转
            rotated_point = point_3d.copy()
        else:
            # 计算旋转角度
            dot_product = np.dot(projection_point, v_vector)
            # 确保点积在有效范围内，避免数值误差
            dot_product = np.clip(dot_product, -1.0, 1.0)
            rotation_angle = np.arccos(dot_product) * 180 / np.pi

            # 创建旋转矩阵（使用罗德里格旋转公式）
            axis_normalized = rotation_axis / axis_norm
            angle_rad = np.radians(rotation_angle)

            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)

            ux, uy, uz = axis_normalized
            rotation_matrix = np.array([
                [cos_angle + ux*ux*(1-cos_angle), ux*uy*(1-cos_angle) - uz*sin_angle, ux*uz*(1-cos_angle) + uy*sin_angle],
                [uy*ux*(1-cos_angle) + uz*sin_angle, cos_angle + uy*uy*(1-cos_angle), uy*uz*(1-cos_angle) - ux*sin_angle],
                [uz*ux*(1-cos_angle) - uy*sin_angle, uz*uy*(1-cos_angle) + ux*sin_angle, cos_angle + uz*uz*(1-cos_angle)]
            ])

            # 应用旋转变换
            rotated_point = np.dot(rotation_matrix, point_3d)

        # 使用与pixel_coloring完全相同的坐标映射公式
        # 参考bubble_rendering.py第445行：mapped_x, mapped_y = (filtered_points[i, 0] - min_x) * scale, (filtered_points[i, 1] - min_y) * scale
        mapped_x = (rotated_point[0] - min_x) * scale_factor
        mapped_y = (rotated_point[1] - min_y) * scale_factor
        depth = rotated_point[2]  # Z坐标作为深度值

        return mapped_x, mapped_y, depth

    except Exception as e:
        print(f"3D到pixel_coloring坐标变换失败: {e}")
        print(f"  输入参数: point_3d={point_3d}, projection_point={projection_point}, v_vector={v_vector}")
        print(f"  坐标参数: min_x={min_x}, min_y={min_y}, scale_factor={scale_factor}")
        return None, None, None
