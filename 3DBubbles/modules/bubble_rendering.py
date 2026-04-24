# bubble_rendering.py
"""
气泡渲染模块

该模块包含气泡渲染相关的函数，主要用于将3D气泡mesh渲染为2D图像，
包括单气泡渲染、像素着色、图像后处理等功能。

主要功能：
- 单气泡渲染
- 像素着色和映射
- 图像后处理和标准化
- 椭球拟合和参数计算
"""

import os
import cv2
import numpy as np
import math
import gc
from scipy.ndimage import median_filter, gaussian_filter
from scipy.spatial import ConvexHull


def cv2_enhance_contrast(img, factor):
    """
    增强图像对比度
    
    Args:
        img: 输入图像
        factor: 对比度增强因子
    
    Returns:
        增强对比度后的图像
    """
    mean = np.uint8(cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0])
    img_deg = np.ones_like(img) * mean
    return cv2.addWeighted(img, factor, img_deg, 1-factor, 0.0)


def ellipsoid_fit(X):
    """
    拟合椭球参数，参考2-projection3d-2d&characterization.py
    
    Args:
        X: 点云数据
    
    Returns:
        tuple: (center, evecs, radii, v) 椭球参数
    """
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array([x * x + y * y - 2 * z * z,
                 x * x + z * z - 2 * y * y,
                 2 * x * y,
                 2 * x * z,
                 2 * y * z,
                 2 * x,
                 2 * y,
                 2 * z,
                 1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                    [v[3], v[1], v[5], v[7]],
                    [v[4], v[5], v[2], v[8]],
                    [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)

    return center, evecs, radii, v


def calculate_dynamic_canvas_size(points, scale_factor=1.3, min_canvas_size=None, enable_true_size=True):
    """
    根据气泡点云计算动态画布大小（支持真实尺寸渲染）

    Args:
        points: 气泡的点云数据
        scale_factor: 缩放因子，确保气泡完全包含在画布内
        min_canvas_size: 最小画布大小（None表示无限制，允许真实尺寸）
        enable_true_size: 是否启用真实尺寸渲染（移除64像素限制）

    Returns:
        canvas_size: 正方形画布的边长（像素）
        scale: 从3D坐标到像素坐标的缩放比例
        offset_x, offset_y: 画布中心偏移量
        size_info: 尺寸信息字典
    """
    # 计算点云的边界框
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    # 计算实际尺寸
    width = x_max - x_min
    height = y_max - y_min

    # 选择较大的尺寸作为基准，确保正方形画布
    max_size = max(width, height)

    # 应用缩放因子
    canvas_size_float = max_size * scale_factor * 100  # 基础缩放100

    # 向上取整到合适的像素值，并确保是偶数（便于中心对齐）
    canvas_size = int(np.ceil(canvas_size_float))
    if canvas_size % 2 != 0:
        canvas_size += 1

    # 记录原始计算的画布尺寸
    original_calculated_size = canvas_size

    # 尺寸限制处理
    if enable_true_size:
        # 启用真实尺寸：移除64像素硬编码限制，但保持合理的最小值
        if min_canvas_size is not None:
            canvas_size = max(canvas_size, min_canvas_size)
        else:
            # 设置一个非常小的最小值，确保至少有几个像素用于渲染
            canvas_size = max(canvas_size, 4)  # 最小4×4像素，保证基本可见性
    else:
        # 保持原有的64像素限制（向后兼容）
        canvas_size = max(canvas_size, 64)

    # 计算缩放比例
    scale = canvas_size / (max_size * scale_factor)

    # 计算偏移量，使气泡居中
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    offset_x = canvas_size / 2 / scale - center_x
    offset_y = canvas_size / 2 / scale - center_y

    # 创建尺寸信息记录
    size_info = {
        'original_calculated_size': original_calculated_size,
        'final_canvas_size': canvas_size,
        'was_size_limited': canvas_size != original_calculated_size,
        'size_limit_applied': min_canvas_size if min_canvas_size and canvas_size == min_canvas_size else None,
        'physical_width': width,
        'physical_height': height,
        'max_physical_size': max_size,
        'scale_factor_applied': scale_factor,
        'enable_true_size': enable_true_size
    }

    return canvas_size, scale, offset_x, offset_y, size_info


def render_single_bubble(mesh, bubble_idx, point_fibonacci, v, output_dir, alpha=8, truncation=0.75,
                        enable_true_size=True, min_canvas_size=None):
    """
    渲染单个气泡（支持真实尺寸渲染）

    Args:
        mesh: 单个气泡的mesh对象
        bubble_idx: 气泡索引
        point_fibonacci: 投影点
        v: 旋转向量
        output_dir: 输出目录
        alpha: 角度权重指数
        truncation: 截断值
        enable_true_size: 是否启用真实尺寸渲染（移除64像素限制）
        min_canvas_size: 最小画布大小（None表示无限制）

    Returns:
        bubble_params: 包含3D和2D参数的字典
    """
    try:
        # 旋转mesh到指定视角
        rot_mesh = mesh.rotate_vector(np.cross(point_fibonacci, v),
                                     np.arccos(np.dot(point_fibonacci, v)) * 180 / np.pi,
                                     inplace=False)

        # 计算3D参数
        volume_3d = mesh.volume
        surface_area_3d = mesh.area

        # 拟合椭球
        try:
            center, evecs, radii, v_ellipsoid = ellipsoid_fit(mesh.points)
            a, b, c = sorted(radii, reverse=True)

            # 计算3D形状参数
            EI_3D = b / a if a > 0 else 0
            FI_3D = c / b if b > 0 else 0
            AR_3D = (EI_3D + FI_3D) / 2

            # 计算球形度
            Sphericity_3D = (np.pi**(1/3) * (6 * volume_3d)**(2/3)) / surface_area_3d if surface_area_3d > 0 else 0

            # 计算凸包凸度
            hull = ConvexHull(mesh.points)
            Convexity_3D = volume_3d / hull.volume if hull.volume > 0 else 0

        except Exception as e:
            print(f"椭球拟合失败，气泡 {bubble_idx}: {e}")
            a, b, c = 0, 0, 0
            EI_3D = FI_3D = AR_3D = Sphericity_3D = Convexity_3D = 0

        # 获取旋转后的点和法向量
        points = rot_mesh.points
        normals = rot_mesh.point_normals

        # 法向量过滤（参考2-projection3d-2d&characterization.py）
        mask = normals[:, 2] > -0
        filtered_points = points[mask]
        filtered_normals = normals[mask]

        if len(filtered_points) == 0:
            print(f"警告：气泡 {bubble_idx} 没有可见的点")
            return None

        # 计算动态画布大小（启用真实尺寸渲染）
        canvas_size, scale, offset_x, offset_y, size_info = calculate_dynamic_canvas_size(
            filtered_points, scale_factor=1.3, min_canvas_size=min_canvas_size, enable_true_size=enable_true_size
        )

        # 角度权重计算
        angles = filtered_normals[:, 2] / np.linalg.norm(filtered_normals, axis=1)
        M = angles ** alpha

        # 初始化画布
        mapped_points = np.ones((canvas_size, canvas_size))

        # 双线性插值映射（参考2-projection3d-2d&characterization.py的逻辑）
        for i in range(filtered_points.shape[0]):
            x, y = filtered_points[i, 0] + offset_x, filtered_points[i, 1] + offset_y
            mapped_x, mapped_y = x * scale, y * scale

            # 边界检查
            if mapped_x < 0 or mapped_x >= canvas_size - 1 or mapped_y < 0 or mapped_y >= canvas_size - 1:
                continue

            # 双线性插值
            l1 = np.square(mapped_x - np.floor(mapped_x)) + np.square(mapped_y - np.floor(mapped_y))
            l2 = np.square(mapped_x - np.ceil(mapped_x)) + np.square(mapped_y - np.floor(mapped_y))
            l3 = np.square(mapped_x - np.floor(mapped_x)) + np.square(mapped_y - np.ceil(mapped_y))
            l4 = np.square(mapped_x - np.ceil(mapped_x)) + np.square(mapped_y - np.ceil(mapped_y))
            total = l1 + l2 + l3 + l4

            if total > 0:
                # 基本四个点
                for x_idx, y_idx, l in [(int(np.floor(mapped_x)), int(np.floor(mapped_y)), l1),
                                       (int(np.ceil(mapped_x)), int(np.floor(mapped_y)), l2),
                                       (int(np.floor(mapped_x)), int(np.ceil(mapped_y)), l3),
                                       (int(np.ceil(mapped_x)), int(np.ceil(mapped_y)), l4)]:
                    if 0 <= x_idx < canvas_size and 0 <= y_idx < canvas_size:
                        if mapped_points[x_idx, y_idx] == 1:
                            mapped_points[x_idx, y_idx] = 0
                        mapped_points[x_idx, y_idx] += M[i] * (total - l) / total

        # 合并3D参数和尺寸信息
        bubble_3d_params = {
            'volume_3d': volume_3d,
            'surface_area_3d': surface_area_3d,
            'a': a, 'b': b, 'c': c,
            'EI_3D': EI_3D, 'FI_3D': FI_3D, 'AR_3D': AR_3D,
            'Sphericity_3D': Sphericity_3D,
            'Convexity_3D': Convexity_3D
        }

        # 添加尺寸信息
        bubble_3d_params.update(size_info)

        return canvas_size, mapped_points, scale, offset_x, offset_y, bubble_3d_params

    except Exception as e:
        print(f"渲染气泡 {bubble_idx} 时发生错误: {e}")
        return None


def process_single_bubble_rendering(canvas_size, mapped_points, scale, offset_x, offset_y,
                                  bubble_params, bubble_idx, output_path, truncation=0.75,
                                  target_size=128, enable_size_tracking=True):
    """
    处理单气泡渲染的后处理和保存（支持真实尺寸处理）

    Args:
        canvas_size: 原始画布大小
        mapped_points: 映射后的点数据
        scale: 缩放因子
        offset_x, offset_y: 偏移量
        bubble_params: 3D参数字典（包含尺寸信息）
        bubble_idx: 气泡索引
        output_path: 输出路径
        truncation: 截断值
        target_size: 目标标准化尺寸（默认128×128）
        enable_size_tracking: 是否启用尺寸追踪

    Returns:
        final_params: 包含3D和2D参数的完整字典
    """
    try:
        # 记录原始画布大小
        original_canvas_size = canvas_size

        # 找到背景像素（值为1的像素）
        indices = np.where(mapped_points == 1)
        mapped_points[indices] = 0

        # 高斯滤波
        mapped_points = gaussian_filter(mapped_points, sigma=1)

        # 归一化和截断
        if mapped_points.max() > 0:
            mapped_points_normalized = np.clip(mapped_points / mapped_points.max(), 0, truncation) / truncation
        else:
            mapped_points_normalized = mapped_points
        mapped_points_normalized[indices] = 1

        # 进一步滤波处理
        mapped_points_normalized = gaussian_filter(mapped_points_normalized, sigma=0.75)
        mapped_points_normalized = median_filter(mapped_points_normalized, size=5)

        # 最终归一化
        if mapped_points_normalized.max() > mapped_points_normalized.min():
            mapped_points_normalized = (mapped_points_normalized - mapped_points_normalized.min()) / \
                                     (mapped_points_normalized.max() - mapped_points_normalized.min())

        # 转换为图像格式
        mapped_points_normalized = (mapped_points_normalized * 255).astype(np.uint8).T
        mapped_points_normalized = cv2.cvtColor(mapped_points_normalized, cv2.COLOR_GRAY2RGB)
        mapped_points_normalized = cv2_enhance_contrast(mapped_points_normalized, 2)
        mapped_points_normalized = cv2.transpose(mapped_points_normalized)

        # 图像尺寸标准化：基于原始尺寸选择处理策略（支持真实尺寸）
        processing_method = ""
        resize_scale_factor = 1.0
        padding_info = None

        if original_canvas_size <= target_size:
            # 小尺寸图像：使用填充策略保持原始尺寸
            processing_method = "padding"
            resize_scale_factor = 1.0

            # 计算填充量
            pad_total = target_size - original_canvas_size
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left

            # 记录填充信息
            padding_info = {
                'pad_top': pad_top,
                'pad_bottom': pad_bottom,
                'pad_left': pad_left,
                'pad_right': pad_right,
                'pad_total': pad_total
            }

            # 使用白色填充（255, 255, 255）
            mapped_points_normalized = cv2.copyMakeBorder(
                mapped_points_normalized,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255]
            )

            # 对于非常小的气泡，记录真实尺寸信息
            if enable_size_tracking and original_canvas_size < 64:
                print(f"  小气泡真实尺寸保持: {bubble_idx} - 原始{original_canvas_size}×{original_canvas_size}像素，填充到{target_size}×{target_size}像素")
        else:
            # 大尺寸图像：使用缩放策略
            processing_method = "resize"
            resize_scale_factor = target_size / original_canvas_size

            # 使用双线性插值进行resize
            mapped_points_normalized = cv2.resize(mapped_points_normalized, (target_size, target_size),
                                                interpolation=cv2.INTER_LINEAR)

            if enable_size_tracking:
                print(f"  大气泡缩放处理: {bubble_idx} - 原始{original_canvas_size}×{original_canvas_size}像素，缩放到{target_size}×{target_size}像素（缩放因子: {resize_scale_factor:.3f}）")

        # 保存图像
        cv2.imwrite(output_path, mapped_points_normalized)

        # 计算2D参数（基于二值化图像）
        gray_img = cv2.cvtColor(mapped_points_normalized, cv2.COLOR_RGB2GRAY)
        _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)

        # 查找轮廓
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # 选择最大的轮廓
            cnt = max(contours, key=cv2.contourArea)

            # 计算2D参数
            area_2d_pixels = cv2.contourArea(cnt)
            area_2d = area_2d_pixels / (scale ** 2)  # 转换为实际面积

            perimeter = cv2.arcLength(cnt, True)

            # 计算凸包
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            convexity_2d = float(area_2d_pixels) / hull_area if hull_area > 0 else 0

            # 拟合椭圆
            try:
                if len(cnt) >= 5:  # 至少需要5个点来拟合椭圆
                    (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                    MA /= scale  # 转换为实际尺寸
                    ma /= scale
                    aspect_ratio_2d = MA / ma if ma > 0 else 0

                    # 计算圆度
                    circularity_2d = (2 * np.sqrt(np.pi * area_2d_pixels)) / perimeter if perimeter > 0 else 0

                    # 计算椭圆面积
                    ellipse_area = np.pi * MA * ma / 4
                    solidity_2d = 1 - abs((area_2d / ellipse_area) - 1) if ellipse_area > 0 else 0
                else:
                    MA = ma = aspect_ratio_2d = circularity_2d = solidity_2d = 0

            except Exception as e:
                print(f"椭圆拟合失败，气泡 {bubble_idx}: {e}")
                MA = ma = aspect_ratio_2d = circularity_2d = solidity_2d = 0
        else:
            area_2d = convexity_2d = MA = ma = aspect_ratio_2d = circularity_2d = solidity_2d = 0

        # 合并所有参数（包含完整的尺寸追踪信息）
        final_params = bubble_params.copy()

        # 添加2D渲染参数
        final_params.update({
            'canvas_size': canvas_size,  # 这里现在是原始画布大小
            'original_canvas_size': original_canvas_size,  # 原始画布边长
            'resized_canvas_size': target_size,  # 标准化后的尺寸（128）
            'processing_method': processing_method,  # 处理方式：padding 或 resize
            'resize_scale_factor': resize_scale_factor,  # resize的缩放比例（填充时为1.0）
            'scale_factor': scale,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'area_2d': area_2d,
            'MA': MA,
            'ma': ma,
            'aspect_ratio_2d': aspect_ratio_2d,
            'circularity_2d': circularity_2d,
            'solidity_2d': solidity_2d,
            'convexity_2d': convexity_2d
        })

        # 添加填充信息（如果使用了填充策略）
        if padding_info:
            final_params['padding_info'] = padding_info

        # 添加尺寸一致性信息
        if enable_size_tracking:
            # 计算尺寸保真度（原始尺寸在最终图像中的占比）
            size_fidelity = (original_canvas_size / target_size) ** 2  # 面积比

            # 判断是否为真实尺寸渲染（未被64像素限制影响）
            is_true_size_rendering = bubble_params.get('enable_true_size', False) and not bubble_params.get('was_size_limited', False)

            final_params.update({
                'size_fidelity': size_fidelity,
                'is_true_size_rendering': is_true_size_rendering,
                'size_tracking_enabled': True,
                'original_to_final_ratio': original_canvas_size / target_size
            })

            # 如果是小气泡的真实尺寸渲染，添加特殊标记
            if is_true_size_rendering and original_canvas_size < 64:
                final_params['small_bubble_true_size'] = True
                final_params['size_improvement_note'] = f"小气泡保持真实尺寸：{original_canvas_size}×{original_canvas_size}像素（未被强制放大到64×64）"

        return final_params

    except Exception as e:
        print(f"处理气泡 {bubble_idx} 的后处理时发生错误: {e}")
        return bubble_params
    finally:
        # 清理内存
        if 'mapped_points_normalized' in locals():
            del mapped_points_normalized
        if 'gray_img' in locals():
            del gray_img
        if 'binary_img' in locals():
            del binary_img
        gc.collect()


def render_single_bubble_with_unified_scaling(mesh, bubble_idx, point_fibonacci, v, output_dir,
                                            scaling_manager=None, bubble_id=None,
                                            alpha=8, truncation=0.75, enable_true_size=True):
    """
    使用统一缩放系统渲染单个气泡

    Args:
        mesh: 单个气泡的mesh对象
        bubble_idx: 气泡索引
        point_fibonacci: 投影点
        v: 旋转向量
        output_dir: 输出目录
        scaling_manager: 统一缩放管理器实例
        bubble_id: 气泡唯一标识
        alpha: 角度权重指数
        truncation: 截断值
        enable_true_size: 是否启用真实尺寸渲染

    Returns:
        final_params: 包含完整尺寸追踪信息的参数字典
    """
    try:
        # 使用新的渲染函数
        result = render_single_bubble(
            mesh, bubble_idx, point_fibonacci, v, output_dir,
            alpha=alpha, truncation=truncation,
            enable_true_size=enable_true_size, min_canvas_size=None
        )

        if result is None:
            return None

        canvas_size, mapped_points, scale, offset_x, offset_y, bubble_3d_params = result

        # 生成输出路径
        output_path = f"{output_dir}/bubble_{bubble_idx:04d}.png"

        # 处理渲染结果
        final_params = process_single_bubble_rendering(
            canvas_size, mapped_points, scale, offset_x, offset_y,
            bubble_3d_params, bubble_idx, output_path, truncation=truncation,
            target_size=128, enable_size_tracking=True
        )

        # 如果提供了统一缩放管理器，记录尺寸信息
        if scaling_manager and bubble_id:
            try:
                # 创建渲染尺寸记录
                rendering_info = {
                    'bubble_id': bubble_id,
                    'original_canvas_size': final_params.get('original_canvas_size', canvas_size),
                    'final_canvas_size': 128,  # 标准化尺寸
                    'processing_method': final_params.get('processing_method', 'unknown'),
                    'size_fidelity': final_params.get('size_fidelity', 1.0),
                    'is_true_size_rendering': final_params.get('is_true_size_rendering', False),
                    'rendering_timestamp': str(np.datetime64('now'))
                }

                # 保存到缩放管理器（如果支持渲染记录）
                if hasattr(scaling_manager, 'add_rendering_record'):
                    scaling_manager.add_rendering_record(bubble_id, rendering_info)

            except Exception as e:
                print(f"记录渲染信息失败 {bubble_id}: {e}")

        return final_params

    except Exception as e:
        print(f"统一缩放系统渲染气泡 {bubble_idx} 失败: {e}")
        return None


def pixel_coloring(masks_path, alpha, all_points, all_vectors, min_x, min_y, scale, canvas_range_x, canvas_range_y):
    """
    像素着色函数，用于多气泡渲染的像素映射

    Args:
        masks_path: 掩码保存路径
        alpha: 角度权重指数
        all_points: 所有气泡的点云数据
        all_vectors: 所有气泡的法向量数据
        min_x, min_y: 最小坐标值
        scale: 缩放比例
        canvas_range_x, canvas_range_y: 画布范围

    Returns:
        tuple: (bboxes, bub_conts, mapped_points) 边界框、轮廓和映射点
    """
    bboxes = []
    bub_conts = []
    mapped_points = np.ones((canvas_range_x, canvas_range_y))

    for points, vectors in zip(all_points, all_vectors):      # 每一个points都是一个气泡
        mask = vectors[:, 2] > -99099999
        filtered_points = points[mask]
        filtered_normals = vectors[mask]
        angles = filtered_normals[:, 2] / np.linalg.norm(filtered_normals, axis=1)
        M = angles ** alpha
        min_mapped_x, max_mapped_x, min_mapped_y, max_mapped_y = float('inf'), float('-inf'), float('inf'), float('-inf')

        for i in range(filtered_points.shape[0]):
            mapped_x, mapped_y = (filtered_points[i, 0] - min_x) * scale, (filtered_points[i, 1] - min_y) * scale
            min_mapped_x, max_mapped_x = min(min_mapped_x, mapped_x), max(max_mapped_x, mapped_x)
            min_mapped_y, max_mapped_y = min(min_mapped_y, mapped_y), max(max_mapped_y, mapped_y)
            l1 = np.square(mapped_x - np.floor(mapped_x)) + np.square(mapped_y - np.floor(mapped_y))
            l2 = np.square(mapped_x - np.ceil(mapped_x)) + np.square(mapped_y - np.floor(mapped_y))
            l3 = np.square(mapped_x - np.floor(mapped_x)) + np.square(mapped_y - np.ceil(mapped_y))
            l4 = np.square(mapped_x - np.ceil(mapped_x)) + np.square(mapped_y - np.ceil(mapped_y))
            total = l1 + l2 + l3 + l4

            for x, y, l in [(int(np.floor(mapped_x)), int(np.floor(mapped_y)), l1),
                            (int(np.ceil(mapped_x)), int(np.floor(mapped_y)), l2),
                            (int(np.floor(mapped_x)), int(np.ceil(mapped_y)), l3),
                            (int(np.ceil(mapped_x)), int(np.ceil(mapped_y)), l4),
                            (int(np.floor(mapped_x)) - 1, int(np.floor(mapped_y)), l1),
                            (int(np.floor(mapped_x)), int(np.floor(mapped_y)) - 1, l1),
                            (int(np.ceil(mapped_x)) + 1, int(np.floor(mapped_y)), l2),
                            (int(np.ceil(mapped_x)), int(np.floor(mapped_y)) - 1, l2),
                            (int(np.floor(mapped_x)) - 1, int(np.ceil(mapped_y)), l3),
                            (int(np.floor(mapped_x)), int(np.ceil(mapped_y)) + 1, l3),
                            (int(np.ceil(mapped_x)) + 1, int(np.ceil(mapped_y)), l4),
                            (int(np.ceil(mapped_x)), int(np.ceil(mapped_y)) + 1, l4)]:
                if x < 0 or x >= canvas_range_x or y < 0 or y >= canvas_range_y:
                    continue
                if mapped_points[x, y] != 1:
                    mapped_points[x, y] = 1
                mapped_points[x, y] += M[i] * (total - l) / total

        bboxes.append((min_mapped_x, max_mapped_x, min_mapped_y, max_mapped_y))
        mask_image = np.zeros((canvas_range_x, canvas_range_y), dtype=np.uint8)

        for i in range(filtered_points.shape[0]):
            mapped_x, mapped_y = (filtered_points[i, 0] - min_x) * scale, (filtered_points[i, 1] - min_y) * scale
            if 0 <= int(mapped_x) < canvas_range_x and 0 <= int(mapped_y) < canvas_range_y:
                mask_image[int(mapped_x), int(mapped_y)] = 255

        mask_image_path = os.path.join(masks_path, f'mask_{len(bboxes)}.png')
        kernel = np.ones((5,5),np.uint8)
        mask_image = cv2.dilate(mask_image, kernel, iterations=1)
        cv2.imwrite(mask_image_path, mask_image)

        ret, mask = cv2.threshold(mask_image, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours) >= 1:
            second_contour = contours[0]
            bub_conts.append(second_contour)

    return bboxes, bub_conts, mapped_points
