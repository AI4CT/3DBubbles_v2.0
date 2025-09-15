# flow_composition.py
"""
流场合成模块

该模块包含流场气泡合成相关的函数，主要用于将单个气泡图像
按照3D位置信息合成到2D画布上，创建完整的流场图像。

主要功能：
- 流场气泡合成
- 气泡图像合成到画布
- 气泡位置信息加载
- 深度排序合成
- 完整流场合成（包含位置采样、重叠检测、视觉增强等）
"""

import os
import csv
import cv2
import numpy as np
import random
import math
from scipy.ndimage import median_filter, gaussian_filter

# 导入相关模块
try:
    from .coordinate_transform import transform_3d_to_2d, sort_bubbles_by_depth, transform_3d_to_pixel_coloring_coords
except ImportError:
    from coordinate_transform import transform_3d_to_2d, sort_bubbles_by_depth, transform_3d_to_pixel_coloring_coords


def calculate_dynamic_canvas_size(bubble_positions, projection_point, v_vector, scale_factor,
                                 bubble_size=128, margin_ratio=0.2, min_size=512, max_size=8192,
                                 volume_size_x=None, volume_size_y=None, volume_height=None):
    """
    根据气泡3D位置和流场物理尺寸动态计算合适的画布尺寸

    Args:
        bubble_positions: 气泡位置列表 [(x, y, z), ...]
        projection_point: 投影点（视角方向）
        v_vector: 旋转向量
        scale_factor: 缩放因子
        bubble_size: 单个气泡图像大小（用于计算边距）
        margin_ratio: 边距比例（相对于内容范围）
        min_size: 最小画布尺寸
        max_size: 最大画布尺寸
        volume_size_x: 流场X方向物理尺寸
        volume_size_y: 流场Y方向物理尺寸
        volume_height: 流场Z方向物理尺寸

    Returns:
        int: 计算出的画布尺寸（正方形）
    """
    try:
        if not bubble_positions:
            return min_size

        # 如果提供了物理尺寸参数，优先使用基于物理尺寸的计算
        if volume_size_x is not None and volume_size_y is not None:
            # 基于物理尺寸计算画布尺寸
            max_physical_size = max(volume_size_x, volume_size_y)

            # 计算基础画布尺寸（基于物理尺寸和缩放因子）
            base_canvas_size = int(max_physical_size * scale_factor)

            # 添加边距
            margin = base_canvas_size * margin_ratio
            canvas_size = int(base_canvas_size + 2 * margin)

            # 确保是偶数（便于中心点计算）
            if canvas_size % 2 != 0:
                canvas_size += 1

            # 限制在合理范围内
            canvas_size = max(min_size, min(canvas_size, max_size))

            print(f"基于物理尺寸计算画布尺寸: {canvas_size}x{canvas_size} (物理尺寸: {volume_size_x:.2f}x{volume_size_y:.2f}, 缩放因子: {scale_factor})")

            return canvas_size

        # 如果没有物理尺寸参数，使用原有的基于投影坐标的计算方法
        # 使用临时画布尺寸进行初步坐标变换
        temp_canvas_size = 1000  # 临时值，用于计算相对坐标

        # 收集所有2D投影坐标
        projected_coords = []
        for pos in bubble_positions:
            x_2d, y_2d, depth = transform_3d_to_2d(pos, projection_point, v_vector,
                                                 temp_canvas_size, scale_factor)
            if x_2d is not None and y_2d is not None:
                # 转换为相对于中心的坐标
                rel_x = x_2d - temp_canvas_size // 2
                rel_y = y_2d - temp_canvas_size // 2
                projected_coords.append((rel_x, rel_y))

        if not projected_coords:
            return min_size

        # 计算边界范围
        x_coords = [coord[0] for coord in projected_coords]
        y_coords = [coord[1] for coord in projected_coords]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # 计算内容范围
        content_width = max_x - min_x
        content_height = max_y - min_y
        content_range = max(content_width, content_height)

        # 添加气泡尺寸的影响（气泡可能延伸到中心点之外）
        bubble_extension = bubble_size // 2
        content_range += 2 * bubble_extension

        # 添加边距
        margin = content_range * margin_ratio
        required_range = content_range + 2 * margin

        # 计算所需的画布尺寸（需要容纳从中心点向两边的最大范围）
        canvas_size = int(required_range)  # 修改：不再乘以2，因为required_range已经包含了完整的范围

        # 确保是偶数（便于中心点计算）
        if canvas_size % 2 != 0:
            canvas_size += 1

        # 限制在合理范围内
        canvas_size = max(min_size, min(canvas_size, max_size))

        print(f"基于投影坐标计算画布尺寸: {canvas_size}x{canvas_size} (内容范围: {content_range:.1f}, 边距: {margin:.1f})")

        return canvas_size

    except Exception as e:
        print(f"动态计算画布尺寸失败: {e}")
        return min_size


def load_bubble_positions(base_path):
    """
    从names_points.csv文件加载气泡位置信息

    Args:
        base_path: 基础路径

    Returns:
        list: 气泡位置列表 [(x, y, z), ...]
    """
    try:
        positions_file = os.path.join(base_path, 'names_points.csv')
        positions = []

        if os.path.exists(positions_file):
            with open(positions_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                for row in reader:
                    if len(row) >= 4:
                        x, y, z = float(row[1]), float(row[2]), float(row[3])
                        positions.append((x, y, z))

        return positions

    except Exception as e:
        print(f"加载气泡位置信息失败: {e}")
        return []


def composite_bubble_to_canvas(canvas, bubble_image, position_2d, bubble_size=128):
    """
    将单个气泡图像合成到画布上

    Args:
        canvas: 目标画布
        bubble_image: 气泡图像
        position_2d: 2D位置 (x, y)
        bubble_size: 气泡图像大小

    Returns:
        bool: 合成是否成功
    """
    try:
        x_center, y_center = int(position_2d[0]), int(position_2d[1])
        half_size = bubble_size // 2

        # 计算在画布上的位置
        x_start = x_center - half_size
        y_start = y_center - half_size
        x_end = x_start + bubble_size
        y_end = y_start + bubble_size

        # 检查边界
        canvas_h, canvas_w = canvas.shape[:2]
        if x_end <= 0 or y_end <= 0 or x_start >= canvas_w or y_start >= canvas_h:
            return False  # 完全在画布外

        # 计算有效区域
        src_x_start = max(0, -x_start)
        src_y_start = max(0, -y_start)
        src_x_end = bubble_size - max(0, x_end - canvas_w)
        src_y_end = bubble_size - max(0, y_end - canvas_h)

        dst_x_start = max(0, x_start)
        dst_y_start = max(0, y_start)
        dst_x_end = min(canvas_w, x_end)
        dst_y_end = min(canvas_h, y_end)

        # 提取有效区域
        bubble_region = bubble_image[src_y_start:src_y_end, src_x_start:src_x_end]

        if bubble_region.size == 0:
            return False

        # 创建掩码（白色背景为透明）
        if len(bubble_region.shape) == 3:
            # 彩色图像
            mask = np.all(bubble_region >= [240, 240, 240], axis=2)
        else:
            # 灰度图像
            mask = bubble_region >= 240

        # 合成到画布上（只有非白色区域）
        canvas_region = canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end]

        if len(canvas.shape) == 3 and len(bubble_region.shape) == 3:
            # 彩色到彩色
            canvas_region[~mask] = bubble_region[~mask]
        elif len(canvas.shape) == 3 and len(bubble_region.shape) == 2:
            # 灰度到彩色
            for c in range(3):
                canvas_region[~mask, c] = bubble_region[~mask]
        elif len(canvas.shape) == 2 and len(bubble_region.shape) == 3:
            # 彩色到灰度
            gray_bubble = cv2.cvtColor(bubble_region, cv2.COLOR_BGR2GRAY)
            canvas_region[~mask] = gray_bubble[~mask]
        else:
            # 灰度到灰度
            canvas_region[~mask] = bubble_region[~mask]

        return True

    except Exception as e:
        print(f"气泡合成失败: {e}")
        return False


def create_flow_field_composition(projection_dir, bubble_positions, projection_point, v_vector,
                                canvas_size=None, scale_factor=100, use_hq_images=True,
                                volume_size_x=None, volume_size_y=None, volume_height=None):
    """
    创建流场气泡合成图像

    Args:
        projection_dir: 投影目录路径
        bubble_positions: 气泡位置列表
        projection_point: 投影点
        v_vector: 旋转向量
        canvas_size: 画布大小（如果为None则动态计算）
        scale_factor: 缩放因子
        use_hq_images: 是否使用高质量图像
        volume_size_x: 流场X方向尺寸
        volume_size_y: 流场Y方向尺寸
        volume_height: 流场Z方向尺寸

    Returns:
        tuple: (合成图像, 成功合成的气泡数量)
    """
    try:
        # 动态计算画布尺寸（如果未指定）
        if canvas_size is None:
            canvas_size = calculate_dynamic_canvas_size(
                bubble_positions, projection_point, v_vector, scale_factor,
                volume_size_x=volume_size_x, volume_size_y=volume_size_y, volume_height=volume_height
            )

        # 创建白色背景画布
        canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

        # 确定气泡图像目录
        if use_hq_images:
            bubble_dir = os.path.join(projection_dir, 'high_quality_renders')
            if not os.path.exists(bubble_dir):
                bubble_dir = os.path.join(projection_dir, 'single_bubble_renders')
                use_hq_images = False
                print("高质量图像目录不存在，使用预渲染图像")
        else:
            bubble_dir = os.path.join(projection_dir, 'single_bubble_renders')

        if not os.path.exists(bubble_dir):
            print(f"气泡图像目录不存在: {bubble_dir}")
            return canvas, 0

        # 获取气泡图像文件列表
        bubble_files = []
        for f in os.listdir(bubble_dir):
            if f.endswith('.png') and ('bubble_' in f):
                bubble_files.append(f)

        bubble_files.sort()  # 确保顺序一致

        # 根据深度排序气泡
        sorted_indices = sort_bubbles_by_depth(bubble_positions, projection_point, v_vector)

        successful_compositions = 0

        # 按深度顺序合成气泡（从远到近）
        for bubble_idx in sorted_indices:
            if bubble_idx >= len(bubble_files) or bubble_idx >= len(bubble_positions):
                continue

            # 加载气泡图像
            bubble_file = bubble_files[bubble_idx]
            bubble_path = os.path.join(bubble_dir, bubble_file)

            if not os.path.exists(bubble_path):
                continue

            bubble_image = cv2.imread(bubble_path)
            if bubble_image is None:
                continue

            # 获取3D位置并转换为2D
            pos_3d = bubble_positions[bubble_idx]
            x_2d, y_2d, depth = transform_3d_to_2d(pos_3d, projection_point, v_vector,
                                                 canvas_size, scale_factor)

            if x_2d is None or y_2d is None:
                continue

            # 合成到画布上
            if composite_bubble_to_canvas(canvas, bubble_image, (x_2d, y_2d)):
                successful_compositions += 1

        print(f"成功合成 {successful_compositions}/{len(bubble_positions)} 个气泡")
        return canvas, successful_compositions

    except Exception as e:
        print(f"创建流场合成图像失败: {e}")
        return np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8), 0


def cut_image_out_of_range(img, cx, cy, pad=None):
    """
    从图像中裁剪指定中心点的区域

    Args:
        img: 输入图像
        cx, cy: 中心点坐标
        pad: 填充参数，如果为None则使用128x128裁剪

    Returns:
        裁剪后的图像
    """
    if pad is None:
        w, h = 128, 128
        a = cx - w/2
        b = cx + w/2
        c = cy - h/2
        d = cy + h/2
        xl = max(0, -a)
        yl = max(0, -c)
        a = max(0, a)
        b = min(img.shape[1], b)
        c = max(0, c)
        d = min(img.shape[0], d)
        img_inner = img[int(c): int(d), int(a): int(b), :]
        ret_img = np.zeros((int(h), int(w), 3), dtype=np.uint8)
        ret_img[int(yl):int(img_inner.shape[0]+yl), int(xl):int(img_inner.shape[1]+xl), :] = img_inner
        return ret_img
    else:
        h = 640 - pad[1] - pad[0]
        w = 640 - pad[3] - pad[2]
        img_inner = img[pad[2]: 640-pad[3], pad[0]: 640-pad[1], :]
        ret_img = np.zeros((int(h), int(w), 3), dtype=np.uint8)
        ret_img[0:h, 0:w, :] = img_inner
        return ret_img


def sample_bubble_position(width_pix, height_pix, pad, location_dis='gaussian',
                          x_e=0.5, x_sd=175, x_pad=24):
    """
    采样气泡位置

    Args:
        width_pix: 画布宽度（像素）
        height_pix: 画布高度（像素）
        pad: 填充大小
        location_dis: 位置分布类型 ('gaussian' 或 'uniform')
        x_e: 水平位置期望值（0-1之间的比例）
        x_sd: 水平位置标准差
        x_pad: 水平边距

    Returns:
        tuple: (x, y) 位置坐标
    """
    # 计算水平位置
    if location_dis == 'gaussian':
        x_e_pix = x_e * width_pix
        x = round(np.random.normal(x_e_pix + pad, x_sd))
        while x < x_pad + pad or x > width_pix - x_pad + pad:
            x = round(np.random.normal(x_e_pix + pad, x_sd))
    elif location_dis == 'uniform':
        x = np.random.randint(x_pad, width_pix - x_pad) + pad
    else:
        # 默认使用均匀分布
        x = np.random.randint(x_pad, width_pix - x_pad) + pad

    # 垂直位置使用均匀分布
    y = np.random.randint(0, height_pix) + pad

    return x, y


def rotate_and_mask_bubble(bubble_img, angle, threshold=50):
    """
    旋转气泡图像并生成掩码

    Args:
        bubble_img: 气泡图像
        angle: 旋转角度
        threshold: 二值化阈值

    Returns:
        tuple: (旋转后的图像, 掩码, 轮廓)
    """
    # 预处理图像
    img = cv2.bitwise_not(bubble_img).reshape(128, 128, 1)
    img = img.repeat(3, -1)
    rows, cols, _ = img.shape

    # 旋转图像
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    M_mask = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    cropped_img = cv2.warpAffine(img, M, (cols, rows), borderValue=(0, 0, 0))
    img_mask = cv2.warpAffine(img, M_mask, (cols, rows), borderValue=(0, 0, 0))

    # 生成掩码
    mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    # 提取轮廓
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) >= 1:
        main_contour = contours[0]
        mask_with_contour = np.zeros_like(mask)
        cv2.drawContours(mask_with_contour, [main_contour], -1, (255), thickness=cv2.FILLED)
        return cropped_img, mask_with_contour, main_contour
    else:
        return None, None, None


def detect_overlaps_and_assign_labels(global_mask, bubble_masks, bubble_labels):
    """
    检测重叠并分配标签

    Args:
        global_mask: 当前气泡的全局掩码
        bubble_masks: 已有气泡掩码列表
        bubble_labels: 已有气泡标签列表

    Returns:
        int: 分配给当前气泡的标签
    """
    overlap_idx = []
    for j, prev_mask in enumerate(bubble_masks):
        overlap = np.logical_and(global_mask, prev_mask)
        if np.count_nonzero(overlap) > 5:
            overlap_idx.append(j)

    if not overlap_idx:
        label_val = 0  # 单独气泡
    else:
        # 更新重叠气泡的标签
        for j in overlap_idx:
            if bubble_labels[j] == 0:
                bubble_labels[j] = 1
        max_label = max([bubble_labels[j] for j in overlap_idx])
        label_val = max_label + 1

    return label_val


def apply_visual_enhancement(roi, img_fg, mask_with_contour, contour,
                           enable_gauss=True, enable_median=True):
    """
    应用视觉增强处理

    Args:
        roi: 背景ROI区域
        img_fg: 前景图像
        mask_with_contour: 轮廓掩码
        contour: 轮廓点
        enable_gauss: 是否启用高斯滤波
        enable_median: 是否启用中值滤波

    Returns:
        dict: 包含不同增强效果的图像字典
    """
    results = {}

    # 基础合成
    mask_inv = cv2.bitwise_not(mask_with_contour)
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    dst = cv2.add(img_bg, img_fg)
    results['basic'] = dst

    # 高斯滤波增强
    if enable_gauss:
        kernel_gauss = 3
        dst_global_gauss = cv2.add(img_bg, img_fg)
        dst_global_gauss = cv2.GaussianBlur(dst_global_gauss, (kernel_gauss, kernel_gauss), 0)
        dst_local_gauss = cv2.add(img_bg, img_fg)

        # 局部高斯滤波
        semi = 5
        origin_contour = contour[:, 0, :].copy()
        for point in origin_contour:
            if (point[1]-semi >= 0 and point[1]+semi < dst_local_gauss.shape[0] and
                point[0]-semi >= 0 and point[0]+semi < dst_local_gauss.shape[1]):
                mini_dst_local_gauss = dst_local_gauss[point[1]-semi:point[1]+semi, point[0]-semi:point[0]+semi,:]
                mini_dst_local_gauss = cv2.GaussianBlur(mini_dst_local_gauss, (kernel_gauss, kernel_gauss), 0)
                dst_local_gauss[point[1]-semi:point[1]+semi, point[0]-semi:point[0]+semi] = mini_dst_local_gauss

        results['global_gauss'] = dst_global_gauss
        results['local_gauss'] = dst_local_gauss

    # 中值滤波增强
    if enable_median:
        kernel_median = 3
        dst_global_median = cv2.add(img_bg, img_fg)
        dst_global_median = cv2.medianBlur(dst_global_median, kernel_median)
        dst_local_median = cv2.add(img_bg, img_fg)

        # 局部中值滤波
        semi = 5
        origin_contour = contour[:, 0, :].copy()
        for point in origin_contour:
            if (point[1]-semi >= 0 and point[1]+semi < dst_local_median.shape[0] and
                point[0]-semi >= 0 and point[0]+semi < dst_local_median.shape[1]):
                mini_dst_local_median = dst_local_median[point[1]-semi:point[1]+semi, point[0]-semi:point[0]+semi,:]
                mini_dst_local_median = cv2.medianBlur(mini_dst_local_median, kernel_median)
                dst_local_median[point[1]-semi:point[1]+semi, point[0]-semi:point[0]+semi] = mini_dst_local_median

        results['global_median'] = dst_global_median
        results['local_median'] = dst_local_median

    return results


def generate_flow_field_composition(projection_dir, bubble_analysis_data,
                                   bubble_positions=None, projection_point=None, v_vector=None, scale_factor=100,
                                   width_pix=800, height_pix=600, pad=128,
                                   location_dis='gaussian', x_e=0.5, x_sd=175, x_pad=24,
                                   threshold=50, enable_gauss=True, enable_median=True,
                                   save_outputs=True, use_3d_positions=True,
                                   min_x=None, min_y=None, auto_adjust_canvas=True):
    """
    生成完整的流场合成图像（参考generate_flow_field_overlap.py的实现）

    Args:
        projection_dir: 投影目录路径
        bubble_analysis_data: 气泡分析数据字典 {image_path: analysis_result}
        bubble_positions: 3D气泡位置列表 [(x, y, z), ...]（用于保持位置一致性）
        projection_point: 投影点（视角方向）
        v_vector: 旋转向量
        scale_factor: 缩放因子
        width_pix: 画布宽度（像素）
        height_pix: 画布高度（像素）
        pad: 填充大小
        location_dis: 位置分布类型（当use_3d_positions=False时使用）
        x_e: 水平位置期望值
        x_sd: 水平位置标准差
        x_pad: 水平边距
        threshold: 二值化阈值
        enable_gauss: 是否启用高斯滤波
        enable_median: 是否启用中值滤波
        save_outputs: 是否保存输出文件
        use_3d_positions: 是否使用3D位置信息（True时与原始渲染位置一致）
        min_x, min_y: 原始渲染的最小坐标值（用于坐标映射一致性）
        auto_adjust_canvas: 是否自动调整画布尺寸和填充以适应所有气泡位置

    Returns:
        dict: 包含合成结果的字典
    """
    try:
        print(f"开始生成完整流场合成，气泡数量: {len(bubble_analysis_data)}")

        # 动态调整画布尺寸和填充（如果启用且使用3D位置）
        if auto_adjust_canvas and use_3d_positions and bubble_positions and min_x is not None and min_y is not None:
            adjusted_params = calculate_optimal_canvas_parameters(
                bubble_positions, min_x, min_y, scale_factor,
                width_pix, height_pix, pad
            )
            width_pix = adjusted_params['width_pix']
            height_pix = adjusted_params['height_pix']
            pad = adjusted_params['pad']
            print(f"  自动调整画布参数: 尺寸={width_pix}x{height_pix}, 填充={pad}")

        # 初始化画布
        background = np.zeros((height_pix + pad * 2, width_pix + pad * 2, 3), np.uint8)
        background_global_gauss = np.zeros((height_pix + pad * 2, width_pix + pad * 2, 3), np.uint8) if enable_gauss else None
        background_local_gauss = np.zeros((height_pix + pad * 2, width_pix + pad * 2, 3), np.uint8) if enable_gauss else None
        background_global_median = np.zeros((height_pix + pad * 2, width_pix + pad * 2, 3), np.uint8) if enable_median else None
        background_local_median = np.zeros((height_pix + pad * 2, width_pix + pad * 2, 3), np.uint8) if enable_median else None

        # 存储气泡信息
        bubble_labels = []
        bubble_masks = []
        bub_conts = []
        bub_conts_pad = []
        processed_bubbles = []

        # 确定气泡处理顺序和位置信息
        if use_3d_positions and bubble_positions and projection_point is not None and v_vector is not None:
            # 使用3D位置信息，按深度排序（与原始渲染保持一致）
            print(f"  使用3D位置信息进行合成，气泡数量: {len(bubble_positions)}")
            sorted_indices = sort_bubbles_by_depth(bubble_positions, projection_point, v_vector)

            # 将分析数据按索引重新组织
            analysis_items = list(bubble_analysis_data.items())
            if len(analysis_items) != len(bubble_positions):
                print(f"  警告：分析数据数量({len(analysis_items)})与位置数据数量({len(bubble_positions)})不匹配")
                # 取较小的数量
                min_count = min(len(analysis_items), len(bubble_positions))
                analysis_items = analysis_items[:min_count]
                sorted_indices = [i for i in sorted_indices if i < min_count]

            # 按深度排序重新组织数据
            sorted_items = [(analysis_items[i][0], analysis_items[i][1], bubble_positions[i], i)
                          for i in sorted_indices if i < len(analysis_items)]
        else:
            # 使用随机位置采样，按边缘锐利度排序
            print(f"  使用随机位置采样进行合成")
            sorted_items = [(path, data, None, i) for i, (path, data) in
                          enumerate(sorted(bubble_analysis_data.items(),
                                         key=lambda x: x[1].get('edge_gradient', 0), reverse=True))]

        successful_compositions = 0

        for item in sorted_items:
            if len(item) == 4:
                image_rel_path, analysis_result, bubble_3d_pos, bubble_idx = item
            else:
                # 兼容旧格式
                image_rel_path, analysis_result = item
                bubble_3d_pos, bubble_idx = None, 0
            try:
                # 强制只使用高质量图像，不提供备选机制
                base_name = os.path.basename(image_rel_path)

                # 转换文件名：bubble_000.png -> hq_bubble_000.png
                if base_name.startswith('bubble_'):
                    hq_filename = 'hq_' + base_name
                else:
                    # 如果文件名不是标准格式，尝试添加hq_前缀
                    hq_filename = 'hq_' + base_name

                bubble_image_path = os.path.join(projection_dir, 'high_quality_renders', hq_filename)

                if not os.path.exists(bubble_image_path):
                    error_msg = f"高质量图像不存在: {bubble_image_path}。增强流场合成要求使用高质量筛选图像，请确保高质量生成功能已启用并成功执行。原始路径: {image_rel_path}"
                    print(f"  错误: {error_msg}")
                    raise FileNotFoundError(error_msg)

                print(f"  使用高质量图像: {hq_filename} (原始: {base_name})")

                # 加载气泡图像
                bubble_img = cv2.imread(bubble_image_path, cv2.IMREAD_GRAYSCALE)
                if bubble_img is None:
                    print(f"  无法加载气泡图像: {bubble_image_path}")
                    continue

                # 确定气泡位置
                if bubble_3d_pos is not None and projection_point is not None and v_vector is not None:
                    # 使用与预渲染阶段完全一致的坐标变换逻辑
                    if min_x is None or min_y is None:
                        error_msg = f"缺少坐标映射参数 min_x={min_x}, min_y={min_y}。无法进行3D到2D坐标转换。"
                        print(f"  错误: {error_msg}")
                        raise ValueError(error_msg)

                    # 使用专用的坐标变换函数，实现与pixel_coloring完全一致的变换逻辑
                    mapped_x, mapped_y, depth = transform_3d_to_pixel_coloring_coords(
                        bubble_3d_pos, projection_point, v_vector, min_x, min_y, scale_factor
                    )

                    if mapped_x is not None and mapped_y is not None:
                        # 转换到画布坐标系（添加pad偏移）
                        x = int(mapped_x) + pad
                        y = int(mapped_y) + pad

                        print(f"  使用3D位置: 气泡{bubble_idx} 3D{bubble_3d_pos} -> 映射坐标({mapped_x:.1f}, {mapped_y:.1f}) -> 画布坐标({x-pad}, {y-pad}), 深度={depth:.2f}")
                    else:
                        # 3D坐标变换失败时的错误处理
                        error_msg = f"3D坐标变换失败: 3D{bubble_3d_pos}, projection_point={projection_point}, v_vector={v_vector}"
                        print(f"  错误: {error_msg}")
                        raise ValueError(error_msg)

                    # 验证坐标并应用回退策略
                    valid_x_range = (64, width_pix + pad * 2 - 64)
                    valid_y_range = (64, height_pix + pad * 2 - 64)

                    if not (valid_x_range[0] <= x <= valid_x_range[1] and valid_y_range[0] <= y <= valid_y_range[1]):
                        print(f"  警告: 3D位置映射超出范围 - 3D{bubble_3d_pos} -> 2D({x}, {y}), 有效范围: x{valid_x_range}, y{valid_y_range}")

                        # 应用坐标约束回退策略
                        x_clamped = max(valid_x_range[0], min(x, valid_x_range[1]))
                        y_clamped = max(valid_y_range[0], min(y, valid_y_range[1]))

                        print(f"  应用坐标约束: ({x}, {y}) -> ({x_clamped}, {y_clamped})")
                        x, y = x_clamped, y_clamped

                    print(f"  使用3D位置: 气泡{bubble_idx} 3D{bubble_3d_pos} -> 2D({x-pad}, {y-pad})")
                else:
                    # 使用随机位置采样
                    x, y = sample_bubble_position(width_pix, height_pix, pad, location_dis, x_e, x_sd, x_pad)
                    print(f"  使用随机位置: ({x-pad}, {y-pad})")

                # 获取旋转角度（从分析结果中获取，或随机生成）
                base_angle = analysis_result.get('angle', np.random.uniform(-180, 180))

                # 修复90度角度偏差：在基础角度上增加90度顺时针旋转
                # 这确保增强流场合成与原始3D渲染的气泡方向完全匹配
                angle = base_angle + 90.0

                # 将角度规范化到[-180, 180]范围内
                while angle > 180:
                    angle -= 360
                while angle <= -180:
                    angle += 360

                print(f"    气泡旋转角度: 基础角度={base_angle:.1f}°, 修正后角度={angle:.1f}°")

                # 旋转气泡并生成掩码
                rotated_img, mask_with_contour, main_contour = rotate_and_mask_bubble(bubble_img, angle, threshold)

                if rotated_img is None or mask_with_contour is None or main_contour is None:
                    continue

                # 生成全局掩码
                global_mask = np.zeros((height_pix + pad * 2, width_pix + pad * 2), dtype=np.uint8)
                x0, y0 = max(0, x - 64), max(0, y - 64)
                x1, y1 = min(width_pix + pad * 2, x + 64), min(height_pix + pad * 2, y + 64)
                mask_x0, mask_y0 = max(0, 64 - x), max(0, 64 - y)
                mask_x1, mask_y1 = min(128, 128 - (x + 64 - (width_pix + pad * 2))), min(128, 128 - (y + 64 - (height_pix + pad * 2)))

                if x1 > x0 and y1 > y0 and mask_x1 > mask_x0 and mask_y1 > mask_y0:
                    global_mask[y0:y1, x0:x1] = mask_with_contour[mask_y0:mask_y1, mask_x0:mask_x1]
                else:
                    continue

                # 检测重叠并分配标签
                label_val = detect_overlaps_and_assign_labels(global_mask, bubble_masks, bubble_labels)
                bubble_labels.append(label_val)
                bubble_masks.append(global_mask)

                # 处理轮廓坐标
                origin_contour = main_contour[:, 0, :].copy()
                bub_conts_pad.append(main_contour.copy())
                adjusted_contour = main_contour.copy()
                adjusted_contour[:, 0, 0] += x - pad - 64
                adjusted_contour[:, 0, 1] += y - pad - 64
                bub_conts.append(adjusted_contour)

                # 提取ROI区域
                roi = cut_image_out_of_range(background, x, y)
                roi_global_gauss = cut_image_out_of_range(background_global_gauss, x, y) if enable_gauss else None
                roi_local_gauss = cut_image_out_of_range(background_local_gauss, x, y) if enable_gauss else None
                roi_global_median = cut_image_out_of_range(background_global_median, x, y) if enable_median else None
                roi_local_median = cut_image_out_of_range(background_local_median, x, y) if enable_median else None

                # 准备前景图像
                img_fg = cv2.bitwise_and(rotated_img, rotated_img, mask=mask_with_contour)

                # 应用视觉增强
                enhanced_results = apply_visual_enhancement(roi, img_fg, mask_with_contour, main_contour, enable_gauss, enable_median)

                # 合成到背景
                background[y - 64 : y + 64, x - 64 : x + 64] = enhanced_results['basic']
                if enable_gauss:
                    background_global_gauss[y - 64 : y + 64, x - 64 : x + 64] = enhanced_results['global_gauss']
                    background_local_gauss[y - 64 : y + 64, x - 64 : x + 64] = enhanced_results['local_gauss']
                if enable_median:
                    background_global_median[y - 64 : y + 64, x - 64 : x + 64] = enhanced_results['global_median']
                    background_local_median[y - 64 : y + 64, x - 64 : x + 64] = enhanced_results['local_median']

                # 记录处理成功的气泡信息
                bubble_info = {
                    'image_path': image_rel_path,
                    'bubble_index': bubble_idx,
                    'position_2d': (x - pad, y - pad),
                    'position_3d': bubble_3d_pos,
                    'angle': angle,
                    'label': label_val,
                    'analysis_result': analysis_result,
                    'used_3d_position': bubble_3d_pos is not None
                }
                processed_bubbles.append(bubble_info)
                successful_compositions += 1

            except (FileNotFoundError, ValueError) as e:
                # 对于关键错误（高质量图像不存在、坐标映射错误），立即停止处理
                print(f"处理气泡 {image_rel_path} 时出错: {e}")
                raise e
            except Exception as e:
                # 对于其他错误，记录并继续处理
                print(f"处理气泡 {image_rel_path} 时出错: {e}")
                continue

        print(f"成功合成 {successful_compositions}/{len(bubble_analysis_data)} 个气泡")

        # 后处理：反转颜色并裁剪
        background = cv2.bitwise_not(background)
        background = background[pad : height_pix + pad, pad : width_pix + pad]

        results = {
            'basic': background,
            'bubble_labels': bubble_labels,
            'bubble_contours': bub_conts,
            'bubble_contours_pad': bub_conts_pad,
            'processed_bubbles': processed_bubbles,
            'successful_compositions': successful_compositions
        }

        if enable_gauss:
            background_global_gauss = cv2.bitwise_not(background_global_gauss)
            background_local_gauss = cv2.bitwise_not(background_local_gauss)
            results['global_gauss'] = background_global_gauss[pad : height_pix + pad, pad : width_pix + pad]
            results['local_gauss'] = background_local_gauss[pad : height_pix + pad, pad : width_pix + pad]

        if enable_median:
            background_global_median = cv2.bitwise_not(background_global_median)
            background_local_median = cv2.bitwise_not(background_local_median)
            results['global_median'] = background_global_median[pad : height_pix + pad, pad : width_pix + pad]
            results['local_median'] = background_local_median[pad : height_pix + pad, pad : width_pix + pad]

        # 保存输出文件
        if save_outputs:
            save_flow_field_outputs(projection_dir, results, bubble_labels, bub_conts, processed_bubbles)

        return results

    except Exception as e:
        print(f"生成完整流场合成失败: {e}")
        return {
            'basic': np.zeros((height_pix, width_pix, 3), dtype=np.uint8),
            'successful_compositions': 0,
            'error': str(e)
        }


def calculate_optimal_canvas_parameters(bubble_positions, min_x, min_y, scale_factor,
                                      original_width, original_height, original_pad):
    """
    计算最优的画布参数以适应所有气泡位置

    Args:
        bubble_positions: 3D气泡位置列表
        min_x, min_y: 坐标映射参数
        scale_factor: 缩放因子
        original_width, original_height: 原始画布尺寸
        original_pad: 原始填充大小

    Returns:
        dict: 包含调整后的画布参数
    """
    try:
        if not bubble_positions:
            return {
                'width_pix': original_width,
                'height_pix': original_height,
                'pad': original_pad
            }

        # 计算所有气泡的映射坐标
        mapped_positions = []
        for pos in bubble_positions:
            mapped_x = (pos[0] - min_x) * scale_factor
            mapped_y = (pos[1] - min_y) * scale_factor
            mapped_positions.append((mapped_x, mapped_y))

        # 计算映射坐标的范围
        mapped_x_coords = [pos[0] for pos in mapped_positions]
        mapped_y_coords = [pos[1] for pos in mapped_positions]

        min_mapped_x = min(mapped_x_coords)
        max_mapped_x = max(mapped_x_coords)
        min_mapped_y = min(mapped_y_coords)
        max_mapped_y = max(mapped_y_coords)

        # 计算所需的画布尺寸（考虑气泡图像大小128x128）
        bubble_size = 128
        safety_margin = 64  # 额外的安全边距

        required_width = int(max_mapped_x - min_mapped_x) + bubble_size + safety_margin * 2
        required_height = int(max_mapped_y - min_mapped_y) + bubble_size + safety_margin * 2

        # 计算所需的填充大小
        # 确保最小映射坐标加上填充后不会小于64（有效范围的最小值）
        min_pad_x = max(64 - int(min_mapped_x), 0) + safety_margin
        min_pad_y = max(64 - int(min_mapped_y), 0) + safety_margin
        required_pad = max(min_pad_x, min_pad_y, original_pad)

        # 调整画布尺寸，确保足够容纳所有气泡
        adjusted_width = max(required_width, original_width)
        adjusted_height = max(required_height, original_height)

        # 确保填充足够大，使所有气泡都在有效范围内
        max_x_with_pad = int(max_mapped_x) + required_pad + bubble_size
        max_y_with_pad = int(max_mapped_y) + required_pad + bubble_size

        if max_x_with_pad > adjusted_width + required_pad * 2 - 64:
            adjusted_width = max_x_with_pad - required_pad * 2 + 64 + safety_margin

        if max_y_with_pad > adjusted_height + required_pad * 2 - 64:
            adjusted_height = max_y_with_pad - required_pad * 2 + 64 + safety_margin

        print(f"    映射坐标范围: X[{min_mapped_x:.1f}, {max_mapped_x:.1f}], Y[{min_mapped_y:.1f}, {max_mapped_y:.1f}]")
        print(f"    原始参数: {original_width}x{original_height}, pad={original_pad}")
        print(f"    调整后参数: {adjusted_width}x{adjusted_height}, pad={required_pad}")

        return {
            'width_pix': adjusted_width,
            'height_pix': adjusted_height,
            'pad': required_pad,
            'mapped_range': {
                'x_min': min_mapped_x, 'x_max': max_mapped_x,
                'y_min': min_mapped_y, 'y_max': max_mapped_y
            }
        }

    except Exception as e:
        print(f"计算最优画布参数失败: {e}")
        return {
            'width_pix': original_width,
            'height_pix': original_height,
            'pad': original_pad
        }


def save_flow_field_outputs(projection_dir, results, bubble_labels, bub_conts, processed_bubbles):
    """
    保存流场合成的输出文件

    Args:
        projection_dir: 投影目录路径
        results: 合成结果字典
        bubble_labels: 气泡标签列表
        bub_conts: 气泡轮廓列表
        processed_bubbles: 处理成功的气泡信息列表
    """
    try:
        # 保存基础流场图像
        cv2.imwrite(os.path.join(projection_dir, 'flow_field_complete_composition.png'), results['basic'])

        # 保存增强版本
        if 'global_gauss' in results:
            cv2.imwrite(os.path.join(projection_dir, 'flow_field_complete_global_gauss.png'), results['global_gauss'])
        if 'local_gauss' in results:
            cv2.imwrite(os.path.join(projection_dir, 'flow_field_complete_local_gauss.png'), results['local_gauss'])
        if 'global_median' in results:
            cv2.imwrite(os.path.join(projection_dir, 'flow_field_complete_global_median.png'), results['global_median'])
        if 'local_median' in results:
            cv2.imwrite(os.path.join(projection_dir, 'flow_field_complete_local_median.png'), results['local_median'])

        # 保存标签可视化图像
        if bubble_labels and bub_conts:
            label_vis_img = results.get('local_median', results['basic']).copy()

            for i, (contour, label) in enumerate(zip(bub_conts, bubble_labels)):
                x, y, w, h = cv2.boundingRect(contour)
                if label == 0:
                    color = (53, 130, 84)  # 绿色 - 单独气泡
                else:
                    intensity = min(255, 80 + label * 30)
                    color = (0, 0, intensity)  # 蓝色系 - 重叠气泡

                cv2.rectangle(label_vis_img, (x, y), (x + w, y + h), color, 2)
                label_text = f"{label}"
                cv2.putText(label_vis_img, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 4)
                cv2.putText(label_vis_img, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imwrite(os.path.join(projection_dir, 'flow_field_complete_label_visualization.png'), label_vis_img)

        # 保存轮廓信息
        if bub_conts:
            with open(os.path.join(projection_dir, 'flow_field_complete_contours.txt'), 'w') as f:
                for contour in bub_conts:
                    f.write(','.join(map(str, contour.reshape(-1))) + '\n')

        # 保存YOLO格式标签
        if bubble_labels and bub_conts:
            img_height, img_width = results['basic'].shape[:2]
            with open(os.path.join(projection_dir, 'flow_field_complete_labels.txt'), 'w') as f:
                for contour, label in zip(bub_conts, bubble_labels):
                    x, y, w, h = cv2.boundingRect(contour)
                    x_center = (x + w/2) / img_width
                    y_center = (y + h/2) / img_height
                    width = w / img_width
                    height = h / img_height
                    f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # 保存处理信息
        if processed_bubbles:
            with open(os.path.join(projection_dir, 'flow_field_complete_info.txt'), 'w') as f:
                f.write("bubble_index\timage_path\tx_2d\ty_2d\tx_3d\ty_3d\tz_3d\tangle\tlabel\tused_3d_position\n")
                for bubble in processed_bubbles:
                    pos_3d = bubble.get('position_3d', (None, None, None))
                    pos_2d = bubble.get('position_2d', (0, 0))
                    f.write(f"{bubble.get('bubble_index', -1)}\t{bubble['image_path']}\t{pos_2d[0]}\t{pos_2d[1]}\t"
                           f"{pos_3d[0] if pos_3d[0] is not None else 'N/A'}\t"
                           f"{pos_3d[1] if pos_3d[1] is not None else 'N/A'}\t"
                           f"{pos_3d[2] if pos_3d[2] is not None else 'N/A'}\t"
                           f"{bubble['angle']:.2f}\t{bubble['label']}\t{bubble.get('used_3d_position', False)}\n")

        print(f"流场合成输出文件已保存到: {projection_dir}")

    except Exception as e:
        print(f"保存流场合成输出文件失败: {e}")
