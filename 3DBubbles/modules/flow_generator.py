# flow_generator.py
"""
流场生成器模块

该模块包含主要的流场生成逻辑，包括气泡mesh的生成、分布、
多视角投影渲染等核心功能。

主要功能：
- 流场生成主函数
- 多视角投影处理
- 气泡mesh上采样和缩放
- 点云生成和分布
"""

import os
import random
import csv
import shutil
import json
import numpy as np
import signal
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading
import pyvista as pv
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import gc
# GPU加速筛选导入
from .bubble_screening import create_bubble_screener, screen_bubble_images_parallel
from .bubble_analysis import analyze_bubble_image_smart, TORCH_AVAILABLE
import multiprocessing as mp
from scipy.ndimage import median_filter, gaussian_filter
from tqdm import tqdm

# 全局变量用于优雅退出
_shutdown_requested = False
_shutdown_lock = threading.Lock()

def _signal_handler(signum, frame):
    """信号处理器，用于优雅退出"""
    global _shutdown_requested
    with _shutdown_lock:
        if not _shutdown_requested:
            _shutdown_requested = True
            print("\n收到中断信号，正在优雅退出...")
            print("等待当前任务完成，请稍候...")

def _is_shutdown_requested():
    """检查是否请求关闭"""
    with _shutdown_lock:
        return _shutdown_requested

def _process_single_bubble(args):
    """
    处理单个气泡的函数，用于多线程并行处理

    Args:
        args: 包含处理参数的元组

    Returns:
        tuple: (image_rel_path, result, success)
    """
    try:
        (image_rel_path, params, base_path, hq_output_dir, base_csv_path,
         bubble_selector, filtering_params) = args

        # 检查是否请求关闭
        if _is_shutdown_requested():
            return image_rel_path, None, False

        # 构建完整的图像路径
        original_image_path = os.path.join(base_path, image_rel_path)

        if not os.path.exists(original_image_path):
            return image_rel_path, {'success': False, 'reason': 'file_not_found'}, False

        # 从图像路径中提取编号信息
        import re
        basename = os.path.basename(image_rel_path)
        match = re.search(r'(\d+)', basename)
        image_index = int(match.group(1)) if match else None

        # 生成高质量图像（使用预生成数据集筛选或回退到GAN）
        from .image_generation import generate_high_quality_bubble_image
        result = generate_high_quality_bubble_image(
            original_image_path=original_image_path,
            original_params=params,
            output_dir=hq_output_dir,
            bubble_selector=bubble_selector,
            filtering_params=filtering_params,
            image_index=image_index
        )

        return image_rel_path, result, result.get('success', False)

    except Exception as e:
        return image_rel_path, {'success': False, 'reason': 'processing_error', 'error': str(e)}, False

# 导入相关模块
try:
    from .bubble_analysis import analyze_bubble_image
    from .image_generation import generate_high_quality_bubble_image
    from .flow_composition import create_flow_field_composition, load_bubble_positions, generate_flow_field_composition
    from .bubble_rendering import render_single_bubble, process_single_bubble_rendering, pixel_coloring, cv2_enhance_contrast
except ImportError:
    from bubble_analysis import analyze_bubble_image
    from image_generation import generate_high_quality_bubble_image
    from flow_composition import create_flow_field_composition, load_bubble_positions, generate_flow_field_composition
    from bubble_rendering import render_single_bubble, process_single_bubble_rendering, pixel_coloring, cv2_enhance_contrast


def generate_uniform_points_on_sphere(N=1000):
    """
    在球面上生成均匀分布的点
    
    Args:
        N: 点的数量
    
    Returns:
        points: 球面上的点坐标
    """
    phi = (np.sqrt(5) - 1) / 2
    n = np.arange(0, N)
    z = ((2*n + 1) / N - 1)
    x = (np.sqrt(1 - z**2)) * np.cos(2 * np.pi * (n + 1) * phi)
    y = (np.sqrt(1 - z**2)) * np.sin(2 * np.pi * (n + 1) * phi)
    points = np.stack([x, y, z], axis=-1)
    return points


def upsample_point_cloud(points, num_clusters, sample_spacing):
    """
    上采样点云
    
    Args:
        points: 原始点云
        num_clusters: 目标点数
        sample_spacing: 采样间距
    
    Returns:
        mesh: 上采样后的mesh
    """
    cloud = pv.PolyData(points)
    sample_spacing = 0.1
    while True:
        mesh = cloud.reconstruct_surface(nbr_sz=10, sample_spacing=sample_spacing)
        new_points = np.asarray(mesh.points)
        if new_points.shape[0] < num_clusters * 1.1:
            sample_spacing *= 0.8
        else:
            break
    return mesh


def upsample_and_scale_mesh(stl_files, num_clusters, chosen_volume, sample_spacing):
    """
    上采样和缩放mesh
    
    Args:
        stl_files: STL文件列表
        num_clusters: 目标点数
        chosen_volume: 目标体积
        sample_spacing: 采样间距
    
    Returns:
        tuple: (stl_file, mesh, mesh_origin, chosen_volume)
    """
    stl_file = random.choice(stl_files)
    mesh_origin = pv.read(stl_file)
    mesh = upsample_point_cloud(mesh_origin.points, num_clusters, sample_spacing)
    mesh.smooth_taubin(n_iter=10, pass_band=5, inplace=True)
    mesh = mesh.fill_holes(100)
    volume = mesh.volume
    scale_factor = (chosen_volume / volume) ** (1/3)
    mesh.points *= scale_factor
    mesh_origin.points *= scale_factor
    return stl_file, mesh, mesh_origin, chosen_volume


def generate_points_in_cube(num_points, cube_size=np.array([100,100,100]), num=100, poisson_max_iter = 100000):
    """
    在立方体中生成泊松圆盘采样点
    
    Args:
        num_points: 点的数量
        cube_size: 立方体尺寸
        num: 密度参数
        poisson_max_iter: 最大迭代次数
    
    Returns:
        points: 生成的点坐标
    """
    rnd_points = []
    Look_up_num = 0
    while len(rnd_points) < num_points:
        if Look_up_num >= poisson_max_iter:
            raise RuntimeError("超过最大迭代次数，可能无法在给定条件下生成足够的点")
        x, y, z = np.random.rand(3) * cube_size
        if all(np.linalg.norm(np.array([x, y, z]) - p) > (cube_size.min() / (num**(1/3)) * 0.5) for p in rnd_points)\
                and np.linalg.norm(np.array([x, y]) - cube_size[:2]/2) < cube_size[0]/2*0.85:
            rnd_points.append([x, y, z])
        Look_up_num += 1
    return np.array(rnd_points)


def process_projection(mp_args):
    """
    处理单个投影的函数（多进程调用）

    Args:
        mp_args: 多进程参数元组

    Returns:
        None
    """
    (i_projection, point_fibonacci, allocated_meshes, allocated_origin_meshes, base_path, gas_holdup, v, scale, alpha, truncation, enable_hq_generation, enable_bubble_composition,
     use_pregenerated_dataset, bubble_selector, filtering_params, gpu_ids, max_gpus, enable_gpu_acceleration, gpu_batch_size) = mp_args
    masks_path = os.path.join(base_path, f'{str(i_projection).zfill(3)}/masks')
    if not os.path.exists(masks_path):
        os.makedirs(masks_path)
    meshes = [mesh.rotate_vector(np.cross(point_fibonacci, v), np.arccos(np.dot(point_fibonacci, v)) * 180 / np.pi, inplace=False) for mesh in allocated_meshes]
    meshes_origin = [mesh.rotate_vector(np.cross(point_fibonacci, v), np.arccos(np.dot(point_fibonacci, v)) * 180 / np.pi, inplace=False) for mesh in allocated_origin_meshes]
    mesh_fibonacci = pv.merge([mesh for mesh in meshes_origin])
    mesh_fibonacci.save(os.path.join(base_path, f'{str(i_projection).zfill(3)}/{gas_holdup}.stl'))

    all_points = [mesh.points for mesh in meshes]
    all_vectors = [mesh.point_normals for mesh in meshes]
    volume_size_x = max([np.max(point[:, 0]) for point in all_points]) - min([np.min(point[:, 0]) for point in all_points])
    volume_size_y = max([np.max(point[:, 1]) for point in all_points]) - min([np.min(point[:, 1]) for point in all_points])
    canvas_range_x = int(scale * volume_size_x)
    canvas_range_y = int(scale * volume_size_y)

    # Sort all_points and all_vectors based on the maximum value of the last column of each array in all_points
    sorted_indices = np.argsort([np.max(point[:, 2]) for point in all_points])[::-1]
    all_points = [all_points[i] for i in sorted_indices]
    all_vectors = [all_vectors[i] for i in sorted_indices]

    # 计算坐标映射参数（原始位置，用于pixel_coloring和后续的增强流场合成）
    min_x, max_x = np.min([np.min(point[:, 0]) for point in all_points]), np.max([np.max(point[:, 0]) for point in all_points])
    min_y, max_y = np.min([np.min(point[:, 1]) for point in all_points]), np.max([np.max(point[:, 1]) for point in all_points])

    bboxes, bub_conts, mapped_points = pixel_coloring(masks_path, alpha, all_points, all_vectors, min_x, min_y, scale, canvas_range_x, canvas_range_y)

    indices = np.where(mapped_points == 1)
    mapped_points[indices] = 0
    mapped_points = gaussian_filter(mapped_points, sigma=1)
    
    mapped_points_normalized = np.clip(mapped_points / mapped_points.max(), 0, truncation) / truncation
    mapped_points_normalized[indices] = 1

    mapped_points_normalized = gaussian_filter(mapped_points_normalized, sigma=0.75)
    mapped_points_normalized = median_filter(mapped_points_normalized, size=5)
    mapped_points_normalized = (mapped_points_normalized - mapped_points_normalized.min()) / (mapped_points_normalized.max() - mapped_points_normalized.min())

    mapped_points_normalized = (mapped_points_normalized * 255).astype(np.uint8).T
    mapped_points_normalized = cv2.cvtColor(mapped_points_normalized, cv2.COLOR_GRAY2RGB)
    mapped_points_normalized = cv2_enhance_contrast(mapped_points_normalized, 2)

    image_with_bboxes = mapped_points_normalized.copy()
    with open(os.path.join(base_path, f'{str(i_projection).zfill(3)}/bubbly_flow.txt'), 'w') as f:
        for i, bbox in enumerate(bboxes):
            min_mapped_x, max_mapped_x, min_mapped_y, max_mapped_y = map(int, bbox)
            is_overlapping = False
            for j, other_bbox in enumerate(bboxes):
                if i != j:
                    other_min_mapped_x, other_max_mapped_x, other_min_mapped_y, other_max_mapped_y = map(int, other_bbox)
                    if (min_mapped_x <= other_max_mapped_x and max_mapped_x >= other_min_mapped_x and
                            min_mapped_y <= other_max_mapped_y and max_mapped_y >= other_min_mapped_y):
                        is_overlapping = True
                        break
            color = (0, 0, 192) if is_overlapping else (53, 130, 84)
            cv2.rectangle(image_with_bboxes, (min_mapped_x - 2, min_mapped_y - 2), (max_mapped_x + 2, max_mapped_y + 2), color, 2)
            f.write(f"{int(is_overlapping)} {(min_mapped_x + min_mapped_x)/ 2 / canvas_range_x} {(min_mapped_y + min_mapped_y)/ 2 / canvas_range_y} {(max_mapped_x-min_mapped_x)/canvas_range_x} {(max_mapped_y-min_mapped_y)/canvas_range_y}\n")
    image_with_bboxes = cv2.transpose(image_with_bboxes)
    mapped_points_normalized = cv2.transpose(mapped_points_normalized)
    cv2.imwrite(os.path.join(base_path, f'{str(i_projection).zfill(3)}/bubbly_flow.png'), mapped_points_normalized)
    cv2.imwrite(os.path.join(base_path, f'{str(i_projection).zfill(3)}/bubbly_flow_bboxes.png'), image_with_bboxes)

    SAM_background_merge = np.zeros((canvas_range_x + 128 * 2, canvas_range_y + 128 * 2, 3), np.uint8)
    colors = []
    for xx in range(len(bub_conts)):
        bub_conts[xx][:, 0, 0] += 128
        bub_conts[xx][:, 0, 1] += 128
    for bub_cont in bub_conts:
        if bub_cont.shape[0] > 6:
            zeros = np.ones((SAM_background_merge.shape), dtype=np.uint8) * 255
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.drawContours(SAM_background_merge, [bub_cont], -1, color=color, thickness=cv2.FILLED)
            colors.append(tuple(c / 255 for c in color))

    SAM_background_merge = SAM_background_merge[128 : canvas_range_x + 128, 128 : canvas_range_y + 128]
    cv2.imwrite(os.path.join(base_path, f'{str(i_projection).zfill(3)}/mask.png'), SAM_background_merge)

    # 创建可视化图像
    image = cv2.imread(os.path.join(base_path, f'{str(i_projection).zfill(3)}/bubbly_flow.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((image.shape[0], image.shape[1], 4))
    img[:,:,3] = 0
    mask_files = [f for f in os.listdir(masks_path) if f.endswith('.png')]
    for (color, mask_file) in zip(colors, mask_files):
        mask_path = os.path.join(masks_path, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        m = mask > 0
        img[m] = np.concatenate([color, [0.4]])
    ax.imshow(img)
    plt.axis('off')
    plt.savefig(os.path.join(base_path, f'{str(i_projection).zfill(3)}/mask_merge.png'), bbox_inches='tight', pad_inches=0, dpi=150)

    image = cv2.imread(os.path.join(base_path, f'{str(i_projection).zfill(3)}/bubbly_flow_bboxes.png'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((image.shape[0], image.shape[1], 4))
    img[:,:,3] = 0
    mask_files = [f for f in os.listdir(masks_path) if f.endswith('.png')]
    for (color, mask_file) in zip(colors, mask_files):
        mask_path = os.path.join(masks_path, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        m = mask > 0
        img[m] = np.concatenate([color, [0.4]])
    ax.imshow(img)
    plt.axis('off')
    plt.savefig(os.path.join(base_path, f'{str(i_projection).zfill(3)}/mask_merge_bboxes.png'), bbox_inches='tight', pad_inches=0, dpi=150)

    # 单气泡渲染功能
    single_bubble_dir = os.path.join(base_path, f'{str(i_projection).zfill(3)}/single_bubble_renders')
    if not os.path.exists(single_bubble_dir):
        os.makedirs(single_bubble_dir)

    # 存储所有气泡的参数
    all_bubble_params = []
    successful_renders = 0

    # 单个气泡处理
    print(f"  开始预渲染 {len(allocated_meshes)} 个气泡...")

    for bubble_idx, mesh in enumerate(allocated_meshes):
        try:
            # 渲染单个气泡
            render_result = render_single_bubble(mesh, bubble_idx, point_fibonacci, v,
                                               single_bubble_dir, alpha, truncation)

            if render_result is not None:
                canvas_size, mapped_points, scale_factor, offset_x, offset_y, bubble_3d_params = render_result

                # 临时输出路径
                temp_bubble_image_path = os.path.join(single_bubble_dir, f'temp_bubble_{bubble_idx:03d}.png')

                # 后处理和保存
                final_params = process_single_bubble_rendering(
                    canvas_size, mapped_points, scale_factor, offset_x, offset_y,
                    bubble_3d_params, bubble_idx, temp_bubble_image_path, truncation
                )

                # 根据处理方式确定最终文件名
                if final_params.get('processing_method') == 'padding':
                    # 小尺寸图像使用填充，文件名不包含原始尺寸
                    final_bubble_image_path = os.path.join(single_bubble_dir, f'bubble_{bubble_idx:03d}.png')
                else:
                    # 大尺寸图像使用缩放，文件名包含原始尺寸
                    final_bubble_image_path = os.path.join(single_bubble_dir, f'bubble_{bubble_idx:03d}_size{canvas_size}.png')

                # 重命名文件
                if os.path.exists(temp_bubble_image_path):
                    os.rename(temp_bubble_image_path, final_bubble_image_path)
                    # 更新参数中的图像路径
                    final_params['image_path'] = os.path.relpath(final_bubble_image_path, base_path)

                # 添加额外信息（image_path已经在上面设置了）
                final_params.update({
                    'bubble_index': bubble_idx,
                    'projection_index': i_projection,
                    'projection_point': point_fibonacci.tolist(),
                    'stl_source': getattr(mesh, 'source_file', 'unknown')  # 如果有源文件信息
                })

                all_bubble_params.append(final_params)
                successful_renders += 1

        except Exception as e:
            # 渲染错误，不输出详细信息
            continue
        finally:
            # 清理内存
            if 'render_result' in locals():
                del render_result
            gc.collect()

    print(f"  预渲染完成，成功渲染 {successful_renders}/{len(allocated_meshes)} 个气泡")

    # 保存参数到JSON文件
    params_json_path = os.path.join(base_path, f'{str(i_projection).zfill(3)}/bubble_params.json')
    try:
        # 添加汇总信息
        summary_info = {
            'projection_index': i_projection,
            'projection_point': point_fibonacci.tolist(),
            'total_bubbles': len(allocated_meshes),
            'successful_renders': successful_renders,
            'failed_renders': len(allocated_meshes) - successful_renders,
            'render_parameters': {
                'alpha': alpha,
                'truncation': truncation,
                'scale': scale
            },
            'image_standardization': {
                'target_size': 128,
                'processing_strategies': {
                    'small_images': 'padding_with_white_background',
                    'large_images': 'bilinear_interpolation_resize'
                },
                'size_threshold': 128,
                'filename_formats': {
                    'small_images': 'bubble_{idx:03d}.png',
                    'large_images': 'bubble_{idx:03d}_size{original_canvas_size}.png'
                }
            },
            'bubbles': all_bubble_params
        }

        with open(params_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_info, f, indent=2, ensure_ascii=False)
    except Exception as e:
        pass  # 不输出详细错误信息

    # 对生成的气泡图像进行结构表征分析和筛选
    analysis_results = []
    analysis_success_count = 0
    screening_results = {'passed': [], 'failed': [], 'analysis_data': {}}

    # 分析single_bubble_renders文件夹中的所有图像
    if os.path.exists(single_bubble_dir):
        bubble_files = [f for f in os.listdir(single_bubble_dir) if f.endswith('.png')]
        bubble_files.sort()  # 确保按顺序处理

        if bubble_files:
            # 构建完整的图像路径列表
            bubble_image_paths = [os.path.join(single_bubble_dir, f) for f in bubble_files]

            # 使用GPU加速筛选机制
            print(f"  开始筛选 {len(bubble_image_paths)} 个气泡图像...")

            # 检查是否启用GPU加速
            use_gpu_screening = enable_gpu_acceleration and TORCH_AVAILABLE

            if use_gpu_screening:
                try:
                    # 使用GPU加速筛选
                    print(f"  使用GPU加速筛选模式")
                    screener = create_bubble_screener(
                        gpu_ids=gpu_ids,
                        max_gpus=max_gpus,
                        batch_size=gpu_batch_size
                    )

                    screening_results = screener.screen_bubble_images(
                        bubble_image_paths,
                        enable_gpu_acceleration=True
                    )

                    # 清理筛选器资源
                    screener.cleanup()

                except Exception as e:
                    print(f"  GPU筛选失败，回退到CPU模式: {e}")
                    use_gpu_screening = False

            if not use_gpu_screening:
                # 使用CPU模式筛选（智能分析）
                print(f"  使用CPU筛选模式")
                for bubble_image_path in bubble_image_paths:
                    # 使用智能分析（自动选择GPU或CPU）
                    analysis_result = analyze_bubble_image_smart(bubble_image_path, standardize_orientation=True)
                    if analysis_result is not None:
                        screening_results['analysis_data'][bubble_image_path] = analysis_result
                        # 简单的筛选标准（可以根据需要调整）
                        if (0.3 <= analysis_result.get('circularity', 0) <= 1.0 and
                            0.5 <= analysis_result.get('solidity', 0) <= 1.0):
                            screening_results['passed'].append(bubble_image_path)
                        else:
                            screening_results['failed'].append(bubble_image_path)
                    else:
                        screening_results['failed'].append(bubble_image_path)

            # 处理筛选结果
            for image_path, analysis_result in screening_results['analysis_data'].items():
                # 使用相对路径
                relative_path = os.path.relpath(image_path, base_path)

                # 格式化分析结果为txt格式
                analysis_line = f"{relative_path}\t{analysis_result['angle']:.6f}\t{analysis_result['major_axis_length']:.6f}\t{analysis_result['minor_axis_length']:.6f}\t{analysis_result['centroid_x']:.6f}\t{analysis_result['centroid_y']:.6f}\t{analysis_result['circularity']:.6f}\t{analysis_result['solidity']:.6f}\t{analysis_result['shadow_ratio']:.6f}\t{analysis_result['edge_gradient']:.6f}"
                analysis_results.append(analysis_line)
                analysis_success_count += 1

            passed_count = len(screening_results['passed'])
            total_count = len(bubble_image_paths)
            print(f"  筛选完成: {passed_count}/{total_count} 个气泡通过筛选")

            # 内存清理
            del bubble_image_paths
            gc.collect()

    # 保存分析结果到txt文件
    if analysis_results:
        analysis_txt_path = os.path.join(base_path, f'{str(i_projection).zfill(3)}/bubble_analysis_results.txt')
        try:
            with open(analysis_txt_path, 'w', encoding='utf-8') as f:
                # 写入分析结果
                for line in analysis_results:
                    f.write(line + "\n")
        except Exception as e:
            pass  # 不输出详细错误信息

    # 高质量图像重新生成功能
    if enable_hq_generation and analysis_results:
        # 创建高质量图像输出目录
        hq_output_dir = os.path.join(base_path, f'{str(i_projection).zfill(3)}/high_quality_renders')
        if not os.path.exists(hq_output_dir):
            os.makedirs(hq_output_dir)

        # StyleGAN2基础CSV文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        base_csv_path = os.path.join(project_root, 'Conditional_StyleGAN2', 'bubble_test', 'path_to_base_csv_file.csv')

        # 生成报告
        hq_generation_report = []
        hq_success_count = 0

        # 解析分析结果
        bubble_analysis_data = {}
        for line in analysis_results:
            parts = line.split('\t')
            if len(parts) >= 10:
                image_path = parts[0]
                bubble_analysis_data[image_path] = {
                    'angle': float(parts[1]),
                    'major_axis_length': float(parts[2]),
                    'minor_axis_length': float(parts[3]),
                    'centroid_x': float(parts[4]),
                    'centroid_y': float(parts[5]),
                    'circularity': float(parts[6]),
                    'solidity': float(parts[7]),
                    'shadow_ratio': float(parts[8]),
                    'edge_gradient': float(parts[9])
                }

        print(f"  开始高质量生成 {len(bubble_analysis_data)} 个气泡图像...")

        # 设置信号处理器用于优雅退出
        global _shutdown_requested
        _shutdown_requested = False
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        # 准备并行处理的参数
        bubble_tasks = []
        for image_rel_path, params in bubble_analysis_data.items():
            task_args = (image_rel_path, params, base_path, hq_output_dir, base_csv_path,
                        bubble_selector, filtering_params)
            bubble_tasks.append(task_args)

        # 确定线程数量（基于CPU核心数，但不超过任务数量）
        max_workers = min(cpu_count(), len(bubble_tasks), 8)  # 限制最大8个线程避免过度并发
        print(f"  使用 {max_workers} 个线程进行并行处理...")

        # 初始化进度条
        hq_progress = tqdm(total=len(bubble_tasks), desc="高质量生成", unit="图像")

        completed_tasks = 0
        start_time = time.time()

        # 使用ThreadPoolExecutor进行并行处理
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_task = {executor.submit(_process_single_bubble, task): task for task in bubble_tasks}

                # 处理完成的任务
                for future in as_completed(future_to_task):
                    if _is_shutdown_requested():
                        print("\n正在取消剩余任务...")
                        # 取消未完成的任务
                        for f in future_to_task:
                            if not f.done():
                                f.cancel()
                        break

                    try:
                        image_rel_path, result, success = future.result()
                        completed_tasks += 1

                        # 更新进度条
                        elapsed_time = time.time() - start_time
                        if completed_tasks > 0:
                            avg_time_per_task = elapsed_time / completed_tasks
                            remaining_tasks = len(bubble_tasks) - completed_tasks
                            eta = avg_time_per_task * remaining_tasks
                            hq_progress.set_postfix({
                                'ETA': f'{eta:.0f}s',
                                'Speed': f'{completed_tasks/elapsed_time:.1f}/s'
                            })
                        hq_progress.update(1)
                        # 处理结果
                        if result and success:
                            hq_success_count += 1

                            # 适应新的返回格式
                            output_path = result.get('best_image_path', result.get('output_path', 'unknown'))
                            similarity_score = result.get('similarity_score', result.get('final_score', 0.0))
                            method = result.get('method', 'unknown')

                            # 兼容旧格式和新格式
                            batches_used = result.get('batches_used', 1 if method == 'pregenerated_dataset_filtering' else 0)
                            total_generated = result.get('total_generated', 1 if method == 'pregenerated_dataset_filtering' else 0)
                            threshold = result.get('final_threshold', result.get('iou_threshold', 0.0))

                            report_line = f"{image_rel_path}\t{output_path}\t{batches_used}\t{total_generated}\t{similarity_score:.6f}\t{threshold:.2f}\t{method}"

                            # 添加详细指标（如果有）
                            if 'metrics' in result:
                                metrics = result['metrics']
                                report_line += f"\t{metrics['mse_similarity']:.6f}\t{metrics['cosine_similarity']:.6f}\t{metrics['mask_iou']:.6f}\t{metrics['fourier_similarity']:.6f}"

                            if 'note' in result:
                                report_line += f"\t{result['note']}"

                            hq_generation_report.append(report_line)
                        elif result:
                            reason = result.get('reason', 'unknown')
                            error_msg = result.get('error', result.get('reason', 'Unknown error'))
                            method = result.get('method', 'unknown')
                            error_line = f"{image_rel_path}\tFAILED\t0\t0\t0.0\t0.0\t{method}\t{error_msg}"
                            hq_generation_report.append(error_line)
                        else:
                            error_line = f"{image_rel_path}\tERROR\t0\t0\t0.0\t0.0\tprocessing_error"
                            hq_generation_report.append(error_line)

                    except Exception as e:
                        error_line = f"{image_rel_path}\tERROR\t0\t0\t0.0\t0.0\t{str(e)}"
                        hq_generation_report.append(error_line)

        except KeyboardInterrupt:
            print("\n收到键盘中断，正在优雅退出...")
            _shutdown_requested = True
        except Exception as e:
            print(f"\n并行处理出现错误: {e}")
        finally:

            # 关闭高质量生成进度条
            hq_progress.close()

            # 输出处理统计信息
            elapsed_time = time.time() - start_time
            print(f"\n  并行处理完成:")
            print(f"    总任务数: {len(bubble_tasks)}")
            print(f"    完成任务数: {completed_tasks}")
            print(f"    成功任务数: {hq_success_count}")
            print(f"    总耗时: {elapsed_time:.1f}秒")
            if completed_tasks > 0:
                print(f"    平均速度: {completed_tasks/elapsed_time:.1f}任务/秒")

            if _is_shutdown_requested():
                print(f"    注意: 由于用户中断，仅完成了 {completed_tasks}/{len(bubble_tasks)} 个任务")
                print(f"    已保存 {hq_success_count} 个成功结果")

        # 保存高质量生成报告
        if hq_generation_report:
            hq_report_path = os.path.join(base_path, f'{str(i_projection).zfill(3)}/hq_generation_report.txt')
            try:
                with open(hq_report_path, 'w', encoding='utf-8') as f:
                    # 写入表头
                    f.write("Original_Image\tHQ_Image\tBatches_Used\tTotal_Generated\tFinal_Score\tFinal_Threshold\tMSE_Similarity\tCosine_Similarity\tMask_IOU\tFourier_Similarity\tNotes\n")
                    # 写入报告数据
                    for line in hq_generation_report:
                        f.write(line + "\n")
            except Exception as e:
                pass  # 不输出详细错误信息

    # 气泡流场合成功能
    if enable_bubble_composition:
        try:
            projection_dir = os.path.join(base_path, f'{str(i_projection).zfill(3)}')

            # 检查是否有分析结果数据
            if analysis_results:
                # 解析分析结果为字典格式
                bubble_analysis_data = {}
                for line in analysis_results:
                    parts = line.split('\t')
                    if len(parts) >= 10:
                        image_path = parts[0]
                        bubble_analysis_data[image_path] = {
                            'angle': float(parts[1]),
                            'major_axis_length': float(parts[2]),
                            'minor_axis_length': float(parts[3]),
                            'centroid_x': float(parts[4]),
                            'centroid_y': float(parts[5]),
                            'circularity': float(parts[6]),
                            'solidity': float(parts[7]),
                            'shadow_ratio': float(parts[8]),
                            'edge_gradient': float(parts[9])
                        }

                if bubble_analysis_data:
                    print(f"  开始完整流场合成，气泡数量: {len(bubble_analysis_data)}")
                    print(f"  坐标映射参数: min_x={min_x:.3f}, min_y={min_y:.3f}, scale={scale}")

                    # 导入完整流场合成函数
                    from .flow_composition import generate_flow_field_composition

                    # 加载气泡位置信息（与原始渲染保持一致）
                    bubble_positions = load_bubble_positions(base_path)

                    # 调用完整流场合成，传递原始渲染的坐标参数
                    composition_results = generate_flow_field_composition(
                        projection_dir=projection_dir,
                        bubble_analysis_data=bubble_analysis_data,
                        bubble_positions=bubble_positions,  # 传递3D位置信息
                        projection_point=point_fibonacci,   # 传递投影点
                        v_vector=v,                        # 传递旋转向量
                        scale_factor=scale,                # 传递缩放因子
                        width_pix=canvas_range_x,
                        height_pix=canvas_range_y,
                        pad=128,
                        location_dis='gaussian',  # 当use_3d_positions=False时使用
                        x_e=0.5,
                        x_sd=canvas_range_x * 0.2,
                        x_pad=24,
                        threshold=50,
                        enable_gauss=True,
                        enable_median=True,
                        save_outputs=True,
                        use_3d_positions=True,  # 启用3D位置模式
                        # 传递原始渲染的坐标参数
                        min_x=min_x,
                        min_y=min_y,
                        auto_adjust_canvas=True  # 启用自动画布调整
                    )

                    success_count = composition_results.get('successful_compositions', 0)
                    print(f"  完整流场合成完成，成功合成 {success_count} 个气泡")

                else:
                    print("  没有可用的气泡分析数据，跳过完整流场合成")
            else:
                print("  没有分析结果，跳过完整流场合成")

            # 保持原有的简单合成作为备选方案
            bubble_positions = load_bubble_positions(base_path)
            if bubble_positions and not analysis_results:
                print("  使用简单流场合成作为备选方案")
                use_hq = enable_hq_generation
                canvas, success_count = create_flow_field_composition(
                    projection_dir=projection_dir,
                    bubble_positions=bubble_positions,
                    projection_point=point_fibonacci,
                    v_vector=v,
                    canvas_size=None,
                    scale_factor=scale,
                    use_hq_images=use_hq,
                    volume_size_x=volume_size_x,
                    volume_size_y=volume_size_y,
                    volume_height=volume_height
                )

                if use_hq:
                    composition_path = os.path.join(projection_dir, f'flow_field_hq_composition.png')
                else:
                    composition_path = os.path.join(projection_dir, f'flow_field_composition.png')

                cv2.imwrite(composition_path, canvas)
                print(f"  简单流场合成完成，成功合成 {success_count} 个气泡")

        except Exception as e:
            print(f"  流场合成失败: {e}")
            # 不输出详细错误信息，但保留基本错误提示


def generater(stl_files, base_path, volume_size_x, volume_size_y, volume_height, gas_holdups, alpha, truncation, poisson_max_iter, sample_spacing, enable_hq_generation=False, enable_bubble_composition=False,
              use_pregenerated_dataset=True, pregenerated_image_dir=None, pregenerated_struct_csv=None,
              a_tolerance=0.1, b_tolerance=0.1, cx_tolerance=5.0, cy_tolerance=5.0, sr_tolerance=0.1,
              enable_iou_filtering=True, iou_threshold=0.3, min_similarity_score=0.5, screening_pool_size=10000,
              gpu_ids=None, max_gpus=4, enable_gpu_acceleration=True, gpu_batch_size=16):
    """
    主要的流场生成函数

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
        a_tolerance, b_tolerance: 几何参数容差
        cx_tolerance, cy_tolerance: 重心坐标容差
        sr_tolerance: 阴影比容差
        enable_iou_filtering: 是否启用IoU筛选
        iou_threshold: IoU阈值
        min_similarity_score: 最小相似度阈值
        screening_pool_size: 气泡筛选时的候选池大小，控制从预生成数据集中筛选的最大候选数量
        gpu_ids: 指定使用的GPU ID列表
        max_gpus: 最大使用GPU数量
        enable_gpu_acceleration: 是否启用GPU加速
        gpu_batch_size: GPU批处理大小
    """
    # 初始化预生成数据集筛选系统
    dataset_manager = None
    bubble_selector = None

    if use_pregenerated_dataset and pregenerated_image_dir and pregenerated_struct_csv:
        try:
            from .pregenerated_dataset_manager import PregeneratedDatasetManager
            from .generate_flow_field_optimized import OptimizedBubbleSelector

            print("正在初始化预生成数据集筛选系统...")
            dataset_manager = PregeneratedDatasetManager(pregenerated_image_dir, pregenerated_struct_csv)
            bubble_selector = OptimizedBubbleSelector(dataset_manager)
            print("预生成数据集筛选系统初始化完成")
        except Exception as e:
            print(f"预生成数据集筛选系统初始化失败: {e}")
            print("回退到传统CPU筛选模式")
            use_pregenerated_dataset = False
    else:
        print("使用传统CPU筛选模式")
        use_pregenerated_dataset = False

    try:
        # 气泡筛选处理
        print(f"开始处理 {len(gas_holdups)} 个气含率...")
        for gas_idx, gas_holdup in enumerate(gas_holdups):
            print(f"正在处理气含率 {gas_idx + 1}/{len(gas_holdups)}: {gas_holdup}")
            expected_volume = volume_size_x * volume_size_y * volume_height * gas_holdup

            names = []
            meshes = []
            meshes_origin = []
            volumes = []
            allocated_meshes = []
            allocated_origin_meshes = []
            total_volume = 0

            chosen_volumes = []
            while total_volume < expected_volume:
                chosen_volume = np.random.lognormal(mean=3.5, sigma=1.0) / 1000
                chosen_volumes.append(chosen_volume)
                total_volume += chosen_volume

            with mp.Pool(processes=mp.cpu_count()) as pool:
                mesh_data = pool.starmap(upsample_and_scale_mesh, [(stl_files, 20000, vol, sample_spacing) for vol in chosen_volumes])

            for stl_file, mesh, mesh_origin, volume in mesh_data:
                names.append(stl_file)
                meshes.append(mesh)
                meshes_origin.append(mesh_origin)
                volumes.append(volume * 10)

            points = generate_points_in_cube(len(meshes),
                                                cube_size=np.array([volume_size_x, volume_size_y, volume_height]) * 1.2, poisson_max_iter = poisson_max_iter)

            for mesh, mesh_origin, point in zip(meshes, meshes_origin, points):
                mesh.points += point
                allocated_meshes.append(mesh)
                mesh_origin.points += point
                allocated_origin_meshes.append(mesh_origin)

            with open(os.path.join(base_path, 'names_points.csv'), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Name', 'X', 'Y', 'Z', 'Volume'])
                for name, point, volume in zip(names, points, volumes):
                    writer.writerow([name, point[0], point[1], point[2], volume])

            mesh = pv.merge([mesh for mesh in allocated_meshes])
            mesh_origin = pv.merge([mesh for mesh in allocated_origin_meshes])
            mesh_origin.save(os.path.join(base_path, f'{gas_holdup}.stl'))

            points_fibonacci = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
            points_output_path = os.path.join(base_path, "points_view.csv")
            np.savetxt(points_output_path, points_fibonacci, delimiter=",")
            v = np.array([0.01, 0.01, 1])
            i_projection = 0
            scale = 100

            # 视角渲染处理
            print(f"开始处理 {len(points_fibonacci)} 个视角...")

            # 准备筛选参数
            filtering_params = {
                'a_tolerance': a_tolerance,
                'b_tolerance': b_tolerance,
                'cx_tolerance': cx_tolerance,
                'cy_tolerance': cy_tolerance,
                'sr_tolerance': sr_tolerance,
                'enable_iou_filtering': enable_iou_filtering,
                'iou_threshold': iou_threshold,
                'min_similarity_score': min_similarity_score,
                'screening_pool_size': screening_pool_size
            }

            for i_projection, point_fibonacci in enumerate(points_fibonacci):
                print(f"正在处理视角 {i_projection + 1}/{len(points_fibonacci)}")

                # 调用单个投影处理函数
                mp_args = (i_projection, point_fibonacci, allocated_meshes, allocated_origin_meshes, base_path, gas_holdup, v, scale, alpha, truncation, enable_hq_generation, enable_bubble_composition,
                          use_pregenerated_dataset, bubble_selector, filtering_params, gpu_ids, max_gpus, enable_gpu_acceleration, gpu_batch_size)
                process_projection(mp_args)

                print(f"视角 {i_projection + 1} 处理完成")

                # 每个视角处理完成后进行内存清理
                gc.collect()

            print(f"气含率 {gas_holdup} 处理完成")

            # 清理大型数据结构和内存
            del allocated_meshes, allocated_origin_meshes
            del meshes, meshes_origin, mesh_data
            if 'mesh' in locals():
                del mesh
            if 'mesh_origin' in locals():
                del mesh_origin
            gc.collect()

            # 保存当前py文件到base_path目录下
            current_file_path = __file__
            destination_path = os.path.join(base_path, 'FlowRenderer.py')
            shutil.copy(current_file_path, destination_path)

    finally:
        # 最终内存清理（GPU资源清理代码已移除）
        gc.collect()
        print("资源清理完成")
