# bubble_analysis.py
"""
气泡图像分析模块

该模块包含气泡图像分析相关的函数，用于提取气泡的结构表征参数、
计算相似度指标和傅里叶描述子等。

主要功能：
- 气泡图像结构参数分析（支持GPU加速）
- 相似度指标计算
- 傅里叶描述子计算
- 余弦相似度计算
- GPU批量处理优化
"""

import cv2
import numpy as np
import math
import warnings
from typing import Optional, Union, List, Dict, Any, Tuple

# GPU加速相关导入
try:
    import torch
    import torch.nn.functional as F
    import kornia
    import kornia.morphology as morph
    import kornia.feature as KF
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch或kornia未安装，将使用CPU模式进行图像分析")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# 导入性能管理器
from .gpu_performance_manager import get_global_performance_manager


def cosine_similarity_manual(a, b):
    """
    手动实现余弦相似度计算

    Args:
        a: 第一个向量
        b: 第二个向量

    Returns:
        float: 余弦相似度值
    """
    a = a.flatten()
    b = b.flatten()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _get_device(device: Optional[str] = None) -> torch.device:
    """获取计算设备"""
    if not TORCH_AVAILABLE:
        return None

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return torch.device(device)


def _image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """将图像转换为GPU张量"""
    if len(image.shape) == 3:
        # BGR to RGB and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    else:
        # Grayscale
        tensor = torch.from_numpy(image).float() / 255.0
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)

    return tensor.unsqueeze(0).to(device)  # Add batch dimension


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """将GPU张量转换回图像"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    if tensor.shape[0] == 1:
        # Grayscale
        image = tensor.squeeze(0).cpu().numpy()
    else:
        # RGB to BGR
        image = tensor.permute(1, 2, 0).cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return (image * 255.0).astype(np.uint8)


def _gpu_threshold(image_tensor: torch.Tensor, threshold: float = 180/255.0) -> torch.Tensor:
    """GPU加速的二值化处理"""
    return (image_tensor > threshold).float()


def _gpu_find_contours(binary_tensor: torch.Tensor) -> List[np.ndarray]:
    """GPU加速的轮廓检测（混合GPU-CPU实现）"""
    # 将二值化结果转回CPU进行轮廓检测
    binary_image = _tensor_to_image(binary_tensor)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)


def _gpu_ellipse_fitting(contour: np.ndarray) -> Tuple[float, float, float, float, float]:
    """GPU加速的椭圆拟合"""
    if len(contour) < 5:
        return 0, 0, 0, 0, 0

    try:
        ellipse = cv2.fitEllipse(contour)
        (cX, cY), (MA, ma), angle = ellipse
        # 修复：使用完整轴长而不是半轴长，与analyze_bubble_image函数保持一致
        a, b = max(MA, ma), min(MA, ma)  # 长轴、短轴（完整轴长）
        return a, b, angle, cX, cY
    except:
        return 0, 0, 0, 0, 0


def _gpu_calculate_moments(image_tensor: torch.Tensor, contour: np.ndarray) -> Tuple[float, float]:
    """GPU加速的重心计算"""
    try:
        # 创建掩码
        mask = np.zeros(image_tensor.shape[-2:], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # 转换为GPU张量
        mask_tensor = torch.from_numpy(mask).float().to(image_tensor.device) / 255.0

        # 反转图像（255 - gray）
        inverted_image = 1.0 - image_tensor.squeeze()

        # 应用掩码
        masked_image = inverted_image * mask_tensor

        # 计算重心
        threshold = 20.0 / 255.0
        valid_pixels = masked_image > threshold

        if valid_pixels.sum() > 0:
            y_coords, x_coords = torch.meshgrid(
                torch.arange(masked_image.shape[0], device=image_tensor.device),
                torch.arange(masked_image.shape[1], device=image_tensor.device),
                indexing='ij'
            )

            weights = masked_image[valid_pixels]
            weighted_sum_x = (x_coords[valid_pixels] * weights).sum()
            weighted_sum_y = (y_coords[valid_pixels] * weights).sum()
            total_weight = weights.sum()

            cX = (weighted_sum_x / total_weight).cpu().item()
            cY = (weighted_sum_y / total_weight).cpu().item()
        else:
            cX = cY = 0

        return cX, cY
    except Exception as e:
        return 0, 0


def _gpu_calculate_shape_metrics(contour: np.ndarray, a: float, b: float) -> Tuple[float, float]:
    """GPU加速的形状指标计算"""
    try:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # 圆形度
        circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        # 实心度（Solidity）
        ellipse_area = math.pi * a * b
        solidity = 1 - abs((area + 1e-6) / (ellipse_area + 1e-6) - 1) if ellipse_area > 0 else 0

        return circularity, solidity
    except:
        return 0, 0


def _gpu_calculate_shadow_ratio(image_tensor: torch.Tensor, contour: np.ndarray) -> float:
    """GPU加速的阴影比计算"""
    try:
        # 创建掩码
        mask = np.zeros(image_tensor.shape[-2:], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # 转换为GPU张量
        mask_tensor = torch.from_numpy(mask).float().to(image_tensor.device) / 255.0

        # 应用掩码
        masked_image = image_tensor.squeeze() * mask_tensor

        # 计算阴影比
        valid_pixels = mask_tensor > 0
        if valid_pixels.sum() > 0:
            bubble_pixels = masked_image[valid_pixels]
            dark_pixels = (bubble_pixels < 128.0/255.0).sum()
            shadow_ratio = (dark_pixels.float() / valid_pixels.sum()).cpu().item()
        else:
            shadow_ratio = 0

        return shadow_ratio
    except:
        return 0


def _gpu_calculate_edge_gradient(image_tensor: torch.Tensor, contour: np.ndarray) -> float:
    """GPU加速的边缘梯度计算"""
    try:
        # 使用Kornia进行边缘检测
        edges = kornia.filters.canny(image_tensor, low_threshold=50/255.0, high_threshold=150/255.0)[1]

        # 创建掩码
        mask = np.zeros(image_tensor.shape[-2:], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mask_tensor = torch.from_numpy(mask).float().to(image_tensor.device) / 255.0

        # 应用掩码
        masked_edges = edges.squeeze() * mask_tensor

        # 计算平均边缘强度
        valid_pixels = mask_tensor > 0
        if valid_pixels.sum() > 0:
            edge_gradient = masked_edges[valid_pixels].mean().cpu().item()
        else:
            edge_gradient = 0

        return edge_gradient
    except:
        return 0


def analyze_bubble_image_gpu(image_path: str, device: Optional[str] = None,
                           standardize_orientation: bool = True) -> Optional[Dict[str, float]]:
    """
    GPU加速的气泡图像分析函数

    Args:
        image_path: 图像文件路径
        device: 计算设备 ('cuda:0', 'cpu' 等)
        standardize_orientation: 是否标准化气泡方向

    Returns:
        dict: 包含分析参数的字典，如果分析失败返回None
    """
    if not TORCH_AVAILABLE:
        # 回退到CPU版本
        return analyze_bubble_image(image_path, standardize_orientation)

    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告：无法读取图像 {image_path}")
            return None

        return analyze_bubble_image_from_array_gpu(img, device, standardize_orientation)

    except Exception as e:
        print(f"GPU分析图像 {image_path} 时发生错误: {e}")
        # 回退到CPU版本
        return analyze_bubble_image(image_path, standardize_orientation)


def analyze_bubble_image_from_array_gpu(image_array: np.ndarray, device: Optional[str] = None,
                                       standardize_orientation: bool = True) -> Optional[Dict[str, float]]:
    """
    GPU加速的从图像数组分析气泡参数

    Args:
        image_array: 图像数组
        device: 计算设备
        standardize_orientation: 是否标准化气泡方向

    Returns:
        dict: 包含分析参数的字典，如果分析失败返回None
    """
    if not TORCH_AVAILABLE:
        # 回退到CPU版本
        return analyze_bubble_image_from_array(image_array)

    try:
        device = _get_device(device)

        # 转换为灰度图像
        if len(image_array.shape) == 3:
            img_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image_array.copy()

        # 转换为GPU张量
        image_tensor = _image_to_tensor(img_gray, device)

        # GPU加速的二值化处理
        binary_tensor = _gpu_threshold(image_tensor, threshold=180/255.0)

        # 轮廓检测（混合GPU-CPU）
        contours = _gpu_find_contours(binary_tensor)

        if len(contours) < 2:
            return None

        # 选择第二大轮廓（第一个通常是整个图像边界）
        cnt = contours[1]
        area = cv2.contourArea(cnt)

        if area < 100:  # 面积太小，可能是噪声
            return None

        # 椭圆拟合
        a, b, angle, ellipse_cX, ellipse_cY = _gpu_ellipse_fitting(cnt)
        if a == 0 or b == 0:
            return None

        # GPU加速的重心计算
        cX, cY = _gpu_calculate_moments(image_tensor, cnt)

        # 形状指标计算
        circularity, solidity = _gpu_calculate_shape_metrics(cnt, a, b)

        # GPU加速的阴影比计算
        shadow_ratio = _gpu_calculate_shadow_ratio(image_tensor, cnt)

        # GPU加速的边缘梯度计算
        edge_gradient = _gpu_calculate_edge_gradient(image_tensor, cnt)

        # 返回分析结果（归一化）
        return {
            'angle': angle,
            'major_axis_length': a / 128,
            'minor_axis_length': b / 128,
            'centroid_x': cX / 128,
            'centroid_y': cY / 128,
            'circularity': circularity,
            'solidity': solidity,
            'shadow_ratio': shadow_ratio,
            'edge_gradient': edge_gradient
        }

    except Exception as e:
        print(f"GPU分析图像数组时发生错误: {e}")
        # 回退到CPU版本
        return analyze_bubble_image_from_array(image_array)


def analyze_bubble_batch_gpu(image_arrays: List[np.ndarray], device: Optional[str] = None,
                           batch_size: int = 8) -> List[Optional[Dict[str, float]]]:
    """
    GPU批量分析气泡图像

    Args:
        image_arrays: 图像数组列表
        device: 计算设备
        batch_size: 批处理大小

    Returns:
        List[Dict]: 分析结果列表
    """
    if not TORCH_AVAILABLE:
        # 回退到逐个CPU分析
        return [analyze_bubble_image_from_array(img) for img in image_arrays]

    results = []
    device = _get_device(device)

    # 分批处理
    for i in range(0, len(image_arrays), batch_size):
        batch = image_arrays[i:i + batch_size]
        batch_results = []

        try:
            # 预处理批次数据
            batch_tensors = []
            valid_indices = []

            for j, img in enumerate(batch):
                try:
                    if len(img.shape) == 3:
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        img_gray = img.copy()

                    tensor = _image_to_tensor(img_gray, device)
                    batch_tensors.append(tensor)
                    valid_indices.append(j)
                except:
                    continue

            if not batch_tensors:
                batch_results = [None] * len(batch)
            else:
                # 批量GPU处理
                batch_tensor = torch.cat(batch_tensors, dim=0)

                # 批量二值化
                binary_batch = _gpu_threshold(batch_tensor, threshold=180/255.0)

                # 逐个处理轮廓检测和参数计算（这部分难以完全批量化）
                for k, tensor_idx in enumerate(valid_indices):
                    try:
                        single_tensor = batch_tensor[k:k+1]
                        single_binary = binary_batch[k:k+1]

                        # 轮廓检测
                        contours = _gpu_find_contours(single_binary)

                        if len(contours) < 2:
                            batch_results.append(None)
                            continue

                        cnt = contours[1]
                        area = cv2.contourArea(cnt)

                        if area < 100:
                            batch_results.append(None)
                            continue

                        # 参数计算
                        a, b, angle, _, _ = _gpu_ellipse_fitting(cnt)
                        if a == 0 or b == 0:
                            batch_results.append(None)
                            continue

                        cX, cY = _gpu_calculate_moments(single_tensor, cnt)
                        circularity, solidity = _gpu_calculate_shape_metrics(cnt, a, b)
                        shadow_ratio = _gpu_calculate_shadow_ratio(single_tensor, cnt)
                        edge_gradient = _gpu_calculate_edge_gradient(single_tensor, cnt)

                        result = {
                            'angle': angle,
                            'major_axis_length': a / 128,
                            'minor_axis_length': b / 128,
                            'centroid_x': cX / 128,
                            'centroid_y': cY / 128,
                            'circularity': circularity,
                            'solidity': solidity,
                            'shadow_ratio': shadow_ratio,
                            'edge_gradient': edge_gradient
                        }
                        batch_results.append(result)

                    except Exception as e:
                        batch_results.append(None)

                # 填充无效索引的结果
                final_batch_results = [None] * len(batch)
                for j, valid_idx in enumerate(valid_indices):
                    if j < len(batch_results):
                        final_batch_results[valid_idx] = batch_results[j]
                batch_results = final_batch_results

        except Exception as e:
            print(f"批量GPU处理失败: {e}")
            # 回退到逐个CPU分析
            batch_results = [analyze_bubble_image_from_array(img) for img in batch]

        results.extend(batch_results)

    return results


def analyze_bubble_image_smart(image_path: str, device: Optional[str] = None,
                              standardize_orientation: bool = True) -> Optional[Dict[str, float]]:
    """
    智能气泡图像分析函数（自动选择GPU或CPU）

    Args:
        image_path: 图像文件路径
        device: 计算设备（可选，自动选择）
        standardize_orientation: 是否标准化气泡方向

    Returns:
        dict: 包含分析参数的字典，如果分析失败返回None
    """
    performance_manager = get_global_performance_manager()

    def gpu_analysis():
        return analyze_bubble_image_gpu(image_path, device, standardize_orientation)

    def cpu_analysis():
        return analyze_bubble_image(image_path, standardize_orientation)

    return performance_manager.execute_with_fallback(gpu_analysis, cpu_analysis)


def analyze_bubble_image_from_array_smart(image_array: np.ndarray, device: Optional[str] = None,
                                         standardize_orientation: bool = True) -> Optional[Dict[str, float]]:
    """
    智能从图像数组分析气泡参数（自动选择GPU或CPU）

    Args:
        image_array: 图像数组
        device: 计算设备（可选，自动选择）
        standardize_orientation: 是否标准化气泡方向

    Returns:
        dict: 包含分析参数的字典，如果分析失败返回None
    """
    performance_manager = get_global_performance_manager()

    def gpu_analysis():
        return analyze_bubble_image_from_array_gpu(image_array, device, standardize_orientation)

    def cpu_analysis():
        return analyze_bubble_image_from_array(image_array)

    return performance_manager.execute_with_fallback(gpu_analysis, cpu_analysis)


def analyze_bubble_batch_smart(image_arrays: List[np.ndarray], device: Optional[str] = None,
                              batch_size: int = 8) -> List[Optional[Dict[str, float]]]:
    """
    智能批量分析气泡图像（自动选择GPU或CPU）

    Args:
        image_arrays: 图像数组列表
        device: 计算设备（可选，自动选择）
        batch_size: 批处理大小

    Returns:
        List[Dict]: 分析结果列表
    """
    performance_manager = get_global_performance_manager()

    def gpu_analysis():
        return analyze_bubble_batch_gpu(image_arrays, device, batch_size)

    def cpu_analysis():
        return [analyze_bubble_image_from_array(img) for img in image_arrays]

    return performance_manager.execute_with_fallback(gpu_analysis, cpu_analysis)


def get_analysis_performance_report() -> Dict[str, Any]:
    """获取分析性能报告"""
    performance_manager = get_global_performance_manager()
    return performance_manager.get_performance_report()


def reset_analysis_performance() -> None:
    """重置分析性能统计"""
    performance_manager = get_global_performance_manager()
    performance_manager.reset_metrics()


def calculate_bubble_similarity(bubble1_params, bubble2_params, weights=None):
    """
    计算两个气泡的综合相似度

    Args:
        bubble1_params: 第一个气泡的参数字典
        bubble2_params: 第二个气泡的参数字典
        weights: 参数权重字典，如果为None则使用默认权重

    Returns:
        float: 综合相似度分数 (0-1之间，1表示完全相似)
    """
    if weights is None:
        weights = {
            'major_axis_length': 0.25,
            'minor_axis_length': 0.25,
            'centroid_x': 0.15,
            'centroid_y': 0.15,
            'shadow_ratio': 0.20
        }

    # 核心参数
    core_params = ['major_axis_length', 'minor_axis_length', 'centroid_x', 'centroid_y', 'shadow_ratio']

    # 计算欧氏距离
    euclidean_distances = []
    for param in core_params:
        if param in bubble1_params and param in bubble2_params:
            diff = abs(bubble1_params[param] - bubble2_params[param])
            euclidean_distances.append(diff * weights[param])

    euclidean_distance = np.sqrt(sum([d**2 for d in euclidean_distances]))
    euclidean_similarity = 1.0 / (1.0 + euclidean_distance)

    # 计算所有参数的余弦相似度
    all_params = ['major_axis_length', 'minor_axis_length', 'centroid_x', 'centroid_y',
                  'circularity', 'solidity', 'shadow_ratio', 'edge_gradient']

    vec1 = []
    vec2 = []
    for param in all_params:
        if param in bubble1_params and param in bubble2_params:
            vec1.append(bubble1_params[param])
            vec2.append(bubble2_params[param])

    if len(vec1) > 0:
        cosine_sim = cosine_similarity_manual(np.array(vec1), np.array(vec2))
    else:
        cosine_sim = 0.0

    # 综合相似度 (欧氏距离相似度权重0.6，余弦相似度权重0.4)
    combined_similarity = 0.6 * euclidean_similarity + 0.4 * cosine_sim

    return max(0.0, min(1.0, combined_similarity))


def standardize_bubble_orientation(image, contour):
    """
    标准化气泡方向，使最长轴水平对齐

    Args:
        image: 输入图像
        contour: 气泡轮廓

    Returns:
        tuple: (标准化后的图像, 旋转角度)
    """
    # 椭圆拟合获取长轴方向
    if len(contour) < 5:
        return image, 0.0

    ellipse = cv2.fitEllipse(contour)
    angle = ellipse[2]  # 椭圆角度

    # 将角度转换为弧度
    angle_rad = np.radians(angle)

    # 计算旋转矩阵
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 执行旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
                                   borderValue=(255, 255, 255))  # 白色填充

    return rotated_image, angle


def analyze_bubble_image(image_path, standardize_orientation=True):
    """
    分析单气泡图像，提取结构表征参数

    Args:
        image_path: 图像文件路径
        standardize_orientation: 是否标准化气泡方向

    Returns:
        dict: 包含分析参数的字典，包括角度信息
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告：无法读取图像 {image_path}")
            return None

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 二值化处理
        _, img_bin = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if len(contours) < 2:
            print(f"警告：图像 {image_path} 中未找到足够的轮廓")
            return None

        # 使用第二大轮廓（第一个通常是背景）
        cnt = contours[1]

        # 记录原始角度
        original_angle = 0.0

        # 如果需要标准化方向
        if standardize_orientation:
            # 标准化气泡方向
            img, original_angle = standardize_bubble_orientation(img, cnt)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 重新进行二值化和轮廓检测
            _, img_bin = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            if len(contours) >= 2:
                cnt = contours[1]

        # 1. 椭圆拟合 - 获取长轴长、短轴长和角度
        ellipse = cv2.fitEllipse(cnt)
        a, b = sorted(ellipse[1], reverse=True)  # 长轴、短轴
        angle = ellipse[2]  # 椭圆长轴与水平轴的夹角（度）

        # 2. 计算灰度重心坐标
        gray_image = 255 - img_gray.copy()
        threshold = 20
        gray_sum = np.sum(gray_image[gray_image > threshold])

        if gray_sum > 0:
            x_coords = np.where(gray_image > threshold)[1]
            y_coords = np.where(gray_image > threshold)[0]
            weighted_sum_x = np.sum(x_coords * gray_image[gray_image > threshold])
            weighted_sum_y = np.sum(y_coords * gray_image[gray_image > threshold])
            cX = weighted_sum_x / gray_sum
            cY = weighted_sum_y / gray_sum
        else:
            cX = cY = 0

        # 3. 计算圆度
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        # 4. 计算Solidity（实心度）
        ellipse_area = math.pi * (a / 2) * (b / 2)
        solidity = 1 - np.abs((area + 1e-6) / (ellipse_area + 1e-6) - 1)

        # 5. 计算阴影比
        p = np.zeros(shape=img_gray.shape)
        cv2.drawContours(p, [cnt], -1, 255, -1)
        pixel = np.zeros(shape=(np.where(p == 255)[0].size, 1))
        xx = np.where(p == 255)[0].reshape(-1, 1)
        yy = np.where(p == 255)[1].reshape(-1, 1)

        for i, (x, y) in enumerate(zip(xx, yy)):
            pixel[i] = img_gray[x, y]

        shade_ratio = 1 - (np.mean(pixel) / 255) if len(pixel) > 0 else 0

        # 6. 计算边缘像素梯度
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.convertScaleAbs(sobely)
        sobelx = abs(sobelx) / 255
        sobely = abs(sobely) / 255
        sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

        contxy = cnt[:, 0, :]
        EG = []
        for ii in range(contxy.shape[0]):
            yy_coord = contxy[ii, 0]
            xx_coord = contxy[ii, 1]
            if 0 <= xx_coord < sobelxy.shape[0] and 0 <= yy_coord < sobelxy.shape[1]:
                EG.append(sobelxy[xx_coord, yy_coord])

        edge_gradient = np.mean(EG) if EG else 0

        # 返回分析结果（前4个参数需要归一化，角度保持原值）
        return {
            'angle': angle,                    # 标准化后的椭圆角度
            'original_angle': original_angle,  # 原始旋转角度
            'major_axis_length': a / 128,      # 归一化
            'minor_axis_length': b / 128,      # 归一化
            'centroid_x': cX / 128,            # 归一化
            'centroid_y': cY / 128,            # 归一化
            'circularity': circularity,        # 保持原值
            'solidity': solidity,              # 保持原值
            'shadow_ratio': shade_ratio,       # 保持原值
            'edge_gradient': edge_gradient     # 保持原值
        }

    except Exception as e:
        print(f"分析图像 {image_path} 时发生错误: {e}")
        return None


def analyze_bubble_image_from_array(image_array):
    """
    从图像数组分析气泡参数（类似analyze_bubble_image但输入是数组）

    Args:
        image_array: 图像数组

    Returns:
        dict: 包含9个分析参数的字典
    """
    try:
        if len(image_array.shape) == 3:
            img_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image_array.copy()

        # 二值化处理
        _, img_bin = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if len(contours) < 2:
            return None

        # 选择第二大轮廓（第一个通常是整个图像边界）
        cnt = contours[1]
        area = cv2.contourArea(cnt)

        if area < 100:  # 面积太小，可能是噪声
            return None

        # 椭圆拟合
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (cX, cY), (MA, ma), angle = ellipse
            # 修复：使用完整轴长而不是半轴长，与analyze_bubble_image函数保持一致
            a, b = sorted([MA, ma], reverse=True)  # 长轴、短轴（完整轴长）
        else:
            return None

        # 计算其他参数
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # 阴影比计算
        mask = np.zeros(img_gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        masked_img = cv2.bitwise_and(img_gray, img_gray, mask=mask)

        bubble_pixels = masked_img[mask == 255]
        if len(bubble_pixels) > 0:
            dark_pixels = np.sum(bubble_pixels < 128)
            shade_ratio = dark_pixels / len(bubble_pixels)
        else:
            shade_ratio = 0

        # 边缘梯度计算
        edges = cv2.Canny(img_gray, 50, 150)
        edge_pixels = edges[mask == 255]
        edge_gradient = np.mean(edge_pixels) / 255.0 if len(edge_pixels) > 0 else 0

        # 返回分析结果（归一化）
        return {
            'angle': angle,
            'major_axis_length': a / 128,
            'minor_axis_length': b / 128,
            'centroid_x': cX / 128,
            'centroid_y': cY / 128,
            'circularity': circularity,
            'solidity': solidity,
            'shadow_ratio': shade_ratio,
            'edge_gradient': edge_gradient
        }

    except Exception as e:
        return None


def calculate_fourier_descriptors(contour, num_descriptors=10):
    """
    计算轮廓的傅里叶描述子

    Args:
        contour: 轮廓点
        num_descriptors: 描述子数量

    Returns:
        descriptors: 傅里叶描述子
    """
    try:
        # 将轮廓转换为复数形式
        contour_complex = contour[:, 0, 0] + 1j * contour[:, 0, 1]

        # 计算傅里叶变换
        fourier_result = np.fft.fft(contour_complex)

        # 取前num_descriptors个描述子的幅值
        descriptors = np.abs(fourier_result[:num_descriptors])

        # 归一化（除以第一个描述子，避免尺度影响）
        if descriptors[0] != 0:
            descriptors = descriptors / descriptors[0]

        return descriptors
    except:
        return np.zeros(num_descriptors)


def calculate_similarity_metrics(original_image_path, generated_image, original_params):
    """
    计算4种相似度指标

    Args:
        original_image_path: 原始图像路径
        generated_image: 生成的图像
        original_params: 原始图像的9个参数

    Returns:
        dict: 包含4种相似度指标的字典
    """
    try:
        # 读取原始图像
        original_image = cv2.imread(original_image_path)
        if original_image is None:
            return None

        # 确保图像尺寸一致
        if original_image.shape != generated_image.shape:
            generated_image = cv2.resize(generated_image, (original_image.shape[1], original_image.shape[0]))

        # 1. MSE（均方误差）
        mse = np.mean((original_image.astype(float) - generated_image.astype(float)) ** 2)
        mse_similarity = 1.0 / (1.0 + mse / 1000.0)  # 归一化到0-1

        # 2. 表征参数余弦相似度
        generated_analysis = analyze_bubble_image_from_array(generated_image)
        if generated_analysis is not None:
            # 提取后8个参数（排除角度）
            original_features = np.array([
                original_params['major_axis_length'],
                original_params['minor_axis_length'],
                original_params['centroid_x'],
                original_params['centroid_y'],
                original_params['circularity'],
                original_params['solidity'],
                original_params['shadow_ratio'],
                original_params['edge_gradient']
            ]).reshape(1, -1)

            generated_features = np.array([
                generated_analysis['major_axis_length'],
                generated_analysis['minor_axis_length'],
                generated_analysis['centroid_x'],
                generated_analysis['centroid_y'],
                generated_analysis['circularity'],
                generated_analysis['solidity'],
                generated_analysis['shadow_ratio'],
                generated_analysis['edge_gradient']
            ]).reshape(1, -1)

            cosine_sim = cosine_similarity_manual(original_features, generated_features)
            cosine_sim = max(0, cosine_sim)  # 确保非负
        else:
            cosine_sim = 0.0

        # 3. Mask IOU
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        generated_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)

        _, original_mask = cv2.threshold(original_gray, 180, 255, cv2.THRESH_BINARY_INV)
        _, generated_mask = cv2.threshold(generated_gray, 180, 255, cv2.THRESH_BINARY_INV)

        intersection = np.logical_and(original_mask, generated_mask)
        union = np.logical_or(original_mask, generated_mask)

        if np.sum(union) > 0:
            iou = np.sum(intersection) / np.sum(union)
        else:
            iou = 0.0

        # 4. 轮廓傅里叶描述子相似度
        original_contours, _ = cv2.findContours(original_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        generated_contours, _ = cv2.findContours(generated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(original_contours) > 0 and len(generated_contours) > 0:
            original_contour = max(original_contours, key=cv2.contourArea)
            generated_contour = max(generated_contours, key=cv2.contourArea)

            original_fourier = calculate_fourier_descriptors(original_contour)
            generated_fourier = calculate_fourier_descriptors(generated_contour)

            # 计算傅里叶描述子的余弦相似度
            if np.linalg.norm(original_fourier) > 0 and np.linalg.norm(generated_fourier) > 0:
                fourier_sim = cosine_similarity_manual(original_fourier, generated_fourier)
                fourier_sim = max(0, fourier_sim)
            else:
                fourier_sim = 0.0
        else:
            fourier_sim = 0.0

        return {
            'mse_similarity': mse_similarity,
            'cosine_similarity': cosine_sim,
            'mask_iou': iou,
            'fourier_similarity': fourier_sim
        }

    except Exception as e:
        print(f"计算相似度指标时发生错误: {e}")
        return None
