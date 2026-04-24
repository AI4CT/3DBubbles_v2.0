"""
尺寸工具模块

该模块提供处理气泡图像尺寸信息的工具函数，包括：
- 从文件名提取原始尺寸信息
- 生成包含尺寸信息的文件名
- 处理图像尺寸恢复逻辑

主要功能：
- 尺寸信息的文件名编码和解码
- 支持padding和resize两种处理方式的尺寸恢复
- 图像尺寸变换的工具函数
"""

import os
import re
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


def extract_size_from_filename(filename: str) -> Optional[int]:
    """
    从文件名中提取原始尺寸信息
    
    支持的格式：
    - bubble_013_size140.png -> 140
    - hq_bubble_005_size64.png -> 64  
    - image_001_size256.jpg -> 256
    
    Args:
        filename: 文件名
        
    Returns:
        原始尺寸（整数），如果没有找到则返回None
    """
    # 使用正则表达式匹配 _size数字 的模式
    size_match = re.search(r'_size(\d+)', filename)
    if size_match:
        return int(size_match.group(1))
    return None


def generate_filename_with_size(base_filename: str, original_size: int) -> str:
    """
    生成包含尺寸信息的文件名
    
    Args:
        base_filename: 基础文件名 (如 "bubble_013.png")
        original_size: 原始尺寸
        
    Returns:
        包含尺寸信息的文件名 (如 "bubble_013_size140.png")
    """
    name, ext = os.path.splitext(base_filename)
    return f"{name}_size{original_size}{ext}"


def remove_size_from_filename(filename: str) -> str:
    """
    从文件名中移除尺寸信息
    
    Args:
        filename: 包含尺寸信息的文件名
        
    Returns:
        移除尺寸信息后的文件名
    """
    return re.sub(r'_size\d+', '', filename)


def get_processing_method_from_size(original_size: int, target_size: int = 128) -> str:
    """
    根据原始尺寸确定处理方法
    
    Args:
        original_size: 原始尺寸
        target_size: 目标尺寸 (默认128)
        
    Returns:
        处理方法: 'padding' 或 'resize'
    """
    return 'padding' if original_size <= target_size else 'resize'


def calculate_padding_params(original_size: int, target_size: int = 128) -> Dict[str, int]:
    """
    计算填充参数
    
    Args:
        original_size: 原始尺寸
        target_size: 目标尺寸
        
    Returns:
        填充参数字典，包含 top, bottom, left, right
    """
    if original_size >= target_size:
        return {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    
    pad_total = target_size - original_size
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    
    return {
        'top': pad_top,
        'bottom': pad_bottom,
        'left': pad_left,
        'right': pad_right
    }


def restore_image_size(image: np.ndarray, original_size: int, current_size: int = 128, 
                      processing_method: Optional[str] = None) -> np.ndarray:
    """
    将图像从当前尺寸恢复到原始尺寸
    
    Args:
        image: 输入图像 (current_size x current_size)
        original_size: 目标原始尺寸
        current_size: 当前图像尺寸
        processing_method: 处理方法 ('padding' 或 'resize')，如果为None则自动判断
        
    Returns:
        恢复到原始尺寸的图像
    """
    if original_size == current_size:
        return image
    
    # 如果没有指定处理方法，自动判断
    if processing_method is None:
        processing_method = get_processing_method_from_size(original_size, current_size)
    
    if processing_method == 'padding':
        # 如果原始图像是通过填充得到的，需要裁剪掉填充部分
        if original_size >= current_size:
            # 这种情况不应该发生，但为了安全起见
            return cv2.resize(image, (original_size, original_size), interpolation=cv2.INTER_LINEAR)
        
        # 计算裁剪区域
        pad_params = calculate_padding_params(original_size, current_size)
        top, bottom, left, right = pad_params['top'], pad_params['bottom'], pad_params['left'], pad_params['right']
        
        # 裁剪掉填充部分
        cropped_image = image[top:current_size-bottom, left:current_size-right]
        return cropped_image
        
    else:  # resize
        # 如果原始图像是通过缩放得到的，直接放大到原始尺寸
        return cv2.resize(image, (original_size, original_size), interpolation=cv2.INTER_LINEAR)


def scale_image_to_standard_size(image: np.ndarray, target_size: int = 128, 
                                fill_color: Tuple[int, int, int] = (255, 255, 255)) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    将图像缩放到标准尺寸，并返回处理信息
    
    Args:
        image: 输入图像
        target_size: 目标尺寸
        fill_color: 填充颜色 (RGB)
        
    Returns:
        (处理后的图像, 处理信息字典)
    """
    original_size = max(image.shape[:2])
    processing_info = {
        'original_size': original_size,
        'target_size': target_size,
        'processing_method': get_processing_method_from_size(original_size, target_size),
        'scale_factor': 1.0
    }
    
    if original_size <= target_size:
        # 小尺寸：使用填充
        pad_params = calculate_padding_params(original_size, target_size)
        
        processed_image = cv2.copyMakeBorder(
            image,
            pad_params['top'], pad_params['bottom'],
            pad_params['left'], pad_params['right'],
            cv2.BORDER_CONSTANT,
            value=fill_color
        )
        processing_info['padding'] = pad_params
        
    else:
        # 大尺寸：使用缩放
        scale_factor = target_size / original_size
        processed_image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        processing_info['scale_factor'] = scale_factor
    
    return processed_image, processing_info


def get_size_info_from_bubble_params(bubble_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    从气泡参数中提取尺寸相关信息
    
    Args:
        bubble_params: 气泡参数字典
        
    Returns:
        尺寸信息字典
    """
    size_info = {}
    
    # 提取现有的尺寸信息
    keys_to_extract = [
        'original_canvas_size', 'resized_canvas_size', 'processing_method',
        'resize_scale_factor', 'canvas_size'
    ]
    
    for key in keys_to_extract:
        if key in bubble_params:
            size_info[key] = bubble_params[key]
    
    return size_info


def validate_size_consistency(filename: str, bubble_params: Dict[str, Any]) -> bool:
    """
    验证文件名中的尺寸信息与参数中的尺寸信息是否一致
    
    Args:
        filename: 文件名
        bubble_params: 气泡参数
        
    Returns:
        是否一致
    """
    filename_size = extract_size_from_filename(filename)
    if filename_size is None:
        return False
    
    param_size = bubble_params.get('original_canvas_size')
    if param_size is None:
        return False
    
    return filename_size == param_size


def create_size_mapping_file(output_dir: str, bubble_params_list: list) -> str:
    """
    创建尺寸映射文件，记录所有气泡的尺寸信息
    
    Args:
        output_dir: 输出目录
        bubble_params_list: 气泡参数列表
        
    Returns:
        映射文件路径
    """
    import json
    
    size_mapping = {}
    
    for params in bubble_params_list:
        if 'image_path' in params:
            filename = os.path.basename(params['image_path'])
            size_info = get_size_info_from_bubble_params(params)
            if size_info:
                size_mapping[filename] = size_info
    
    mapping_file = os.path.join(output_dir, 'bubble_size_mapping.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(size_mapping, f, indent=2, ensure_ascii=False)
    
    return mapping_file


def load_size_mapping(mapping_file: str) -> Dict[str, Dict[str, Any]]:
    """
    加载尺寸映射文件
    
    Args:
        mapping_file: 映射文件路径
        
    Returns:
        尺寸映射字典
    """
    import json
    
    if not os.path.exists(mapping_file):
        return {}
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载尺寸映射文件失败: {e}")
        return {}


# 测试函数
def test_size_utils():
    """测试尺寸工具函数"""
    print("测试尺寸工具函数...")
    
    # 测试文件名处理
    test_filename = "bubble_013.png"
    filename_with_size = generate_filename_with_size(test_filename, 140)
    print(f"生成带尺寸文件名: {test_filename} -> {filename_with_size}")
    
    extracted_size = extract_size_from_filename(filename_with_size)
    print(f"提取尺寸: {filename_with_size} -> {extracted_size}")
    
    filename_without_size = remove_size_from_filename(filename_with_size)
    print(f"移除尺寸: {filename_with_size} -> {filename_without_size}")
    
    # 测试处理方法判断
    method1 = get_processing_method_from_size(64)
    method2 = get_processing_method_from_size(256)
    print(f"处理方法: size=64 -> {method1}, size=256 -> {method2}")
    
    # 测试填充参数计算
    pad_params = calculate_padding_params(64, 128)
    print(f"填充参数 (64->128): {pad_params}")
    
    print("测试完成!")


if __name__ == "__main__":
    test_size_utils()
