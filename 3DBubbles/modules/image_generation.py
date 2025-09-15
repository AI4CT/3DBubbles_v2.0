# image_generation.py
"""
高质量图像生成模块

该模块包含高质量气泡图像生成相关的函数，主要用于调用StyleGAN2
生成高质量的气泡图像，并进行质量评估和筛选。

主要功能：
- 创建StyleGAN2所需的prompts文件
- 生成高质量气泡图像
- 图像质量评估和筛选
"""

import os
import sys
import tempfile
import shutil
import cv2
from tqdm import tqdm

# 导入相关模块
from .bubble_analysis import calculate_similarity_metrics
from .coordinate_transform import rotate_image


def create_bubble_prompts_file(bubble_params_list, output_file):
    """
    创建StyleGAN2需要的bubble_prompts.txt文件

    Args:
        bubble_params_list: 气泡参数列表，每个元素包含8个结构参数
        output_file: 输出文件路径
    """
    try:
        with open(output_file, 'w') as f:
            for i, params in enumerate(bubble_params_list):
                # 文件名格式：{i}.png
                filename = f"{i:02d}.png"

                # 写入8个参数（排除角度）
                line = f"{filename}\t{params['major_axis_length']:.9f}\t{params['minor_axis_length']:.9f}\t{params['centroid_x']:.9f}\t{params['centroid_y']:.9f}\t{params['circularity']:.9f}\t{params['solidity']:.9f}\t{params['shadow_ratio']:.9f}\t{params['edge_gradient']:.9f}\n"
                f.write(line)
    except Exception as e:
        print(f"创建bubble_prompts.txt文件时发生错误: {e}")


def _fallback_copy_original_image(original_image_path, output_dir):
    """
    回退方案：复制原始图像作为高质量图像

    Args:
        original_image_path: 原始图像路径
        output_dir: 输出目录

    Returns:
        dict: 结果信息
    """
    try:
        import shutil
        output_path = os.path.join(output_dir, 'hq_bubble.png')
        shutil.copy2(original_image_path, output_path)

        return {
            'success': True,
            'best_image_path': output_path,
            'similarity_score': 1.0,  # 原始图像相似度为1
            'method': 'original_copy_fallback',
            'note': 'Used original image as fallback'
        }
    except Exception as e:
        return {
            'success': False,
            'reason': 'copy_failed',
            'error': str(e),
            'method': 'original_copy_fallback'
        }


def generate_high_quality_bubble_image_from_pregenerated(original_image_path, original_params, output_dir,
                                                       bubble_selector, filtering_params, image_index=None):
    """
    从预生成数据集中筛选高质量气泡图像（替代GAN生成）

    Args:
        original_image_path: 原始预渲染图像路径
        original_params: 原始图像的参数字典
        output_dir: 输出目录
        bubble_selector: 预生成数据集筛选器
        filtering_params: 筛选参数字典
        image_index: 图像编号，用于生成唯一的输出文件名

    Returns:
        dict: 筛选结果信息
    """
    try:
        # 从原始参数中提取标准化后的参数
        target_params = {
            'major_axis_length': original_params.get('major_axis_length', 1.0),
            'minor_axis_length': original_params.get('minor_axis_length', 1.0),
            'centroid_x': original_params.get('centroid_x', 0.5),
            'centroid_y': original_params.get('centroid_y', 0.5),
            'shadow_ratio': original_params.get('shadow_ratio', 0.4),
            'circularity': original_params.get('circularity', 0.8),
            'solidity': original_params.get('solidity', 0.8),
            'edge_gradient': original_params.get('edge_gradient', 0.5)
        }

        print(f"    从预生成数据集筛选气泡: a={target_params['major_axis_length']:.3f}, "
              f"b={target_params['minor_axis_length']:.3f}, "
              f"cx={target_params['centroid_x']:.3f}, "
              f"cy={target_params['centroid_y']:.3f}")

        # 使用简化的筛选算法
        dataset_manager = bubble_selector.dataset_manager
        max_candidates = filtering_params.get('screening_pool_size', 10000)
        print(f"    使用筛选候选池大小: {max_candidates}")
        best_idx = dataset_manager.find_best_match_simplified(target_params, max_candidates=max_candidates)

        if best_idx is not None:
            # 获取匹配的气泡图像数据和参数
            bubble_image_data = dataset_manager.get_image_data(best_idx)
            bubble_params = dataset_manager.get_struct_params_normalized(best_idx)

            # 计算相似度分数
            from .bubble_analysis import calculate_bubble_similarity
            similarity_score = calculate_bubble_similarity(target_params, bubble_params)

            if similarity_score >= filtering_params.get('min_similarity_score', 0.1):
                # 保存筛选出的高质量图像
                if bubble_image_data is not None:
                    import cv2
                    import os

                    # 生成唯一的输出文件名
                    if image_index is not None:
                        # 使用提供的图像编号
                        output_filename = f'hq_bubble_{image_index:03d}.png'
                    else:
                        # 尝试从原始图像路径中提取编号
                        import re
                        basename = os.path.basename(original_image_path)
                        # 匹配文件名中的数字（如bubble_001.png, image_123.jpg等）
                        match = re.search(r'(\d+)', basename)
                        if match:
                            number = int(match.group(1))
                            output_filename = f'hq_bubble_{number:03d}.png'
                        else:
                            # 如果无法提取编号，使用时间戳确保唯一性
                            import time
                            timestamp = int(time.time() * 1000) % 100000
                            output_filename = f'hq_bubble_{timestamp:05d}.png'

                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, bubble_image_data)

                    print(f"    筛选成功: 相似度={similarity_score:.3f}, 索引={best_idx}")

                    return {
                        'success': True,
                        'best_image_path': output_path,
                        'similarity_score': similarity_score,
                        'selected_index': best_idx,
                        'selected_params': bubble_params,
                        'method': 'simplified_pregenerated_filtering'
                    }
                else:
                    print(f"    无法获取匹配的气泡图像数据: 索引={best_idx}")
                    return {
                        'success': False,
                        'reason': 'image_data_error',
                        'method': 'simplified_pregenerated_filtering'
                    }
            else:
                print(f"    筛选失败: 相似度过低 ({similarity_score:.3f})")
                return {
                    'success': False,
                    'reason': 'low_similarity_score',
                    'best_score': similarity_score,
                    'method': 'simplified_pregenerated_filtering'
                }
        else:
            print(f"    筛选失败: 未找到满足条件的气泡")
            return {
                'success': False,
                'reason': 'no_suitable_bubble_found',
                'best_score': 0.0,
                'method': 'simplified_pregenerated_filtering'
            }

    except Exception as e:
        print(f"    预生成数据集筛选失败: {e}")
        return {
            'success': False,
            'reason': 'filtering_error',
            'error': str(e),
            'method': 'simplified_pregenerated_filtering'
        }


def generate_high_quality_bubble_image(original_image_path, original_params, output_dir,
                                     bubble_selector=None, filtering_params=None, image_index=None,
                                     # 保持向后兼容性的已弃用参数
                                     base_csv_path=None, batch_size=64, max_batches=20,
                                     initial_threshold=0.9, threshold_decay=0.05, min_threshold=0.7):
    """
    生成高质量气泡图像（优化版本）

    优先使用预生成数据集筛选，如果不可用则回退到传统GAN生成逻辑。

    Args:
        original_image_path: 原始预渲染图像路径
        original_params: 原始图像的参数字典
        output_dir: 输出目录
        bubble_selector: 预生成数据集筛选器
        filtering_params: 筛选参数字典（包含screening_pool_size等参数）
        image_index: 图像编号，用于生成唯一的输出文件名

        # 以下参数已弃用，仅为向后兼容性保留
        base_csv_path: StyleGAN2基础CSV文件路径（已弃用）
        batch_size: 每批生成的图像数量（已弃用）
        max_batches: 最大批次数（已弃用）
        initial_threshold: 初始相似度阈值（已弃用）
        threshold_decay: 阈值衰减率（已弃用）
        min_threshold: 最小阈值（已弃用）

    Returns:
        dict: 生成结果信息
    """
    # 优先使用预生成数据集筛选
    if bubble_selector is not None and filtering_params is not None:
        return generate_high_quality_bubble_image_from_pregenerated(
            original_image_path, original_params, output_dir, bubble_selector, filtering_params, image_index
        )

    # 回退到原有的GAN生成逻辑（已弃用）
    print("    警告: 预生成数据集筛选不可用，回退到原有GAN生成逻辑")
    return _generate_high_quality_bubble_image_legacy(
        original_image_path, original_params, output_dir, base_csv_path,
        batch_size, max_batches, initial_threshold, threshold_decay, min_threshold
    )


def _generate_high_quality_bubble_image_legacy(original_image_path, original_params, output_dir,
                                             base_csv_path, batch_size=64, max_batches=20,
                                             initial_threshold=0.9, threshold_decay=0.05, min_threshold=0.7):
    """
    原有的GAN生成逻辑（已弃用，保留用于兼容性）
    """
    try:
        print("    警告: 使用已弃用的GAN生成逻辑，建议使用预生成数据集筛选")

        # 导入StyleGAN2相关模块（已弃用）
        # 获取项目根目录路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        stylegan2_path = os.path.join(project_root, 'Conditional_StyleGAN2')

        if not os.path.exists(stylegan2_path):
            print(f"    Conditional_StyleGAN2目录不存在: {stylegan2_path}")
            print("    回退到复制原始图像")
            return _fallback_copy_original_image(original_image_path, output_dir)

        if stylegan2_path not in sys.path:
            sys.path.append(stylegan2_path)

        try:
            from generate_1bubble import Generate_bubble_images
        except ImportError as e:
            print(f"    无法导入generate_1bubble模块: {e}")
            print("    回退到复制原始图像")
            return _fallback_copy_original_image(original_image_path, output_dir)

        # 创建临时工作目录
        temp_dir = tempfile.mkdtemp()
        temp_prompts_file = os.path.join(temp_dir, 'bubble_prompts.txt')
        temp_output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(temp_output_dir, exist_ok=True)

        current_threshold = initial_threshold
        best_image = None
        best_score = -1
        best_metrics = None
        total_generated = 0

        print(f"开始为图像 {os.path.basename(original_image_path)} 生成高质量版本...")

        for batch_idx in range(max_batches):
            print(f"  批次 {batch_idx + 1}/{max_batches}, 当前阈值: {current_threshold:.2f}")

            # 创建当前批次的参数列表（重复原始参数）
            batch_params = [original_params] * batch_size

            # 创建prompts文件
            create_bubble_prompts_file(batch_params, temp_prompts_file)

            # 清空输出目录
            for f in os.listdir(temp_output_dir):
                if f.endswith('.png'):
                    os.remove(os.path.join(temp_output_dir, f))

            # 调用StyleGAN2生成图像
            try:
                Generate_bubble_images(base_csv_path, temp_prompts_file, temp_output_dir)
                total_generated += batch_size
            except Exception as e:
                print(f"    StyleGAN2生成失败: {e}")
                continue

            # 处理生成的图像
            generated_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.png') and f != 'image_grid.png']

            for gen_file in generated_files:
                gen_image_path = os.path.join(temp_output_dir, gen_file)

                # 读取生成的图像
                gen_image = cv2.imread(gen_image_path)
                if gen_image is None:
                    continue

                # 旋转图像（使用原始参数中的角度）
                rotated_image = rotate_image(gen_image, original_params['angle'])

                # 计算相似度指标
                metrics = calculate_similarity_metrics(original_image_path, rotated_image, original_params)
                if metrics is None:
                    continue

                # 综合评分（4个指标的加权平均）
                composite_score = (
                    metrics['mse_similarity'] * 0.25 +
                    metrics['cosine_similarity'] * 0.25 +
                    metrics['mask_iou'] * 0.25 +
                    metrics['fourier_similarity'] * 0.25
                )

                # 检查是否满足阈值要求
                if composite_score >= current_threshold:
                    # 找到满足要求的图像
                    final_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(original_image_path))[0]}_hq.png")
                    cv2.imwrite(final_output_path, rotated_image)

                    # 清理临时文件
                    shutil.rmtree(temp_dir)

                    return {
                        'success': True,
                        'output_path': final_output_path,
                        'batches_used': batch_idx + 1,
                        'total_generated': total_generated,
                        'final_score': composite_score,
                        'final_threshold': current_threshold,
                        'metrics': metrics
                    }

                # 更新最佳结果
                if composite_score > best_score:
                    best_score = composite_score
                    best_image = rotated_image.copy()
                    best_metrics = metrics.copy()

            # 每5个批次降低阈值
            if (batch_idx + 1) % 5 == 0:
                current_threshold = max(min_threshold, current_threshold - threshold_decay)
                print(f"    阈值降低到: {current_threshold:.2f}")

        # 如果没有找到满足阈值的图像，使用最佳结果
        if best_image is not None:
            final_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(original_image_path))[0]}_hq.png")
            cv2.imwrite(final_output_path, best_image)

            # 清理临时文件
            shutil.rmtree(temp_dir)

            return {
                'success': True,
                'output_path': final_output_path,
                'batches_used': max_batches,
                'total_generated': total_generated,
                'final_score': best_score,
                'final_threshold': current_threshold,
                'metrics': best_metrics,
                'note': 'Used best result (threshold not met)'
            }

        # 清理临时文件
        shutil.rmtree(temp_dir)

        return {
            'success': False,
            'batches_used': max_batches,
            'total_generated': total_generated,
            'final_threshold': current_threshold,
            'error': 'No valid images generated'
        }

    except Exception as e:
        print(f"生成高质量图像时发生错误: {e}")
        return {
            'success': False,
            'error': str(e)
        }
