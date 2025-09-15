#!/usr/bin/env python3
# example_gpu_usage.py
"""
GPU加速功能使用示例

该脚本展示如何使用重构后的GPU加速功能进行气泡筛选和分析。

主要示例：
- 基本GPU加速分析
- 批量GPU处理
- 智能分析（自动GPU/CPU选择）
- 性能监控和报告
"""

import os
import sys
import time
import cv2
import numpy as np
from typing import List

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.bubble_analysis import (
    analyze_bubble_image_smart,
    analyze_bubble_batch_smart,
    get_analysis_performance_report,
    reset_analysis_performance
)

from modules.bubble_screening import create_bubble_screener
from modules.gpu_manager import get_global_gpu_manager


def example_basic_gpu_analysis():
    """基本GPU加速分析示例"""
    print("=" * 50)
    print("基本GPU加速分析示例")
    print("=" * 50)
    
    # 假设您有一些气泡图像文件
    image_paths = [
        # 'path/to/your/bubble1.png',
        # 'path/to/your/bubble2.png',
        # 添加您的图像路径
    ]
    
    if not image_paths:
        print("请在image_paths列表中添加您的气泡图像路径")
        return
    
    print(f"分析 {len(image_paths)} 张图像...")
    
    results = []
    total_time = 0
    
    for i, image_path in enumerate(image_paths):
        print(f"\n分析图像 {i+1}: {os.path.basename(image_path)}")
        
        start_time = time.time()
        # 使用智能分析（自动选择GPU或CPU）
        result = analyze_bubble_image_smart(image_path)
        analysis_time = time.time() - start_time
        total_time += analysis_time
        
        if result:
            results.append(result)
            print(f"  分析时间: {analysis_time:.4f}s")
            print(f"  长轴长度: {result['major_axis_length']:.3f}")
            print(f"  短轴长度: {result['minor_axis_length']:.3f}")
            print(f"  圆形度: {result['circularity']:.3f}")
            print(f"  阴影比: {result['shadow_ratio']:.3f}")
        else:
            print(f"  分析失败")
    
    print(f"\n总分析时间: {total_time:.4f}s")
    print(f"平均每张图像: {total_time/len(image_paths):.4f}s")
    print(f"成功率: {len(results)}/{len(image_paths)} ({len(results)/len(image_paths)*100:.1f}%)")


def example_batch_processing():
    """批量处理示例"""
    print("=" * 50)
    print("批量处理示例")
    print("=" * 50)
    
    # 创建一些示例图像数据
    print("创建示例图像数据...")
    test_images = []
    for i in range(10):
        # 创建128x128的示例图像
        img = np.ones((128, 128, 3), dtype=np.uint8) * 255
        
        # 添加一个椭圆
        center = (64 + np.random.randint(-20, 20), 64 + np.random.randint(-20, 20))
        axes = (np.random.randint(20, 40), np.random.randint(15, 35))
        angle = np.random.randint(0, 180)
        cv2.ellipse(img, center, axes, angle, 0, 360, (100, 100, 100), -1)
        
        test_images.append(img)
    
    print(f"批量分析 {len(test_images)} 张图像...")
    
    start_time = time.time()
    # 使用智能批量分析
    results = analyze_bubble_batch_smart(test_images, batch_size=8)
    batch_time = time.time() - start_time
    
    successful_results = [r for r in results if r is not None]
    
    print(f"批量分析时间: {batch_time:.4f}s")
    print(f"平均每张图像: {batch_time/len(test_images)*1000:.2f}ms")
    print(f"成功分析: {len(successful_results)}/{len(test_images)} 张")
    
    # 显示一些统计信息
    if successful_results:
        avg_circularity = np.mean([r['circularity'] for r in successful_results])
        avg_shadow_ratio = np.mean([r['shadow_ratio'] for r in successful_results])
        
        print(f"\n统计信息:")
        print(f"  平均圆形度: {avg_circularity:.3f}")
        print(f"  平均阴影比: {avg_shadow_ratio:.3f}")


def example_screening_with_gpu():
    """使用GPU加速的筛选示例"""
    print("=" * 50)
    print("GPU加速筛选示例")
    print("=" * 50)
    
    # 假设您有一个包含气泡图像的目录
    image_directory = "path/to/your/bubble/images"
    
    if not os.path.exists(image_directory):
        print(f"图像目录不存在: {image_directory}")
        print("请修改image_directory变量指向您的图像目录")
        return
    
    # 获取所有图像文件
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_paths = []
    
    for filename in os.listdir(image_directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(image_directory, filename))
    
    if not image_paths:
        print(f"在目录 {image_directory} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 创建GPU加速的筛选器
    screener = create_bubble_screener(
        gpu_ids=None,  # 自动检测GPU
        max_gpus=4,    # 最多使用4个GPU
        batch_size=16  # 批处理大小
    )
    
    print("开始筛选...")
    start_time = time.time()
    
    # 执行筛选
    screening_results = screener.screen_bubble_images(
        image_paths, 
        enable_gpu_acceleration=True
    )
    
    screening_time = time.time() - start_time
    
    print(f"筛选完成，耗时: {screening_time:.2f}s")
    print(f"通过筛选: {len(screening_results['passed'])} 张")
    print(f"未通过筛选: {len(screening_results['failed'])} 张")
    print(f"筛选率: {len(screening_results['passed'])/len(image_paths)*100:.1f}%")
    
    # 获取性能报告
    performance_report = screener.get_performance_report()
    print("\n性能报告:")
    print(f"  GPU状态: {performance_report['gpu_manager_status']}")
    print(f"  分析性能: {performance_report['analysis_performance']}")
    
    # 清理资源
    screener.cleanup()


def example_performance_monitoring():
    """性能监控示例"""
    print("=" * 50)
    print("性能监控示例")
    print("=" * 50)
    
    # 重置性能统计
    reset_analysis_performance()
    
    # 创建测试数据
    test_images = []
    for i in range(20):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        test_images.append(img)
    
    print("执行多轮分析以收集性能数据...")
    
    # 执行多轮分析
    for round_num in range(3):
        print(f"\n第 {round_num + 1} 轮分析:")
        
        start_time = time.time()
        results = analyze_bubble_batch_smart(test_images, batch_size=8)
        round_time = time.time() - start_time
        
        successful = sum(1 for r in results if r is not None)
        print(f"  时间: {round_time:.4f}s")
        print(f"  成功: {successful}/{len(test_images)}")
        
        # 获取当前性能报告
        report = get_analysis_performance_report()
        if 'current_mode' in report:
            print(f"  当前模式: {report['current_mode']}")
    
    # 最终性能报告
    final_report = get_analysis_performance_report()
    print("\n最终性能报告:")
    for key, value in final_report.items():
        print(f"  {key}: {value}")


def example_gpu_memory_management():
    """GPU内存管理示例"""
    print("=" * 50)
    print("GPU内存管理示例")
    print("=" * 50)
    
    # 获取GPU管理器
    gpu_manager = get_global_gpu_manager()
    
    if not gpu_manager.use_gpu:
        print("GPU不可用，跳过内存管理示例")
        return
    
    print("GPU状态:")
    print(f"  可用GPU: {gpu_manager.available_gpus}")
    print(f"  使用GPU: {gpu_manager.gpu_ids}")
    
    # 获取内存信息
    memory_info = gpu_manager.get_gpu_memory_info()
    print("\nGPU内存信息:")
    for gpu_id, info in memory_info.items():
        print(f"  GPU {gpu_id}:")
        print(f"    总内存: {info['total']:.1f} MB")
        print(f"    已分配: {info['allocated']:.1f} MB")
        print(f"    已缓存: {info['cached']:.1f} MB")
        print(f"    可用: {info['free']:.1f} MB")
    
    # 执行一些GPU操作
    print("\n执行GPU操作...")
    test_images = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(10)]
    results = analyze_bubble_batch_smart(test_images)
    
    # 再次检查内存
    memory_info_after = gpu_manager.get_gpu_memory_info()
    print("\n操作后GPU内存信息:")
    for gpu_id, info in memory_info_after.items():
        print(f"  GPU {gpu_id}:")
        print(f"    已分配: {info['allocated']:.1f} MB")
        print(f"    已缓存: {info['cached']:.1f} MB")
    
    # 清理GPU内存
    print("\n清理GPU内存...")
    gpu_manager.cleanup()
    
    memory_info_cleaned = gpu_manager.get_gpu_memory_info()
    print("清理后GPU内存信息:")
    for gpu_id, info in memory_info_cleaned.items():
        print(f"  GPU {gpu_id}:")
        print(f"    已分配: {info['allocated']:.1f} MB")
        print(f"    已缓存: {info['cached']:.1f} MB")


def main():
    """主函数"""
    print("3DBubbles GPU加速功能使用示例")
    print("=" * 60)
    
    # 检查GPU可用性
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("PyTorch未安装，将使用CPU模式")
    
    print("\n选择要运行的示例:")
    print("1. 基本GPU加速分析")
    print("2. 批量处理")
    print("3. GPU加速筛选")
    print("4. 性能监控")
    print("5. GPU内存管理")
    print("6. 运行所有示例")
    
    choice = input("\n请输入选择 (1-6): ").strip()
    
    if choice == '1':
        example_basic_gpu_analysis()
    elif choice == '2':
        example_batch_processing()
    elif choice == '3':
        example_screening_with_gpu()
    elif choice == '4':
        example_performance_monitoring()
    elif choice == '5':
        example_gpu_memory_management()
    elif choice == '6':
        example_basic_gpu_analysis()
        example_batch_processing()
        example_screening_with_gpu()
        example_performance_monitoring()
        example_gpu_memory_management()
    else:
        print("无效选择")


if __name__ == '__main__':
    main()
