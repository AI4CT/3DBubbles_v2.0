# FlowRenderer.py
"""
3D气泡流场渲染器 - 重构版本

该文件是3DBubbles_v2.0的主入口文件，已重构为模块化架构。
原有的功能已分解到以下模块中：
- bubble_analysis.py: 气泡图像分析
- image_generation.py: 高质量图像生成
- coordinate_transform.py: 坐标变换
- flow_composition.py: 流场合成
- bubble_rendering.py: 气泡渲染
- flow_generator.py: 流场生成器

保持向后兼容性，所有原有的命令行参数和功能接口保持不变。
"""

import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入重构后的模块
from modules.bubble_analysis import (
    analyze_bubble_image,
    analyze_bubble_image_from_array,
    analyze_bubble_image_smart,
    analyze_bubble_image_from_array_smart,
    analyze_bubble_batch_smart,
    get_analysis_performance_report,
    reset_analysis_performance,
    calculate_fourier_descriptors,
    calculate_similarity_metrics,
    cosine_similarity_manual,
    TORCH_AVAILABLE
)

from modules.bubble_screening import (
    BubbleScreener,
    create_bubble_screener,
    screen_bubble_images_parallel
)

from modules.gpu_manager import (
    get_global_gpu_manager,
    cleanup_global_gpu_manager
)

from modules.gpu_performance_manager import (
    get_global_performance_manager,
    cleanup_global_performance_manager
)

from modules.image_generation import (
    create_bubble_prompts_file,
    generate_high_quality_bubble_image
)

from modules.coordinate_transform import (
    rotate_image,
    transform_3d_to_2d,
    sort_bubbles_by_depth
)

from modules.flow_composition import (
    create_flow_field_composition,
    composite_bubble_to_canvas,
    load_bubble_positions,
    calculate_dynamic_canvas_size
)

from modules.bubble_rendering import (
    render_single_bubble,
    process_single_bubble_rendering,
    pixel_coloring,
    cv2_enhance_contrast,
    ellipsoid_fit,
    calculate_dynamic_canvas_size
)

from modules.flow_generator import (
    generater,
    process_projection,
    generate_uniform_points_on_sphere,
    upsample_point_cloud,
    upsample_and_scale_mesh,
    generate_points_in_cube
)

# 为了向后兼容性，将所有函数导入到全局命名空间
# 这样原有的代码调用方式仍然有效


if __name__ == '__main__':
    """
    主程序入口点，保持原有的命令行接口
    """

    parser = argparse.ArgumentParser(description='流场生成器与渲染器')
    parser.add_argument('--stl_path', type=str, default=r"/home/yubd/mount/dataset/dataset_3Dbubble/mesh_20250619", help='STL文件的路径')
    parser.add_argument('--save_path', type=str, default=r"/home/yubd/mount/codebase/3DBubbles_v2.0/3DBubbles/3Dbubbleflowrender/", help='保存路径')
    parser.add_argument('-num','--flow_num', type=int, default=10, help='生成数量')
    parser.add_argument('-x','--volume_size_x', type=int, default=5, help='流场宽度[mm]')
    parser.add_argument('-y','--volume_size_y', type=int, default=5, help='流场深度[mm]')
    parser.add_argument('-hh','--volume_height', type=int, default=15, help='流场高度[mm]')
    parser.add_argument('--gas_holdup', type=float, default=0.005, help='气含率')
    parser.add_argument('-a','--alpha', type=int, default=4, help='向量指数:Alpha')
    parser.add_argument('-t','--truncation', type=float, default=0.75, help='截断值')
    parser.add_argument('--poisson_max_iter', type=int, default=100000, help='泊松圆盘采样最大迭代次数')
    parser.add_argument('--sample_spacing', type=int, default=0.02, help='点云上采样的采样距离')
    parser.add_argument('--enable_hq_generation', action='store_true', default=True, help='启用高质量图像重新生成功能（默认开启）')
    parser.add_argument('--enable_bubble_composition', action='store_true', default=True, help='启用三维流场气泡位置映射与合成功能（默认开启）')

    # 预生成数据集筛选参数
    parser.add_argument('--use_pregenerated_dataset', action='store_true', default=True, help='使用预生成数据集筛选（默认开启）')
    parser.add_argument('--pregenerated_image_dir', type=str, default='/home/yubd/mount/dataset/dataset_BubStyle/generated_bubble_images', help='预生成气泡图像目录路径')
    parser.add_argument('--pregenerated_struct_csv', type=str, default='/home/yubd/mount/dataset/dataset_BubStyle/generated_bubble_structs.csv', help='预生成气泡结构参数CSV文件路径')

    # 分层筛选参数
    parser.add_argument('--a_tolerance', type=float, default=0.1, help='长轴长容差（相对）')
    parser.add_argument('--b_tolerance', type=float, default=0.1, help='短轴长容差（相对）')
    parser.add_argument('--cx_tolerance', type=float, default=5.0, help='重心横坐标容差（绝对）')
    parser.add_argument('--cy_tolerance', type=float, default=5.0, help='重心纵坐标容差（绝对）')
    parser.add_argument('--sr_tolerance', type=float, default=0.1, help='阴影比容差（绝对）')
    parser.add_argument('--enable_iou_filtering', action='store_true', default=True, help='启用IoU筛选（默认开启）')
    parser.add_argument('--iou_threshold', type=float, default=0.3, help='IoU阈值')
    parser.add_argument('--min_similarity_score', type=float, default=0.5, help='最小相似度阈值')
    parser.add_argument('--screening_pool_size', type=int, default=10000, help='气泡筛选时的候选池大小，控制从预生成数据集中筛选的最大候选数量')

    # GPU加速参数（重新启用）
    parser.add_argument('--gpu-ids', type=int, nargs='*', default=None, help='指定使用的GPU ID列表（例如：--gpu-ids 0 1 2）')
    parser.add_argument('--max-gpus', type=int, default=4, help='最大使用GPU数量')
    parser.add_argument('--enable-gpu-acceleration', action='store_true', default=True, help='启用GPU加速（默认开启）')
    parser.add_argument('--disable-gpu-acceleration', action='store_true', help='禁用GPU加速')
    parser.add_argument('--gpu-batch-size', type=int, default=16, help='GPU批处理大小')
    parser.add_argument('--gpu-performance-report', action='store_true', help='显示GPU性能报告')

    args = parser.parse_args()

    # 处理预生成数据集参数
    use_pregenerated = getattr(args, 'use_pregenerated_dataset', True)
    pregenerated_image_dir = getattr(args, 'pregenerated_image_dir', '/home/yubd/mount/dataset/dataset_BubStyle/generated_bubble_images')
    pregenerated_struct_csv = getattr(args, 'pregenerated_struct_csv', '/home/yubd/mount/dataset/dataset_BubStyle/generated_bubble_structs.csv')

    # 处理GPU参数
    gpu_ids = getattr(args, 'gpu_ids', None)
    max_gpus = getattr(args, 'max_gpus', 4)
    enable_gpu_acceleration = getattr(args, 'enable_gpu_acceleration', True)
    gpu_batch_size = getattr(args, 'gpu_batch_size', 16)
    show_gpu_report = getattr(args, 'gpu_performance_report', False)

    # 如果指定了禁用GPU加速，则覆盖默认设置
    if getattr(args, 'disable_gpu_acceleration', False):
        enable_gpu_acceleration = False

    # 初始化GPU管理和性能监控
    print("=" * 60)
    print("3DBubbles v2.0 GPU加速流场生成器")
    print("=" * 60)

    # 检查GPU可用性
    gpu_available = False
    if TORCH_AVAILABLE:
        import torch
        gpu_available = torch.cuda.is_available()
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {gpu_available}")
        if gpu_available:
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"检测到GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("PyTorch未安装，将使用CPU模式")

    # 根据GPU可用性调整设置
    if not gpu_available:
        enable_gpu_acceleration = False
        print("警告: GPU不可用，自动切换到CPU模式")

    # 初始化GPU管理器
    if enable_gpu_acceleration:
        gpu_manager = get_global_gpu_manager(gpu_ids=gpu_ids, max_gpus=max_gpus)
        performance_manager = get_global_performance_manager()
        reset_analysis_performance()  # 重置性能统计
        print(f"GPU管理器初始化完成:")
        print(f"  使用GPU: {gpu_manager.gpu_ids}")
        print(f"  批处理大小: {gpu_batch_size}")

    print(f"\n数据集配置:")
    print(f"  使用预生成数据集: {'是' if use_pregenerated else '否'}")
    if use_pregenerated:
        print(f"  图像目录: {pregenerated_image_dir}")
        print(f"  结构参数文件: {pregenerated_struct_csv}")
    print(f"  GPU加速: {'启用' if enable_gpu_acceleration else '禁用'}")
    if enable_gpu_acceleration:
        print(f"  GPU设备: {gpu_ids if gpu_ids else '自动检测'}")
        print(f"  最大GPU数: {max_gpus}")
        print(f"  批处理大小: {gpu_batch_size}")

    # 获取STL文件列表
    stl_files = [os.path.join(args.stl_path, f) for f in os.listdir(args.stl_path) if f.endswith('.stl')]

    # 主循环：生成指定数量的流场
    print(f"开始生成 {args.flow_num} 个流场...")
    for num in range(args.flow_num):
        print(f"正在生成第 {num + 1}/{args.flow_num} 个流场...")

        # 创建时间戳目录
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S-%f')
        base_path = f'{args.save_path}/{timestamp}'
        os.makedirs(base_path, exist_ok=True)

        # 调用主要的生成函数
        generater(
            stl_files=stl_files,
            base_path=base_path,
            volume_size_x=args.volume_size_x,
            volume_size_y=args.volume_size_y,
            volume_height=args.volume_height,
            gas_holdups=[args.gas_holdup],
            alpha=args.alpha,
            truncation=args.truncation,
            poisson_max_iter=args.poisson_max_iter,
            sample_spacing=args.sample_spacing,
            enable_hq_generation=args.enable_hq_generation,
            enable_bubble_composition=args.enable_bubble_composition,
            # 预生成数据集参数
            use_pregenerated_dataset=use_pregenerated,
            pregenerated_image_dir=pregenerated_image_dir,
            pregenerated_struct_csv=pregenerated_struct_csv,
            a_tolerance=args.a_tolerance,
            b_tolerance=args.b_tolerance,
            cx_tolerance=args.cx_tolerance,
            cy_tolerance=args.cy_tolerance,
            sr_tolerance=args.sr_tolerance,
            enable_iou_filtering=args.enable_iou_filtering,
            iou_threshold=args.iou_threshold,
            min_similarity_score=args.min_similarity_score,
            screening_pool_size=args.screening_pool_size,
            # GPU加速参数
            gpu_ids=gpu_ids,
            max_gpus=max_gpus,
            enable_gpu_acceleration=enable_gpu_acceleration,
            gpu_batch_size=gpu_batch_size
        )

        print(f"第 {num + 1} 个流场生成完成，保存至: {base_path}")

    print(f"所有 {args.flow_num} 个流场生成完成！")

    # 显示GPU性能报告
    if enable_gpu_acceleration and (show_gpu_report or args.flow_num > 1):
        print("\n" + "="*60)
        print("GPU性能报告")
        print("="*60)

        try:
            # 获取分析性能报告
            analysis_report = get_analysis_performance_report()
            print("分析性能统计:")
            for key, value in analysis_report.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

            # 获取GPU内存信息
            gpu_manager = get_global_gpu_manager()
            memory_info = gpu_manager.get_gpu_memory_info()
            if memory_info:
                print("\nGPU内存使用情况:")
                for gpu_id, info in memory_info.items():
                    print(f"  GPU {gpu_id}:")
                    print(f"    总内存: {info['total']:.1f} MB")
                    print(f"    已使用: {info['allocated']:.1f} MB")
                    print(f"    已缓存: {info['cached']:.1f} MB")
                    print(f"    使用率: {info['allocated']/info['total']*100:.1f}%")

        except Exception as e:
            print(f"获取GPU性能报告失败: {e}")

    # 清理GPU资源
    if enable_gpu_acceleration:
        print("\n清理GPU资源...")
        try:
            cleanup_global_gpu_manager()
            cleanup_global_performance_manager()
            print("GPU资源清理完成")
        except Exception as e:
            print(f"GPU资源清理失败: {e}")

    # 输出运行参数信息
    print("\n" + "="*60)
    print("3DBubbles_v2.0 流场生成完成")
    print("="*60)
    print("生成数量:", args.flow_num)
    print("流场宽度[mm]:", args.volume_size_x)
    print("流场深度[mm]:", args.volume_size_y)
    print("流场高度[mm]:", args.volume_height)
    print("气含率:", args.gas_holdup)
    print("向量指数:Alpha:", args.alpha)
    print("截断值:", args.truncation)
    print("采样距离:", args.sample_spacing)
    print("高质量生成:", "启用" if args.enable_hq_generation else "禁用")
    print("流场合成:", "启用" if args.enable_bubble_composition else "禁用")
    print("预生成数据集筛选:", "启用" if use_pregenerated else "禁用")
    if use_pregenerated:
        print("筛选参数:")
        print(f"  几何参数容差: a={args.a_tolerance}, b={args.b_tolerance}")
        print(f"  重心坐标容差: cx={args.cx_tolerance}, cy={args.cy_tolerance}")
        print(f"  IoU筛选: {'启用' if args.enable_iou_filtering else '禁用'}, 阈值={args.iou_threshold}")
        print(f"  最小相似度: {args.min_similarity_score}")
        print(f"  筛选候选池大小: {args.screening_pool_size}")
    print("GPU加速:", "启用" if enable_gpu_acceleration else "禁用")
    if enable_gpu_acceleration:
        print("GPU配置:")
        print(f"  使用的GPU: {gpu_ids if gpu_ids else '自动检测'}")
        print(f"  最大GPU数: {max_gpus}")
        print(f"  批处理大小: {gpu_batch_size}")

        # 显示最终性能统计
        if show_gpu_report:
            try:
                final_report = get_analysis_performance_report()
                if final_report.get('total_processed', 0) > 0:
                    print("最终性能统计:")
                    if 'performance_ratio' in final_report:
                        print(f"  GPU/CPU性能比: {final_report['performance_ratio']:.2f}")
                    if 'avg_gpu_time' in final_report and final_report['avg_gpu_time'] > 0:
                        print(f"  平均GPU处理时间: {final_report['avg_gpu_time']*1000:.2f}ms")
                    if 'avg_cpu_time' in final_report and final_report['avg_cpu_time'] > 0:
                        print(f"  平均CPU处理时间: {final_report['avg_cpu_time']*1000:.2f}ms")
            except:
                pass
    print("="*60)
