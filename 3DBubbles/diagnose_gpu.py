#!/usr/bin/env python3
# diagnose_gpu.py
"""
GPU诊断脚本 - 全面诊断3DBubbles_v2.0的GPU使用情况

使用方法:
python diagnose_gpu.py [--detailed] [--stress-test] [--fix-issues]
"""

import os
import sys
import time
import argparse
import subprocess
from typing import Dict, List, Any

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def check_system_environment():
    """检查系统环境"""
    print("=== 系统环境检查 ===")
    
    # 检查NVIDIA驱动
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA驱动正常")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version:' in line:
                    driver_version = line.split('Driver Version:')[1].split()[0]
                    print(f"   驱动版本: {driver_version}")
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].split()[0]
                    print(f"   CUDA版本: {cuda_version}")
        else:
            print("❌ NVIDIA驱动检查失败")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi命令未找到，请检查NVIDIA驱动安装")
        return False
    
    # 检查PyTorch
    try:
        import torch
        print("✅ PyTorch已安装")
        print(f"   PyTorch版本: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"   GPU {i}: {name} ({memory_gb:.1f}GB)")
        return True
    except ImportError:
        print("❌ PyTorch未安装")
        return False


def check_kornia():
    """检查Kornia库"""
    print("\n=== Kornia库检查 ===")
    try:
        import kornia
        print("✅ Kornia已安装")
        print(f"   Kornia版本: {kornia.__version__}")
        
        # 测试Kornia GPU功能
        import torch
        if torch.cuda.is_available():
            try:
                x = torch.randn(1, 3, 64, 64, device='cuda:0')
                edges = kornia.filters.canny(x, low_threshold=0.1, high_threshold=0.2)
                print("✅ Kornia GPU功能正常")
                del x, edges
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ Kornia GPU功能异常: {e}")
        return True
    except ImportError:
        print("❌ Kornia未安装")
        return False


def test_gpu_managers():
    """测试GPU管理器"""
    print("\n=== GPU管理器测试 ===")
    
    try:
        from modules.gpu_manager import get_global_gpu_manager
        from modules.gpu_performance_manager import get_global_performance_manager
        
        # 测试GPU管理器
        print("测试GPU管理器...")
        gpu_manager = get_global_gpu_manager()
        print(f"✅ GPU管理器初始化成功")
        print(f"   可用GPU: {gpu_manager.available_gpus}")
        print(f"   使用GPU: {gpu_manager.gpu_ids}")
        print(f"   GPU数量: {gpu_manager.num_gpus}")
        print(f"   启用GPU: {gpu_manager.use_gpu}")
        
        # 测试性能管理器
        print("测试性能管理器...")
        perf_manager = get_global_performance_manager()
        print(f"✅ 性能管理器初始化成功")
        print(f"   应该使用GPU: {perf_manager.should_use_gpu()}")
        
        return True
    except Exception as e:
        print(f"❌ GPU管理器测试失败: {e}")
        return False


def test_gpu_analysis_functions():
    """测试GPU分析函数"""
    print("\n=== GPU分析函数测试 ===")
    
    try:
        import numpy as np
        import cv2
        from modules.bubble_analysis import (
            analyze_bubble_image_from_array,
            analyze_bubble_image_from_array_gpu,
            analyze_bubble_image_from_array_smart,
            get_analysis_performance_report,
            reset_analysis_performance
        )
        
        # 创建测试图像
        print("创建测试图像...")
        img = np.ones((128, 128, 3), dtype=np.uint8) * 255
        center = (64, 64)
        axes = (30, 20)
        angle = 45
        cv2.ellipse(img, center, axes, angle, 0, 360, (100, 100, 100), -1)
        
        # 重置性能统计
        reset_analysis_performance()
        
        # 测试CPU分析
        print("测试CPU分析...")
        start_time = time.time()
        cpu_result = analyze_bubble_image_from_array(img)
        cpu_time = time.time() - start_time
        print(f"   CPU分析时间: {cpu_time:.4f}s")
        print(f"   CPU结果: {'成功' if cpu_result else '失败'}")
        
        # 测试GPU分析
        print("测试GPU分析...")
        start_time = time.time()
        gpu_result = analyze_bubble_image_from_array_gpu(img, device='cuda:0')
        gpu_time = time.time() - start_time
        print(f"   GPU分析时间: {gpu_time:.4f}s")
        print(f"   GPU结果: {'成功' if gpu_result else '失败'}")
        
        # 测试智能分析
        print("测试智能分析...")
        start_time = time.time()
        smart_result = analyze_bubble_image_from_array_smart(img)
        smart_time = time.time() - start_time
        print(f"   智能分析时间: {smart_time:.4f}s")
        print(f"   智能结果: {'成功' if smart_result else '失败'}")
        
        # 性能报告
        print("性能报告:")
        report = get_analysis_performance_report()
        for key, value in report.items():
            print(f"   {key}: {value}")
        
        # 加速比计算
        if gpu_time > 0 and cpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"   GPU加速比: {speedup:.2f}x")
        
        return True
    except Exception as e:
        print(f"❌ GPU分析函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def monitor_gpu_during_operation():
    """在操作期间监控GPU使用情况"""
    print("\n=== GPU使用监控测试 ===")
    
    try:
        import torch
        from gpu_monitoring_patch import GPUUsageTracker
        
        print("执行GPU密集型操作并监控...")
        
        with GPUUsageTracker("GPU密集型测试", device_id=0, verbose=True):
            # 执行一些GPU操作
            device = torch.device('cuda:0')
            
            # 创建大量张量进行计算
            tensors = []
            for i in range(10):
                a = torch.randn(500, 500, device=device)
                b = torch.randn(500, 500, device=device)
                c = torch.matmul(a, b)
                tensors.append(c)
                time.sleep(0.1)  # 短暂延迟以便观察
            
            # 清理
            del tensors
            torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"❌ GPU监控测试失败: {e}")
        return False


def stress_test_gpu():
    """GPU压力测试"""
    print("\n=== GPU压力测试 ===")
    
    try:
        from gpu_monitoring_patch import stress_test_gpu
        stress_test_gpu(device_id=0, duration=5.0)
        return True
    except Exception as e:
        print(f"❌ GPU压力测试失败: {e}")
        return False


def provide_optimization_recommendations():
    """提供优化建议"""
    print("\n=== 优化建议 ===")
    
    try:
        from gpu_optimization_config import GPUOptimizationConfig, get_optimized_gpu_args
        
        # 打印优化报告
        GPUOptimizationConfig.print_optimization_report()
        
        # 推荐启动参数
        print("\n推荐的FlowRenderer.py启动参数:")
        args = get_optimized_gpu_args()
        cmd_args = []
        for key, value in args.items():
            if key != 'gpu_memory_fraction':  # 这个参数不是命令行参数
                cmd_args.append(f"--{key.replace('_', '-')} {value}")
        
        print(f"python FlowRenderer.py {' '.join(cmd_args)}")
        
        return True
    except Exception as e:
        print(f"❌ 优化建议生成失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='3DBubbles GPU诊断工具')
    parser.add_argument('--detailed', action='store_true', help='详细诊断')
    parser.add_argument('--stress-test', action='store_true', help='执行GPU压力测试')
    parser.add_argument('--fix-issues', action='store_true', help='尝试修复发现的问题')
    
    args = parser.parse_args()
    
    print("3DBubbles_v2.0 GPU诊断工具")
    print("=" * 50)
    
    # 基础检查
    results = []
    results.append(("系统环境", check_system_environment()))
    results.append(("Kornia库", check_kornia()))
    results.append(("GPU管理器", test_gpu_managers()))
    results.append(("GPU分析函数", test_gpu_analysis_functions()))
    
    if args.detailed:
        results.append(("GPU使用监控", monitor_gpu_during_operation()))
    
    if args.stress_test:
        results.append(("GPU压力测试", stress_test_gpu()))
    
    # 优化建议
    provide_optimization_recommendations()
    
    # 总结
    print("\n" + "=" * 50)
    print("诊断结果总结:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！GPU功能正常工作。")
        print("\n如果您觉得GPU利用率不高，这可能是因为：")
        print("1. 当前任务的GPU计算部分占比较小")
        print("2. GPU操作时间很短，nvidia-smi可能捕捉不到")
        print("3. 主要瓶颈在CPU密集型的数据筛选操作上")
    else:
        print("⚠️  发现问题，请根据上述错误信息进行修复。")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
