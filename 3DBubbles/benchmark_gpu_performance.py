#!/usr/bin/env python3
# benchmark_gpu_performance.py
"""
GPU性能基准测试脚本 - 验证完整流程并记录性能数据

使用方法:
python benchmark_gpu_performance.py [--test-dataset-dir path] [--output-dir results]
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import psutil
import threading

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

class GPUMonitor:
    """GPU监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.gpu_data = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.gpu_data = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 获取GPU信息
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    timestamp = time.time()
                    for i, line in enumerate(lines):
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            gpu_util, mem_used, mem_total, temp = parts[:4]
                            self.gpu_data.append({
                                'timestamp': timestamp,
                                'gpu_id': i,
                                'gpu_utilization': float(gpu_util),
                                'memory_used_mb': float(mem_used),
                                'memory_total_mb': float(mem_total),
                                'temperature': float(temp)
                            })
            except Exception as e:
                print(f"GPU监控错误: {e}")
            
            time.sleep(0.5)  # 每0.5秒采样一次
    
    def get_summary(self):
        """获取监控摘要"""
        if not self.gpu_data:
            return {}
        
        # 按GPU分组
        gpu_summaries = {}
        for data in self.gpu_data:
            gpu_id = data['gpu_id']
            if gpu_id not in gpu_summaries:
                gpu_summaries[gpu_id] = {
                    'utilization': [],
                    'memory_used': [],
                    'temperature': []
                }
            
            gpu_summaries[gpu_id]['utilization'].append(data['gpu_utilization'])
            gpu_summaries[gpu_id]['memory_used'].append(data['memory_used_mb'])
            gpu_summaries[gpu_id]['temperature'].append(data['temperature'])
        
        # 计算统计信息
        summary = {}
        for gpu_id, data in gpu_summaries.items():
            summary[f'gpu_{gpu_id}'] = {
                'max_utilization': max(data['utilization']),
                'avg_utilization': sum(data['utilization']) / len(data['utilization']),
                'max_memory_used_mb': max(data['memory_used']),
                'avg_memory_used_mb': sum(data['memory_used']) / len(data['memory_used']),
                'max_temperature': max(data['temperature']),
                'avg_temperature': sum(data['temperature']) / len(data['temperature']),
                'samples': len(data['utilization'])
            }
        
        return summary


def run_flowrenderer_benchmark(test_dataset_dir: str, output_dir: str):
    """运行FlowRenderer基准测试"""
    
    print("=== GPU性能基准测试 ===")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 测试配置
    test_configs = [
        {
            'name': 'GPU加速测试',
            'args': [
                '--pregenerated_image_dir', f'{test_dataset_dir}/generated_bubble_images',
                '--pregenerated_struct_csv', f'{test_dataset_dir}/generated_bubble_structs.csv',
                '--enable-gpu-acceleration',
                '--gpu-performance-report',
                '--gpu-batch-size', '32',
                '--max-gpus', '4',
                '-num', '1',
                '-x', '4',
                '-y', '4',
                '-hh', '8',
                '--gas_holdup', '0.01'
            ]
        },
        {
            'name': 'CPU对比测试',
            'args': [
                '--pregenerated_image_dir', f'{test_dataset_dir}/generated_bubble_images',
                '--pregenerated_struct_csv', f'{test_dataset_dir}/generated_bubble_structs.csv',
                '-num', '1',
                '-x', '4',
                '-y', '4',
                '-hh', '8',
                '--gas_holdup', '0.01'
            ]
        }
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n--- {config['name']} ---")
        
        # 启动GPU监控
        monitor = GPUMonitor()
        monitor.start_monitoring()
        
        # 记录系统信息
        cpu_percent_before = psutil.cpu_percent(interval=1)
        memory_before = psutil.virtual_memory()
        
        # 运行测试
        start_time = time.time()
        
        try:
            cmd = ['python', 'FlowRenderer.py'] + config['args']
            print(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(__file__) or '.',
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 停止监控
            monitor.stop_monitoring()
            
            # 记录系统信息
            cpu_percent_after = psutil.cpu_percent(interval=1)
            memory_after = psutil.virtual_memory()
            
            # 收集结果
            test_result = {
                'config_name': config['name'],
                'execution_time_seconds': execution_time,
                'return_code': result.returncode,
                'success': result.returncode == 0,
                'stdout_lines': len(result.stdout.split('\n')),
                'stderr_lines': len(result.stderr.split('\n')),
                'cpu_usage_before': cpu_percent_before,
                'cpu_usage_after': cpu_percent_after,
                'memory_used_mb_before': memory_before.used / 1024 / 1024,
                'memory_used_mb_after': memory_after.used / 1024 / 1024,
                'gpu_monitoring': monitor.get_summary(),
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存详细输出
            output_file = output_path / f"{config['name'].replace(' ', '_').lower()}_output.txt"
            with open(output_file, 'w') as f:
                f.write(f"=== {config['name']} 输出 ===\n")
                f.write(f"执行时间: {execution_time:.2f}秒\n")
                f.write(f"返回码: {result.returncode}\n")
                f.write(f"命令: {' '.join(cmd)}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            
            print(f"执行时间: {execution_time:.2f}秒")
            print(f"返回码: {result.returncode}")
            print(f"成功: {'是' if result.returncode == 0 else '否'}")
            
            # 显示GPU监控摘要
            gpu_summary = monitor.get_summary()
            if gpu_summary:
                print("GPU使用摘要:")
                for gpu_name, stats in gpu_summary.items():
                    print(f"  {gpu_name}: 最大利用率 {stats['max_utilization']:.1f}%, "
                          f"平均利用率 {stats['avg_utilization']:.1f}%, "
                          f"最大显存 {stats['max_memory_used_mb']:.0f}MB")
            
            results[config['name']] = test_result
            
        except subprocess.TimeoutExpired:
            monitor.stop_monitoring()
            print(f"测试超时 (600秒)")
            results[config['name']] = {
                'config_name': config['name'],
                'execution_time_seconds': 600,
                'return_code': -1,
                'success': False,
                'error': 'timeout',
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            monitor.stop_monitoring()
            print(f"测试失败: {e}")
            results[config['name']] = {
                'config_name': config['name'],
                'execution_time_seconds': 0,
                'return_code': -1,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # 保存结果
    results_file = output_path / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== 基准测试完成 ===")
    print(f"结果保存到: {results_file}")
    
    # 生成报告
    generate_report(results, output_path)
    
    return results


def generate_report(results: dict, output_path: Path):
    """生成测试报告"""
    
    report_file = output_path / 'benchmark_report.md'
    
    with open(report_file, 'w') as f:
        f.write("# 3DBubbles GPU性能基准测试报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 测试摘要\n\n")
        f.write("| 测试配置 | 执行时间(秒) | 成功 | 最大GPU利用率 | 平均GPU利用率 |\n")
        f.write("|---------|-------------|------|-------------|-------------|\n")
        
        for name, result in results.items():
            success = "✅" if result.get('success', False) else "❌"
            exec_time = result.get('execution_time_seconds', 0)
            
            gpu_summary = result.get('gpu_monitoring', {})
            max_util = 0
            avg_util = 0
            if gpu_summary:
                for gpu_stats in gpu_summary.values():
                    max_util = max(max_util, gpu_stats.get('max_utilization', 0))
                    avg_util = max(avg_util, gpu_stats.get('avg_utilization', 0))
            
            f.write(f"| {name} | {exec_time:.2f} | {success} | {max_util:.1f}% | {avg_util:.1f}% |\n")
        
        f.write("\n## 详细结果\n\n")
        
        for name, result in results.items():
            f.write(f"### {name}\n\n")
            f.write(f"- **执行时间**: {result.get('execution_time_seconds', 0):.2f}秒\n")
            f.write(f"- **成功状态**: {'成功' if result.get('success', False) else '失败'}\n")
            f.write(f"- **返回码**: {result.get('return_code', 'N/A')}\n")
            
            if 'error' in result:
                f.write(f"- **错误**: {result['error']}\n")
            
            gpu_summary = result.get('gpu_monitoring', {})
            if gpu_summary:
                f.write(f"- **GPU监控数据**:\n")
                for gpu_name, stats in gpu_summary.items():
                    f.write(f"  - {gpu_name}: 最大利用率 {stats['max_utilization']:.1f}%, "
                           f"平均利用率 {stats['avg_utilization']:.1f}%, "
                           f"最大显存 {stats['max_memory_used_mb']:.0f}MB\n")
            
            f.write("\n")
        
        # 性能对比
        if len(results) >= 2:
            gpu_result = None
            cpu_result = None
            
            for name, result in results.items():
                if 'GPU' in name and result.get('success', False):
                    gpu_result = result
                elif 'CPU' in name and result.get('success', False):
                    cpu_result = result
            
            if gpu_result and cpu_result:
                gpu_time = gpu_result['execution_time_seconds']
                cpu_time = cpu_result['execution_time_seconds']
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                
                f.write("## 性能对比\n\n")
                f.write(f"- **GPU加速时间**: {gpu_time:.2f}秒\n")
                f.write(f"- **CPU处理时间**: {cpu_time:.2f}秒\n")
                f.write(f"- **加速比**: {speedup:.2f}x\n")
                f.write(f"- **时间节省**: {cpu_time - gpu_time:.2f}秒 ({((cpu_time - gpu_time) / cpu_time * 100):.1f}%)\n\n")
    
    print(f"测试报告保存到: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='3DBubbles GPU性能基准测试')
    parser.add_argument('--test-dataset-dir', type=str,
                       default='/home/yubd/mount/dataset/dataset_BubStyle_test_fast',
                       help='测试数据集目录')
    parser.add_argument('--output-dir', type=str,
                       default='benchmark_results',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    # 检查测试数据集
    if not os.path.exists(args.test_dataset_dir):
        print(f"错误: 测试数据集目录不存在: {args.test_dataset_dir}")
        return False
    
    csv_file = os.path.join(args.test_dataset_dir, 'generated_bubble_structs.csv')
    img_dir = os.path.join(args.test_dataset_dir, 'generated_bubble_images')
    
    if not os.path.exists(csv_file):
        print(f"错误: CSV文件不存在: {csv_file}")
        return False
    
    if not os.path.exists(img_dir):
        print(f"错误: 图像目录不存在: {img_dir}")
        return False
    
    print(f"使用测试数据集: {args.test_dataset_dir}")
    print(f"结果输出目录: {args.output_dir}")
    
    # 运行基准测试
    results = run_flowrenderer_benchmark(args.test_dataset_dir, args.output_dir)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
