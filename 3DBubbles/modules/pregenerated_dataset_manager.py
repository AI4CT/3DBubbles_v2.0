import numpy as np
import pandas as pd
import cv2
import os
import math
import threading
from typing import List, Tuple, Dict, Optional
import pickle
from tqdm import tqdm
import scipy.io as sio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import OrderedDict
import heapq
from functools import lru_cache
import gc

# GPU相关导入
try:
    import torch
    from .gpu_manager import get_global_gpu_manager, GPUManager
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("警告: PyTorch或GPU管理器不可用，将使用CPU模式")


def cosine_similarity_manual(a: np.ndarray, b: np.ndarray) -> float:
    """手动计算余弦相似度，避免sklearn依赖"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class LRUCache:
    """线程安全的LRU缓存实现"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                # 移动到末尾（最近使用）
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.capacity:
                    # 删除最久未使用的项
                    self.cache.popitem(last=False)
            self.cache[key] = value

    def clear(self):
        with self.lock:
            self.cache.clear()

    def size(self):
        with self.lock:
            return len(self.cache)


class PregeneratedDatasetManager:
    """预生成数据集管理器

    用于高效管理和筛选100万张预生成的气泡图像及其结构参数
    """

    # CSV列名到标准参数名的映射
    COLUMN_MAPPING = {
        '0': 'a',    # 长轴长
        '1': 'b',    # 短轴长
        '2': 'cx',   # 灰度重心横坐标
        '3': 'cy',   # 灰度重心纵坐标
        '4': 'C',    # 圆形度
        '5': 'S',    # 椭圆度
        '6': 'SR',   # 阴影比
        '7': 'EG'    # 边缘梯度
    }

    # 标准参数名到CSV列名的反向映射
    REVERSE_MAPPING = {v: k for k, v in COLUMN_MAPPING.items()}

    def __init__(self, image_dir: str, struct_csv_path: str, cache_dir: str = None,
                 max_workers: int = 4, enable_parallel: bool = True, enable_gpu_acceleration: bool = True,
                 gpu_batch_size: int = 1024):
        """
        初始化数据集管理器

        Args:
            image_dir: 图像目录路径
            struct_csv_path: 结构参数CSV文件路径
            cache_dir: 缓存目录，用于存储预处理的索引数据
            max_workers: 最大并行工作线程数
            enable_parallel: 是否启用并行处理
            enable_gpu_acceleration: 是否启用GPU加速
            gpu_batch_size: GPU批处理大小
        """
        self.image_dir = image_dir
        self.struct_csv_path = struct_csv_path
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(image_dir), 'cache')
        self.max_workers = max_workers
        self.enable_parallel = enable_parallel
        self.enable_gpu_acceleration = enable_gpu_acceleration and GPU_AVAILABLE
        self.gpu_batch_size = gpu_batch_size

        # 数据存储
        self.struct_params = None
        self.mat_file_info = None  # 存储MAT文件信息 {idx: (mat_file_path, index_in_mat)}

        # 使用LRU缓存替代简单字典
        self.mat_cache = LRUCache(capacity=50)  # 限制MAT文件缓存数量
        self.contour_cache = LRUCache(capacity=10000)  # 限制轮廓缓存数量

        # GPU管理器（延迟初始化）
        self.gpu_manager = None
        if self.enable_gpu_acceleration:
            try:
                self.gpu_manager = get_global_gpu_manager()
                print(f"GPU加速已启用，批处理大小: {self.gpu_batch_size}")
            except Exception as e:
                print(f"GPU管理器初始化失败，将使用CPU模式: {e}")
                self.enable_gpu_acceleration = False

        # 线程安全锁
        self.mat_cache_lock = threading.RLock()  # 用于MAT文件缓存的线程安全
        self.contour_cache_lock = threading.RLock()  # 用于轮廓缓存的线程安全

        # 优化的多维索引结构
        self.a_index = None      # 长轴长索引
        self.b_index = None      # 短轴长索引
        self.cx_index = None     # 灰度重心横坐标索引
        self.cy_index = None     # 灰度重心纵坐标索引

        # 辅助索引
        self.sr_index = None     # 阴影比索引

        # 新增：KD树索引用于快速几何参数查询
        self.geometric_kdtree = None
        self.geometric_data = None

        # 参数范围缓存（用于快速范围查询）
        self.param_ranges = {}

        # 新增：预计算的几何距离缓存
        self.geometric_distance_cache = LRUCache(capacity=1000)

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

        # 加载数据
        self._load_dataset()
    
    def _load_dataset(self):
        """加载数据集"""
        print("正在加载预生成数据集...")
        
        # 检查是否有缓存的索引
        cache_file = os.path.join(self.cache_dir, 'dataset_index.pkl')
        if os.path.exists(cache_file):
            print("发现缓存索引，正在加载...")
            self._load_from_cache(cache_file)
        else:
            print("未发现缓存，正在构建索引...")
            self._build_index()
            self._save_to_cache(cache_file)
    
    def _build_index(self):
        """构建数据集索引"""
        # 加载结构参数
        print("加载结构参数...")
        self.struct_params = pd.read_csv(self.struct_csv_path)
        print(f"加载了 {len(self.struct_params)} 个结构参数")

        # 验证CSV格式
        self._validate_csv_format()

        # 构建MAT文件信息映射
        print("构建MAT文件信息索引...")
        self.mat_file_info = {}

        # 根据实际的图像文件组织方式构建MAT文件映射
        # 每250行数据对应一个.mat文件
        for idx in range(len(self.struct_params)):
            # 计算对应的.mat文件编号（每250行一个文件）
            mat_file_number = ((idx // 250) + 1) * 250
            mat_filename = f"{mat_file_number}.mat"

            # 在.mat文件中的索引（0-249）
            index_in_mat = idx % 250

            # 构建MAT文件路径
            mat_file_name = f"{mat_file_number}.mat"
            mat_file_path = os.path.join(self.image_dir, mat_file_name)

            # 存储映射信息
            self.mat_file_info[idx] = (mat_file_path, index_in_mat)

        # 构建快速检索索引
        self._build_search_indices()

    def _validate_csv_format(self):
        """验证CSV文件格式是否符合预期"""
        expected_columns = set(self.COLUMN_MAPPING.keys())
        actual_columns = set(self.struct_params.columns.astype(str))

        if not expected_columns.issubset(actual_columns):
            missing_columns = expected_columns - actual_columns
            raise ValueError(f"CSV文件格式不正确，缺少列: {missing_columns}. "
                           f"期望的列名: {sorted(expected_columns)}, "
                           f"实际的列名: {sorted(actual_columns)}")

        print(f"CSV格式验证通过，包含 {len(self.struct_params)} 行数据")
        print(f"列名映射: {self.COLUMN_MAPPING}")
    
    def _build_search_indices(self):
        """构建基于核心几何参数的多维搜索索引"""
        print("构建多维搜索索引...")

        # 使用CSV列名访问数据，构建核心几何参数的排序索引
        print("  构建长轴长(a)索引...")
        self.a_index = np.argsort(self.struct_params['0'].values)  # 第0列是长轴长

        print("  构建短轴长(b)索引...")
        self.b_index = np.argsort(self.struct_params['1'].values)  # 第1列是短轴长

        print("  构建重心横坐标(cx)索引...")
        self.cx_index = np.argsort(self.struct_params['2'].values)  # 第2列是重心横坐标

        print("  构建重心纵坐标(cy)索引...")
        self.cy_index = np.argsort(self.struct_params['3'].values)  # 第3列是重心纵坐标

        # 构建阴影比索引
        print("  构建阴影比(SR)索引...")
        self.sr_index = np.argsort(self.struct_params['6'].values)  # 第6列是阴影比

        # 构建KD树索引用于快速几何参数查询
        print("  构建KD树索引...")
        self._build_kdtree_index()

        # 缓存参数范围以加速查询
        self._cache_parameter_ranges()

    def _build_kdtree_index(self):
        """构建KD树索引用于快速几何参数查询"""
        try:
            from sklearn.neighbors import KDTree

            # 准备几何参数数据 (a, b, cx, cy)
            self.geometric_data = np.column_stack([
                self.struct_params['0'].values,  # a
                self.struct_params['1'].values,  # b
                self.struct_params['2'].values,  # cx
                self.struct_params['3'].values   # cy
            ])

            # 构建KD树
            self.geometric_kdtree = KDTree(self.geometric_data, leaf_size=30)
            print("  KD树索引构建完成")

        except ImportError:
            print("  警告: sklearn不可用，跳过KD树索引构建")
            self.geometric_kdtree = None
            self.geometric_data = None
    
    def _cache_parameter_ranges(self):
        """缓存参数范围以加速查询"""
        print("  缓存参数范围...")

        # 使用CSV列名和对应的索引
        param_mapping = {
            'a': ('0', self.a_index),
            'b': ('1', self.b_index),
            'cx': ('2', self.cx_index),
            'cy': ('3', self.cy_index)
        }

        for param_name, (csv_col, index) in param_mapping.items():
            values = self.struct_params[csv_col].values
            self.param_ranges[param_name] = {
                'min': np.min(values),
                'max': np.max(values),
                'sorted_values': values[index]
            }

        # 缓存阴影比范围
        sr_values = self.struct_params['6'].values  # 第6列是阴影比
        self.param_ranges['SR'] = {
            'min': np.min(sr_values),
            'max': np.max(sr_values),
            'sorted_values': sr_values[self.sr_index]
        }

    def _find_range_candidates(self, param_name: str, target_value: float,
                              tolerance: float, is_relative: bool = False) -> np.ndarray:
        """
        使用二分搜索找到参数范围内的候选索引

        Args:
            param_name: 参数名称
            target_value: 目标值
            tolerance: 容差
            is_relative: 是否为相对容差

        Returns:
            候选索引数组
        """
        if param_name not in self.param_ranges:
            return np.array([])

        # 计算搜索范围
        if is_relative:
            min_val = target_value * (1 - tolerance)
            max_val = target_value * (1 + tolerance)
        else:
            min_val = target_value - tolerance
            max_val = target_value + tolerance

        # 获取排序后的值和对应的索引
        sorted_values = self.param_ranges[param_name]['sorted_values']
        param_index = getattr(self, f'{param_name}_index')

        # 使用二分搜索找到范围
        left_idx = np.searchsorted(sorted_values, min_val, side='left')
        right_idx = np.searchsorted(sorted_values, max_val, side='right')

        # 返回原始索引
        return param_index[left_idx:right_idx]
    
    def _save_to_cache(self, cache_file: str):
        """保存索引到缓存"""
        print("保存索引到缓存...")
        cache_data = {
            'struct_params': self.struct_params,
            'mat_file_info': self.mat_file_info,
            'a_index': self.a_index,
            'b_index': self.b_index,
            'cx_index': self.cx_index,
            'cy_index': self.cy_index,
            'sr_index': self.sr_index,
            'param_ranges': self.param_ranges,
            'geometric_kdtree': self.geometric_kdtree,
            'geometric_data': self.geometric_data
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _load_from_cache(self, cache_file: str):
        """从缓存加载索引"""
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        self.struct_params = cache_data['struct_params']
        self.mat_file_info = cache_data['mat_file_info']
        self.a_index = cache_data['a_index']
        self.b_index = cache_data['b_index']
        self.cx_index = cache_data['cx_index']
        self.cy_index = cache_data['cy_index']
        self.sr_index = cache_data['sr_index']
        self.param_ranges = cache_data['param_ranges']

        # 加载KD树索引（如果存在）
        self.geometric_kdtree = cache_data.get('geometric_kdtree', None)
        self.geometric_data = cache_data.get('geometric_data', None)
    
    def find_similar_bubbles(self, target_a: float, target_b: float, target_cx: float, target_cy: float,
                           a_tolerance: float = 0.1, b_tolerance: float = 0.1,
                           cx_tolerance: float = 5.0, cy_tolerance: float = 5.0,
                           target_sr: float = None, sr_tolerance: float = 0.1,
                           enable_iou_filtering: bool = True, iou_threshold: float = 0.3,
                           max_candidates: int = 1000, max_iou_candidates: int = 100) -> List[int]:
        """
        优化的分层筛选算法，基于新的优先级体系找到相似的气泡候选

        筛选优先级：
        1. 主要几何参数(a, b, cx, cy) - 最重要，使用KD树加速
        2. 早期终止机制 - 避免处理明显不匹配的候选
        3. 轮廓相似度(IoU) - 并行计算
        4. 阴影特征(SR) - 快速过滤
        5. 综合结构相似度 - 最终排序

        Args:
            target_a: 目标长轴长
            target_b: 目标短轴长
            target_cx: 目标重心横坐标
            target_cy: 目标重心纵坐标
            a_tolerance: 长轴长容差（相对）
            b_tolerance: 短轴长容差（相对）
            cx_tolerance: 重心横坐标容差（绝对）
            cy_tolerance: 重心纵坐标容差（绝对）
            target_sr: 目标阴影比（可选）
            sr_tolerance: 阴影比容差（绝对）
            enable_iou_filtering: 是否启用IoU筛选
            iou_threshold: IoU阈值
            max_candidates: 最大候选数量
            max_iou_candidates: 进行IoU计算的最大候选数量

        Returns:
            按相似度排序的候选气泡索引列表
        """
        print(f"开始优化分层筛选: 目标参数 a={target_a:.3f}, b={target_b:.3f}, cx={target_cx:.1f}, cy={target_cy:.1f}")

        # 第一层：使用KD树进行快速几何参数筛选
        print("  第1层: KD树几何参数筛选...")
        geometric_candidates = self._filter_by_geometric_params_optimized(
            target_a, target_b, target_cx, target_cy,
            a_tolerance, b_tolerance, cx_tolerance, cy_tolerance,
            max_candidates * 3  # 减少粗筛选候选数量
        )

        if len(geometric_candidates) == 0:
            print("  几何参数筛选无结果")
            return []

        print(f"  几何参数筛选得到 {len(geometric_candidates)} 个候选")

        # 早期终止检查：如果候选数量很少，直接返回
        if len(geometric_candidates) <= max_candidates // 2:
            print("  候选数量较少，跳过后续筛选步骤")
            return self._quick_rank_candidates(geometric_candidates, target_a, target_b, target_cx, target_cy, target_sr)

        # 第二层：并行IoU筛选（如果启用）
        iou_candidates = geometric_candidates
        if enable_iou_filtering and len(geometric_candidates) > 1:
            print("  第2层: 并行IoU筛选...")
            iou_candidates = self._filter_by_contour_iou_parallel(
                geometric_candidates, target_a, target_b, target_cx, target_cy,
                iou_threshold, max_iou_candidates
            )
            print(f"  IoU筛选得到 {len(iou_candidates)} 个候选")

        # 第三层：快速阴影特征筛选
        shadow_candidates = iou_candidates
        if target_sr is not None:
            print("  第3层: 快速阴影特征筛选...")
            shadow_candidates = self._filter_by_shadow_ratio_fast(
                iou_candidates, target_sr, sr_tolerance
            )
            print(f"  阴影筛选得到 {len(shadow_candidates)} 个候选")

        # 第四层：并行综合相似度排序
        print("  第4层: 并行综合相似度排序...")
        final_candidates = self._rank_by_structural_similarity_parallel(
            shadow_candidates, target_a, target_b, target_cx, target_cy, target_sr,
            max_candidates
        )

        print(f"  最终筛选得到 {len(final_candidates)} 个候选")
        return final_candidates
    
    def calculate_contour_iou(self, image_path1: str, image_path2: str) -> float:
        """
        计算两个气泡图像的轮廓IoU
        
        Args:
            image_path1: 第一个图像路径
            image_path2: 第二个图像路径
            
        Returns:
            IoU值
        """
        # 获取轮廓
        contour1 = self._get_contour(image_path1)
        contour2 = self._get_contour(image_path2)
        
        if contour1 is None or contour2 is None:
            return 0.0
        
        # 创建掩码
        height, width = 128, 128  # 假设图像尺寸为128x128
        mask1 = np.zeros((height, width), dtype=np.uint8)
        mask2 = np.zeros((height, width), dtype=np.uint8)
        
        # 绘制轮廓
        cv2.drawContours(mask1, [contour1], -1, 255, -1)
        cv2.drawContours(mask2, [contour2], -1, 255, -1)
        
        # 计算IoU
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _get_contour(self, image_path: str):
        """获取图像的主要轮廓（优化版本）"""
        # 检查缓存
        cached_contour = self.contour_cache.get(image_path)
        if cached_contour is not None:
            return cached_contour

        # 读取图像
        if not os.path.exists(image_path):
            return None

        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None

            # 优化的二值化处理
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            # 使用自适应阈值或Otsu方法获得更好的二值化效果
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # 选择最大的轮廓
            main_contour = max(contours, key=cv2.contourArea)

            # 轮廓简化以减少内存使用
            epsilon = 0.02 * cv2.arcLength(main_contour, True)
            simplified_contour = cv2.approxPolyDP(main_contour, epsilon, True)

            # 缓存结果
            self.contour_cache.put(image_path, simplified_contour)

            return simplified_contour

        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            return None

    def _get_contour_from_data(self, image_data: np.ndarray, idx: int):
        """从图像数据获取主要轮廓（线程安全）"""
        cache_key = f"data_{idx}"

        # 使用线程锁确保轮廓缓存的线程安全
        with self.contour_cache_lock:
            # 检查缓存
            cached_contour = self.contour_cache.get(cache_key)
            if cached_contour is not None:
                return cached_contour

            try:
                # 确保图像数据格式正确
                if len(image_data.shape) == 3:
                    # 如果是彩色图像，转换为灰度
                    image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                else:
                    image = image_data.copy()

                # 优化的二值化处理
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)

                # 使用自适应阈值或Otsu方法获得更好的二值化效果
                _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # 查找轮廓
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    return None

                # 选择最大的轮廓
                main_contour = max(contours, key=cv2.contourArea)

                # 轮廓简化以减少内存使用
                epsilon = 0.02 * cv2.arcLength(main_contour, True)
                simplified_contour = cv2.approxPolyDP(main_contour, epsilon, True)

                # 缓存结果
                self.contour_cache.put(cache_key, simplified_contour)

                return simplified_contour

            except Exception as e:
                print(f"处理图像数据 idx={idx} 时出错: {e}")
                return None
    
    def calculate_structural_similarity(self, idx1: int, idx2: int, 
                                      param_columns: List[str] = None) -> float:
        """
        计算结构参数的余弦相似度
        
        Args:
            idx1: 第一个气泡的索引
            idx2: 第二个气泡的索引
            param_columns: 用于计算相似度的参数列名
            
        Returns:
            余弦相似度值
        """
        if param_columns is None:
            # 默认使用所有8个结构参数列
            param_columns = ['0', '1', '2', '3', '4', '5', '6', '7']
        
        # 获取参数向量
        params1 = self.struct_params.iloc[idx1][param_columns].values.reshape(1, -1)
        params2 = self.struct_params.iloc[idx2][param_columns].values.reshape(1, -1)
        
        # 计算余弦相似度
        similarity = cosine_similarity_manual(params1.flatten(), params2.flatten())
        
        return similarity
    
    def get_image_data(self, idx: int) -> Optional[np.ndarray]:
        """获取指定索引的图像数据（线程安全，使用LRU缓存）"""
        if idx >= len(self.struct_params):
            return None

        mat_file_path, index_in_mat = self.mat_file_info[idx]

        # 使用线程锁确保MAT文件缓存的线程安全
        with self.mat_cache_lock:
            # 检查MAT文件缓存
            cached_data = self.mat_cache.get(mat_file_path)
            if cached_data is None:
                try:
                    print(f"加载MAT文件: {os.path.basename(mat_file_path)}")
                    mat_data = sio.loadmat(mat_file_path)
                    # 转置数据以匹配正确的维度顺序
                    bubble_images = mat_data['generated_bubble_images'].transpose(2, 1, 0)
                    self.mat_cache.put(mat_file_path, bubble_images)
                    cached_data = bubble_images
                except Exception as e:
                    print(f"无法加载MAT文件 {mat_file_path}: {e}")
                    return None

            # 从缓存中获取图像数据
            if index_in_mat >= cached_data.shape[0]:
                print(f"索引超出范围: {index_in_mat} >= {cached_data.shape[0]}")
                return None

            return cached_data[index_in_mat]

    def get_image_path(self, idx: int) -> str:
        """获取指定索引的MAT文件路径（保持兼容性）"""
        if idx >= len(self.struct_params):
            return ""
        mat_file_path, index_in_mat = self.mat_file_info[idx]
        return f"{mat_file_path}[{index_in_mat}]"
    
    def get_struct_params(self, idx: int) -> Dict:
        """获取指定索引的结构参数，返回标准参数名格式"""
        row = self.struct_params.iloc[idx]

        # 将CSV列名转换为标准参数名
        result = {}
        for csv_col, param_name in self.COLUMN_MAPPING.items():
            if csv_col in row.index:
                result[param_name] = row[csv_col]

        return result
    
    def get_dataset_size(self) -> int:
        """获取数据集大小"""
        return len(self.struct_params)

    def clear_contour_cache(self):
        """清理轮廓缓存以释放内存（线程安全）"""
        with self.contour_cache_lock:
            self.contour_cache.clear()

    def clear_mat_cache(self):
        """清理MAT文件缓存以释放内存（线程安全）"""
        with self.mat_cache_lock:
            self.mat_cache.clear()
        print("MAT文件缓存已清理")

    def clear_all_caches(self):
        """清理所有缓存以释放内存（线程安全）"""
        self.clear_contour_cache()
        self.clear_mat_cache()
        self.geometric_distance_cache.clear()
        print("所有缓存已清理")

    def get_memory_usage_info(self) -> Dict[str, int]:
        """获取内存使用信息"""
        return {
            'struct_params_size': len(self.struct_params),
            'mat_file_info_size': len(self.mat_file_info),
            'mat_cache_size': self.mat_cache.size(),
            'contour_cache_size': self.contour_cache.size(),
            'geometric_distance_cache_size': self.geometric_distance_cache.size(),
            'a_index_size': len(self.a_index) if self.a_index is not None else 0,
            'b_index_size': len(self.b_index) if self.b_index is not None else 0,
            'cx_index_size': len(self.cx_index) if self.cx_index is not None else 0,
            'cy_index_size': len(self.cy_index) if self.cy_index is not None else 0,
            'sr_index_size': len(self.sr_index) if self.sr_index is not None else 0,
            'kdtree_available': self.geometric_kdtree is not None
        }

    def batch_preload_contours(self, image_indices: List[int], batch_size: int = 100):
        """批量预加载轮廓以提高后续访问速度"""
        print(f"批量预加载 {len(image_indices)} 个轮廓...")

        for i in range(0, len(image_indices), batch_size):
            batch_indices = image_indices[i:i+batch_size]

            for idx in tqdm(batch_indices, desc=f"预加载批次 {i//batch_size + 1}"):
                if idx < len(self.struct_params):
                    image_data = self.get_image_data(idx)
                    if image_data is not None:
                        self._get_contour_from_data(image_data, idx)  # 这会自动缓存结果

    def _filter_by_geometric_params_optimized(self, target_a: float, target_b: float,
                                             target_cx: float, target_cy: float,
                                             a_tolerance: float, b_tolerance: float,
                                             cx_tolerance: float, cy_tolerance: float,
                                             max_candidates: int) -> List[int]:
        """优化的第一层筛选：使用KD树进行快速几何参数筛选"""

        # 如果有KD树索引，使用KD树进行快速查询
        if self.geometric_kdtree is not None:
            return self._kdtree_geometric_search(
                target_a, target_b, target_cx, target_cy,
                a_tolerance, b_tolerance, cx_tolerance, cy_tolerance,
                max_candidates
            )

        # 回退到原始方法
        return self._filter_by_geometric_params_fallback(
            target_a, target_b, target_cx, target_cy,
            a_tolerance, b_tolerance, cx_tolerance, cy_tolerance,
            max_candidates
        )

    def _kdtree_geometric_search(self, target_a: float, target_b: float,
                                target_cx: float, target_cy: float,
                                a_tolerance: float, b_tolerance: float,
                                cx_tolerance: float, cy_tolerance: float,
                                max_candidates: int) -> List[int]:
        """使用KD树进行快速几何参数搜索"""

        # 构建查询点
        query_point = np.array([[target_a, target_b, target_cx, target_cy]])

        # 计算搜索半径（使用最大容差）
        max_radius = max(
            target_a * a_tolerance,
            target_b * b_tolerance,
            cx_tolerance,
            cy_tolerance
        )

        # 使用KD树进行半径搜索
        indices = self.geometric_kdtree.query_radius(query_point, r=max_radius * 2)[0]

        if len(indices) == 0:
            return []

        # 精确筛选：检查每个参数是否在容差范围内
        valid_candidates = []
        for idx in indices:
            params = self.struct_params.iloc[idx]

            # 检查每个参数是否在容差范围内
            if (abs(params['0'] - target_a) <= target_a * a_tolerance and
                abs(params['1'] - target_b) <= target_b * b_tolerance and
                abs(params['2'] - target_cx) <= cx_tolerance and
                abs(params['3'] - target_cy) <= cy_tolerance):

                valid_candidates.append(idx)

        # 如果候选太多，按几何距离排序
        if len(valid_candidates) > max_candidates:
            valid_candidates = self._rank_by_geometric_distance(
                valid_candidates, target_a, target_b, target_cx, target_cy
            )[:max_candidates]

        return valid_candidates

    def _filter_by_geometric_params_fallback(self, target_a: float, target_b: float,
                                           target_cx: float, target_cy: float,
                                           a_tolerance: float, b_tolerance: float,
                                           cx_tolerance: float, cy_tolerance: float,
                                           max_candidates: int) -> List[int]:
        """回退的几何参数筛选方法（原始实现的优化版本）"""

        # 使用二分搜索找到每个参数的候选集合
        a_candidates = self._find_range_candidates('a', target_a, a_tolerance, is_relative=True)
        b_candidates = self._find_range_candidates('b', target_b, b_tolerance, is_relative=True)
        cx_candidates = self._find_range_candidates('cx', target_cx, cx_tolerance, is_relative=False)
        cy_candidates = self._find_range_candidates('cy', target_cy, cy_tolerance, is_relative=False)

        # 优化：先找到最小的候选集合，减少交集计算量
        candidate_sets = [
            (len(a_candidates), set(a_candidates)),
            (len(b_candidates), set(b_candidates)),
            (len(cx_candidates), set(cx_candidates)),
            (len(cy_candidates), set(cy_candidates))
        ]
        candidate_sets.sort(key=lambda x: x[0])  # 按大小排序

        # 从最小的集合开始计算交集
        intersection = candidate_sets[0][1]
        for _, candidate_set in candidate_sets[1:]:
            intersection = intersection.intersection(candidate_set)
            # 早期终止：如果交集已经很小，不需要继续
            if len(intersection) <= max_candidates:
                break

        intersection_list = list(intersection)

        # 如果候选太多，基于几何距离进行排序和筛选
        if len(intersection_list) > max_candidates:
            intersection_list = self._rank_by_geometric_distance(
                intersection_list, target_a, target_b, target_cx, target_cy
            )[:max_candidates]

        return intersection_list

    def _rank_by_geometric_distance(self, candidates: List[int], target_a: float,
                                   target_b: float, target_cx: float, target_cy: float) -> List[int]:
        """按几何距离对候选进行排序"""

        # 检查缓存
        cache_key = f"{target_a:.3f}_{target_b:.3f}_{target_cx:.1f}_{target_cy:.1f}"
        cached_result = self.geometric_distance_cache.get(cache_key)
        if cached_result is not None:
            # 过滤出仍然有效的候选
            return [idx for idx in cached_result if idx in candidates]

        # 计算几何距离
        distances = []
        for idx in candidates:
            params = self.struct_params.iloc[idx]

            # 使用CSV列名访问数据，计算标准化距离
            a_dist = abs(params['0'] - target_a) / max(target_a, 0.1)
            b_dist = abs(params['1'] - target_b) / max(target_b, 0.1)
            cx_dist = abs(params['2'] - target_cx) / 64.0
            cy_dist = abs(params['3'] - target_cy) / 64.0

            # 加权几何距离（优化权重）
            combined_dist = 0.35 * a_dist + 0.35 * b_dist + 0.15 * cx_dist + 0.15 * cy_dist
            distances.append((idx, combined_dist))

        # 按距离排序
        distances.sort(key=lambda x: x[1])
        sorted_candidates = [idx for idx, _ in distances]

        # 缓存结果
        self.geometric_distance_cache.put(cache_key, sorted_candidates)

        return sorted_candidates

    def _filter_by_contour_iou(self, candidates: List[int], target_a: float, target_b: float,
                              target_cx: float, target_cy: float, iou_threshold: float,
                              max_candidates: int) -> List[int]:
        """第二层筛选：基于轮廓IoU"""

        if len(candidates) <= 1:
            return candidates

        # 创建目标轮廓（理想椭圆）
        target_contour = self._create_target_contour(target_a, target_b, target_cx, target_cy)

        iou_results = []
        processed_count = 0

        for idx in candidates:
            if processed_count >= max_candidates:
                break

            image_path = self.get_image_path(idx)
            candidate_contour = self._get_contour(image_path)

            if candidate_contour is not None:
                iou = self._calculate_contour_iou_direct(target_contour, candidate_contour)
                if iou >= iou_threshold:
                    iou_results.append((idx, iou))

            processed_count += 1

        # 按IoU降序排序
        iou_results.sort(key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in iou_results]

    def _filter_by_shadow_ratio(self, candidates: List[int], target_sr: float,
                               sr_tolerance: float) -> List[int]:
        """第三层筛选：基于阴影比"""

        if not candidates or target_sr is None:
            return candidates

        filtered_candidates = []

        for idx in candidates:
            params = self.struct_params.iloc[idx]
            candidate_sr = params.get('6', 0.5)  # 第6列是阴影比，默认值0.5

            if abs(candidate_sr - target_sr) <= sr_tolerance:
                filtered_candidates.append(idx)

        return filtered_candidates

    def _rank_by_structural_similarity(self, candidates: List[int], target_a: float,
                                     target_b: float, target_cx: float, target_cy: float,
                                     target_sr: float, max_candidates: int) -> List[int]:
        """第四层筛选：基于综合结构相似度排序"""

        if len(candidates) <= max_candidates:
            return candidates

        # 构建目标参数向量（使用所有8个结构参数）
        target_vector = self._build_target_vector(target_a, target_b, target_cx, target_cy, target_sr)

        similarity_results = []

        for idx in candidates:
            params = self.struct_params.iloc[idx]
            candidate_vector = self._build_candidate_vector(params)

            # 计算余弦相似度
            similarity = self._calculate_cosine_similarity(target_vector, candidate_vector)
            similarity_results.append((idx, similarity))

        # 按相似度降序排序
        similarity_results.sort(key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in similarity_results[:max_candidates]]

    def _create_target_contour(self, target_a: float, target_b: float,
                              target_cx: float, target_cy: float) -> np.ndarray:
        """创建目标椭圆轮廓"""

        # 创建128x128的图像
        img = np.zeros((128, 128), dtype=np.uint8)

        # 椭圆参数
        center = (int(target_cx), int(target_cy))
        axes = (int(target_a * 20), int(target_b * 20))  # 缩放因子
        angle = 0

        # 绘制椭圆
        cv2.ellipse(img, center, axes, angle, 0, 360, 255, -1)

        # 查找轮廓
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            return max(contours, key=cv2.contourArea)
        else:
            # 返回一个简单的矩形轮廓作为后备
            return np.array([[[50, 50]], [[78, 50]], [[78, 78]], [[50, 78]]], dtype=np.int32)

    def _calculate_contour_iou_direct(self, contour1: np.ndarray, contour2: np.ndarray) -> float:
        """直接计算两个轮廓的IoU"""

        if contour1 is None or contour2 is None:
            return 0.0

        # 创建掩码
        height, width = 128, 128
        mask1 = np.zeros((height, width), dtype=np.uint8)
        mask2 = np.zeros((height, width), dtype=np.uint8)

        # 绘制轮廓
        cv2.drawContours(mask1, [contour1], -1, 255, -1)
        cv2.drawContours(mask2, [contour2], -1, 255, -1)

        # 计算IoU
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)

        intersection_area = np.sum(intersection)
        union_area = np.sum(union)

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def _build_target_vector(self, target_a: float, target_b: float, target_cx: float,
                           target_cy: float, target_sr: float) -> np.ndarray:
        """构建目标参数向量"""

        # 计算其他参数的估计值
        target_c = max(target_a, target_b) * 0.8  # 圆形度估计
        target_s = min(target_a, target_b) / max(target_a, target_b)  # 椭圆度估计
        target_eg = 0.5  # 边缘梯度默认值

        if target_sr is None:
            target_sr = 0.4  # 阴影比默认值

        return np.array([target_a, target_b, target_cx, target_cy,
                        target_c, target_s, target_sr, target_eg])

    def _build_candidate_vector(self, params: pd.Series) -> np.ndarray:
        """构建候选参数向量，使用CSV列名访问数据"""

        return np.array([
            params.get('0', 1.0),    # 长轴长
            params.get('1', 1.0),    # 短轴长
            params.get('2', 64.0),   # 灰度重心横坐标
            params.get('3', 64.0),   # 灰度重心纵坐标
            params.get('4', 0.8),    # 圆形度
            params.get('5', 0.8),    # 椭圆度
            params.get('6', 0.4),    # 阴影比
            params.get('7', 0.5)     # 边缘梯度
        ])

    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""

        # 归一化向量
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # 计算余弦相似度
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)

        # 确保结果在[0, 1]范围内
        return max(0.0, similarity)

    def find_best_match_simplified(self, target_params: Dict, max_candidates: int = 1000) -> Optional[int]:
        """
        简化的筛选算法：直接计算相似度找到最佳匹配

        Args:
            target_params: 目标气泡参数字典，包含标准化后的参数
            max_candidates: 最大候选数量

        Returns:
            int: 最佳匹配的索引，如果没有找到则返回None
        """
        if len(self.struct_params) == 0:
            return None

        print(f"开始简化筛选，目标参数: a={target_params.get('major_axis_length', 0):.3f}, "
              f"b={target_params.get('minor_axis_length', 0):.3f}, "
              f"cx={target_params.get('centroid_x', 0):.3f}, "
              f"cy={target_params.get('centroid_y', 0):.3f}")

        # 核心参数权重
        weights = {
            'major_axis_length': 0.25,
            'minor_axis_length': 0.25,
            'centroid_x': 0.15,
            'centroid_y': 0.15,
            'shadow_ratio': 0.20
        }

        best_similarity = 0.0
        best_idx = None
        similarities = []

        # 限制候选数量以提高效率
        total_bubbles = len(self.struct_params)
        if max_candidates < total_bubbles:
            # 随机采样max_candidates个候选，确保多样性
            import random
            candidate_indices = random.sample(range(total_bubbles), max_candidates)
            print(f"  从 {total_bubbles} 个气泡中随机采样 {max_candidates} 个候选进行筛选")
        else:
            candidate_indices = list(range(total_bubbles))
            print(f"  筛选所有 {total_bubbles} 个气泡")

        # 批量计算相似度
        batch_size = 10000
        total_candidates = len(candidate_indices)

        for start_idx in range(0, total_candidates, batch_size):
            end_idx = min(start_idx + batch_size, total_candidates)

            # 收集当前批次的候选参数
            batch_candidates = []
            batch_indices = []

            for i in range(start_idx, end_idx):
                idx = candidate_indices[i]
                candidate_params = self.get_struct_params_normalized(idx)
                if candidate_params is not None:
                    batch_candidates.append(candidate_params)
                    batch_indices.append(idx)

            if not batch_candidates:
                continue

            # 使用GPU批量计算或CPU逐个计算
            if self.enable_gpu_acceleration and len(batch_candidates) >= 32:  # 最小GPU批次大小
                # 将大批次分割为GPU可处理的小批次
                gpu_batch_size = min(self.gpu_batch_size, len(batch_candidates))
                batch_similarities = []

                for gpu_start in range(0, len(batch_candidates), gpu_batch_size):
                    gpu_end = min(gpu_start + gpu_batch_size, len(batch_candidates))
                    gpu_candidates = batch_candidates[gpu_start:gpu_end]
                    gpu_indices = batch_indices[gpu_start:gpu_end]

                    try:
                        # GPU批量计算
                        gpu_similarities = self._calculate_bubble_similarity_gpu_batch(
                            target_params, gpu_candidates, weights
                        )

                        # 组合结果
                        for i, similarity in enumerate(gpu_similarities):
                            idx = gpu_indices[i]
                            batch_similarities.append((idx, similarity))

                            # 更新最佳匹配
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_idx = idx

                    except Exception as e:
                        print(f"GPU批量计算失败，回退到CPU模式: {e}")
                        # CPU回退
                        for i, candidate_params in enumerate(gpu_candidates):
                            idx = gpu_indices[i]
                            similarity = self._calculate_bubble_similarity_fast(target_params, candidate_params, weights)
                            batch_similarities.append((idx, similarity))

                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_idx = idx
            else:
                # CPU逐个计算（小批次或GPU不可用）
                batch_similarities = []
                for i, candidate_params in enumerate(batch_candidates):
                    idx = batch_indices[i]
                    similarity = self._calculate_bubble_similarity_fast(target_params, candidate_params, weights)
                    batch_similarities.append((idx, similarity))

                    # 更新最佳匹配
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_idx = idx

            similarities.extend(batch_similarities)

            if (start_idx // batch_size + 1) % 10 == 0:
                print(f"  已处理 {end_idx}/{total_candidates} 个候选气泡，当前最佳相似度: {best_similarity:.3f}")

        print(f"筛选完成，最佳匹配索引: {best_idx}, 相似度: {best_similarity:.3f}")

        return best_idx if best_similarity > 0.1 else None  # 设置最低相似度阈值

    def get_struct_params_normalized(self, idx: int) -> Optional[Dict]:
        """获取标准化的结构参数"""
        if idx >= len(self.struct_params):
            return None

        row = self.struct_params.iloc[idx]

        # 转换为标准化格式
        return {
            'major_axis_length': row.get('0', 1.0),      # 长轴长
            'minor_axis_length': row.get('1', 1.0),      # 短轴长
            'centroid_x': row.get('2', 0.5),             # 重心x
            'centroid_y': row.get('3', 0.5),             # 重心y
            'circularity': row.get('4', 0.8),            # 圆形度
            'solidity': row.get('5', 0.8),               # 椭圆度
            'shadow_ratio': row.get('6', 0.4),           # 阴影比
            'edge_gradient': row.get('7', 0.5)           # 边缘梯度
        }

    def _calculate_bubble_similarity_fast(self, params1: Dict, params2: Dict, weights: Dict) -> float:
        """快速计算气泡相似度"""
        # 核心参数
        core_params = ['major_axis_length', 'minor_axis_length', 'centroid_x', 'centroid_y', 'shadow_ratio']

        # 计算加权欧氏距离
        weighted_distances = []
        for param in core_params:
            if param in params1 and param in params2 and param in weights:
                diff = abs(params1[param] - params2[param])
                weighted_distances.append(diff * weights[param])

        if not weighted_distances:
            return 0.0

        euclidean_distance = np.sqrt(sum([d**2 for d in weighted_distances]))
        euclidean_similarity = 1.0 / (1.0 + euclidean_distance)

        # 计算余弦相似度
        all_params = ['major_axis_length', 'minor_axis_length', 'centroid_x', 'centroid_y',
                      'circularity', 'solidity', 'shadow_ratio', 'edge_gradient']

        vec1 = []
        vec2 = []
        for param in all_params:
            if param in params1 and param in params2:
                vec1.append(params1[param])
                vec2.append(params2[param])

        if len(vec1) > 0:
            cosine_sim = cosine_similarity_manual(np.array(vec1), np.array(vec2))
        else:
            cosine_sim = 0.0

        # 综合相似度
        combined_similarity = 0.6 * euclidean_similarity + 0.4 * cosine_sim

        return max(0.0, min(1.0, combined_similarity))

    def _calculate_bubble_similarity_gpu_batch(self, target_params: Dict,
                                             candidate_params_list: List[Dict],
                                             weights: Dict, device: str = None) -> List[float]:
        """
        GPU批量计算气泡相似度

        Args:
            target_params: 目标气泡参数
            candidate_params_list: 候选气泡参数列表
            weights: 权重字典
            device: GPU设备

        Returns:
            相似度列表
        """
        if not GPU_AVAILABLE or not self.enable_gpu_acceleration:
            # CPU回退
            return [self._calculate_bubble_similarity_fast(target_params, candidate_params, weights)
                   for candidate_params in candidate_params_list]

        try:
            if device is None:
                device = self.gpu_manager.get_device(0) if self.gpu_manager else 'cpu'

            if device == 'cpu':
                # CPU回退
                return [self._calculate_bubble_similarity_fast(target_params, candidate_params, weights)
                       for candidate_params in candidate_params_list]

            # 设置GPU设备
            if device.startswith('cuda'):
                torch.cuda.set_device(device)

            batch_size = len(candidate_params_list)
            if batch_size == 0:
                return []

            # 核心参数和所有参数定义
            core_params = ['major_axis_length', 'minor_axis_length', 'centroid_x', 'centroid_y', 'shadow_ratio']
            all_params = ['major_axis_length', 'minor_axis_length', 'centroid_x', 'centroid_y',
                         'circularity', 'solidity', 'shadow_ratio', 'edge_gradient']

            # 准备目标参数张量
            target_core_values = []
            target_all_values = []
            target_weights = []

            for param in core_params:
                if param in target_params and param in weights:
                    target_core_values.append(target_params[param])
                    target_weights.append(weights[param])
                else:
                    target_core_values.append(0.0)
                    target_weights.append(0.0)

            for param in all_params:
                if param in target_params:
                    target_all_values.append(target_params[param])
                else:
                    target_all_values.append(0.0)

            # 准备候选参数张量
            candidate_core_batch = []
            candidate_all_batch = []
            valid_core_mask = []
            valid_all_mask = []

            for candidate_params in candidate_params_list:
                # 核心参数
                core_values = []
                core_mask = []
                for param in core_params:
                    if param in candidate_params and param in target_params and param in weights:
                        core_values.append(candidate_params[param])
                        core_mask.append(1.0)
                    else:
                        core_values.append(0.0)
                        core_mask.append(0.0)

                candidate_core_batch.append(core_values)
                valid_core_mask.append(core_mask)

                # 所有参数
                all_values = []
                all_mask = []
                for param in all_params:
                    if param in candidate_params and param in target_params:
                        all_values.append(candidate_params[param])
                        all_mask.append(1.0)
                    else:
                        all_values.append(0.0)
                        all_mask.append(0.0)

                candidate_all_batch.append(all_values)
                valid_all_mask.append(all_mask)

            # 转换为PyTorch张量
            target_core_tensor = torch.tensor(target_core_values, dtype=torch.float32, device=device)
            target_all_tensor = torch.tensor(target_all_values, dtype=torch.float32, device=device)
            target_weights_tensor = torch.tensor(target_weights, dtype=torch.float32, device=device)

            candidate_core_tensor = torch.tensor(candidate_core_batch, dtype=torch.float32, device=device)
            candidate_all_tensor = torch.tensor(candidate_all_batch, dtype=torch.float32, device=device)
            valid_core_tensor = torch.tensor(valid_core_mask, dtype=torch.float32, device=device)
            valid_all_tensor = torch.tensor(valid_all_mask, dtype=torch.float32, device=device)

            # 批量计算加权欧氏距离
            core_diff = torch.abs(candidate_core_tensor - target_core_tensor.unsqueeze(0))
            weighted_diff = core_diff * target_weights_tensor.unsqueeze(0) * valid_core_tensor
            euclidean_distances = torch.sqrt(torch.sum(weighted_diff ** 2, dim=1))
            euclidean_similarities = 1.0 / (1.0 + euclidean_distances)

            # 批量计算余弦相似度
            # 只考虑有效的参数
            target_all_masked = target_all_tensor.unsqueeze(0) * valid_all_tensor
            candidate_all_masked = candidate_all_tensor * valid_all_tensor

            # 计算点积
            dot_products = torch.sum(target_all_masked * candidate_all_masked, dim=1)

            # 计算范数
            target_norms = torch.sqrt(torch.sum(target_all_masked ** 2, dim=1))
            candidate_norms = torch.sqrt(torch.sum(candidate_all_masked ** 2, dim=1))

            # 避免除零
            norms_product = target_norms * candidate_norms
            cosine_similarities = torch.where(
                norms_product > 1e-8,
                dot_products / norms_product,
                torch.zeros_like(dot_products)
            )

            # 综合相似度
            combined_similarities = 0.6 * euclidean_similarities + 0.4 * cosine_similarities
            combined_similarities = torch.clamp(combined_similarities, 0.0, 1.0)

            # 转换回CPU并返回列表
            result = combined_similarities.cpu().numpy().tolist()

            return result

        except Exception as e:
            print(f"GPU批量计算失败，回退到CPU模式: {e}")
            # CPU回退
            return [self._calculate_bubble_similarity_fast(target_params, candidate_params, weights)
                   for candidate_params in candidate_params_list]
        finally:
            # 清理GPU内存
            if device and device.startswith('cuda'):
                torch.cuda.empty_cache()
            gc.collect()

    def _filter_by_contour_iou_parallel(self, candidates: List[int], target_a: float, target_b: float,
                                       target_cx: float, target_cy: float, iou_threshold: float,
                                       max_candidates: int) -> List[int]:
        """并行IoU筛选"""

        if len(candidates) <= 1:
            return candidates

        # 创建目标轮廓（理想椭圆）
        target_contour = self._create_target_contour(target_a, target_b, target_cx, target_cy)

        if not self.enable_parallel or len(candidates) < 10:
            # 对于小数据集，使用串行处理
            return self._filter_by_contour_iou(candidates, target_a, target_b, target_cx, target_cy,
                                             iou_threshold, max_candidates)

        # 并行处理IoU计算
        def calculate_iou_for_candidate(idx):
            try:
                image_data = self.get_image_data(idx)
                if image_data is not None:
                    candidate_contour = self._get_contour_from_data(image_data, idx)
                    if candidate_contour is not None:
                        iou = self._calculate_contour_iou_direct(target_contour, candidate_contour)
                        return (idx, iou) if iou >= iou_threshold else None
            except Exception as e:
                print(f"计算IoU时出错 (idx={idx}): {e}")
            return None

        # 使用线程池并行计算IoU
        iou_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 限制处理的候选数量以避免过多的I/O
            process_candidates = candidates[:max_candidates * 2]
            futures = [executor.submit(calculate_iou_for_candidate, idx) for idx in process_candidates]

            for future in futures:
                result = future.result()
                if result is not None:
                    iou_results.append(result)
                    # 早期终止：如果已经找到足够的候选
                    if len(iou_results) >= max_candidates:
                        break

        # 按IoU降序排序
        iou_results.sort(key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in iou_results[:max_candidates]]

    def _filter_by_shadow_ratio_fast(self, candidates: List[int], target_sr: float,
                                   sr_tolerance: float) -> List[int]:
        """快速阴影比筛选"""

        if not candidates or target_sr is None:
            return candidates

        # 向量化操作，一次性筛选所有候选
        candidate_indices = np.array(candidates)
        sr_values = self.struct_params.iloc[candidate_indices]['6'].values  # 第6列是阴影比

        # 使用向量化操作进行筛选
        mask = np.abs(sr_values - target_sr) <= sr_tolerance
        filtered_candidates = candidate_indices[mask].tolist()

        return filtered_candidates

    def _rank_by_structural_similarity_parallel(self, candidates: List[int], target_a: float,
                                              target_b: float, target_cx: float, target_cy: float,
                                              target_sr: float, max_candidates: int) -> List[int]:
        """并行综合结构相似度排序"""

        if len(candidates) <= max_candidates:
            return candidates

        # 构建目标参数向量
        target_vector = self._build_target_vector(target_a, target_b, target_cx, target_cy, target_sr)

        if not self.enable_parallel or len(candidates) < 20:
            # 对于小数据集，使用串行处理
            return self._rank_by_structural_similarity(candidates, target_a, target_b, target_cx, target_cy,
                                                     target_sr, max_candidates)

        def calculate_similarity_for_candidate(idx):
            try:
                params = self.struct_params.iloc[idx]
                candidate_vector = self._build_candidate_vector(params)
                similarity = self._calculate_cosine_similarity(target_vector, candidate_vector)
                return (idx, similarity)
            except Exception as e:
                print(f"计算相似度时出错 (idx={idx}): {e}")
                return (idx, 0.0)

        # 使用线程池并行计算相似度
        similarity_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(calculate_similarity_for_candidate, idx) for idx in candidates]
            similarity_results = [future.result() for future in futures]

        # 按相似度降序排序
        similarity_results.sort(key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in similarity_results[:max_candidates]]

    def _quick_rank_candidates(self, candidates: List[int], target_a: float, target_b: float,
                             target_cx: float, target_cy: float, target_sr: float) -> List[int]:
        """快速候选排序（用于候选数量较少的情况）"""

        if len(candidates) <= 1:
            return candidates

        # 简化的相似度计算，只考虑几何参数
        scores = []
        for idx in candidates:
            params = self.struct_params.iloc[idx]

            # 计算几何相似度
            a_sim = 1.0 - abs(params['0'] - target_a) / max(target_a, 0.1)
            b_sim = 1.0 - abs(params['1'] - target_b) / max(target_b, 0.1)
            cx_sim = 1.0 - abs(params['2'] - target_cx) / 64.0
            cy_sim = 1.0 - abs(params['3'] - target_cy) / 64.0

            # 加权平均
            geometric_score = 0.35 * max(0, a_sim) + 0.35 * max(0, b_sim) + 0.15 * max(0, cx_sim) + 0.15 * max(0, cy_sim)

            # 如果有阴影比，加入阴影相似度
            if target_sr is not None:
                sr_sim = 1.0 - abs(params['6'] - target_sr) / 0.5
                geometric_score = 0.8 * geometric_score + 0.2 * max(0, sr_sim)

            scores.append((idx, geometric_score))

        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in scores]

    def cleanup_gpu_resources(self):
        """清理GPU资源"""
        if self.gpu_manager:
            try:
                self.gpu_manager.cleanup()
            except Exception as e:
                print(f"清理GPU资源时出错: {e}")

    def __del__(self):
        """析构函数，确保GPU资源被正确清理"""
        self.cleanup_gpu_resources()
