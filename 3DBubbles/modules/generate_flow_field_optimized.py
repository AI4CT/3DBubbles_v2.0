import scipy.io as sio
import numpy as np
import cv2
import math
import argparse
import matplotlib
from tqdm.contrib import tzip
import os
import matplotlib.pyplot as plt
import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from .pregenerated_dataset_manager import PregeneratedDatasetManager


class Timer:
    def __init__(self):
        self.times = {}
        self.start_time = None
        
    def start(self):
        self.start_time = time.time()
        
    def end(self, name):
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(time.time() - self.start_time)
        
    def get_average_time(self, name):
        if name in self.times:
            return sum(self.times[name]) / len(self.times[name])
        return 0
    
    def print_times(self):
        print("\n时间统计:")
        print("-" * 50)
        for name, times in self.times.items():
            avg_time = sum(times) / len(times)
            print(f"{name}: {avg_time:.2f}秒")
        print("-" * 50)


class OptimizedBubbleSelector:
    """优化的气泡筛选器，支持并行处理和缓存优化"""

    def __init__(self, dataset_manager: PregeneratedDatasetManager, max_workers: int = 4):
        self.dataset_manager = dataset_manager
        self.iou_weight = 0.7  # IoU权重
        self.structural_weight = 0.3  # 结构相似度权重
        self.max_workers = max_workers

        # 性能统计
        self.selection_stats = {
            'total_selections': 0,
            'cache_hits': 0,
            'average_candidates': 0,
            'average_time': 0
        }
        
    def select_best_bubble(self, target_a: float, target_b: float, target_cx: float, target_cy: float,
                          a_tolerance: float = 0.1, b_tolerance: float = 0.1,
                          cx_tolerance: float = 5.0, cy_tolerance: float = 5.0,
                          target_sr: float = None, sr_tolerance: float = 0.1,
                          enable_iou_filtering: bool = True, iou_threshold: float = 0.3,
                          max_candidates: int = 100) -> tuple:
        """
        优化的气泡选择方法，支持并行处理和性能统计

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

        Returns:
            (best_idx, best_score, bubble_image, bubble_params)
        """
        start_time = time.time()

        # 使用优化的分层筛选机制获取候选气泡
        candidates = self.dataset_manager.find_similar_bubbles(
            target_a, target_b, target_cx, target_cy,
            a_tolerance, b_tolerance, cx_tolerance, cy_tolerance,
            target_sr, sr_tolerance,
            enable_iou_filtering, iou_threshold,
            max_candidates
        )

        if not candidates:
            return None, 0.0, None, None

        # 并行评估多个候选（如果候选数量足够多）
        if len(candidates) > 3:
            best_idx, best_score = self._parallel_candidate_evaluation(
                candidates[:min(5, len(candidates))], target_a, target_b, target_cx, target_cy, target_sr
            )
        else:
            # 对于少量候选，使用串行处理
            best_idx = candidates[0]
            best_score = self._calculate_comprehensive_score(
                best_idx, target_a, target_b, target_cx, target_cy, target_sr
            )

        # 加载最佳匹配的气泡图像
        bubble_image = self._load_bubble_image_optimized(best_idx)
        bubble_params = self.dataset_manager.get_struct_params(best_idx)

        # 更新性能统计
        self._update_selection_stats(len(candidates), time.time() - start_time)

        return best_idx, best_score, bubble_image, bubble_params

    def _parallel_candidate_evaluation(self, candidates: list, target_a: float, target_b: float,
                                     target_cx: float, target_cy: float, target_sr: float) -> tuple:
        """并行评估候选气泡"""

        def evaluate_candidate(idx):
            score = self._calculate_comprehensive_score(idx, target_a, target_b, target_cx, target_cy, target_sr)
            return (idx, score)

        # 使用线程池并行评估
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(candidates))) as executor:
            results = list(executor.map(evaluate_candidate, candidates))

        # 选择最佳候选
        best_idx, best_score = max(results, key=lambda x: x[1])
        return best_idx, best_score

    def _update_selection_stats(self, num_candidates: int, selection_time: float):
        """更新选择统计信息"""
        self.selection_stats['total_selections'] += 1
        self.selection_stats['average_candidates'] = (
            (self.selection_stats['average_candidates'] * (self.selection_stats['total_selections'] - 1) + num_candidates) /
            self.selection_stats['total_selections']
        )
        self.selection_stats['average_time'] = (
            (self.selection_stats['average_time'] * (self.selection_stats['total_selections'] - 1) + selection_time) /
            self.selection_stats['total_selections']
        )
    
    def _calculate_comprehensive_score(self, candidate_idx: int, target_a: float, target_b: float,
                                     target_cx: float, target_cy: float, target_sr: float = None) -> float:
        """计算综合评分"""

        candidate_params = self.dataset_manager.get_struct_params(candidate_idx)

        # 几何参数相似度（权重最高）
        geometric_score = self._calculate_geometric_similarity(
            candidate_params, target_a, target_b, target_cx, target_cy
        )

        # 阴影特征相似度
        shadow_score = 1.0  # 默认值
        if target_sr is not None and 'SR' in candidate_params:
            shadow_diff = abs(candidate_params['SR'] - target_sr)
            shadow_score = max(0.0, 1.0 - shadow_diff / 0.5)  # 归一化到[0,1]

        # 综合评分（可调整权重）
        comprehensive_score = 0.8 * geometric_score + 0.2 * shadow_score

        return comprehensive_score

    def _calculate_geometric_similarity(self, candidate_params: dict, target_a: float,
                                      target_b: float, target_cx: float, target_cy: float) -> float:
        """计算几何参数相似度"""

        # 提取候选参数
        cand_a = candidate_params.get('a', 1.0)
        cand_b = candidate_params.get('b', 1.0)
        cand_cx = candidate_params.get('cx', 64.0)
        cand_cy = candidate_params.get('cy', 64.0)

        # 计算各参数的相似度
        a_similarity = 1.0 - abs(cand_a - target_a) / max(target_a, 0.1)
        b_similarity = 1.0 - abs(cand_b - target_b) / max(target_b, 0.1)
        cx_similarity = 1.0 - abs(cand_cx - target_cx) / 64.0  # 归一化到图像尺寸
        cy_similarity = 1.0 - abs(cand_cy - target_cy) / 64.0

        # 确保相似度在[0,1]范围内
        similarities = [max(0.0, min(1.0, sim)) for sim in [a_similarity, b_similarity, cx_similarity, cy_similarity]]

        # 加权平均（长短轴权重更高）
        geometric_score = 0.3 * similarities[0] + 0.3 * similarities[1] + 0.2 * similarities[2] + 0.2 * similarities[3]

        return geometric_score
    
    def _load_bubble_image_optimized(self, idx: int):
        """优化的气泡图像加载方法，直接从数据集管理器获取"""
        try:
            # 直接从数据集管理器获取图像数据，避免文件I/O
            image_data = self.dataset_manager.get_image_data(idx)
            if image_data is not None:
                # 确保图像格式正确
                if len(image_data.shape) == 3:
                    image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                else:
                    image = image_data.copy()

                # 确保图像尺寸为128x128
                if image.shape != (128, 128):
                    image = cv2.resize(image, (128, 128))

                return image
        except Exception as e:
            print(f"加载图像数据时出错 (idx={idx}): {e}")

        return None

    def _load_bubble_image(self, image_path: str):
        """传统的气泡图像加载方法（保持兼容性）"""
        if not os.path.exists(image_path):
            return None

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        # 确保图像尺寸为128x128
        if image.shape != (128, 128):
            image = cv2.resize(image, (128, 128))

        return image

    def get_selection_stats(self) -> dict:
        """获取选择统计信息"""
        return self.selection_stats.copy()

    def reset_selection_stats(self):
        """重置选择统计信息"""
        self.selection_stats = {
            'total_selections': 0,
            'cache_hits': 0,
            'average_candidates': 0,
            'average_time': 0
        }


def cut_image_out_of_range(img, cx, cy, pad=None):
    """从原代码保留的图像裁剪函数"""
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


def generate_flow_field_optimized(args):
    """优化的流场生成函数"""
    timer = Timer()
    timer.start()
    
    # 初始化预生成数据集管理器
    print("初始化预生成数据集管理器...")
    dataset_manager = PregeneratedDatasetManager(
        image_dir=args.image_path,
        struct_csv_path=args.BubInfo_path,
        cache_dir=args.cache_path if hasattr(args, 'cache_path') else None
    )
    
    # 初始化优化的气泡筛选器
    bubble_selector = OptimizedBubbleSelector(dataset_manager)
    
    print(f"数据集大小: {dataset_manager.get_dataset_size()}")
    timer.end('数据集初始化')
    
    # 计算流场参数
    if args.channel == 'cylinder':
        TotVol = math.pi * args.Width * args.Width * args.Height / 4        # mm
    elif args.channel == 'rectangle':
        TotVol = args.Depth * args.Width * args.Height                      # mm
    Height_pix = round(args.Height / args.pixtomm)
    Width_pix = round(args.Width / args.pixtomm)

    for ImgNum in range(args.ImageGenNum):
        print(f'正在生成第{ImgNum + 1}/{args.ImageGenNum}张流场图像')
        
        #---------------------------------------------------------#
        #   优化的气泡筛选
        #---------------------------------------------------------#
        timer.start()
        Bubs_Volume = 0
        BubList = []
        ImgLabel = []
        bub_conts_pad = []
        bub_conts = []
        
        print('|----------(1/2)开始筛选气泡!')
        
        # 创建气泡图像保存目录
        save_path = f'{args.save_path}/{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        selection_count = 0
        max_attempts = 10000  # 最大尝试次数，避免无限循环
        attempts = 0
        
        while Bubs_Volume < args.gas_holdup * TotVol and attempts < max_attempts:
            attempts += 1
            
            # 生成目标几何参数
            target_a = np.random.uniform(0.5, 2.0)  # 长轴长
            target_b = np.random.uniform(0.5, 2.0)  # 短轴长
            target_cx = np.random.uniform(20, 108)  # 重心横坐标
            target_cy = np.random.uniform(20, 108)  # 重心纵坐标
            target_sr = np.random.uniform(0.0, 0.8) if hasattr(args, 'use_shadow_ratio') and args.use_shadow_ratio else None

            # 使用优化的筛选器选择最佳匹配气泡
            best_idx, best_score, bubble_image, bubble_params = bubble_selector.select_best_bubble(
                target_a=target_a,
                target_b=target_b,
                target_cx=target_cx,
                target_cy=target_cy,
                a_tolerance=args.a_tolerance if hasattr(args, 'a_tolerance') else 0.1,
                b_tolerance=args.b_tolerance if hasattr(args, 'b_tolerance') else 0.1,
                cx_tolerance=args.cx_tolerance if hasattr(args, 'cx_tolerance') else 5.0,
                cy_tolerance=args.cy_tolerance if hasattr(args, 'cy_tolerance') else 5.0,
                target_sr=target_sr,
                sr_tolerance=args.sr_tolerance if hasattr(args, 'sr_tolerance') else 0.1,
                enable_iou_filtering=args.enable_iou_filtering if hasattr(args, 'enable_iou_filtering') else True,
                iou_threshold=args.iou_threshold if hasattr(args, 'iou_threshold') else 0.3
            )
            
            if bubble_image is not None and best_score > args.min_similarity_score:
                # 获取实际的几何参数
                a = bubble_params.get('a', target_a)
                b = bubble_params.get('b', target_b)
                cx = bubble_params.get('cx', target_cx)
                cy = bubble_params.get('cy', target_cy)

                # 计算体积和离心率（用于兼容性）
                Bub_Volume = 128**3 * args.pixtomm**3 * (a * b * max(a, b) * math.pi / 6)
                c = math.sqrt(abs(a**2 - b**2))
                eccentricity = c / max(a, b) if max(a, b) > 0 else 0
                
                # 生成旋转角度
                angle_rad = np.random.normal(args.phi_e, args.phi_sd)
                while angle_rad < -1 or angle_rad > 1:
                    angle_rad = np.random.normal(args.phi_e, args.phi_sd)
                angle = round(90 - angle_rad * 180 / math.pi)
                
                Bubs_Volume += Bub_Volume
                
                # 构建标签（保持与原代码兼容的格式）
                label_data = [
                    a,  # 使用实际获取的参数
                    b,
                    cx,
                    cy,
                    bubble_params.get('C', 0.8),
                    bubble_params.get('S', 0.8),
                    bubble_params.get('SR', 0.5),
                    bubble_params.get('EG', 0.5),
                    angle_rad,
                    angle,
                    Bub_Volume,
                    eccentricity,
                    0,  # x position (will be set later)
                    0,  # y position (will be set later)
                    0   # label (will be set later)
                ]
                
                ImgLabel.append(label_data)
                BubList.append(bubble_image)
                selection_count += 1
                
                selection_process = 100 * Bubs_Volume / (args.gas_holdup * TotVol)
                selection_process = min(selection_process, 100)
                print(f'|----------气泡筛选已进行{selection_process:.1f}% (筛选了{selection_count}个气泡, 相似度: {best_score:.3f})')
                print(f'            匹配参数: a={a:.3f}, b={b:.3f}, cx={cx:.1f}, cy={cy:.1f}')
        
        if attempts >= max_attempts:
            print(f"警告: 达到最大尝试次数 {max_attempts}，可能需要调整筛选参数")
        
        print(f'|----------已成功筛选{len(BubList)}个气泡!')
        timer.end('气泡筛选')
        
        #---------------------------------------------------------#
        #   生成流场
        #---------------------------------------------------------#
        timer.start()
        print('|----------(2/2)开始生成流场!')
        ImgLabel_new = np.array(ImgLabel)
        BubImg_new = np.array(BubList)

        if len(BubImg_new) == 0:
            print("警告: 没有筛选到合适的气泡，跳过此次生成")
            continue

        # 按边缘锐利度排序（如果有EG参数）
        if ImgLabel_new.shape[1] > 7:
            BubInfo_index = np.argsort(ImgLabel_new[:, 7])
            ImgLabel_new = ImgLabel_new[BubInfo_index]
            BubImg_new = BubImg_new[BubInfo_index]

        background = np.zeros((Height_pix + args.pad * 2, Width_pix + args.pad * 2, 3), np.uint8)

        # 合并标签生成、掩码生成、流场合成
        bubble_labels = []
        bubble_masks = []
        bub_conts = []
        bub_conts_pad = []

        x_e = 0.5 * Width_pix + args.pad  # 简化的位置期望

        for i, (img, label) in enumerate(zip(BubImg_new, ImgLabel_new)):
            # 位置采样
            x = np.random.randint(64, Width_pix - 64) + args.pad
            y = np.random.randint(64, Height_pix - 64) + args.pad
            ImgLabel_new[i, -3:-1] = x - args.pad, y - args.pad

            # 处理图像
            if len(img.shape) == 2:
                img = img.reshape(128, 128, 1)
            img = cv2.bitwise_not(img).reshape(128, 128, 1)
            img = img.repeat(3, -1)

            # 旋转图像
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), label[9], 1)
            cropped_img = cv2.warpAffine(img, M, (cols, rows), borderValue=(0, 0, 0))

            # 创建掩码
            mask = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

            # 查找轮廓
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            if len(contours) >= 1:
                main_contour = contours[0]
                mask_with_contour = np.zeros_like(mask)
                cv2.drawContours(mask_with_contour, [main_contour], -1, 255, thickness=cv2.FILLED)
            else:
                continue

            # 生成全局掩码
            global_mask = np.zeros((Height_pix + args.pad * 2, Width_pix + args.pad * 2), dtype=np.uint8)
            x0, y0 = max(0, x - 64), max(0, y - 64)
            x1, y1 = min(Width_pix + args.pad * 2, x + 64), min(Height_pix + args.pad * 2, y + 64)
            mask_x0, mask_y0 = max(0, 64 - x), max(0, 64 - y)
            mask_x1, mask_y1 = min(128, 128 - (x + 64 - (Width_pix + args.pad * 2))), min(128, 128 - (y + 64 - (Height_pix + args.pad * 2)))

            if x1 > x0 and y1 > y0 and mask_x1 > mask_x0 and mask_y1 > mask_y0:
                global_mask[y0:y1, x0:x1] = mask_with_contour[mask_y0:mask_y1, mask_x0:mask_x1]
            else:
                continue

            # 计算重叠标签
            overlap_idx = []
            for j, prev_mask in enumerate(bubble_masks):
                overlap = np.logical_and(global_mask, prev_mask)
                if np.count_nonzero(overlap) > 5:
                    overlap_idx.append(j)

            if not overlap_idx:
                label_val = 0
            else:
                for j in overlap_idx:
                    if bubble_labels[j] == 0:
                        bubble_labels[j] = 1
                max_label = max([bubble_labels[j] for j in overlap_idx])
                label_val = max_label + 1

            bubble_labels.append(label_val)
            bubble_masks.append(global_mask)

            # 合成流场
            roi = cut_image_out_of_range(background, x, y)
            origin_contour = main_contour[:, 0, :].copy()
            bub_conts_pad.append(main_contour)

            main_contour[:, 0, 0] += x - args.pad - 64
            main_contour[:, 0, 1] += y - args.pad - 64
            bub_conts.append(main_contour)

            mask_inv = cv2.bitwise_not(mask_with_contour)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            img_fg = cv2.bitwise_and(cropped_img, cropped_img, mask=mask_with_contour)
            dst = cv2.add(img_bg, img_fg)

            background[y - 64 : y + 64, x - 64 : x + 64] = dst

        # 更新标签
        if len(bubble_labels) > 0:
            ImgLabel_new = np.hstack([ImgLabel_new[:len(bubble_labels), :-1],
                                    np.array(bubble_labels).reshape(-1, 1)])

        # 处理最终图像
        background = cv2.bitwise_not(background)
        background = background[args.pad : Height_pix + args.pad, args.pad : Width_pix + args.pad]

        # 保存结果
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cv2.imwrite(os.path.join(save_path, 'Image.jpg'), background)

        # 保存标签文件
        if len(ImgLabel_new) > 0:
            header = "a,b,cx,cy,C,S,SR,EG,angle_rad,angle,V,e,x,y,label"
            formats = ['%.6f'] * 14 + ['%d']
            np.savetxt(os.path.join(save_path, 'structure_info.csv'), ImgLabel_new,
                      delimiter=',', header=header, comments='', fmt=formats)

        # 保存轮廓信息
        if bub_conts:
            with open(os.path.join(save_path, 'bub_conts.txt'), 'w') as file:
                for bub_cont in bub_conts:
                    file.write(','.join(map(str, bub_cont.reshape(-1))) + '\n')

        timer.end('流场生成')
        print(f"流场生成完成! 保存路径: {save_path}")
    
    # 打印时间统计
    timer.print_times()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="优化的流场生成器")
    
    # 基本参数
    parser.add_argument("--channel", type=str, default='cylinder', help="通道类型")
    parser.add_argument("--Depth", type=float, default=100, help="深度(mm)")
    parser.add_argument("--Width", type=float, default=100, help="宽度(mm)")
    parser.add_argument("--Height", type=float, default=64, help="高度(mm)")
    parser.add_argument("--pixtomm", type=float, default=0.080128, help="像素尺寸换算")
    
    # 分布参数
    parser.add_argument("--phi_e", type=float, default=-0.0, help="角度分布期望")
    parser.add_argument("--phi_sd", type=float, default=0.29769662022590637, help="角度分布标准差")
    parser.add_argument("--V_sd", type=float, default=50, help="气泡体积分布标准差")
    parser.add_argument("--e_sd", type=float, default=0.27194802937871304, help="离心率分布标准差")
    parser.add_argument("--gas_holdup", type=float, default=0.004, help="固定气含率")
    
    # 新的筛选参数（基于几何参数优先级）
    parser.add_argument("--a_tolerance", type=float, default=0.1, help="长轴长容差（相对）")
    parser.add_argument("--b_tolerance", type=float, default=0.1, help="短轴长容差（相对）")
    parser.add_argument("--cx_tolerance", type=float, default=5.0, help="重心横坐标容差（绝对）")
    parser.add_argument("--cy_tolerance", type=float, default=5.0, help="重心纵坐标容差（绝对）")
    parser.add_argument("--sr_tolerance", type=float, default=0.1, help="阴影比容差（绝对）")
    parser.add_argument("--use_shadow_ratio", type=lambda x: x.lower() == 'true', default=False, help="是否使用阴影比筛选")
    parser.add_argument("--enable_iou_filtering", type=lambda x: x.lower() == 'true', default=True, help="是否启用IoU筛选")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="IoU阈值")
    parser.add_argument("--min_similarity_score", type=float, default=0.5, help="最小相似度阈值")
    
    # 路径参数
    parser.add_argument("--image_path", type=str, 
                        default='/home/yubd/mount/dataset/dataset_BubStyle/generated_bubble_images',
                        help="预生成图像路径")
    parser.add_argument("--BubInfo_path", type=str,
                        default='/home/yubd/mount/dataset/dataset_BubStyle/generated_bubble_structs.csv',
                        help="结构参数CSV路径")
    parser.add_argument("--save_path", type=str,
                        default='./optimized_flow_field_output',
                        help="输出保存路径")
    parser.add_argument("--cache_path", type=str,
                        default='./dataset_cache',
                        help="数据集缓存路径")
    
    # 生成参数
    parser.add_argument("--ImageGenNum", type=int, default=10, help="生成流场数量")
    parser.add_argument("--pad", type=int, default=128, help="背景填充")
    
    args = parser.parse_args()
    print("优化的流场生成器参数:")
    print(args)
    
    generate_flow_field_optimized(args)
