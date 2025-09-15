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

def cut_image_out_of_range(img, cx, cy, pad = None):
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

def precompute_bubble_info(bub_info, pixtomm):
    """预计算气泡信息"""
    volumes = []
    eccentricities = []
    for info in bub_info:
        volume = 128**3 * pixtomm**3 * (info[0] * info[1] * max(info[0], info[1]) * math.pi / 6)
        c = math.sqrt(abs(info[0]**2 - info[1]**2))
        eccentricity = c / info[0]
        volumes.append(volume)
        eccentricities.append(eccentricity)
    return np.array(volumes), np.array(eccentricities)

def find_matching_bubble(volumes, eccentricities, V_sd, e_sd, million_bub, random_int, BubInfo_rand):
    """查找匹配的气泡"""
    Bub_Volume = volumes[random_int]
    eccentricity = eccentricities[random_int]
    
    V_sd_rand = np.random.rayleigh(V_sd)
    e_sd_rand = 1 - np.random.rayleigh(e_sd)
    
    if abs(Bub_Volume - V_sd_rand) < 0.001 * V_sd_rand and abs(eccentricity - e_sd_rand) < 0.01 * e_sd_rand:
        return million_bub[random_int], BubInfo_rand, Bub_Volume
    return None, None, None

def calculate_overlap(contour1, contour2, height, width, pad):
    """计算两个轮廓的重叠面积比例"""
    # 创建掩码
    mask1 = np.zeros((height + pad * 2, width + pad * 2), dtype=np.uint8)
    mask2 = np.zeros((height + pad * 2, width + pad * 2), dtype=np.uint8)
    
    # 绘制轮廓
    cv2.drawContours(mask1, [contour1], -1, 255, -1)
    cv2.drawContours(mask2, [contour2], -1, 255, -1)
    
    # 计算重叠
    overlap = cv2.bitwise_and(mask1, mask2)
    overlap_area = np.sum(overlap > 0)
    contour1_area = np.sum(mask1 > 0)
    
    return overlap_area / contour1_area if contour1_area > 0 else 0

def generate_flow_field(args):
    timer = Timer()
    timer.start()
    
    # 读取 MillionBub.mat 文件
    mat_folders = [f for f in os.listdir(args.image_path)]
    # 读取 generated_bubble_structs.csv 文件
    BubInfo = np.genfromtxt(args.BubInfo_path, delimiter=',', skip_header=1)
    print("标准流场生成所用数据集维度为:", BubInfo.shape)
    
    if args.channel == 'cylinder':
        TotVol = math.pi * args.Width * args.Width * args.Height / 4        # mm
    elif args.channel == 'rectangle':
        TotVol = args.Depth * args.Width * args.Height                      # mm
    Height_pix = round(args.Height / args.pixtomm)
    Width_pix = round(args.Width / args.pixtomm)

    for ImgNum in range(args.ImageGenNum):
        print(f'正在生成第{ImgNum + 1}/{args.ImageGenNum}张流场图像')
        #---------------------------------------------------------#
        #   气泡筛选
        #---------------------------------------------------------#
        timer.start()
        Bubs_Volume = 0
        BubList = []
        ImgLabel = []
        bub_conts_pad = []
        bub_conts = []
        sample_mat_count = 0
        cycles_count = 0
        mat_name = np.random.choice(mat_folders)
        million_bub = sio.loadmat(os.path.join(args.image_path, mat_name))
        million_bub = million_bub['generated_bubble_images'].transpose(2, 1, 0)
        print('|----------(1/2)开始筛选气泡!')
        
        # 创建气泡图像保存目录
        save_path = f'{args.save_path}/{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        while Bubs_Volume < args.gas_holdup * TotVol:
            if sample_mat_count > 10 or cycles_count > 250:
                mat_name = np.random.choice(mat_folders)
                million_bub = sio.loadmat(os.path.join(args.image_path, mat_name))
                million_bub = million_bub['generated_bubble_images'].transpose(2, 1, 0)
                sample_mat_count = 0
                cycles_count = 0

            random_int = np.random.randint(0, million_bub.shape[0])
            mat_name_int = int(os.path.basename(mat_name)[:-4])
            BubInfo_rand = BubInfo[mat_name_int - 250 + random_int]
            V_sd = np.random.rayleigh(args.V_sd)

            Bub_Volume = 128**3 * args.pixtomm**3 *(BubInfo_rand[0] * BubInfo_rand[1] * max(BubInfo_rand[0], BubInfo_rand[1]) * math.pi / 6)

            if abs(Bub_Volume - V_sd) < 0.001 * V_sd:
                c = math.sqrt(abs(BubInfo_rand[0]**2 - BubInfo_rand[1]**2))
                eccentricity = c / BubInfo_rand[0]
                e_sd = 1 - np.random.rayleigh(args.e_sd)
                if abs(eccentricity - e_sd) < 0.01 * e_sd:
                    angle_rad = np.random.normal(args.phi_e, args.phi_sd)
                    while angle_rad < -1 or angle_rad > 1:
                        angle_rad = np.random.normal(args.phi_e, args.phi_sd)
                    angle = round(90 - angle_rad * 180 / math.pi)
                    Bubs_Volume += Bub_Volume
                    
                    # 保存气泡图像
                    bubble_img = million_bub[random_int]
                    
                    # 添加标签，初始化为0（single）
                    ImgLabel.append(np.append(BubInfo_rand, [angle_rad, angle, Bub_Volume, eccentricity, 0, 0, 0]))
                    BubList.append(bubble_img)
                    sample_mat_count += 1
                    selection_process = 100 * Bubs_Volume / (args.gas_holdup * TotVol) if (100 * Bubs_Volume / (args.gas_holdup * TotVol)) < 100 else 100
                    print(f'|----------气泡筛选已进行{selection_process:.1f}%')
            cycles_count += 1
        print(f'|----------已成功筛选{len(BubList)}个气泡!')
        timer.end('气泡筛选')

        #---------------------------------------------------------#
        #   生成流场
        #---------------------------------------------------------#
        timer.start()
        print('|----------(2/2)开始生成流场!')
        ImgLabel_new = np.array(ImgLabel)
        BubImg_new = np.array(BubList)
        BubInfo_index = np.argsort(ImgLabel_new[:, 7])      # 按边缘锐利度排序
        ImgLabel_new = ImgLabel_new[BubInfo_index]
        # ImgLabel_new[:, -3:] = ImgLabel_new[:, -3:] - args.pad
        BubImg_new = BubImg_new[BubInfo_index]
        
        background = np.zeros((Height_pix + args.pad * 2, Width_pix + args.pad * 2, 3), np.uint8)
        if args.visual_enhance_gauss:
            background_global_gauss = np.zeros((Height_pix + args.pad * 2, Width_pix + args.pad * 2, 3), np.uint8)
            background_local_gauss = np.zeros((Height_pix + args.pad * 2, Width_pix + args.pad * 2, 3), np.uint8)
        if args.visual_enhance_median:
            background_global_median = np.zeros((Height_pix + args.pad * 2, Width_pix + args.pad * 2, 3), np.uint8)
            background_local_median = np.zeros((Height_pix + args.pad * 2, Width_pix + args.pad * 2, 3), np.uint8)

        # 合并标签生成、掩码生成、流场合成、bub_conts和可视化
        bubble_labels = []
        bubble_masks = []
        bub_conts = []
        bub_conts_pad = []
        label_vis_img = background_local_median.copy()
        i = 0
        x_e = args.x_e * args.Width / args.pixtomm
        for img, label in tzip(BubImg_new, ImgLabel_new):
            # 1. 位置采样
            if args.Location_dis == 'gaussian':
                x = round(np.random.normal(x_e + args.pad, args.x_sd))
                while x < args.x_pad + args.pad or x > Width_pix - args.x_pad + args.pad:
                    x = round(np.random.normal(x_e + args.pad, args.x_sd))
            elif args.Location_dis == 'uniform':
                x = np.random.randint(args.x_pad, Width_pix - args.x_pad) + args.pad
            y = np.random.randint(0, Height_pix) + args.pad
            ImgLabel_new[i, -3:-1] = x - args.pad, y - args.pad
            img = cv2.bitwise_not(img).reshape(128, 128, 1)
            roi = cut_image_out_of_range(background, x, y)
            if args.visual_enhance_gauss:
                roi_global_gauss = cut_image_out_of_range(background_global_gauss, x, y)
                roi_local_gauss = cut_image_out_of_range(background_local_gauss, x, y)
            if args.visual_enhance_median:
                roi_global_median = cut_image_out_of_range(background_global_median, x, y)
                roi_local_median = cut_image_out_of_range(background_local_median, x, y)

            img = img.repeat(3, -1)
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), label[9], 1)
            M_mask = cv2.getRotationMatrix2D((cols/2, rows/2), label[9], 1)
            cropped_img = cv2.warpAffine(img, M, (cols, rows), borderValue=(0, 0, 0))
            img_mask = cv2.warpAffine(img, M_mask, (cols, rows), borderValue=(0, 0, 0))

            mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(mask, args.threshold, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            if len(contours) >= 1:
                second_contour = contours[0]
                mask_with_contour = np.zeros_like(mask)
                cv2.drawContours(mask_with_contour, [second_contour], -1, (255), thickness=cv2.FILLED)
            else:
                continue

            # 1. 生成掩码
            global_mask = np.zeros((Height_pix + args.pad * 2, Width_pix + args.pad * 2), dtype=np.uint8)
            x0, y0 = max(0, x - 64), max(0, y - 64)
            x1, y1 = min(Width_pix + args.pad * 2, x + 64), min(Height_pix + args.pad * 2, y + 64)
            mask_x0, mask_y0 = max(0, 64 - x), max(0, 64 - y)
            mask_x1, mask_y1 = min(128, 128 - (x + 64 - (Width_pix + args.pad * 2))), min(128, 128 - (y + 64 - (Height_pix + args.pad * 2)))
            
            if x1 > x0 and y1 > y0 and mask_x1 > mask_x0 and mask_y1 > mask_y0:
                # 使用mask_with_contour替代mask来确保内部填充为白色
                global_mask[y0:y1, x0:x1] = mask_with_contour[mask_y0:mask_y1, mask_x0:mask_x1]
                
                # 保存完整掩码
                if args.save_complete_masks:
                    if not os.path.exists(os.path.join(save_path, 'complete_masks')):
                        os.makedirs(os.path.join(save_path, 'complete_masks'))
                    complete_mask = global_mask[args.pad : Height_pix + args.pad, args.pad : Width_pix + args.pad]
                    cv2.imwrite(os.path.join(save_path, f'complete_masks/{str(i).zfill(3)}.png'), complete_mask)
            else:
                continue

            # 2. 计算标签
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

            # 3. 合成流场、bub_conts等
            origin_contour = second_contour[:, 0, :].copy()
            bub_conts_pad.append(second_contour)
            second_contour[:, 0, 0] += x - args.pad * 1 - 64
            second_contour[:, 0, 1] += y - args.pad * 1 - 64
            bub_conts.append(second_contour)

            mask_inv = cv2.bitwise_not(mask_with_contour)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            if args.visual_enhance_gauss:
                img_bg_global_gauss = cv2.bitwise_and(roi_global_gauss, roi_global_gauss, mask=mask_inv)
                img_bg_local_gauss = cv2.bitwise_and(roi_local_gauss, roi_local_gauss, mask=mask_inv)
            if args.visual_enhance_median:
                img_bg_global_median = cv2.bitwise_and(roi_global_median, roi_global_median, mask=mask_inv)
                img_bg_local_median = cv2.bitwise_and(roi_local_median, roi_local_median, mask=mask_inv)

            img_fg = cv2.bitwise_and(cropped_img, cropped_img, mask=mask_with_contour)
            kernel_gauss = 3
            dst = cv2.add(img_bg, img_fg)
            if args.visual_enhance_gauss:
                dst_global_gauss = cv2.add(img_bg_global_gauss, img_fg)
                dst_global_gauss = cv2.GaussianBlur(dst_global_gauss, (kernel_gauss, kernel_gauss), 0)
                dst_local_gauss = cv2.add(img_bg_local_gauss, img_fg)

            if args.visual_enhance_median:
                dst_global_median = cv2.add(img_bg_global_median, img_fg)
                dst_global_median = cv2.medianBlur(dst_global_median, kernel_gauss)
                dst_local_median = cv2.add(img_bg_local_median, img_fg)

            semi = 5
            for point in origin_contour:
                if args.visual_enhance_gauss:
                    mini_dst_local_gauss = dst_local_gauss[point[1]-semi:point[1]+semi, point[0]-semi:point[0]+semi,:]
                    mini_dst_local_gauss = cv2.GaussianBlur(mini_dst_local_gauss, (kernel_gauss, kernel_gauss), 0)
                    dst_local_gauss[point[1]-semi:point[1]+semi, point[0]-semi:point[0]+semi] = mini_dst_local_gauss
                if args.visual_enhance_median:
                    mini_dst_local_median = dst_local_median[point[1]-semi:point[1]+semi, point[0]-semi:point[0]+semi,:]
                    mini_dst_local_median = cv2.medianBlur(mini_dst_local_median, kernel_gauss)
                    dst_local_median[point[1]-semi:point[1]+semi, point[0]-semi:point[0]+semi] = mini_dst_local_median

            background[y - 64 : y + 64, x - 64 : x + 64] = dst
            if args.visual_enhance_gauss:
                background_global_gauss[y - 64 : y + 64, x - 64 : x + 64] = dst_global_gauss
                background_local_gauss[y - 64 : y + 64, x - 64 : x + 64] = dst_local_gauss
            if args.visual_enhance_median:
                background_global_median[y - 64 : y + 64, x - 64 : x + 64] = dst_global_median
                background_local_median[y - 64 : y + 64, x - 64 : x + 64] = dst_local_median
            
            i += 1  # 只在所有处理都成功后才自增i

        # 最终保存bubble_labels到csv
        ImgLabel_new = np.hstack([ImgLabel_new[:, :-1], np.array(bubble_labels).reshape(-1, 1)])

        background = cv2.bitwise_not(background)
        background = background[args.pad : Height_pix + args.pad, args.pad : Width_pix + args.pad]
        if args.visual_enhance_gauss:
            background_global_gauss = cv2.bitwise_not(background_global_gauss)
            background_global_gauss = background_global_gauss[args.pad : Height_pix + args.pad, args.pad : Width_pix + args.pad]
            background_local_gauss = cv2.bitwise_not(background_local_gauss)
            background_local_gauss = background_local_gauss[args.pad : Height_pix + args.pad, args.pad : Width_pix + args.pad]
        if args.visual_enhance_median:
            background_global_median = cv2.bitwise_not(background_global_median)
            background_global_median = background_global_median[args.pad : Height_pix + args.pad, args.pad : Width_pix + args.pad]
            background_local_median = cv2.bitwise_not(background_local_median)
            background_local_median = background_local_median[args.pad : Height_pix + args.pad, args.pad : Width_pix + args.pad]

        # 在最终图像上绘制边界框和标签
        # 创建新的图像用于保存标签信息
        label_vis_img = background_local_median.copy()

        # 可视化气泡簇标签
        for i, contour in enumerate(bub_conts):
            x, y, w, h = cv2.boundingRect(contour)
            label = bubble_labels[i]  # 用正确的标签索引
            if label == 0:
                color = (53, 130, 84)  # 绿色
            else:
                intensity = min(255, 80 + label * 30)
                color = (0, 0, intensity)
            cv2.rectangle(label_vis_img, (x, y), (x + w, y + h), color, 2)
            label_text = f"{label}"
            cv2.putText(label_vis_img, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 4)
            cv2.putText(label_vis_img, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if args.isOverwrite or (not os.path.exists(os.path.join(save_path, 'Image.jpg'))):
            cv2.imwrite(os.path.join(save_path, 'Image.jpg'), background)
            if args.visual_enhance_gauss:
                cv2.imwrite(os.path.join(save_path, 'Image_global_gauss.jpg'), background_global_gauss)
                cv2.imwrite(os.path.join(save_path, 'Image_local_gauss.jpg'), background_local_gauss)
            if args.visual_enhance_median:
                cv2.imwrite(os.path.join(save_path, 'Image_global_median.jpg'), background_global_median)
                cv2.imwrite(os.path.join(save_path, 'Image_local_median.jpg'), background_local_median)
            
            # 保存标签可视化图像
            cv2.imwrite(os.path.join(save_path, 'Label_Visualization.jpg'), label_vis_img)
            
            # 保存原始单气泡图像
            if args.save_original_bubbles:
                original_bubble_path = os.path.join(save_path, 'original_bubble_images')
                if not os.path.exists(original_bubble_path):
                    os.makedirs(original_bubble_path)
                
                for i, img in enumerate(BubImg_new):
                    cv2.imwrite(os.path.join(original_bubble_path, f'bubble_{i:03d}.png'), img)

            # 保存旋转后的单气泡图像
            if args.save_single_bubbles:
                single_bubble_path = os.path.join(save_path, 'single_bubble_images')
                if not os.path.exists(single_bubble_path):
                    os.makedirs(single_bubble_path)
                
                for i, (img, label) in enumerate(zip(BubImg_new, ImgLabel_new)):
                    # 旋转图像
                    img = cv2.bitwise_not(img).reshape(128, 128, 1)
                    img = img.repeat(3, -1)
                    rows, cols, _ = img.shape
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), label[9], 1)
                    rotated_img = cv2.warpAffine(img, M, (cols, rows), borderValue=(0, 0, 0))
                    
                    # 反转颜色回正常
                    rotated_img = cv2.bitwise_not(rotated_img)
                    
                    # 保存旋转后的图像
                    cv2.imwrite(os.path.join(single_bubble_path, f'bubble_{i:03d}.png'), rotated_img)

            # 保存带有表头的标签文件，使用指定格式
            header = "a,b,cx,cy,C,S,SR,EG,angle_rad,angle,V,e,x,y,label"
            # 设置不同列的格式
            formats = ['%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', 
                      '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%d']
            np.savetxt(os.path.join(save_path, 'structure_info.csv'), ImgLabel_new, 
                      delimiter=',', header=header, comments='', fmt=formats)
            
            with open(os.path.join(save_path, 'bub_conts.txt'), 'w') as file:
                for bub_cont in bub_conts:
                    file.write(','.join(map(str, bub_cont.reshape(-1))) + '\n')
            
            # 获取图像尺寸
            img_height, img_width = background.shape[:2]
            
            # 生成标签文件
            label_file = os.path.join(save_path, 'labels.txt')
            with open(label_file, 'w') as f:
                for i, (contour, label) in enumerate(zip(bub_conts, bubble_labels)):
                    # 计算边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 归一化坐标
                    x_center = (x + w/2) / img_width
                    y_center = (y + h/2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    # 写入YOLO格式：class_id x_center y_center width height
                    f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # a|b|cx|cy|C|S|SR|EG|angle_rad|angle| V| e| x| y
        # 0|1| 2| 3|4|5| 6| 7|        8|    9|10|11|12|13
        # ---------------------------------------------------------#
        #   生成流场信息图：旋转角度
        # ---------------------------------------------------------#
        info_list = [(8, r'Rotation angle $\varphi$ [rad]', plt.cm.cividis, 'Angle_Info.jpg', '(b)Rotation angle Info'),
                        (10, r'Bubble volume $V$ $[mm^3]$', plt.cm.coolwarm, 'V_Info.jpg', '(c)Bubble volume Info'),
                        (11, r'Eccentricity $e$ [-]', plt.cm.PuOr, 'e_Info.jpg', '(d)Eccentricity Info'),
                        (4, r'Circularity $\Psi$ [-]', plt.cm.RdGy, 'C_Info.jpg', '(e)Circularity Info'),
                        (5, r'Ellipticity $\varepsilon$ [-]', plt.cm.RdBu, 'EE_Info.jpg', '(f)Ellipticity Info'),
                        (6, r'Shadow Ratio $SR$ [-]', plt.cm.YlGnBu, 'SR_Info.jpg', '(g)Shadow Ratio Info'),
                        (7, r'Edge Gradient $EG$ [-]', plt.cm.Spectral, 'EG_Info.jpg', '(h)Edge Gradient Info')]
        
        for info in info_list:
            for idx, contour in enumerate(bub_conts):
                label_value = ImgLabel_new[idx, info[0]]
                color = (label_value - np.min(ImgLabel_new[:, info[0]])) / (np.max(ImgLabel_new[:, info[0]]) - np.min(ImgLabel_new[:, info[0]]))
                color = info[2](color)
                plt.fill(contour[:, 0, 0], Height_pix - contour[:, 0, 1], facecolor=color, edgecolor='k', linewidth=0.5)
                plt.scatter(np.mean(contour[:, 0, 0]), np.mean(Height_pix - contour[:, 0, 1]), color='red', s=0.5)
            norm = matplotlib.colors.Normalize(vmin=np.min(ImgLabel_new[:, info[0]]), vmax=np.max(ImgLabel_new[:, info[0]]))
            ax = plt.gca()
            cb = plt.colorbar(plt.cm.ScalarMappable(cmap=info[2], norm=norm), ax=ax)
            ax.set_aspect(1)
            ax.set_xlabel(f'Total number of bubbles:{i}', fontsize = 9)
            cb.set_label(info[1])
            plt.title(info[4])
            plt.xlim(0, Width_pix)
            plt.ylim(0, Height_pix) 
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f'{save_path}/{info[3]}', dpi=600, bbox_inches='tight')
            plt.close()

        #---------------------------------------------------------#
        #   生成流场信息图：气泡尺寸分布
        #---------------------------------------------------------#
        plt.figure()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.hist(ImgLabel_new[:, 10], bins=20, density=False, color='lightgreen', edgecolor='black', alpha=1, label=r'$BSD$')
        # x = np.linspace(0, 1.1 * np.max(ImgLabel_new[:, 10]), 100)
        x = np.linspace(0, 125 if 1.1 * np.max(ImgLabel_new[:, 10]) < 125 else 1.1 * np.max(ImgLabel_new[:, 10]), 100)      # 自适应范围
        y = (x / args.V_sd**2) * np.exp(-x**2 / (2 * args.V_sd**2))
        ax2.plot(x, y, 'k:', label=r'Expected $BSD$')
        # ax1.legend(['BSD'], loc='upper left')
        # ax2.legend(['Expected BSD'], loc='upper right')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2)

        ax1.set_title(r'Bubble Size Distribution $BSD$')
        ax1.set_xlabel(r'Bubble volume $[mm^3]$')
        ax1.set_ylabel('Number of bubbles')
        ax2.yaxis.set_major_locator(plt.NullLocator())
        # ax2.set_ylabel('Expected BSD')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.savefig(f'{save_path}/BSD_Info.png', dpi=600)
        plt.close()

        #---------------------------------------------------------#
        #   生成流场信息图：气泡离心率分布
        #---------------------------------------------------------#
        plt.figure()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.hist(ImgLabel_new[:, 11], bins=20, density=False, color='Peru', edgecolor='black', alpha=1, label=r'$BED$')
        x = np.linspace(-0.05, 1.05, 100)
        y = (x / args.e_sd**2) * np.exp(-x**2 / (2 * args.e_sd**2))
        ax2.plot(1 - x, y, 'k:', label=r'Expected $BED$')
        # ax1.legend(['BSD'], loc='upper left')
        # ax2.legend(['Expected BSD'], loc='upper right')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2)

        ax1.set_title(r'Bubble Eccentricity Distribution $BED$')
        ax1.set_xlabel(r'Bubble eccentricity $[-]$')
        ax1.set_ylabel('Number of bubbles')
        ax2.yaxis.set_major_locator(plt.NullLocator())
        # ax2.set_ylabel('Expected BED')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.savefig(f'{save_path}/BED_Info.png', dpi=600)
        plt.close()
        
        #---------------------------------------------------------#
        #   生成流场信息图：气泡水平位置分布
        #---------------------------------------------------------#
        plt.figure()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.hist(ImgLabel_new[:, 12], bins=20, density=False, color='skyblue', edgecolor='black', alpha=1, label=r'$BHPD$')
        x = np.linspace(0, Height_pix, 100)
        if args.Location_dis == 'gaussian':
            y = (1 / np.sqrt(2 * math.pi) * args.x_sd) * np.exp(-(x - x_e)**2 / (2 * args.x_sd**2))
        elif args.Location_dis == 'uniform':
            y = np.full_like(x, np.mean(ImgLabel_new[:, 12]))
        
        ax2.plot(x, y, 'k:', label=r'Expected $BHPD$')
        # ax1.legend(['BHPD'], loc='upper left')
        # ax2.legend(['Expected BHPD'], loc='upper right')
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2)
        ax1.set_title(r'Bubble Horizontal Position Distribution $BHPD$')
        ax1.set_xlabel(r'Horizontal pixel $[pix]$')
        ax1.set_ylabel('Number of bubbles')
        ax2.yaxis.set_major_locator(plt.NullLocator())
        # ax2.set_ylabel('Expected BHPD')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.savefig(f'{save_path}/BHPD_Info.png', dpi=600)
        plt.close()

        #---------------------------------------------------------#
        #   生成流场信息图：气泡旋转角度分布
        #---------------------------------------------------------#
        plt.figure()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.hist(ImgLabel_new[:, 8], bins=20, density=False, color='lightpink', edgecolor='black', alpha=1, label=r'$BRAD$')
        x = np.linspace(-1.05, 1.05, 100)
        y = (1 / (np.sqrt(2 * math.pi) * args.phi_sd)) * np.exp(-(x - args.phi_e)**2 / (2 * args.phi_sd**2))
        ax2.plot(x, y, 'k:', label=r'Expected $BRAD$')
        # ax1.legend(['BRAD'], loc='upper left')
        # ax2.legend(['Expected BRAD'], loc='upper right')
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2)
        ax1.set_title(r'Bubble Rotation Angle Distribution $BRAD$')
        ax1.set_xlabel(r'Rotation angle $[rad]$')
        ax1.set_ylabel('Number of bubbles')
        ax2.yaxis.set_major_locator(plt.NullLocator())
        # ax2.set_ylabel('Expected BRAD')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.savefig(f'{save_path}/BRAD_Info.png', dpi=600)
        plt.close()

        timer.end('流场生成')
        
        #---------------------------------------------------------#
        #   生成统计信息图
        #---------------------------------------------------------#
        timer.start()
        # ... 统计信息图生成代码 ...
        timer.end('统计信息图生成')
        
        #---------------------------------------------------------#
        #   生成分割数据集
        #---------------------------------------------------------#
        if args.save_overlap_masks:
            timer.start()
            if not os.path.exists(os.path.join(save_path, 'overlap_masks')):
                os.makedirs(os.path.join(save_path, 'overlap_masks'))
            bounding_boxes = []
            # TODO
            #     1.循环边界框()，并绘制。从当前气泡绘制白色填充，从当前气泡开始往后，一直绘制黑色，作为ground truth masks；
            #     2.寻找当前气泡的轮廓的外接矩形，作为bounding boxes；
            #     3.按顺序保存ground truth masks，写入bounding boxes为txt格式。
            for xx in range(len(bub_conts)):
                bub_conts[xx][:, 0, 0] += args.pad
                bub_conts[xx][:, 0, 1] += args.pad
                
            segment_background_merge = np.zeros((Height_pix + args.pad * 2, Width_pix + args.pad * 2, 1), np.uint8)
            for cont_num, (bub_cont, bub_cont_pad) in enumerate(zip(bub_conts, bub_conts_pad)):
                segment_background = np.zeros((Height_pix + args.pad * 2, Width_pix + args.pad * 2, 1), np.uint8)
                x, y, w, h = cv2.boundingRect(bub_cont)
                bounding_boxe = np.array([x, y, x + w, y + h])

                cv2.drawContours(segment_background, [bub_cont], -1, (255), thickness=cv2.FILLED)
                for cover_i in range(cont_num+1, len(bub_conts)):
                    cv2.drawContours(segment_background, [bub_conts[cover_i]], -1, (0), thickness=cv2.FILLED)
                segment_background = segment_background[args.pad : Height_pix + args.pad, args.pad : Width_pix + args.pad]
                contours, _ = cv2.findContours(segment_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) == 1 and len(contours[0]) > 15:
                    cv2.imwrite(os.path.join(save_path, f'overlap_masks/{str(cont_num).zfill(3)}.png'), segment_background)
                    cv2.drawContours(segment_background_merge, [bub_cont], -1, (255), thickness=cv2.FILLED)
                    cv2.drawContours(segment_background_merge, [bub_cont], -1, (0), thickness=1)
                    bounding_boxes.append(bounding_boxe)
            
            segment_background_merge = segment_background_merge[args.pad : Height_pix + args.pad, args.pad : Width_pix + args.pad]
            cv2.imwrite(os.path.join(save_path, f'overlap_masks/segment_merge.png'), segment_background_merge)
            with open(os.path.join(save_path, 'overlap_masks/bounding_boxes.txt'), 'w') as file:
                for bounding_boxe in bounding_boxes:
                    file.write(','.join(map(str, bounding_boxe.reshape(-1))) + '\n')
            timer.end('分割数据集生成')
        
        # 保存配置信息
        argsDict = args.__dict__
        with open(f'{save_path}/config.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            f.writelines('Generation time: ' + str(datetime.datetime.now()) + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
    
    # 打印时间统计
    timer.print_times()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate_flow_field")
    # parser.add_argument("--Depth", type=float, default=100/2, help="深度(mm)")
    # parser.add_argument("--Width", type=float, default=100/2, help="宽度(mm)")
    # parser.add_argument("--Height", type=float, default=100/2, help="高度(mm)")
    parser.add_argument("--channel", type=str, default='cylinder', help="通道的类型,可选rectangle和cylinder,默认高斯分布")
    parser.add_argument("--Depth", type=float, default=100, help="深度(mm),仅当channel为rectangle时有效")
    parser.add_argument("--Width", type=float, default=100, help="宽度(mm),通道为rectangle是代表宽度,cylinder时代表直径")
    parser.add_argument("--Height", type=float, default=64, help="高度(mm)")
    parser.add_argument("--pixtomm", type=float, default=0.080128, help="像素尺寸换算")
    parser.add_argument("--phi_e", type=float, default=-0.0, help="角度分布的期望,服从正态分布")
    parser.add_argument("--phi_sd", type=float, default=0.29769662022590637, help="角度分布的标准差")
    parser.add_argument("--Location_dis", type=str, default='gaussian', help="水平位置的分布方式,可选uniform与gaussian,默认高斯分布")
    parser.add_argument("--x_e", type=float, default=0.5, help="水平位置分布的期望,取值(0,1),服从正态分布,仅当Location_dis=gaussian有效")
    # parser.add_argument("--x_e", type=float, default=301.954585, help="水平位置分布的期望,服从正态分布,仅当Location_dis=gaussian有效")
    parser.add_argument("--x_sd", type=float, default=175, help="水平位置分布的标准差,服从正态分布,仅当Location_dis=gaussian有效")
    parser.add_argument("--x_pad", type=float, default=24, help="左右两侧的留白像素宽度。")
    parser.add_argument("--V_dis", type=str, default='Rayleigh', help="气泡体积分布方式,可选Rayleigh与gaussian,默认瑞利分布")
    parser.add_argument("--V_sd", type=float, default=50, help="气泡体积分布的标准差,仅当V_dis=Rayleigh有效,气泡尺寸单位mm^3")
    parser.add_argument("--V_e", type=float, default=10, help="气泡体积分布的均值,仅当V_dis=gaussian有效")
    parser.add_argument("--e_sd", type=float, default=0.27194802937871304, help="1-离心率(1-e)分布的标准差,服从瑞利分布")
    parser.add_argument("--gas_holdup", type=float, default=0.004, help="固定气含率")
    parser.add_argument("--pad", type=int, default=128, help="生成背景的四周填充")
    parser.add_argument("--bckgrdint", type=int, default=248, help="背景色")
    parser.add_argument("--threshold", type=int, default=50, help="边缘检测阈值")
    parser.add_argument("--visual_enhance_gauss", type=lambda x: x.lower() == 'true', default=False, help="拼接时选择高斯滤波器进行视觉增强")
    parser.add_argument("--visual_enhance_median", type=lambda x: x.lower() == 'true', default=True, help="拼接时选择中值滤波器进行视觉增强")
    parser.add_argument("--ImageGenNum", type=int, default=100, help="生成流场数量")
    parser.add_argument("--isOverwrite", help="是否覆盖", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--image_path",
                        type=str,
                        # default='D:/fooling_around/BubGAN/bubble_dataset_tinyab/generated_bubble_tinyab.mat',
                        # default='D:/DataSet_DOOR/BubStyle/generated_bubble_images',
                        # default='C:/DataSet_DOOR/dataset_BubStyle/generated_bubble_images',
                        # default='/Users/mac/Desktop/DynamicFlow/dataset_BubStyle/generated_bubble_images',
                        default='/home/yubd/mount/dataset/dataset_BubStyle/generated_bubble_images',
                        help="图像路径",
                    )
    parser.add_argument("--BubInfo_path",
                        type=str, 
                        # default='D:/fooling_around/BubGAN/bubble_dataset_tinyab/generated_bubble_structs.csv',
                        # default='D:/DataSet_DOOR/BubStyle/generated_bubble_structs.csv',
                        # default='C:/DataSet_DOOR/dataset_BubStyle/generated_bubble_structs.csv',
                        # default='/Users/mac/Desktop/DynamicFlow/dataset_BubStyle/generated_bubble_structs.csv',
                        default='/home/yubd/mount/dataset/dataset_BubStyle/generated_bubble_structs.csv',
                        help="图像信息路径",
                    )
    parser.add_argument("--save_path",
                        type=str,
                        # default='C:/Users/glanc/Desktop/Bubble_flow_field_reconstruction/generate_flow_field',
                        # default='D:\DataSet_DOOR\dataset_SAM_finetuning',
                        # default='D:\DataSet_DOOR\dataset_BubStyle',
                        # default='C:/codebase/Bubble_flow_field_reconstruction/generate_flow_field',
                        # default='C:/Users/Administrator/Desktop/DynamicFlow/generate_flow_field',
                        # default='/Users/mac/Desktop/DynamicFlow/generate_flow_field',
                        # default='C:/DataSet_DOOR/dataset_overlap_yolo/origin_synthetic/gas_holdup_0.004',
                        default='C:/DataSet_DOOR/dataset_overlap_yolo/origin_synthetic/gas_holdup_0.004',
                        help="流场保存路径",
                    )
    parser.add_argument("--save_original_bubbles", type=lambda x: x.lower() == 'False', default=True, help="是否保存原始单气泡图像")
    parser.add_argument("--save_single_bubbles", type=lambda x: x.lower() == 'True', default=True, help="是否保存旋转后的单气泡图像")
    parser.add_argument("--save_overlap_masks", type=lambda x: x.lower() == 'True', default=True, help="是否保存包含遮挡关系的mask")
    parser.add_argument("--save_complete_masks", type=lambda x: x.lower() == 'True', default=True, help="是否保存气泡的完整mask")


    args = parser.parse_args()
    print(args)
    generate_flow_field(args)