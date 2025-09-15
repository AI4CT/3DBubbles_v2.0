import cv2
import numpy as np
import time
import math
import os
# import pandas as pd
threshold_bubgan = 120
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def normalization(data):
    # 图像归一化处理
    range = np.max(data) - np.min(data)
    return (data - np.min(data)) / range


class Label_Bubble():
    def __init__(self):
        pass

    def ellipse_fitting(self, img):
        self.img = img.copy()
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, self.img_bin = cv2.threshold(self.img_gray, 180, 255, cv2.THRESH_BINARY)
        # cv_show('self.img_bin', self.img_bin)

        contours, _ = cv2.findContours(self.img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = sorted(contours, key=cv2.contourArea, reverse=True)

        self.scales = []
        self.fftCnts = []
        for i, cnt in enumerate(self.contours[1:]):
            cntPoints = np.squeeze(cnt)  # 删除维度为 1 的数组维度，(2867, 1, 2)->(2867,2)
            if len(cntPoints.shape) == 1:
                continue
            self.scales.append(cntPoints.max())    # 尺度系数
            cntComplex = np.empty(cntPoints.shape[0], dtype = complex)  # 声明复数数组 (2867,)
            cntComplex = cntPoints[:, 0] + 1j * cntPoints[:, 1]  # (xk,yk)->xk+j*yk
            # print("cntComplex", cntComplex.shape)
            self.fftCnts.append(np.fft.fft(cntComplex))  # 离散傅里叶变换，生成傅里叶描述子
            
        # 对该轮廓进行操作
        # draw_img = cv2.drawContours(self.img, self.contours, 1, (255, 0, 255), 2)
        # cv_show('draw_img', draw_img)
        self.Area = cv2.contourArea(self.contours[1])
        self.Perimeter = cv2.arcLength(self.contours[1], True)
        ellipse = cv2.fitEllipse(self.contours[1])

        self.a, self.b = sorted(ellipse[1], reverse=True)
        # self.angle = ellipse[2]

        return self.a, self.b

    def centroid(self):
        gray_image = 255 - self.img_gray.copy()
        threshold = 20
        gray_sum = np.sum(gray_image[gray_image > threshold])
        x_coords = np.where(gray_image > threshold)[1]
        y_coords = np.where(gray_image > threshold)[0]
        weighted_sum_x = np.sum(x_coords * gray_image[gray_image > threshold])
        weighted_sum_y = np.sum(y_coords * gray_image[gray_image > threshold])
        # print(f"重心坐标: ({weighted_sum_x / gray_sum}, {weighted_sum_y / gray_sum})")
        # cX = int(weighted_sum_x / gray_sum)
        # cY = int(weighted_sum_y / gray_sum)
        cX = weighted_sum_x / gray_sum
        cY = weighted_sum_y / gray_sum
        return cX, cY
    
    def circularity(self):
        Circularity = 4 * math.pi * self.Area / self.Perimeter ** 2
        return Circularity

    def solidity(self):
        # 计算拟合椭圆的面积
        ellipse_area = math.pi * (self.a / 2) * (self.b / 2)
        # 计算重合率,以拟合出的标准椭圆面积进行比较
        Solidity = 1 - np.abs((self.Area + 1e-6) / (ellipse_area + 1e-6) - 1)
        return Solidity

    def shade_radio(self):
        '''
        input:
            img_rgb:生成和图像同样尺寸的空白图，填充
            contours:图像轮廓
        output:
            shade_radio:阴影比例值
        '''
        p = np.zeros(shape = self.img_gray.shape)
        # contourIdx=-1时，会将轮廓中的每个点是为单独的轮廓，因此需要包装为一个列表contours→[contours]
        # https://blog.csdn.net/csqqscCSQQSC/article/details/128225315?ops_request_misc=&request_id=&biz_id=102&utm_term=opencv%E8%8E%B7%E5%8F%96%E8%BD%AE%E5%BB%93%E5%86%85%E9%83%A8%E6%89%80%E6%9C%89%E7%82%B9&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-128225315.142^v71^one_line,201^v4^add_ask&spm=1018.2226.3001.4187
        cv2.drawContours(p, [self.contours[1]], -1, 255, -1)
        # cv_show('p', p)
        pixel = np.zeros(shape = (np.where(p == 255)[0].size, 1))
        xx = np.where(p == 255)[0].reshape(-1, 1)
        yy = np.where(p == 255)[1].reshape(-1, 1)
        for i, (x, y) in enumerate(zip(xx, yy)):
            pixel[i] = self.img_gray[x, y]
        shade_radio = np.mean(pixel) / 255
        # coordinate = np.concatenate([x, y], axis = 1).tolist()
        # # 生成一个list[tuple]类型的轮廓内部所有点
        # inside = [tuple(x) for x in coordinate]
        return 1 - shade_radio
    
    def edge_grad(self):
        img_gray = self.img_gray.copy()
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.convertScaleAbs(sobely)
        sobelx = abs(sobelx) / 255
        sobely = abs(sobely) / 255
        sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        contxy = self.contours[1][:, 0, :]

        EG = []
        for ii in range(contxy.shape[0]):
            yy = contxy[ii, 0]
            xx = contxy[ii, 1]
            EG.append(sobelxy[xx, yy])
        EG = np.mean(EG)
        return EG
    
    def truncFFT(self, fftCnt, pLowF = 64):  # 截短傅里叶描述子
        fftShift = np.fft.fftshift(fftCnt)  # 中心化，将低频分量移动到频域中心
        center = int(len(fftShift)/2)
        low, high = center - int(pLowF/2), center + int(pLowF/2)
        fftshiftLow = fftShift[low:high]
        fftLow = np.fft.ifftshift(fftshiftLow)  # 逆中心化
        return fftLow

    def reconstruct(self, ratio = 1.0, fill = None):  # 由傅里叶描述子重建轮廓图
        fftCnts = self.fftCnts[0]
        scales = self.scales[0]
        rebuild = np.ones(self.img.shape, np.uint8) * 255  # 创建空白图像
        if isinstance(fftCnts, list):   # 如果传入的是一个列表的轮廓
            for i, (fftCnt, scale) in enumerate(zip(fftCnts, scales)):
                # pLowF = max(int(fftCnt.shape[0] * ratio), 2)  # 截短长度 P<=K
                pLowF = 32
                fftLow = self.truncFFT(fftCnt, pLowF)  # 截短傅里叶描述子，删除高频系数
                ifft = np.fft.ifft(fftLow)  # 傅里叶逆变换 (P,)
                # cntRebuild = np.array([ifft.real, ifft.imag])  # 复数转为数组 (2, P)
                # cntRebuild = np.transpose(cntRebuild)  # (P, 2)
                cntRebuild = np.stack((ifft.real, ifft.imag), axis=-1)  # 复数转为数组 (P, 2)
                if cntRebuild.min() < 0:
                    cntRebuild -= cntRebuild.min()
                cntRebuild *= scale / cntRebuild.max()
                # cntRebuild = cntRebuild.astype(np.int32)
                # print("ratio={}, fftCNT:{}, fftLow:{}".format(ratio, fftCnt.shape, fftLow.shape))
                # if fill is not None:        # 判断是否要画填充
                #     if (fill[cntRebuild[0, 1], cntRebuild[0, 0], 0] == 0):  # 判断要画的轮廓是否在最大轮廓的内部
                #         if i == 0:         # 判断是最大轮廓（填充0）还是其余轮廓（填充255）
                #             cv2.fillPoly(rebuild, [cntRebuild], 0)
                #         else:
                #             cv2.fillPoly(rebuild, [cntRebuild], (255, 255, 255))
                #     else:
                #         print(f'第{i}个没画上!，一共{len(scales)}个')
                # else:
                #     cv2.polylines(rebuild, [cntRebuild], True, 0, thickness=1)  # 绘制多边形，闭合曲线
        else:
            fftCnt, scale = fftCnts, scales
            # pLowF = int(fftCnt.shape[0] * ratio)  # 截短长度 P<=K
            pLowF = 32
            fftLow = self.truncFFT(fftCnt, pLowF)  # 截短傅里叶描述子，删除高频系数
            ifft = np.fft.ifft(fftLow)  # 傅里叶逆变换 (P,)
            # cntRebuild = np.array([ifft.real, ifft.imag])  # 复数转为数组 (2, P)
            # cntRebuild = np.transpose(cntRebuild)  # (P, 2)
            cntRebuild = np.stack((ifft.real, ifft.imag), axis = -1)  # 复数转为数组 (P, 2)
            if cntRebuild.min() < 0:
                cntRebuild -= cntRebuild.min()
            cntRebuild *= scale / cntRebuild.max()
            # cntRebuild = cntRebuild.astype(np.int32)
            # print("ratio={}, fftCNT:{}, fftLow:{}".format(ratio, fftCnt.shape, fftLow.shape))
            # if fill is not None:
            #     cv2.fillPoly(rebuild, [cntRebuild], 0)  # 填充多边形
            # else:
            #     cv2.polylines(rebuild, [cntRebuild], True, 0, thickness = 1)  # 绘制多边形，闭合曲线
        return rebuild, cntRebuild / 128

label_bubble = Label_Bubble()

if __name__ == "__main__":
    i = 0
    path = r'D:/DataSet_DOOR/Characterization/bubble_dataset_nophi/'      # 输入文件夹地址
    # path = r'D:/DataSet_DOOR/Cosine_similarity_threshold/C0.03/train/'      # 输入文件夹地址
    files = os.listdir(path)   # 读入文件夹
    num_png = len(files)       # 统计文件夹中的文件个数

    with open('D:/DataSet_DOOR/Characterization/bub_label_nophi.txt', 'w') as f, open('D:/DataSet_DOOR/Characterization/bub_label_bubgan.txt', 'w') as f_bubgan:
    # with open('D:/DataSet_DOOR/Characterization/bub_label_nophi_F.txt', 'w') as f:
        for file in files:
            img_rgb = cv2.imread(path + file)
            # cv_show('img_rgb', img_rgb)

            a, b = label_bubble.ellipse_fitting(img_rgb)
            cX, cY = label_bubble.centroid()
            Circularity = label_bubble.circularity()
            Solidity = label_bubble.solidity()
            sr = label_bubble.shade_radio()
            eg = label_bubble.edge_grad()
            # F = label_bubble.reconstruct()[1].flatten()
            # print(F)
            # FF = '\t'.join(map(str, F))
            # print(f"{file}\t{a/128}\t{b/128}\t{cX/128}\t{cY/128}\t{Circularity}\t{Solidity}\t{sr}\t{eg}\n")
            f.write(f"{file}\t{a/128}\t{b/128}\t{cX/128}\t{cY/128}\t{Circularity}\t{Solidity}\t{sr}\t{eg}\n")
            # f.write(f"{file}\t{a/128}\t{b/128}\t{cX/128}\t{cY/128}\t{Circularity}\t{Solidity}\t{sr}\t{eg}\t{FF}\n")
            
            # BubGAN参数
            E = b / a
            phi = 0
            f_bubgan.write(f"{file}\t{E}\t{phi}\t{Circularity}\t{sr}\n")
            # if i % 10 == 0:
            #     break
            if i % 1000 == 0:
                print(f'保存成功{i}/{num_png}个!')
            i += 1