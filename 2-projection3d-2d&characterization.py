import os
import cv2
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy.ndimage import median_filter, gaussian_filter
import concurrent.futures
import threading

def data_regularize(data, type="spherical", divs=10):
    limits = np.array([
        [min(data[:, 0]), max(data[:, 0])],
        [min(data[:, 1]), max(data[:, 1])],
        [min(data[:, 2]), max(data[:, 2])]])
        
    regularized = []

    if type == "cubic": # take mean from points in the cube
        
        X = np.linspace(*limits[0], num=divs)
        Y = np.linspace(*limits[1], num=divs)
        Z = np.linspace(*limits[2], num=divs)

        for i in range(divs-1):
            for j in range(divs-1):
                for k in range(divs-1):
                    points_in_sector = []
                    for point in data:
                        if (point[0] >= X[i] and point[0] < X[i+1] and
                                point[1] >= Y[j] and point[1] < Y[j+1] and
                                point[2] >= Z[k] and point[2] < Z[k+1]):
                            points_in_sector.append(point)
                    if len(points_in_sector) > 0:
                        regularized.append(np.mean(np.array(points_in_sector), axis=0))

    elif type == "spherical": #take mean from points in the sector
        divs_u = divs 
        divs_v = divs * 2

        center = np.array([
            0.5 * (limits[0, 0] + limits[0, 1]),
            0.5 * (limits[1, 0] + limits[1, 1]),
            0.5 * (limits[2, 0] + limits[2, 1])])
        d_c = data - center
    
        #spherical coordinates around center
        r_s = np.sqrt(d_c[:, 0]**2. + d_c[:, 1]**2. + d_c[:, 2]**2.)
        d_s = np.array([
            r_s,
            np.arccos(d_c[:, 2] / r_s),
            np.arctan2(d_c[:, 1], d_c[:, 0])]).T

        u = np.linspace(0, np.pi, num=divs_u)
        v = np.linspace(-np.pi, np.pi, num=divs_v)

        for i in range(divs_u - 1):
            for j in range(divs_v - 1):
                points_in_sector = []
                for k, point in enumerate(d_s):
                    if (point[1] >= u[i] and point[1] < u[i + 1] and
                            point[2] >= v[j] and point[2] < v[j + 1]):
                        points_in_sector.append(data[k])

                if len(points_in_sector) > 0:
                    regularized.append(np.mean(np.array(points_in_sector), axis=0))
# Other strategy of finding mean values in sectors
#                    p_sec = np.array(points_in_sector)
#                    R = np.mean(p_sec[:,0])
#                    U = (u[i] + u[i+1])*0.5
#                    V = (v[j] + v[j+1])*0.5
#                    x = R*math.sin(U)*math.cos(V)
#                    y = R*math.sin(U)*math.sin(V)
#                    z = R*math.cos(U)
#                    regularized.append(center + np.array([x,y,z]))
    return np.array(regularized)


# https://github.com/minillinim/ellipsoid
def ellipsoid_plot(center, radii, rotation, ax, plot_axes=False, cage_color='b', cage_alpha=0.2):
    """Plot an ellipsoid"""
        
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    
    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # x = np.outer(np.cos(u), np.sin(v))
    # y = np.outer(np.sin(u), np.sin(v))
    # z = np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    if plot_axes:
        # make some purdy axes
        axes = np.array([[radii[0],0.0,0.0],
                         [0.0,radii[1],0.0],
                         [0.0,0.0,radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)

        # plot axes
        for i, p in enumerate(axes):
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=['r', 'g', 'b'][i])

    # plot ellipsoid
    ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color=cage_color, alpha=cage_alpha)


# http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
# for arbitrary axes
def ellipsoid_fit(X):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array([x * x + y * y - 2 * z * z,
                 x * x + z * z - 2 * y * y,
                 2 * x * y,
                 2 * x * z,
                 2 * y * z,
                 2 * x,
                 2 * y,
                 2 * z,
                 1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                    [v[3], v[1], v[5], v[7]],
                    [v[4], v[5], v[2], v[8]],
                    [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)

    return center, evecs, radii, v

def sphere_to_cartesian(r, lat, lon):
    x = r * np.sin(lat) * np.cos(lon)
    y = r * np.sin(lat) * np.sin(lon)
    z = r * np.cos(lat)
    return x, y, z
 
def cartesian_to_sphere(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arccos(z / r)
    lon = np.arctan2(y, x)
    return r, lat, lon

def generate_fibonacci_points_on_sphere(N=1000):
    phi = (np.sqrt(5) - 1) / 2
    n = np.arange(0, N)
    z = ((2*n + 1) / N - 1)
    x = (np.sqrt(1 - z**2)) * np.cos(2 * np.pi * (n + 1) * phi)
    y = (np.sqrt(1 - z**2)) * np.sin(2 * np.pi * (n + 1) * phi)

    points = np.stack([x, y, z], axis=-1)
    # fig = plt.figure("sphere points uniform")
    # ax = fig.add_subplot(111,projection='3d')
    # ax.scatter(points[:,0], points[:,1], points[:,2], color='r')
    # plt.show()
    return points

def generate_points_on_sphere(N = 1000):
    sphere = pv.Sphere(radius=1, theta_resolution = 4, phi_resolution = 3)
    points = np.array(sphere.points)
    # print(f"输出点的数量: {len(points)}")
    return points

def cv2_enhance_contrast(img, factor):
    mean = np.uint8(cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0])
    img_deg = np.ones_like(img) * mean
    return cv2.addWeighted(img, factor, img_deg, 1-factor, 0.0)

def ImageRotate_2(image, angle):
    # 要有中心坐标、旋转角度、缩放系数
    h, w = image.shape[:2]  # 输入(H,W,C)，取 H，W 的值
    center = (w // 2, h // 2)  # 绕图片中心进行旋转

    # 1. 获得旋转矩阵
    M = cv2.getRotationMatrix2D(center=center, angle=angle, scale = 1)  # 当angle为负值时，则表示为顺时针

    # -----------------------计算图像的新边界尺寸、调整旋转矩阵以考虑平移-------------------- #
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像的新边界尺寸
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵以考虑平移
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    # ------------------------------------------------------------------------------ #

    # 2. 进行仿射变换，边界填充为255，即白色，borderValue 缺省，默认是黑色（0, 0 , 0）
    image_rotation = cv2.warpAffine(src=image, M=M, dsize=(nW, nH), borderValue=(0, 0, 0))

    return image_rotation

def upsample_point_cloud(mesh, num_clusters):
    # current_points = mesh.n_points
    # # 根据目标点数决定是上采样还是下采样
    # if current_points > num_clusters:
    #     # 下采样 - 使用网格简化
    #     mesh = mesh.decimate(target_reduction=1.0 - (num_clusters / current_points))
    # elif current_points < num_clusters:
    #     # 上采样 - 使用细分
    #     iterations = int(np.log2(num_clusters / current_points)) + 1
    #     for i in range(iterations):
    #         mesh = mesh.subdivide(1, 'loop')
        
    #     # 如果超出目标点数，再次进行轻微降采样
    #     if mesh.n_points > num_clusters:
    #         mesh = mesh.decimate(target_reduction=1.0 - (num_clusters / mesh.n_points))

    cloud = pv.PolyData(mesh.points)
    sample_spacing = 0.2
    while(1):
        mesh = cloud.reconstruct_surface(nbr_sz = 10, sample_spacing = sample_spacing)
        # 提取顶点作为新的点云
        new_points = np.asarray(mesh.points)
        if new_points.shape[0] < num_clusters * 1.1:
            sample_spacing *= 0.9
        else:
            break
    return mesh

def process_stl_file(stl_file, output_folder, points, projection_model, param_file = None, lock=None):
    scale = 100
    # 第一步：读取STL文件并进行基本预处理
    stl_mesh = pv.read(stl_file)

    # 第二步：使用原始mesh计算三维结构参数
    stl_points = stl_mesh.points
    Volume_3D = stl_mesh.volume
    Surface_area_3D = stl_mesh.area

    # 拟合椭球
    center, evecs, radii, v = ellipsoid_fit(stl_points)
    a, b, c = sorted(radii, reverse=True)

    # 计算EI, FI, AR
    EI_3D = b / a
    FI_3D = c / b
    AR_3D = (EI_3D + FI_3D) / 2

    # 计算sphericity (SI)
    Sphericity_3D = (np.pi**(1/3) * (6 * Volume_3D)**(2/3)) / Surface_area_3D

    # 计算凸包convexity (C_X)
    hull = ConvexHull(stl_points)
    Convexity_3D = Volume_3D / hull.volume

    # 计算Angularity_3D
    Angularity_3D = 0
    a, b, c = radii
    for orth in stl_points:
        point = (orth - center) @ np.linalg.inv(evecs)
        cos_x = point[0] / np.linalg.norm(point)
        cos_y = point[1] / np.linalg.norm(point)
        cos_z = point[2] / np.linalg.norm(point)
        distance_ellipsoid = 1 / np.sqrt(cos_x**2 / a**2 + cos_y**2 / b**2 + cos_z**2 / c**2)
        r, lat, lon = cartesian_to_sphere(*point)
        r = r * distance_ellipsoid / np.linalg.norm(point)
        xx, yy, zz = sphere_to_cartesian(r, lat, lon)
        [xx, yy, zz] = np.dot([xx, yy, zz], evecs)

        distance_to_center = np.linalg.norm(orth)
        theoretical1 = np.linalg.norm([xx, yy, zz])
        ratio = abs(distance_to_center - theoretical1) / theoretical1
        Angularity_3D += ratio
    Angularity_3D /= stl_points.shape[0]

    # 第三步：进行点云上采样，用于二维投影
    stl_mesh_upsampled = upsample_point_cloud(stl_mesh, 20000)
    stl_mesh_upsampled.smooth_taubin(n_iter=10, pass_band=5, inplace = True)
    stl_mesh_upsampled = stl_mesh_upsampled.fill_holes(100)
    
    # 第四步：开始二维投影循环
    v = np.array([0.01, 0.01, 1])
    j = 0
    # progress2 = tqdm(points, desc=f"Projection of {os.path.splitext(stl_file)[0]}", leave=False)
    progress2 = tqdm(points, desc=f"Projection of {stl_file[0:-4]}", leave=False)
    
    # 创建存储2D投影结果的列表
    results = []
    
    for point in progress2:
        # Rotate mesh (使用上采样后的mesh进行旋转)
        rot = stl_mesh_upsampled.rotate_vector(np.cross(point, v), np.arccos(np.dot(point, v)) * 180 / np.pi, inplace=False)
        
        if projection_model == 'fibonacci':
            image_path = os.path.join(output_folder, f"Fibonacci_{point[0]:.2f}_{point[1]:.2f}_{point[2]:.2f}.png")
        elif projection_model == 'sphere':
            image_path = os.path.join(output_folder, f"Sphere_{point[0]:.2f}_{point[1]:.2f}_{point[2]:.2f}.png")
                
        normals = rot.point_normals
        # 舍弃z<0
        mask = normals[:, 2] > -0
        filtered_points = rot.points[mask]
        filtered_normals = normals[mask]
        angles = filtered_normals[:, 2] / np.linalg.norm(filtered_normals, axis=1)
        
        M = angles ** 8

        mapped_points = np.ones((128, 128))
        try:
            for i in range(filtered_points.shape[0]):
                x, y = filtered_points[i, 0], filtered_points[i, 1]
                mapped_x, mapped_y = (x + 1) *  64, (y + 1) *  64
                l1 = np.square(mapped_x - np.floor(mapped_x)) + np.square(mapped_y - np.floor(mapped_y))
                l2 = np.square(mapped_x - np.ceil(mapped_x)) + np.square(mapped_y - np.floor(mapped_y))
                l3 = np.square(mapped_x - np.floor(mapped_x)) + np.square(mapped_y - np.ceil(mapped_y))
                l4 = np.square(mapped_x - np.ceil(mapped_x)) + np.square(mapped_y - np.ceil(mapped_y))
                total = l1 + l2 + l3 + l4
                mapped_points[int(np.floor(mapped_x)), int(np.floor(mapped_y))] += M[i] * (total - l1) / total
                mapped_points[int(np.ceil(mapped_x)), int(np.floor(mapped_y))] += M[i] * (total - l2) / total
                mapped_points[int(np.floor(mapped_x)), int(np.ceil(mapped_y))] += M[i] * (total - l3) / total
                mapped_points[int(np.ceil(mapped_x)), int(np.ceil(mapped_y))] += M[i] * (total - l4) / total
                # Additional points
                mapped_points[int(np.floor(mapped_x)) - 1, int(np.floor(mapped_y))] += M[i] * (total - l1) / total
                mapped_points[int(np.floor(mapped_x)), int(np.floor(mapped_y)) - 1] += M[i] * (total - l1) / total
                mapped_points[int(np.ceil(mapped_x)) + 1, int(np.floor(mapped_y))] += M[i] * (total - l2) / total
                mapped_points[int(np.ceil(mapped_x)), int(np.floor(mapped_y)) - 1] += M[i] * (total - l2) / total
                mapped_points[int(np.floor(mapped_x)) - 1, int(np.ceil(mapped_y))] += M[i] * (total - l3) / total
                mapped_points[int(np.floor(mapped_x)), int(np.ceil(mapped_y)) + 1] += M[i] * (total - l3) / total
                mapped_points[int(np.ceil(mapped_x)) + 1, int(np.ceil(mapped_y))] += M[i] * (total - l4) / total
                mapped_points[int(np.ceil(mapped_x)), int(np.ceil(mapped_y)) + 1] += M[i] * (total - l4) / total
        except Exception as e:
            print(f"渲染错误发生在文件：{image_path}，点序号：{j}\n")
            continue
        indices = np.where(mapped_points == 1)

        mapped_points[indices] = 0
        mapped_points = gaussian_filter(mapped_points, sigma=1)
        truncation = 0.5
        mapped_points_normalized = np.clip(mapped_points / mapped_points.max(), 0, truncation) / truncation
        mapped_points_normalized[indices] = 1

    
        # mapped_points_normalized[mapped_points_normalized > 1] = 1
        mapped_points_normalized = gaussian_filter(mapped_points_normalized, sigma=0.75)
        mapped_points_normalized = median_filter(mapped_points_normalized, size=5)

        mapped_points_normalized = np.rot90(mapped_points_normalized, -3)
        # mapped_points_normalized = np.log(mapped_points_normalized + 1)
        # mapped_points_normalized = (mapped_points_normalized - mapped_points_normalized.min()) / (mapped_points_normalized.max() - mapped_points_normalized.min())
        # plt.imsave(image_path, mapped_points_normalized, cmap='gray')

        mapped_points_normalized = (mapped_points_normalized * 255).astype(np.uint8).T
        mapped_points_normalized = cv2.cvtColor(mapped_points_normalized, cv2.COLOR_GRAY2RGB)
        mapped_points_normalized = cv2_enhance_contrast(mapped_points_normalized, 1.5)
        mapped_points_normalized = cv2.transpose(mapped_points_normalized)
        cv2.imwrite(image_path, mapped_points_normalized)

        # 计算投影的x和y范围
        faces = rot.irregular_faces
        vertuces = rot.points
        # 计算x和y的最大最小值以确定图像尺寸
        x_min = np.min(vertuces[:, 0])
        x_max = np.max(vertuces[:, 0])
        y_min = np.min(vertuces[:, 1])
        y_max = np.max(vertuces[:, 1])
    
        # 计算图像的宽度和高度
        width = int((x_max - x_min) * scale)
        height = int((y_max - y_min) * scale)

        # 创建一个空白的图像
        projection_image = np.zeros((height, width), dtype=np.uint8)

        # 遍历所有的三角面片
        for face in faces:
            # 获取三角面片的顶点索引
            indices = face
            # 获取顶点的xy坐标
            tri_vertices = vertuces[indices, :2]
            # 将坐标转换为图像上的像素位置
            tri_vertices[:, 0] = (tri_vertices[:, 0] - x_min) * scale
            tri_vertices[:, 1] = (tri_vertices[:, 1] - y_min) * scale
            # 绘制填充的三角形
            cv2.fillConvexPoly(projection_image, np.array(tri_vertices, 'int32'), 1)

        # # 保存图像
        projection_image = np.pad(projection_image, pad_width=10, mode='constant', constant_values=0)

        # cv2.imwrite(os.path.join(output_folder, f"Binary_{str(j).zfill(3)}.png"), projection_image * 255)

        # 计算二维结构参数
        contours, _ = cv2.findContours(projection_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        Area_2D = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            Convexity_2D = 0
        else:
            Convexity_2D = float(Area_2D) / hull_area
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        except Exception as e:
            print(f"错误发生在文件：{image_path}，点序号：{j}\n")
            continue
        MA /= scale
        ma /= scale
        Aspect_Ratio_2D = MA / ma
        # 计算椭圆的面积
        ellipse_area = np.pi * MA * ma / 4
        Circularity_2D = (2 * np.sqrt(np.pi * Area_2D)) / perimeter
        Area_2D /= scale**2
        Solidity_2D = 1 - abs((Area_2D / ellipse_area) - 1)

        # 计算Radius Angularity Index参数
        # projection_image图片逆向旋转angle度，多余地方使用0值填充
        projection_draw = ImageRotate_2(cv2.cvtColor(projection_image * 255, cv2.COLOR_GRAY2BGR), angle=angle)
        contours, _ = cv2.findContours(cv2.cvtColor(projection_draw, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        try:
            (x, y), (MA_rotate, ma_rotate), angle = cv2.fitEllipse(cnt)
        except Exception as e:
            print(f"错误发生在文件：{image_path}，点序号：{j}\n")
            continue
        MA_rotate /= 2
        ma_rotate /= 2
        sum_ratio = 0
        cnt = cnt[::10]
        for contour_point in cnt:
            # 计算当前点到椭圆中心的距离
            dx = contour_point[0][0] - x
            dy = contour_point[0][1] - y
            distance_to_center = np.sqrt(dx**2 + dy**2)

            # 计算当前点的方向角度
            theta = np.arctan2(dy, dx)

            # 计算椭圆在该方向上的半径
            ellipse_radius = MA_rotate * ma_rotate / np.sqrt((ma_rotate**2) * (np.cos(theta)**2) + (MA_rotate**2) * (np.sin(theta)**2))
            # print(ellipse_radius)
            # 计算差值与椭圆半径的比
            ratio = abs(distance_to_center - ellipse_radius) / ellipse_radius
            sum_ratio += ratio
            # print(ratio)
            
        radius_angularity_index_2D = sum_ratio / len(cnt)

        # 收集结果
        result = {
            "image_path": image_path,
            "a": a, "b": b, "c": c,
            "Volume_3D": Volume_3D,
            "Surface_area_3D": Surface_area_3D,
            "EI_3D": EI_3D,
            "FI_3D": FI_3D,
            "AR_3D": AR_3D,
            "Sphericity_3D": Sphericity_3D,
            "Convexity_3D": Convexity_3D,
            "Angularity_3D": Angularity_3D,
            "ma": ma,
            "MA": MA,
            "Area_2D": Area_2D,
            "Aspect_Ratio_2D": Aspect_Ratio_2D,
            "Circularity_2D": Circularity_2D,
            "Solidity_2D": Solidity_2D,
            "Convexity_2D": Convexity_2D,
            "radius_angularity_index_2D": radius_angularity_index_2D
        }
        results.append(result)
        j += 1
    
    # 线程安全地写入参数文件
    if lock:
        lock.acquire()
    try:
        with open(parameters_path, 'a') as param_file:
            for result in results:
                param_file.write(f"{result['image_path']},{result['a']},{result['b']},{result['c']},{result['Volume_3D']},{result['Surface_area_3D']},"
                                f"{result['EI_3D']},{result['FI_3D']},{result['AR_3D']},{result['Sphericity_3D']},{result['Convexity_3D']},"
                                f"{result['Angularity_3D']},{result['ma']},{result['MA']},{result['Area_2D']},{result['Aspect_Ratio_2D']},"
                                f"{result['Circularity_2D']},{result['Solidity_2D']},{result['Convexity_2D']},{result['radius_angularity_index_2D']}\n")
    finally:
        if lock:
            lock.release()
    
    return f"已完成 {stl_file} 的处理"

# stl_folder = r"mesh"
# output_base_folder = r"projection"
stl_folder = r"mesh_20250619"
output_base_folder = r"projection_20250619"
os.makedirs(output_base_folder, exist_ok=True)

# projection_model = 'fibonacci'
projection_model = 'sphere'

# 准备保存结构参数的CSV文件
parameters_path = os.path.join(output_base_folder, "parameters_20250619.csv")
with open(parameters_path, 'w') as param_file:
    param_file.write("Filename,a,b,c,Volume_3D,Surface_area_3D,EI_3D,FI_3D,AR_3D,Sphericity_3D,Convexity_3D,Angularity_3D,Principal_Dimension_2D,Secondary_Dimension_2D,Area_2D,Aspect_Ratio_2D,Circularity_2D,Solidity_2D,Convexity_2D,radius_angularity_index_2D\n")
    
# Generate uniform points on sphere
if projection_model == 'fibonacci':
    points = generate_fibonacci_points_on_sphere(N = 50)
elif projection_model == 'sphere':
    points = generate_points_on_sphere(N = 50)
# 保存点到文件
points_output_path = os.path.join(output_base_folder, f"{projection_model}_points.csv")
np.savetxt(points_output_path, points, delimiter=",")

# 创建一个线程安全的锁，用于保护CSV文件写入
csv_lock = threading.Lock()

# 获取需要处理的STL文件列表
stl_files = [f for f in os.listdir(stl_folder) if f.endswith(".stl")][:] 
print(f"总共需要处理 {len(stl_files)} 个STL文件")

# 使用线程池并行处理STL文件
def process_file(stl_file):
    try:
        stl_path = os.path.join(stl_folder, stl_file)
        output_folder = os.path.join(output_base_folder, os.path.splitext(stl_file)[0])
        os.makedirs(output_folder, exist_ok=True)
        return process_stl_file(stl_path, output_folder, points, projection_model, lock=csv_lock)
    except Exception as e:
        return f"处理 {stl_file} 时发生错误: {str(e)}"

# 获取CPU物理核心数和逻辑核心数
import psutil
physical_cores = psutil.cpu_count(logical=False)  # 物理核心数
logical_cores = psutil.cpu_count(logical=True)    # 逻辑核心数（包括超线程）

# 设置线程池最大工作线程数，使用逻辑核心数以最大化并行性能
# 对于IO密集型任务，可以使用比核心数更多的线程
max_workers = logical_cores + 4  # IO密集型任务可以使用更多线程

print(f"检测到物理核心数: {physical_cores}, 逻辑核心数: {logical_cores}")
print(f"使用 {max_workers} 个工作线程进行并行处理以最大化性能")

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 提交所有任务到线程池
    future_to_file = {executor.submit(process_file, stl_file): stl_file for stl_file in stl_files}
    
    # 显示进度条
    with tqdm(total=len(future_to_file), desc="处理STL文件", dynamic_ncols=True) as progress_bar:
        for future in concurrent.futures.as_completed(future_to_file):
            stl_file = future_to_file[future]
            try:
                result = future.result()
                print(f"\n完成: {stl_file} - {result}")
            except Exception as e:
                print(f"\n错误: {stl_file} - {str(e)}")
            finally:
                progress_bar.update(1)

print("所有STL文件处理完成！")