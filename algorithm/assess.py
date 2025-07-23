import os
import numpy as np
import re
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
import logging
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用第2块GPU

def read_xyz(file_path):
    """读取 XYZ 格式的点云文件"""
    return np.loadtxt(file_path, usecols=(0, 1, 2))
def downsample_point_cloud(point_cloud, target_num_points=10000):
    """
    对点云进行下采样
    :param point_cloud: 输入的点云数据（形状为 Nx3 的数组）
    :param target_num_points: 目标点数
    :return: 下采样后的点云
    """
    # 如果点云的数量大于目标数量，则进行下采样
    num_points = len(point_cloud)
    if num_points > target_num_points:
        # 使用 numpy 随机选择点云中的点
        indices = np.random.choice(num_points, target_num_points, replace=False)
        downsampled_points = point_cloud[indices]
        logging.info(f"点云数量过多（{num_points}），已下采样为 {target_num_points} 个点。")
        return downsampled_points
    else:
        logging.info(f"点云数量（{num_points}）无需下采样。")
        return point_cloud
def load_mesh(file_path):
    """加载网格文件（支持顶点和面片）"""
    try:
        return trimesh.load(file_path)
    except Exception as e:
        logging.error(f"加载网格文件失败: {file_path}, 错误: {e}")
        raise

def calculate_normals(points, k=30):
    """基于K近邻的法线估计（使用PCA），增强异常处理"""
    n_samples = len(points)
    if n_samples < 3:
        # 点数太少，无法估计法线
        return np.zeros((n_samples, 3))

    k = min(k, n_samples - 1)  # 保证 k < n_samples
    neigh = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = neigh.kneighbors(points)

    normals = []
    for neigh_indices in indices:
        neighbors = points[neigh_indices]
        if len(neighbors) < 3:
            # 点数不足，返回默认法线
            normals.append([0, 0, 1])
            continue
        cov = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # 最小特征值方向
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            normals.append([0, 0, 1])  # 避免除以0
        else:
            normals.append(normal / norm)

    return np.array(normals)



def calculate_curvature(point_cloud, k=20):
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    curvatures = []
    if point_cloud.shape[0] <= k:
        # 如果点太少无法计算邻居，返回全0
        return np.zeros(point_cloud.shape[0])

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(point_cloud)
    distances, indices = nbrs.kneighbors(point_cloud)

    for i in range(len(point_cloud)):
        neighbors = point_cloud[indices[i][1:]]  # skip itself
        if neighbors.shape[0] < 3:
            # 邻居太少，无法构成协方差矩阵
            curvatures.append(0.0)
            continue
        try:
            cov = np.cov(neighbors.T)
            eigvals, _ = np.linalg.eigh(cov)
            eigvals = np.sort(eigvals)
            curvature = eigvals[0] / (eigvals.sum() + 1e-8)
            curvatures.append(curvature)
        except np.linalg.LinAlgError:
            # 有可能矩阵不正定，报错时返回0
            curvatures.append(0.0)

    return np.array(curvatures)


def average_distance(point_cloud, mesh, curvature_weight=0.3, normal_angle_thresh=45, trunc_threshold=1.0, huber_delta=0.1):
    """改进版平均距离计算"""
    num_samples = point_cloud.shape[0]
    sampled_points, face_ids = mesh.sample(num_samples, return_index=True)

    sampled_normals = mesh.face_normals[face_ids]

    cloud_normals = calculate_normals(point_cloud)
    cloud_curv = calculate_curvature(point_cloud)

    tree = KDTree(sampled_points)
    dists, indices = tree.query(point_cloud)
    matched_normals = sampled_normals[indices]

    curv_weights = 1 + curvature_weight * cloud_curv
    cos_sim = np.sum(cloud_normals * matched_normals, axis=1)
    angle_diff = np.arccos(np.clip(cos_sim, -1, 1))
    normal_weights = np.where(np.degrees(angle_diff) < normal_angle_thresh, 1.0, 0.2)

    valid_mask = dists < trunc_threshold
    valid_dists = np.where(valid_mask, dists, trunc_threshold)

    huber_loss = np.where(valid_dists < huber_delta,
                          0.5 * valid_dists ** 2,
                          huber_delta * (valid_dists - 0.5 * huber_delta))

    total_weights = curv_weights * normal_weights
    weighted_sum = np.sum(total_weights * huber_loss)

    return weighted_sum / np.sum(total_weights)

def hausdorff_distance(point_cloud, mesh, p=0.95):
    """计算部分豪斯多夫距离"""
    distances_p_to_q = np.abs(mesh.nearest.signed_distance(point_cloud))
    kdtree_cloud = KDTree(point_cloud)
    distances_q_to_p, _ = kdtree_cloud.query(mesh.vertices)

    q_pq = np.quantile(distances_p_to_q, p)
    q_qp = np.quantile(distances_q_to_p, p)

    return max(q_pq, q_qp)

def enhanced_chamfer(cloud_A, cloud_B, normals_A=None, normals_B=None, curvature_weight=0.3, normal_angle_thresh=45, huber_delta=0.1):
    """增强版倒角距离"""
    if normals_A is None:
        normals_A = calculate_normals(cloud_A)
    if normals_B is None:
        normals_B = calculate_normals(cloud_B)

    curv_A = calculate_curvature(cloud_A)
    curv_B = calculate_curvature(cloud_B)

    tree_B = KDTree(cloud_B)
    dist_A, idx_B = tree_B.query(cloud_A)
    matched_normals_B = normals_B[idx_B]

    cos_sim = np.sum(normals_A * matched_normals_B, axis=1)
    angle_diff = np.arccos(np.clip(cos_sim, -1, 1))
    normal_weight = np.where(np.degrees(angle_diff) < normal_angle_thresh, 1.0, 0.3)

    curv_term = curv_B[idx_B] * curvature_weight

    huber_loss = np.where(dist_A < huber_delta,
                          0.5 * (dist_A ** 2),
                          huber_delta * (dist_A - 0.5 * huber_delta))

    term_A = np.mean(normal_weight * (huber_loss + curv_term))

    tree_A = KDTree(cloud_A)
    dist_B, idx_A = tree_A.query(cloud_B)
    matched_normals_A = normals_A[idx_A]
    cos_sim = np.sum(normals_B * matched_normals_A, axis=1)
    angle_diff = np.arccos(np.clip(cos_sim, -1, 1))
    normal_weight = np.where(np.degrees(angle_diff) < normal_angle_thresh, 1.0, 0.3)
    curv_term = curv_A[idx_A] * curvature_weight
    huber_loss = np.where(dist_B < huber_delta,
                          0.5 * (dist_B ** 2),
                          huber_delta * (dist_B - 0.5 * huber_delta))

    term_B = np.mean(normal_weight * (huber_loss + curv_term))

    return term_A + term_B

def inlier_proportion(point_cloud, mesh, threshold=0.02):
    """计算点云与网格的内点比例"""
    if point_cloud.shape[0] < 2:
        raise ValueError("点云需要至少两个点以计算最远距离。")

    max_distance = pdist(point_cloud).max()
    dynamic_threshold = max_distance * threshold

    # 使用KDTree查询每个点到网格表面采样点的最近距离
    surface_points, _ = mesh.sample(10000, return_index=True)  # 采样一些表面点
    kdtree = KDTree(surface_points)
    distances, _ = kdtree.query(point_cloud)

    inlier_mask = distances < dynamic_threshold
    return np.mean(inlier_mask)

def get_files_in_order(folder, ext):
    """获取文件夹中指定扩展名的文件列表（自然顺序，不做数字提取排序）"""
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(ext)])
def extract_number(filename):
    """从文件名中提取数字进行排序"""
    numbers = re.findall(r'\d+', filename)
    return int(''.join(numbers)) if numbers else 0


def get_sorted_files(folder, ext):
    """按数字排序文件"""
    files = [f for f in os.listdir(folder) if f.lower().endswith(ext)]
    return sorted(files, key=lambda x: (extract_number(x), x))
def compute_projection_iou(point_cloud, mesh):
    """
    计算点云和网格顶点投影到XY平面的凸包面积比，作为投影IoU指标
    """
    # 取点云前两维
    points_2d = point_cloud[:, :2]
    # 计算点云凸包多边形
    try:
        hull_pc = ConvexHull(points_2d)
        pc_poly = Polygon(points_2d[hull_pc.vertices])
    except Exception as e:
        raise ValueError(f"点云凸包计算失败: {e}")

    # 网格顶点投影XY
    vertices_2d = mesh.vertices[:, :2]
    try:
        hull_mesh = ConvexHull(vertices_2d)
        mesh_poly = Polygon(vertices_2d[hull_mesh.vertices])
    except Exception as e:
        raise ValueError(f"网格凸包计算失败: {e}")

    # 计算交集与并集面积，做严格IoU
    inter_area = pc_poly.intersection(mesh_poly).area
    union_area = pc_poly.union(mesh_poly).area
    if union_area == 0:
        raise ValueError("投影多边形面积为0")

    iou = inter_area / union_area
    return iou

def nonlinear_score_transform(q, a=6.0, s=0.05, min_val=0.5):
    """
    非线性分数增强函数：
    将输入分数 q ∈ [0.1, 1.0] 映射到更靠近 [0.5, 1.0] 的区间。
    a: 控制上升速率
    s: 偏移项，避免除0
    min_val: 映射后的最小值（靠近0.5）
    """
    q = np.clip(q, 0.1, 1.0)
    base = 1 - 1.0 / (a * (q - s) + 1.0)  # 先得到类似反比例的非线性增长曲线 ∈ [≈0, 1)
    scaled = min_val + (1 - min_val) * base  # 将其线性映射到 [0.5, 1.0]
    return np.clip(scaled, min_val, 1.0)


def pinggu(mesh_dir, point_cloud_dir, p=0.95, downsample_threshold=10000, target_num_points=10000):
    """按数字提取文件名进行匹配，未匹配的文件记分为0"""
    mesh_files = get_sorted_files(mesh_dir, '.obj')
    pc_files = get_sorted_files(point_cloud_dir, '.xyz')

    # 提取编号与路径映射
    mesh_dict = {extract_number(f): f for f in mesh_files}
    pc_dict = {extract_number(f): f for f in pc_files}

    # 取并集编号
    all_keys = sorted(set(mesh_dict.keys()).union(set(pc_dict.keys())))

    scores = []
    os.makedirs("output", exist_ok=True)

    with open("output/assessment_results_mesh_oabj.txt", "w", encoding="utf-8") as file:
        for key in all_keys:
            mesh_file = mesh_dict.get(key)
            pc_file = pc_dict.get(key)

            if mesh_file is None or pc_file is None:
                msg = f"{key}: 0.0000"
                print(msg)
                file.write(msg + "\n")
                scores.append((mesh_file or f"missing_{key}.obj", pc_file or f"missing_{key}.xyz", 0.0))
                continue

            try:
                mesh_path = os.path.join(mesh_dir, mesh_file)
                pc_path = os.path.join(point_cloud_dir, pc_file)

                # 加载网格
                mesh = trimesh.load(mesh_path)
                if isinstance(mesh, trimesh.Scene):
                    meshes = list(mesh.geometry.values())
                    mesh = trimesh.util.concatenate(meshes)
                if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty or mesh.faces.shape[0] == 0:
                    raise ValueError("无效或空网格")

                # 加载并下采样点云
                point_cloud = np.loadtxt(pc_path, usecols=(0, 1, 2))
                point_cloud = downsample_point_cloud(point_cloud, target_num_points=target_num_points)

                # 计算指标
                avg_dist = average_distance(point_cloud, mesh)
                hausdorff_dist = hausdorff_distance(point_cloud, mesh, p)
                chamfer_dist = enhanced_chamfer(point_cloud, mesh.vertices)
                iou = inlier_proportion(point_cloud, mesh, threshold=0.02)
                proj_iou = compute_projection_iou(point_cloud, mesh)

                quality_score = (1/(1+avg_dist) + 1/(1+hausdorff_dist) + 1/(1+chamfer_dist)+1)*(iou)*proj_iou/4
                adjusted_score = nonlinear_score_transform(quality_score)


                model_name = os.path.splitext(mesh_file)[0]
                result = f"{model_name}: {adjusted_score:.4f}"
                print(result)
                file.write(result + "\n")
                scores.append((mesh_file, pc_file, adjusted_score))

            except Exception as e:
                error_msg = f"{key}: 处理失败 | 错误: {str(e)}"
                print(error_msg)
                file.write(error_msg + "\n")
                scores.append((mesh_file, pc_file, 0.0))

    # 写 CSV 汇总
    with open("output/Gxassessment_summary.csv", "w", encoding="utf-8") as csv_file:
        csv_file.write("MeshFile,PointCloudFile,QualityScore\n")
        for mesh_file, pc_file, score in scores:
            csv_file.write(f"{mesh_file},{pc_file},{score:.4f}\n")

def main():
    mesh_dir = r'input/roof_mesh'
    point_cloud_dir = r'input/roof_points'
    pinggu(mesh_dir, point_cloud_dir, p=0.95)

if __name__ == "__main__":
    input()
    main()
