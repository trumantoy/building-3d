import numpy as np
import os
from PIL import Image
from fastsam import FastSAM, FastSAMPrompt
import torch
import sys
import time
import cv2
import open3d as o3d
from scipy.signal import find_peaks

#提取屋顶
def extract_roof_by_z_density(points, bin_size=0.6, top_percent=0.8,
                               peak_prominence=0.95, max_peaks=10, extend_below=1.0):
    """
    从输入点云中根据 z 分布提取屋顶点。
    返回: 屋顶点数组 (M, 3)，如果为噪声或无主峰则返回 None
    """
    if points.shape[0] == 0:
        return None

    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    bins = np.arange(z_min, z_max + bin_size, bin_size)
    hist, _ = np.histogram(z_vals, bins=bins)

    # 判断噪声：z 分布过于平坦/多峰
    peaks, _ = find_peaks(hist, prominence=peak_prominence * hist.max())
    if len(peaks) > max_peaks:
        return None

    # 从顶部区域找主峰
    top_start = int(len(hist) * (1 - top_percent))
    top_hist = hist[top_start:]
    top_bins = bins[top_start:top_start + len(top_hist)]

    top_peaks, _ = find_peaks(top_hist, prominence=peak_prominence * top_hist.max())
    if len(top_peaks) == 0:
        return None

    peak_idx = top_peaks[np.argmax(top_hist[top_peaks])]
    peak_z = top_bins[peak_idx]

    # 提取 z >= (主峰 z - extend_below)
    z_thresh = peak_z - extend_below              # 屋顶阈值：主峰往下延伸
    mask = z_vals >= z_thresh
    roof_points = points[mask]                    # 满足 z >= 阈值的屋顶点
    return roof_points, points
#合并重叠掩码
def merge_overlapping_masks(masks, iou_thresh=0.5):
    """
    合并高度重叠的掩码（IoU > iou_thresh），加入 bbox 快速过滤加速。
    参数:
        masks: List[Tensor(H, W)] 每个元素是 0/1 掩码
        iou_thresh: float, 两掩码 IoU 超过此值则合并
    返回:
        merged_masks: Tensor(N, H, W)
    """
    def get_bbox(mask):
        """计算掩码的边界框 [ymin, ymax, xmin, xmax]"""
        ys, xs = mask.nonzero(as_tuple=True)
        return ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item()

    def bbox_iou(b1, b2):
        """边界框快速 IoU，用于提前跳过不可能合并的掩码"""
        y_min1, y_max1, x_min1, x_max1 = b1
        y_min2, y_max2, x_min2, x_max2 = b2

        xi1 = max(x_min1, x_min2)
        yi1 = max(y_min1, y_min2)
        xi2 = min(x_max1, x_max2)
        yi2 = min(y_max1, y_max2)

        inter_w = max(0, xi2 - xi1 + 1)
        inter_h = max(0, yi2 - yi1 + 1)
        inter_area = inter_w * inter_h

        area1 = (x_max1 - x_min1 + 1) * (y_max1 - y_min1 + 1)
        area2 = (x_max2 - x_min2 + 1) * (y_max2 - y_min2 + 1)
        union_area = area1 + area2 - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area

    merged = []
    used = [False] * len(masks)
    bboxes = [get_bbox(mask) for mask in masks]

    for i in range(len(masks)):
        if used[i]:
            continue
        current_mask = masks[i].clone()
        current_bbox = bboxes[i]
        used[i] = True

        for j in range(i + 1, len(masks)):
            if used[j]:
                continue
            if bbox_iou(current_bbox, bboxes[j]) < 0.1:
                continue  # 明显不重叠，跳过

            other_mask = masks[j]
            intersection = torch.logical_and(current_mask, other_mask).sum().item()
            union = torch.logical_or(current_mask, other_mask).sum().item()
            if union == 0:
                continue
            iou = intersection / union
            if iou > iou_thresh:
                current_mask = torch.logical_or(current_mask, other_mask)
                used[j] = True

        merged.append(current_mask.unsqueeze(0))

    return torch.cat(merged, dim=0) if merged else torch.empty((0, *masks[0].shape), device=masks[0].device)
#计算粗糙度
def compute_roughness_cloudcompare_style(pcd, radius=1.0, min_neighbors=10):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    roughness = []

    for i, point in enumerate(points):
        _, idx, _ = kdtree.search_radius_vector_3d(point, radius)
        if len(idx) < min_neighbors:
            roughness.append(np.nan)
            continue

        neighbors = points[idx]
        centroid = np.mean(neighbors, axis=0)
        centered = neighbors - centroid

        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[2]  # 最小奇异值方向
            dist = np.abs(np.dot(point - centroid, normal))
            roughness.append(dist)
        except:
            roughness.append(np.nan)

    return np.array(roughness)


if __name__ == '__main__':
    # pyinstaller.exe divide.py --contents-directory _torch_cpu; cp -r weights dist/divide

    input()

    temp_file_path = 'input/to_divide.npy'

    # === step 0 载入点云数据 ===
    t1 = time.time()
    positions = np.load(temp_file_path)
    t2 = time.time()

    # === step 1 构建BEV图像 ===
    bev_dir = './bev_dir'
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    scale = 0.7
    x_norm = ((x - x.min()) / scale).astype(np.int32)
    y_norm = ((y - y.min()) / scale).astype(np.int32)
    img_w, img_h = x_norm.max() + 1, y_norm.max() + 1
    y_norm_flip = img_h - 1 - y_norm

    Bev_pixels = img_w * img_h
    ratio_thresh = (700 * 600/Bev_pixels)*0.2

    bev = np.zeros((img_h, img_w), dtype=np.uint8)
    z_norm = ((z - z.min()) / (z.max() - z.min()) * 255).astype(np.uint8)
    for i in range(len(x_norm)):
        xx, yy = x_norm[i], y_norm_flip[i]
        bev[yy, xx] = max(bev[yy, xx], z_norm[i])

    out_prefix = os.path.splitext(os.path.basename(temp_file_path))[0]
    bev_img_path = os.path.join('output/divided', f"{out_prefix}.png")

    bev_img = Image.fromarray(bev).convert('RGB')
    bev_img.save(bev_img_path)
    t3 = time.time()

    # === step 2 FastSAM 推理 ===
    model = FastSAM("weights/FastSAM-x.pt")
    everything_results = model(
        bev_img,
        device='cpu',
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9    
    )
    t4 = time.time()

    # === step 3 掩码过滤 ===
    prompt_process = FastSAMPrompt(bev_img, everything_results, device='cpu')
    ann = prompt_process.everything_prompt()
    N, H, W = ann.shape
    total_pixels = H * W
    filtered_masks = []


    bev_gray = np.array(bev_img.convert("L"))
    for i in range(N):
        mask = ann[i]
        area = (mask > 0).sum().item()
        ratio = area / total_pixels
        if ratio <= ratio_thresh:
            mask_np = mask.cpu().numpy().astype(np.uint8)
            masked_region = bev_gray[mask_np == 1]
            if masked_region.size == 0:
                continue

            black_ratio = (masked_region < 30).sum() / masked_region.size
            if black_ratio >= 0.6:
                continue  # 黑色占比过高，跳过

            
            # --- 轮廓分析 ---
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            (_, _), (w, h), angle = rect
            rect_area = w * h
            contour_area = cv2.contourArea(cnt)
            fill_ratio = contour_area / rect_area if rect_area > 0 else 0
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

            # 添加形状合理性判断
            if fill_ratio > 0.5 and aspect_ratio < 5:
                filtered_masks.append(mask.unsqueeze(0))
    # 如果没有有效掩码，直接返回

    if filtered_masks:
        filtered_masks = [m.squeeze(0) if m.dim() == 3 else m for m in filtered_masks]
        ann = merge_overlapping_masks(filtered_masks, iou_thresh=0.5)
        
    else:
        ann = torch.empty((0, H, W), device=ann.device)

    
    t5 = time.time()

    if len(ann) == 0:
        print("No valid masks.",file=sys.stderr)
        exit(0)

    print(len(ann))

    # === step 4 掩码点云提取 + 保存 ===
    t_mask_start_all = time.time()
    point_pixels = np.stack([y_norm_flip, x_norm], axis=1)  # shape: (N, 2)

    mask_time_total = 0.0
    for mask_index in range(len(ann)):
        t_mask_start = time.time()

        mask_np = ann[mask_index].cpu().numpy().astype(bool)  # (H, W)
        yy = point_pixels[:, 0]  # 图像中的 Y
        xx = point_pixels[:, 1]  # 图像中的 X

        # 使用掩码直接筛选有效点索引
        valid_mask = mask_np[yy, xx]
        mask_indices = np.where(valid_mask)[0]

        if len(mask_indices) == 0:
            continue

        # 保存掩码点云
        masked_points = np.vstack((x[mask_indices], y[mask_indices], z[mask_indices])).T


        # 粗糙度仅用上方 75% 点计算
        z_values = masked_points[:, 2]
        z_thresh = np.percentile(z_values, 15)
        filtered_points = masked_points[z_values > z_thresh]

        if len(filtered_points) < 20:
            continue

        if len(filtered_points) > 5000:
            filtered_points = filtered_points[np.random.choice(len(filtered_points), 5000, replace=False)]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)

        roughness = compute_roughness_cloudcompare_style(pcd, radius=0.8, min_neighbors=10)
        mean_rough = np.mean(roughness)
        if mean_rough > 0.09:
            continue

        # 提取屋顶点和整栋建筑（屋顶点用于后续重建，而整栋建筑物用于展示）
        roof_building = extract_roof_by_z_density(masked_points)
        if roof_building is None:
            continue
        roof_points, building_points = roof_building

        if roof_points is None or len(roof_points) < 10:
            continue

        file_name = os.path.splitext(os.path.basename(temp_file_path))[0]
        out_file_path = os.path.join('output', f"divided/{mask_index}.npy")
        np.save(out_file_path,building_points)
        print(out_file_path)

        # out_file_path = os.path.join('output', f"divided/{mask_index}.npy")
        # np.save(out_file_path,roof_points)
        # print(out_file_path)

        t_mask_end = time.time()
        print(f"掩码 {mask_index} 处理时间: {t_mask_end - t_mask_start:.4f} 秒",file=sys.stderr)
        mask_time_total += (t_mask_end - t_mask_start)

