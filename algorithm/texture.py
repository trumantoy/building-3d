import os
import numpy as np
import random
from PIL import Image
import shutil

# ─── 解析 OBJ 文件 ──────────
def parse_obj_keep_faces(obj_path):
    vertices = []
    faces = []

    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = []
                for part in parts:
                    idx = int(part.split("/")[0])
                    face.append(idx - 1)
                faces.append(face)

    return np.array(vertices), faces

# ─── 计算法向量 ──────────
def compute_normal(vs):
    v0, v1, v2 = vs[0], vs[1], vs[2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 0 else normal

# ─── 每个面独立 UV 坐标 ──────────
def face_uv_fill(face_vertices):
    n = len(face_vertices)
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    if n == 3:
        return uvs[:3]
    elif n == 4:
        return uvs
    else:
        indices = np.linspace(0, 3, n, dtype=int)
        return uvs[indices]

# === 配置 ===
obj_dir = "input/obj"
output_dir = "output/final_output"
facade_img_dir = "material/facade"
roof_img_dir = "material/roof"

os.makedirs(output_dir, exist_ok=True)

# 将所有材质图片复制到 output_dir（只复制一次）
facade_imgs = os.listdir(facade_img_dir)
roof_imgs = os.listdir(roof_img_dir)

used_facade_imgs = set()
used_roof_imgs = set()

for img_name in facade_imgs:
    dst_path = os.path.join(output_dir, img_name)
    if not os.path.exists(dst_path):
        shutil.copy(os.path.join(facade_img_dir, img_name), dst_path)

for img_name in roof_imgs:
    dst_path = os.path.join(output_dir, img_name)
    if not os.path.exists(dst_path):
        shutil.copy(os.path.join(roof_img_dir, img_name), dst_path)

# 遍历 OBJ 文件夹
for idx, file in enumerate(os.listdir(obj_dir)):
    name = os.path.splitext(file)[0]
    if file.endswith(".obj"):
        obj_file = os.path.join(obj_dir, file)
        vertices, faces = parse_obj_keep_faces(obj_file)

        # 随机选择 facade 和 roof 贴图
        chosen_facade = random.choice(facade_imgs)
        chosen_roof = random.choice(roof_imgs)

        used_facade_imgs.add(chosen_facade)
        used_roof_imgs.add(chosen_roof)

        roof_faces = []
        facade_faces = []

        for face in faces:
            vs = vertices[face]
            n = compute_normal(vs)
            if n[2] < 0.01:
                facade_faces.append(face)
            else:
                roof_faces.append(face)

        print(f"✅ 文件 {file} → 屋顶面: {len(roof_faces)}, 立面面: {len(facade_faces)}")
        print(f"   使用 facade: {chosen_facade}, roof: {chosen_roof}")

        # 写当前 MTL 文件
        mtl_name = f"{name}.mtl"
        mtl_path = os.path.join(output_dir, mtl_name)

        with open(mtl_path, "w") as f:
            f.write("newmtl material_facade\n")
            f.write(f"map_Kd {chosen_facade}\n\n")
            f.write("newmtl material_roof\n")
            f.write(f"map_Kd {chosen_roof}\n")

        # 写当前 OBJ 文件
        obj_out_path = os.path.join(output_dir, f"{name}.obj")
        with open(obj_out_path, "w") as f:
            f.write(f"mtllib {mtl_name}\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            vt_index = [1]

            def write_faces(faces_list, material_name):
                for face in faces_list:
                    f.write(f"usemtl {material_name}\n")
                    vs = vertices[face]
                    uvs = face_uv_fill(vs)

                    face_vt_indices = []
                    for uv in uvs:
                        f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
                        face_vt_indices.append(vt_index[0])
                        vt_index[0] += 1

                    f.write("f")
                    for i, idx_v in enumerate(face):
                        f.write(f" {idx_v+1}/{face_vt_indices[i]}")
                    f.write("\n")

            write_faces(roof_faces, "material_roof")
            write_faces(facade_faces, "material_facade")

print("🎉✅ 所有 OBJ 文件处理完成，每个 OBJ 独立 mtl，统一材质库，图片不重复保存！")
