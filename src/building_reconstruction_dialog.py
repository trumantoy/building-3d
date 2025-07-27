import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Gio

from typing import List
import subprocess as sp
import shutil
import numpy as np
import os
import pygfx as gfx
import threading
import time
import io

from simtoy import *

class BuildingReconstructionDialog (Gtk.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.progress = Gtk.ProgressBar()
        self.progress.set_fraction(0)  # 设置50%进度
        
        self.set_child(self.progress)
        self.set_resizable(False)
        self.progress.set_valign(Gtk.Align.CENTER)
        self.progress.set_margin_start(20)
        self.progress.set_margin_end(20)
        self.progress.set_margin_top(20)
        self.progress.set_margin_bottom(20)

        bar = Gtk.HeaderBar()
        title_label = Gtk.Label()
        title_label.set_text('点云重建')
        bar.set_title_widget(title_label)
        
        bar.set_show_title_buttons(False)
        self.set_titlebar(bar)

        self.working_dir = 'algorithm'

        # pyinstaller reconstruct.py --contents-directory _reconstruct --collect-all=scipy --collect-all=skimage --exclude-module=cupy; mkdir dist/reconstruct/input; cp -r input/pth_files dist/reconstruct/input
        self.process = sp.Popen([f'{self.working_dir}/reconstruct.exe'],stdin=sp.PIPE,encoding='utf-8',text=True,cwd=self.working_dir)



    def __del__(self):
        self.process.kill()

    def input(self, obj : PointCloud):
        if not obj: return
        self.src_obj = obj 
        self.thread = threading.Thread(target=self.reconstructing, args=[obj])
        self.thread.start()
    
    def reconstructing(self,obj):
        building_input_dir = os.path.join(self.working_dir,'input','full_xyz_files')
        roof_input_dir = os.path.join(self.working_dir,'input','roof_xyz_files')
        self.building_output_dir = building_output_dir = os.path.join(self.working_dir,'output','complete_mesh')
        self.roof_output_dir = roof_output_dir = os.path.join(self.working_dir,'output','roof_mesh')

        shutil.rmtree(building_input_dir,ignore_errors=True)
        shutil.rmtree(roof_input_dir,ignore_errors=True)
        shutil.rmtree(building_output_dir,ignore_errors=True)
        shutil.rmtree(roof_output_dir,ignore_errors=True)
        os.makedirs(building_input_dir,exist_ok=True)
        os.makedirs(roof_input_dir,exist_ok=True)
        os.makedirs(building_output_dir,exist_ok=True)
        os.makedirs(roof_output_dir,exist_ok=True)
        self.positions = dict()
        
        count = 0
        for i, sub_obj in enumerate(obj.children):
            if type(sub_obj) != PointCloud:
                continue
            name = os.path.splitext(sub_obj.name)[0]
            self.positions[name] = sub_obj.local.position
            count+=1
            points = sub_obj.geometry.positions.data
            roof_points, building_points = extract_roof_by_z_density(points)
            if roof_points is None: continue
            np.savetxt(os.path.join(building_input_dir, f'{name}.xyz'), building_points)
            np.savetxt(os.path.join(roof_input_dir, f'{name}.xyz'), roof_points)

        self.process.stdin.write(f'\n')
        self.process.stdin.flush()

        i = 1
        while self.process.poll() is None:
            i = len([e.name for e in os.scandir(self.building_output_dir) if e.is_file()])
            fraction = i / count
            self.progress.set_fraction(fraction)
            time.sleep(1)


        self.objs = []
        for file_path in [e.path for e in os.scandir(self.building_output_dir) if e.is_file()]:
            mesh = gfx.load_mesh(file_path)[0]

            pc = mesh.geometry.positions.data
          
            mesh.geometry.positions.data[:] = pc.astype(np.float32)
            mesh.material.side = gfx.VisibleSide.both
            mesh.material.color = (0.8, 0.8, 0.8)  # 设置材质颜色为白色
            mesh.material.shininess = 0  # 降低高光强度
            mesh.material.specular = (0.0, 0.0, 0.0, 1.0)  # 降低高光色
            mesh.material.emissive = (0.8, 0.8, 0.8)  # 设置微弱自发光
            mesh.material.flat_shading = True  # 启用平面着色
            mesh.material.pick_write = True

            building = Building()
            building.geometry = mesh.geometry
            building.material = mesh.material
            building.name = os.path.basename(file_path)
            name = os.path.splitext(building.name)[0]
            building.local.position = self.positions[name]
            roof_file_path = os.path.join(self.roof_output_dir, f'{name}.obj')

            if not os.path.exists(roof_file_path):
                print(file_path, 'Roof file not found:', roof_file_path)
                building.roof_mesh_content = None
                continue

            with open(roof_file_path, 'rb') as f:
                roof_content = f.read()
                building.roof_mesh_content = roof_content
            
            self.src_obj.add(building)
            self.objs.append(building)

        self.process = sp.Popen([f'{self.working_dir}/reconstruct.exe'],stdin=sp.PIPE,encoding='utf-8',text=True,cwd=self.working_dir)
        GLib.idle_add(self.unmap)

    def output(self):
        return self.objs


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