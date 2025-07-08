import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Gio

from typing import List
import subprocess as sp
import shutil
import numpy as np
import os
import pygfx as gfx

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
    
    def input(self, pcs : List[PointCloud]):
        if not pcs: return
        self.working_directory = working_directory = 'algorithm'
        building_input_dir = os.path.join(working_directory,'input','full_xyz_files')
        roof_input_dir = os.path.join(working_directory,'input','roof_xyz_files')
        self.building_output_dir = building_output_dir = os.path.join(working_directory,'output','complete_mesh')
        self.roof_output_dir = roof_output_dir = os.path.join(working_directory,'output','roof_wireframe')
        self.assessment_output_file = os.path.join(working_directory,'output','assessment_results_mesh_oabj.txt')

        shutil.rmtree(building_input_dir,ignore_errors=True)
        shutil.rmtree(roof_input_dir,ignore_errors=True)
        shutil.rmtree(building_output_dir,ignore_errors=True)
        shutil.rmtree(roof_output_dir,ignore_errors=True)
        os.makedirs(building_input_dir,exist_ok=True)
        os.makedirs(roof_input_dir,exist_ok=True)
        os.makedirs(building_output_dir,exist_ok=True)
        os.makedirs(roof_output_dir,exist_ok=True)
        
        for i, pc in enumerate(pcs):
            points = pc.geometry.positions.data + pc.local.position
            roof_points, building_points = extract_roof_by_z_density(points)
            if roof_points is None: continue
            np.savetxt(os.path.join(building_input_dir, f'{i}.xyz'), building_points)
            np.savetxt(os.path.join(roof_input_dir, f'{i}.xyz'), roof_points)           

        # pyinstaller --contents-directory _reconstruct --collect-all=scipy --collect-all=skimage
        p = sp.Popen([f'{working_directory}/reconstruct.exe'],cwd=working_directory)
        GLib.idle_add(self.reconstruct, len(pcs), p)

    def reconstruct(self,count: int, p : sp.Popen):
        if p.poll() is not None:
            self.close()
            return

        i = len([e.name for e in os.scandir(self.building_output_dir) if e.is_file()])

        fraction = i / count
        self.progress.set_fraction(fraction)
        # if i >= count:
        #     self.close()
        #     return

        GLib.idle_add(self.reconstruct, count,p)

    def output(self):
        assessment = dict()
        with open(self.assessment_output_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line: break
                parts = line.split(':')
                index = parts[0].strip()
                value = float(parts[1].strip()) 
                assessment[index] = value

        objs = []
        for file_path in [e.path for e in os.scandir(self.building_output_dir) if e.is_file()]:
            print(file_path)
            index = os.path.splitext(os.path.basename(file_path))[0]
            mesh = gfx.load_mesh(file_path)[0]

            pc = mesh.geometry.positions.data
            x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
            offset = [(x.max()-x.min())/2,(y.max()-y.min())/2,0]
            origin = np.array([np.min(x),np.min(y),0])
            x = x - np.min(x)
            y = y - np.min(y)
            z = 0
            pc = np.column_stack([x,y,z]) - [(x.max()-x.min())/2,(y.max()-y.min())/2,0]
            mesh.geometry.positions.data[:] = pc.astype(np.float32)
            
            building = Building()
            building.geometry = mesh.geometry
            building.material = mesh.material
            building.local.position = origin + offset
            building.update_assessment(assessment[index])
            objs.append(building)
        return objs


from scipy.signal import find_peaks

#提取屋顶
def extract_roof_by_z_density(points, bin_size=0.2, top_percent=0.7,
                            peak_prominence=0.1, max_peaks=10, extend_below=1.0):
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