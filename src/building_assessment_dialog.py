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
import trimesh
from simtoy import *

class BuildingAssessmentDialog (Gtk.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.progress = Gtk.Spinner()
        self.progress.start()

        self.set_child(self.progress)
        self.set_resizable(False)
        self.set_size_request(200,100)
        self.progress.set_valign(Gtk.Align.CENTER)
        self.progress.set_margin_start(20)
        self.progress.set_margin_end(20)
        self.progress.set_margin_top(20)
        self.progress.set_margin_bottom(20)

        bar = Gtk.HeaderBar()
        title_label = Gtk.Label()
        title_label.set_text('评估')
        bar.set_title_widget(title_label)
        bar.set_show_title_buttons(False)      
        self.set_titlebar(bar)
    
        self.connect('map', self.on_map)

    def input(self,obj):
        self.src_obj = obj
        self.thread = threading.Thread(target=self.assessing,args=[obj])

    def on_map(self,*args):
        self.thread.start()
    
    def assessing(self,src_obj):
        working_directory = 'algorithm'
        mesh_input_dir = os.path.join(working_directory,'input','roof_mesh')
        points_input_dir = os.path.join(working_directory,'input','roof_points')
        self.assessment_output_file = assessment_output_file = os.path.join(working_directory,'output','assessment_results_mesh_oabj.txt')

        shutil.rmtree(mesh_input_dir,ignore_errors=True)
        shutil.rmtree(points_input_dir,ignore_errors=True)
        if os.path.exists(assessment_output_file): os.remove(assessment_output_file)

        os.makedirs(mesh_input_dir,exist_ok=True)
        os.makedirs(points_input_dir,exist_ok=True)

        for i, sub_obj in enumerate(src_obj.children):
            if type(sub_obj) == PointCloud:
                points = sub_obj.geometry.positions.data + sub_obj.local.position
                roof_points, building_points = extract_roof_by_z_density(points)
                if roof_points is None: continue
                name = os.path.splitext(sub_obj.name)[0]
                np.savetxt(os.path.join(points_input_dir, f'{name}.xyz'), roof_points)
            elif type(sub_obj) == Building:
                name = os.path.splitext(sub_obj.name)[0]
                if sub_obj.roof_mesh_content:            
                    with open(os.path.join(mesh_input_dir, f'{name}.obj'), 'w') as f:
                        f.write(sub_obj.roof_mesh_content.getvalue())
                    
        # pyinstaller.exe assess.py
        p = sp.Popen([f'{working_directory}/assess.exe'],cwd=working_directory)
        p.wait()

        GLib.idle_add(self.close)

    def output(self):
        assessment = dict()
        if not os.path.exists(self.assessment_output_file):
            return assessment
            
        with open(self.assessment_output_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if not line: break
                parts = line.split(':')
                index = parts[0].strip()
                try:
                    value = float(parts[1].strip()) 
                except ValueError:
                    continue
                assessment[index] = value

        for sub_obj in self.src_obj.children:
            if type(sub_obj) != Building:
                continue
            
            name = os.path.splitext(sub_obj.name)[0]
            if name in assessment:
                sub_obj.update_assessment(assessment[name])

        return assessment


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