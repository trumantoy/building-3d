import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Gio

from typing import List
import subprocess as sp
import numpy as np
import os
import pygfx as gfx
import threading
import shutil
from simtoy import *

class BuildingDivisionDialog (Gtk.Window):
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
        title_label.set_text('点云分割')
        bar.set_title_widget(title_label)
        
        bar.set_show_title_buttons(False)
        self.set_titlebar(bar)
    
    def input(self, obj):
        self.result = []
        self.src_obj = obj
        pc = obj.geometry.positions.data + obj.local.position


        self.progress.set_fraction(0)

        # working_directory = '../pc_seg'
        working_directory = 'algorithm'

        input_file =  os.path.join(working_directory,'input','to_divide.npy')
        output_dir = os.path.join(working_directory,'output','divided')
        
        if os.path.exists(input_file): os.remove(input_file)
        shutil.rmtree(output_dir,ignore_errors=True)
        os.makedirs(output_dir,exist_ok=True)

        np.save(input_file, pc)

        # pyinstaller.exe divide.py --contents-directory _divide; mkdir dist/divide/input; cp -r weights dist/divide/input
        p = sp.Popen([f'{working_directory}/divide.exe'],stdout=sp.PIPE,encoding='utf-8',text=True,cwd=working_directory)

        self.working_directory = working_directory

        self.stdout_thread = threading.Thread(target=self.divide,args=[p])
        self.stdout_thread.start()


    def divide(self,p : sp.Popen):
        count = None
        j = 1

        while p.poll() is None:
            line = p.stdout.readline().strip()
            if not line: continue
            if not count:
                count = int(line)
                continue
            
            j += 1
            fraction = j / count
            self.progress.set_fraction(fraction)
            file_path = os.path.join(self.working_directory,line)
            print(file_path)
            positions = np.load(file_path).astype(np.float32)
            self.result.append([positions,file_path])

        GLib.idle_add(self.close)

    def output(self):
        self.src_obj.material.opacity = 0
        
        pc = np.vstack([i[0] for i in self.result])

        z = pc[:,2]
        z_min = np.min(z)
        z_max = np.max(z)

        import colorsys  # 导入 colorsys 模块用于 HSV 到 RGB 的转换
        objs = []
        for pc,file_path in self.result:
            z = pc[:,2]
            # 归一化 z 坐标
            z_normalized = (z - z_min) / (z_max - z_min) if z_max != z_min else 0.5
            # 绿色色相为 120/360，红色色相为 0，根据 z 坐标线性插值
            hsv_hues = 240 / 360 * (1 - z_normalized)
            # 固定饱和度和明度
            saturation = 1.0
            value = 1.0
            # 转换为 RGB 颜色
            colors = np.array([colorsys.hsv_to_rgb(h, saturation, value) for h in hsv_hues], dtype=np.float32)

            geometry = gfx.Geometry(positions=pc, colors=colors)
            material = gfx.PointsMaterial(color_mode="vertex", size=1)
            points = PointCloud(geometry,material)
            points.name = os.path.basename(file_path)
            
            objs.append(points)

        self.src_obj.add(*objs)
        return objs