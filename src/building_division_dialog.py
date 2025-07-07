import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Gio

from typing import List
import subprocess as sp
import tempfile
import numpy as np
import os
import pygfx as gfx

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
    
    def set_point_cloud(self, pcs : List[PointCloud]):
        if not pcs: return
        self.points = []

        GLib.idle_add(self.task, pcs, 0)

    def task(self,pcs : List[PointCloud],i):
        if i == len(pcs): 
            self.close()
            return

        fraction = i / len(pcs)
        self.progress.set_fraction(fraction)

        pc = pcs[i]
        working_directory = '../pc_seg'
        temp_file_path = os.path.join(working_directory, 'input', f'{i}.npy')

        data = pc.points.geometry.positions.data + pc.local.position
        np.save(temp_file_path, data.astype(np.float32))

        p = sp.Popen(["c:/Users/SLTru/AppData/Local/Programs/Python/Python312/python.exe","../pc_seg/divide.py",temp_file_path],stdin=sp.PIPE,stdout=sp.PIPE,encoding='utf-8', cwd=working_directory)
        
        # working_directory = 'algorithm'
        # p = sp.Popen([f'{working_directory}/divide.exe',temp_file_path],stdin=sp.PIPE,stdout=sp.PIPE,encoding='utf-8',text=True,cwd=working_directory)
        
        self.working_directory = working_directory
        GLib.idle_add(self.divide, pcs, i, p, None, 1)

    def divide(self,pcs : List[np.ndarray], i,p : sp.Popen, count,j):
        line = p.stdout.readline().strip()
        if not line and p.poll() is not None:         
            GLib.idle_add(self.task, pcs, i+1)
            return

        if not line:        
            GLib.idle_add(self.divide, pcs,i,p,count,j)
            return

        print(line)
        
        if not count: 
            count = int(line)
            GLib.idle_add(self.divide, pcs,i,p,count,j)
            return
        
        fraction = i / len(pcs) + ((j / count) / 10)
        self.progress.set_fraction(fraction)

        self.points.append(np.load(os.path.join(self.working_directory,line)))

        GLib.idle_add(self.divide, pcs,i,p,count,j+1)

    def get_point_cloud(self):
        return self.points