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
    
    def input(self, pcs : List[PointCloud]):
        if not pcs: return
        self.points = []

        positions = []
        for pc in pcs: 
            positions.append(pc.geometry.positions.data + pc.local.position)
        pc = np.vstack(positions).astype(np.float32)

        self.progress.set_fraction(0)

        # working_directory = '../pc_seg'
        working_directory = 'algorithm'
        
        input_file = 'input/to_divide.npy'
        np.save(os.path.join(working_directory, input_file), pc)

        # pyinstaller.exe divide.py --contents-directory _divide
        # p = sp.Popen(["c:/Users/SLTru/AppData/Local/Programs/Python/Python312/python.exe","../pc_seg/divide.py",temp_file_path],stdin=sp.PIPE,stdout=sp.PIPE,encoding='utf-8', cwd=working_directory)
        p = sp.Popen([f'{working_directory}/divide.exe',input_file],stdin=sp.PIPE,stdout=sp.PIPE,encoding='utf-8',text=True,cwd=working_directory)
        
        self.working_directory = working_directory
        GLib.idle_add(self.divide, p, None, 1)

    def divide(self,p : sp.Popen, count, j):
        line = p.stdout.readline().strip()
        if not line and p.poll() is not None:         
            self.close()
            return

        if not line:        
            GLib.idle_add(self.divide, p,count,j)
            return
        
        if not count: 
            count = int(line)
            GLib.idle_add(self.divide, p,count,j)
            return
        
        file_path = os.path.join(self.working_directory,line)
        print(file_path)
        
        fraction = j / count
        self.progress.set_fraction(fraction)

        positions = np.load(file_path)
        self.points.append(positions)
        GLib.idle_add(self.divide, p,count,j+1)

    def output(self):
        return self.points