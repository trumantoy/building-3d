import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Gio

from typing import List
import subprocess as sp
import numpy as np
import os

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
    
    def input(self, pcs : List[np.ndarray]):
        if not pcs: return
        working_directory = 'algorithm2'
        input_dir = os.path.join(working_directory,'data','tallinn','raw','test','xyz')
        idx_input_dir = os.path.join(working_directory,'data','tallinn','processed','test_list.txt')

        with open(idx_input_dir,mode='w') as f:
            for i, pc in enumerate(pcs):
                f.write(f'{i}\n')
                np.savetxt(os.path.join(input_dir, f'{i}.xyz'), pc)

        # pyinstaller --collect-all=scipy --collect-all=skimage
        p = sp.Popen([f'{working_directory}/reconstruct.exe'],stdin=sp.PIPE,stdout=sp.PIPE,encoding='utf-8',text=True,cwd=working_directory)
        GLib.idle_add(self.reconstruct, len(pcs), 1, p)

    def reconstruct(self,count: int, i: int, p : sp.Popen):
        line = p.stdout.readline().strip()
        if not line and p.poll() is not None:
            self.close()
            return

        if not line:        
            GLib.idle_add(self.reconstruct, count,i,p)
            return

        print(line)

        fraction = i / count
        self.progress.set_fraction(fraction)
        GLib.idle_add(self.reconstruct, count,i+1,p)

    def output(self):
        dir_path = 'algorithm2/experiments/result/tallinn/checkpoint_guangxi_ephs700'
        
        wireframes = []

        # 遍历目录下的所有条目
        for entry in os.listdir(dir_path):
            file_path = os.path.join(dir_path, entry)
            
            vertices = []
            edges = []
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        vertices.append(list(map(float, line.split()[1:4])))
                    elif line.startswith('l '):
                        edges.append([int(index) - 1 for index in line.split()[1:3]])
            
            wireframes.append((vertices,edges))
        return wireframes