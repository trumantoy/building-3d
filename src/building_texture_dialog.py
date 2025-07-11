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

class BuildingTextureDialog (Gtk.Window):
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
        title_label.set_text('模型纹理')
        bar.set_title_widget(title_label)
        bar.set_show_title_buttons(False)      
        self.set_titlebar(bar)
    
        self.connect('map', self.on_map)

    def input(self,obj):
        self.src_obj = obj
        self.thread = threading.Thread(target=self.texturing,args=[obj])

    def on_map(self,*args):
        self.thread.start()
    
    def texturing(self,src_obj):
        working_directory = 'algorithm'
        building_input_dir = os.path.join(working_directory,'input','obj')
        self.building_output_dir = building_output_dir = os.path.join(working_directory,'output','final_output')

        shutil.rmtree(building_input_dir,ignore_errors=True)
        shutil.rmtree(building_output_dir,ignore_errors=True)

        os.makedirs(building_input_dir,exist_ok=True)
        os.makedirs(building_output_dir,exist_ok=True)

        for i, sub_obj in enumerate(src_obj.children):
            if type(sub_obj) == Building:
                positions = sub_obj.geometry.positions.data + sub_obj.local.position
                faces = sub_obj.geometry.indices.data if sub_obj.geometry.indices is not None else None
                tm = trimesh.Trimesh(vertices=positions, faces=faces)
                name = os.path.splitext(sub_obj.name)[0]
                tm.export(os.path.join(building_input_dir, f'{name}.obj'))

        # pyinstaller.exe texture.py --contents-directory _texture
        p = sp.Popen([f'{working_directory}/texture.exe'],cwd=working_directory)
        p.wait()

        GLib.idle_add(self.close)

    def output(self):
        objs = []
        for file_path in [e.path for e in os.scandir(self.building_output_dir) if e.is_file()]:
            if not file_path.endswith('.obj'):
                continue

            mesh = gfx.load_mesh(file_path)[0]

            pc = mesh.geometry.positions.data
            x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
            offset = [(x.max()-x.min())/2,(y.max()-y.min())/2,0]
            origin = np.array([np.min(x),np.min(y),0])
            x = x - np.min(x)
            y = y - np.min(y)
            z = z
            pc = np.column_stack([x,y,z]) - [(x.max()-x.min())/2,(y.max()-y.min())/2,0]
            mesh.geometry.positions.data[:] = pc.astype(np.float32)

            name = os.path.basename(file_path)
            for sub_obj in self.src_obj.children:
                if type(sub_obj) == Building and sub_obj.name == name:
                    self.src_obj.remove(sub_obj)
                    break

            mesh.material = gfx.MeshBasicMaterial(map=mesh.material.map, side=gfx.VisibleSide.both)

            building = Building()
            building.geometry = mesh.geometry
            building.material = mesh.material
            building.local.position = origin + offset
            building.name = name

            self.src_obj.add(building)
            objs.append(building)

        return objs
