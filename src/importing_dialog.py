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

from simtoy import *

class ImportingDialog (Gtk.Window):
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
        title_label.set_text('导入中')
        bar.set_title_widget(title_label)
        bar.set_show_title_buttons(False)      
        self.set_titlebar(bar)
    
        self.connect('map', self.on_map)

    def input(self,file_path):
        self.thread = threading.Thread(target=self.importing,args=[file_path])
        self.content = None

    def output(self):
        return self.content
    
    def importing(self,file_path):
        content = PointCloud()
        content.set_from_file(file_path)
        content.name = os.path.basename(file_path)
        self.content = content
        GLib.idle_add(self.close)

    def on_map(self,*args):
        self.thread.start()
    