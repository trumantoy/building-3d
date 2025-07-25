import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Gio

import time
import cairo
import numpy as np
import pygfx as gfx
from pathlib import Path
import os
import sys
import shutil
import io
import trimesh
import zipfile
import tempfile

from simtoy import *
from panel import *
from bar import *

@Gtk.Template(filename='ui/app_window.ui')
class AppWindow (Gtk.ApplicationWindow):
    __gtype_name__ = "AppWindow"

    stack : Gtk.Stack = Gtk.Template.Child('panel')
    widget : Gtk.DrawingArea = Gtk.Template.Child('widget')
    actionbar : Actionbar = Gtk.Template.Child('actionbar')
    hotbar : Hotbar = Gtk.Template.Child('hotbar')
    viewbar : Viewbar = Gtk.Template.Child('viewbar')

    def __init__(self):
        self.editor = Editor()
        self.ortho_camera = gfx.OrthographicCamera()
        self.ortho_camera.local.position = [0,0,10]
        self.ortho_camera.show_pos([0,0,0],up=[0,0,1])

        self.persp_camera = gfx.PerspectiveCamera()
        self.persp_camera.local.position = [0,0,10]
        self.persp_camera.show_pos([0,0,0],up=[0,0,1])

        self.canvas = wgpu.gui.offscreen.WgpuCanvas(size=(1024,768))
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.view_controller = gfx.OrbitController()
        self.view_controller.register_events(self.renderer)
        self.view_controller.add_camera(self.persp_camera)
        self.view_controller.add_camera(self.ortho_camera)

        self.panel : Panel = self.stack.get_visible_child()
        self.widget.set_draw_func(self.draw, self.editor)
        self.viewbar.set_controller(self.view_controller)
        self.hotbar.set_viewbar(self.widget, self.viewbar, self.panel, self.editor)
        self.panel.set_viewbar(self.viewbar)

        zoom_controller = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags(Gtk.EventControllerScrollFlags.VERTICAL))
        zoom_controller.connect("scroll", lambda sender,dx,dy: self.renderer.dispatch_event(gfx.WheelEvent('wheel',dx=0.0,dy=dy*100,x=0,y=0,time_stamp=time.perf_counter())))
        
        # click_controller = Gtk.GestureClick.new()
        # click_controller.set_button(1)
        # click_controller.connect("pressed", lambda sender,n_press,x,y: self.renderer.dispatch_event(gfx.PointerEvent('pointer_down',x=x ,y=y,button=3,buttons=(3,),time_stamp=time.perf_counter())))
        # click_controller.connect("released", lambda sender,n_press,x,y: self.renderer.dispatch_event(gfx.PointerEvent('pointer_up',x=x ,y=y,button=3,buttons=(3,),time_stamp=time.perf_counter())))

        rotation_controller = Gtk.GestureClick.new()
        rotation_controller.set_button(2)
        rotation_controller.connect("pressed", lambda sender,n_press,x,y: self.renderer.dispatch_event(gfx.PointerEvent('pointer_down',x=x ,y=y,button=1,buttons=(1,),time_stamp=time.perf_counter())))
        rotation_controller.connect("released", lambda sender,n_press,x,y: self.renderer.dispatch_event(gfx.PointerEvent('pointer_up',x=x ,y=y,button=1,buttons=(1,),time_stamp=time.perf_counter())))

        pan_controller = Gtk.GestureClick.new()
        pan_controller.set_button(3)
        pan_controller.connect("pressed", lambda sender,n_press,x,y: self.renderer.dispatch_event(gfx.PointerEvent('pointer_down',x=x,y=y,button=2,buttons=(2,),time_stamp=time.perf_counter())))
        pan_controller.connect("released", lambda sender,n_press,x,y: self.renderer.dispatch_event(gfx.PointerEvent('pointer_up',x=x,y=y,button=2,buttons=(2,),time_stamp=time.perf_counter())))

        motion_controller = Gtk.EventControllerMotion()
        motion_controller.connect("motion", lambda sender,x,y: self.renderer.dispatch_event(gfx.PointerEvent('pointer_move',x=x ,y=y,time_stamp=time.perf_counter())))

        if rotation_controller: self.add_controller(rotation_controller)
        if pan_controller: self.add_controller(pan_controller)
        if zoom_controller: self.add_controller(zoom_controller)
        if motion_controller: self.add_controller(motion_controller)

        action = Gio.SimpleAction.new('import', None)
        action.connect('activate', self.file_import)
        self.add_action(action)

        action = Gio.SimpleAction.new('export', None)
        action.connect('activate', self.file_export)
        self.add_action(action)

        action = Gio.SimpleAction.new('close', None)
        action.connect('activate', self.file_close)
        self.add_action(action)

        action = Gio.SimpleAction.new('file_add', None)
        action.connect('activate', self.file_add)
        self.add_action(action)

        action = Gio.SimpleAction.new('building_divide', None)
        action.connect('activate', self.building_divide)
        self.add_action(action)

        action = Gio.SimpleAction.new('building_reconstruct', None)
        action.connect('activate', self.building_reconstruct)
        self.add_action(action)

        action = Gio.SimpleAction.new('building_assess', None)
        action.connect('activate', self.building_assess)
        self.add_action(action)

        action = Gio.SimpleAction.new('building_texture', None)
        action.connect('activate', self.building_texture)
        self.add_action(action)

        from building_division_dialog import BuildingDivisionDialog
        self.division_dlg = BuildingDivisionDialog()
        self.division_dlg.set_transient_for(self)  # 设置父窗口
        
        from building_reconstruction_dialog import BuildingReconstructionDialog
        self.reconstruction_dlg = BuildingReconstructionDialog()
        self.reconstruction_dlg.set_transient_for(self)  # 设置父窗口

        from building_assessment_dialog import BuildingAssessmentDialog
        self.assessment_dlg = BuildingAssessmentDialog()
        self.assessment_dlg.set_transient_for(self)  # 设置父窗口

    def draw(self,receiver, cr, area_w, area_h, editor : Editor):
        width,height = self.canvas.get_physical_size()

        if width != area_w or height != area_h: 
            self.canvas = wgpu.gui.offscreen.WgpuCanvas(size=(area_w,area_h))
            self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
            self.view_controller.register_events(self.renderer)

        if '透视' == self.viewbar.view_mode.get_label():
            self.renderer.render(self.editor, self.persp_camera)
        else:
            self.renderer.render(self.editor, self.ortho_camera)
        
        img : np.ndarray = np.asarray(self.canvas.draw())
        img_h,img_w,img_ch = img.shape
        img = np.asarray(img[..., [2, 1, 0, 3]]).copy()
        
        stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, img_w)
        surface = cairo.ImageSurface.create_for_data(img.data, cairo.FORMAT_ARGB32, img_w, img_h, stride)
        cr.set_source_surface(surface, 0, 0)
        cr.paint()

        self.editor.step()
        GLib.idle_add(receiver.queue_draw)

    def file_import(self, sender, args):
        dialog = Gtk.FileDialog()
        dialog.set_modal(True)

        filter_text = Gtk.FileFilter()
        filter_text.set_name("ZIP 文件")
        filter_text.add_pattern("*.zip")
        
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(filter_text)
        dialog.set_filters(filters)
        dialog.set_default_filter(filter_text)

        def open_file(dialog, result): 
            file_path = None
            try:
                file = dialog.open_finish(result)
                file_path = file.get_path()
            except:
                return

            from import_dialog import ImportDialog
            import_dlg = ImportDialog()
            import_dlg.set_transient_for(self)  # 设置父窗口
            import_dlg.set_modal(True)
            import_dlg.input(file_path,self.editor,self.panel)
            import_dlg.present()

        dialog.open(None, None, open_file) 

    def file_export(self,sender, *args):
        dialog = Gtk.FileDialog()
        dialog.set_modal(True)

        filter_text = Gtk.FileFilter()
        filter_text.set_name("ZIP 文件")
        filter_text.add_pattern("*.zip")
        
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(filter_text)
        dialog.set_filters(filters)
        dialog.set_default_filter(filter_text)

        def save_file(dialog, result):
            file_path = None

            try:
                file = dialog.save_finish(result)
                file_path = file.get_path()       
                file_name = file.get_basename()         
            except:
                return
            
            from export_dialog import ExportDialog
            export_dlg = ExportDialog()
            export_dlg.set_transient_for(self)  # 设置父窗口
            export_dlg.set_modal(True)
            export_dlg.input(file_path,self.editor,self.panel)
            export_dlg.present()
                
        dialog.save(None, None, save_file)


    def file_close(self,sender, *args):
        for i,item in enumerate(self.panel.model):
            self.editor.remove(item.obj)
        self.panel.model.remove_all()

    def file_add(self, sender, args):
        dialog = Gtk.FileDialog()
        dialog.set_modal(True)

        filter_text = Gtk.FileFilter()
        filter_text.set_name("点云")
        filter_text.add_pattern('*.las')
        filter_text.add_pattern('*.ply')
        
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(filter_text)
        dialog.set_filters(filters)
        dialog.set_default_filter(filter_text)

        def select_file(dialog, result):
            file_path = None

            try:
                file = dialog.open_finish(result)
                file_path = file.get_path()
            except:
                return
           
            from importing_dialog import ImportingDialog
            dlg = ImportingDialog()
            dlg.input(file_path)
                    
            def do_close_request(win):
                obj = dlg.output()
                self.editor.add(obj)
                self.panel.add(obj)

            dlg.connect('close_request', do_close_request)
            dlg.set_modal(True)  # 设置为模态窗口
            dlg.set_transient_for(self.get_root())  # 设置父窗口

            dlg.present()

        dialog.open(None, None, select_file)
    
    def building_divide(self,sender, *args):
        i = self.panel.selection_model.get_selected()
        if i == Gtk.INVALID_LIST_POSITION:
            return

        tree_row = self.panel.selection_model.get_item(i)
        if tree_row.get_depth():
            return

        item = tree_row.get_item()
        if item.model.get_n_items():
            return
        
        self.division_dlg.input(item.obj)
        
        def do_close_request(win):
            self.panel.add_sub(item,self.division_dlg.output())
            self.division_dlg.set_modal(False)  # 设置为模态窗口

        if 'division_dlg_slot_id' in vars(self):
            self.division_dlg.disconnect(self.division_dlg_slot_id)

        self.division_dlg_slot_id = self.division_dlg.connect('unmap', do_close_request)
        self.division_dlg.set_modal(True)  # 设置为模态窗口
        self.division_dlg.present()
        self.division_dlg.map()

    def building_reconstruct(self,sender, *args):
        i = self.panel.selection_model.get_selected()
        if i == Gtk.INVALID_LIST_POSITION:
            return

        tree_row = self.panel.selection_model.get_item(i)
        if tree_row.get_depth():
            return

        item = tree_row.get_item()
        if 0 == item.model.get_n_items():
            return

        self.reconstruction_dlg.input(item.obj)

        def do_close_request(win):
            self.reconstruction_dlg.set_modal(False)  # 设置为模态窗口
            self.panel.add_sub(item,self.reconstruction_dlg.output())
            
            for i, sub_obj in enumerate(item.obj.children):
                if type(sub_obj) != PointCloud:
                    continue
                sub_obj.material.opacity = 0

        if 'reconstruction_dlg_slot_id' in vars(self):
            self.reconstruction_dlg.disconnect(self.reconstruction_dlg_slot_id)

        self.reconstruction_dlg_slot_id = self.reconstruction_dlg.connect('unmap', do_close_request)
        self.reconstruction_dlg.set_modal(True)  # 设置为模态窗口
        self.reconstruction_dlg.present()
        self.reconstruction_dlg.map()

    def building_assess(self,sender, *args):
        i = self.panel.selection_model.get_selected()
        if i == Gtk.INVALID_LIST_POSITION:
            return

        tree_row = self.panel.selection_model.get_item(i)
        if tree_row.get_depth():
            return

        item = tree_row.get_item()
        if 0 == item.model.get_n_items():
            return

        self.assessment_dlg.input(item.obj)

        def do_close_request(win):
            self.assessment_dlg.set_modal(False)  # 设置为模态窗口
            self.assessment_dlg.output()

        if 'assessment_dlg_slot_id' in vars(self):
            self.assessment_dlg.disconnect(self.assessment_dlg_slot_id)

        self.assessment_dlg_slot_id = self.assessment_dlg.connect('unmap', do_close_request)
        self.assessment_dlg.set_modal(True)  # 设置为模态窗口
        self.assessment_dlg.present()
        self.assessment_dlg.map()

    def building_texture(self,sender, *args):
        i = self.panel.selection_model.get_selected()
        if i == Gtk.INVALID_LIST_POSITION:
            return

        tree_row = self.panel.selection_model.get_item(i)
        if tree_row.get_depth():
            return

        item = tree_row.get_item()
        if 0 == item.model.get_n_items():
            return

        from building_texture_dialog import BuildingTextureDialog
        dlg = BuildingTextureDialog()
        dlg.input(item.obj)

        def do_close_request(win):
            dlg.output()

        dlg.connect('close_request', do_close_request)
        dlg.set_modal(True)  # 设置为模态窗口
        dlg.set_transient_for(self)  # 设置父窗口
        dlg.present()

    