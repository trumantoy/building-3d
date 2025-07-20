import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Gio

import time
import cairo
import numpy as np
import pygfx as gfx
from pathlib import Path
import os
import shutil
import trimesh

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

        def select_folder(dialog, result): 
            file_path = None
            try:
                file = dialog.select_folder_finish(result)
                file_path = file.get_path()
            except:
                return

            # 使用 os.walk 遍历文件夹
            for e in [e for e in os.scandir(file_path) if e.is_file()]:
                print(e.name)
                if not e.path.endswith('.las'):
                    continue

                points = np.loadtxt(e.path,dtype=np.float32,usecols=(0, 1, 2))
                colors = np.loadtxt(e.path+'.colors',dtype=np.float32,usecols=(0, 1, 2))
                
                geometry = gfx.Geometry(positions=points, colors=colors)
                material = gfx.PointsMaterial(color_mode="vertex", size=1)
                obj = PointCloud(geometry,material)
                obj.name = e.name

                self.editor.add(obj)
                item = self.panel.add(obj)

                children_dir = os.path.join(file_path,f'_{e.name}')
                sub_objs = []
                for e in [e for e in os.scandir(children_dir) if e.is_file]:
                    print(e.name)
                    if e.path.endswith('.npy'):
                        sub_points = np.loadtxt(e.path,dtype=np.float32,usecols=[0,1,2])
                        sub_colors = np.loadtxt(e.path+'.colors',dtype=np.float32,usecols=[0,1,2])
                        geometry = gfx.Geometry(positions=sub_points, colors=sub_colors)
                        material = gfx.PointsMaterial(color_mode="vertex", size=1)
                        sub_obj = PointCloud(geometry,material)
                        sub_obj.name = e.name
                        sub_obj.label.visible = False
                        sub_objs.append(sub_obj)
                        continue

                    if e.path.endswith('.obj') and not e.path.endswith('.roof.obj'):                        
                        mesh = gfx.load_mesh(e.path)[0]
                        pc = mesh.geometry.positions.data
                        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
                        offset = [(x.max()-x.min())/2,(y.max()-y.min())/2,0]
                        origin = np.array([np.min(x),np.min(y),0])
                        x = x - np.min(x)
                        y = y - np.min(y)
                        z = z
                        pc = np.column_stack([x,y,z]) - [(x.max()-x.min())/2,(y.max()-y.min())/2,0]
                        mesh.geometry.positions.data[:] = pc.astype(np.float32)

                        mesh.material.side = gfx.VisibleSide.both
                        mesh.material.color = (0.8, 0.8, 0.8)  # 设置材质颜色为白色
                        mesh.material.shininess = 0  # 降低高光强度
                        mesh.material.specular = (0.0, 0.0, 0.0, 1.0)  # 降低高光色
                        mesh.material.emissive = (0.8, 0.8, 0.8)  # 设置微弱自发光
                        mesh.material.flat_shading = True  # 启用平面着色

                        building = Building()
                        building.geometry = mesh.geometry
                        building.material = mesh.material
                        building.local.position = origin + offset
                        building.name = e.name

                        if os.path.exists(e.path+'.roof.obj'):         
                            with open(e.path+ '.roof.obj', 'r') as f:
                                building.roof_mesh_content = f.read()
                        sub_objs.append(building)
                        continue
                
                obj.add(*sub_objs)
                self.panel.add_sub(item,sub_objs)

        dialog.select_folder(None, None, select_folder) 

    def file_export(self,sender, *args):
        dialog = Gtk.FileDialog()
        dialog.set_modal(True)

        filter_text = Gtk.FileFilter()
        filter_text.set_name("点云")
        
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(filter_text)
        dialog.set_filters(filters)
        dialog.set_default_filter(filter_text)

        def save_file(dialog, result):
            file_path = None

            try:
                file = dialog.save_finish(result)
                file_path = file.get_path()
            except:
                return
            
            shutil.rmtree(file_path, ignore_errors=True)
            os.makedirs(file_path, exist_ok=True)
           

            for i, item in enumerate(self.panel.model):
                print(item.obj.name)
                points = item.obj.geometry.positions.data + item.obj.local.position
                np.savetxt(os.path.join(file_path, item.obj.name), points)
                np.savetxt(os.path.join(file_path, item.obj.name+'.colors'), item.obj.geometry.colors.data)

                children_dir = os.path.join(file_path, f'_{item.obj.name}')
                os.makedirs(children_dir, exist_ok=True)
        
                for j, sub_item in enumerate(item.model):
                    print(sub_item.obj.name)
                    if type(sub_item.obj) == PointCloud:
                        points = sub_item.obj.geometry.positions.data + sub_item.obj.local.position
                        np.savetxt(os.path.join(children_dir, sub_item.obj.name), points)
                        np.savetxt(os.path.join(children_dir, sub_item.obj.name+'.colors'), sub_item.obj.geometry.colors.data)
                        continue

                    if type(sub_item.obj) == Building:
                        positions = sub_item.obj.geometry.positions.data + sub_item.obj.local.position
                        faces = sub_item.obj.geometry.indices.data if sub_item.obj.geometry.indices is not None else None
                        tm = trimesh.Trimesh(vertices=positions, faces=faces)
                        tm.export(os.path.join(children_dir, sub_item.obj.name))
                        if sub_item.obj.roof_mesh_content:            
                            with open(os.path.join(children_dir, sub_item.obj.name+'.roof.obj'),'w') as f:
                                f.write(sub_item.obj.roof_mesh_content.getvalue())
                        continue
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

    