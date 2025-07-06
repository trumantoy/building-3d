import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Gio

import time
import cairo
import numpy as np
import pygfx as gfx
from pathlib import Path

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

        self.geom_panel = self.stack.get_visible_child()
        self.widget.set_draw_func(self.draw, self.editor)
        self.viewbar.set_controller(self.view_controller)
        self.hotbar.set_viewbar(self.widget, self.viewbar, self.geom_panel, self.editor)

        action = Gio.SimpleAction.new('import', None)
        action.connect('activate', self.file_import)
        self.add_action(action)

        action = Gio.SimpleAction.new('export', None)
        action.connect('activate', self.file_export)
        self.add_action(action)

        action = Gio.SimpleAction.new('close', None)
        action.connect('activate', self.file_close)
        self.add_action(action)

        action = Gio.SimpleAction.new('算法', None)
        action.connect('activate', lambda _: True)
        self.add_action(action)

        action = Gio.SimpleAction.new('building_divide', None)
        action.connect('activate', self.building_divide)
        self.add_action(action)

        action = Gio.SimpleAction.new('building_reconstruct', None)
        action.connect('activate', self.building_reconstruct)
        self.add_action(action)

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

    def file_close(self,sender, *args):
        pass

    def file_export(self,sender, *args):
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
            try:
                file = dialog.save_finish(result)
            except:
                return
            
            self.content.to_file(file.get_path())
        dialog.save(None, None, select_file) 

    def file_import(self, sender, args):
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
            try:
                file = dialog.open_finish(result)
            except:
                return

            content = PointCloud()
            aabb = content.get_bounding_box()
            content.local.z = aabb[1][2]
            self.editor.add(content)

            file_path = file.get_path()
            file_path = Path(file_path).as_posix()
            content.set_from_file(file_path)

            self.geom_panel.add('点云-'+str(content.id), content)

        dialog.open(None, None, select_file)

    def building_divide(self,sender, *args):
        pcs = []
        for obj in self.editor.children:
            if type(obj) != PointCloud:
                continue
            pcs.append(obj.points.geometry.positions.data)
        
        from building_division_dialog import BuildingDivisionDialog
        dlg = BuildingDivisionDialog()
        dlg.set_point_cloud(pcs)

        def do_close_request(win):
            pcs = dlg.get_point_cloud()
            pc = np.vstack(pcs)
            z = pc[:,2]
            z_min = np.min(z)
            z_max = np.max(z)

            import colorsys  # 导入 colorsys 模块用于 HSV 到 RGB 的转换
            for pc in pcs:
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
                material = gfx.PointsMaterial(color_mode="vertex", size=2)
                points = gfx.Points(geometry,material)
                self.editor.add(points)

            for obj in self.editor.children:
                if type(obj) != PointCloud:
                    continue

                self.editor.remove(obj)

        dlg.connect('close_request', do_close_request)
        dlg.set_modal(True)  # 设置为模态窗口
        dlg.set_transient_for(self)  # 设置父窗口
        dlg.present()

    
    def building_reconstruct(self,sender, *args):
        pcs = []
        for obj in self.editor.children:
            if type(obj) != gfx.Points:
                continue
            pcs.append(obj.geometry.positions.data)
        
        from building_reconstruction_dialog import BuildingReconstructionDialog
        dlg = BuildingReconstructionDialog()
        dlg.input(pcs)

        def do_close_request(win):
            wireframes = dlg.output()

            for vertices, edges in wireframes:
                 # 将顶点转换为 numpy 数组
                obj = gfx.WorldObject()
                vertices = np.array(vertices, dtype=np.float32)

                for edge in edges:
                    start_vertex = vertices[edge[0]]
                    end_vertex = vertices[edge[1]]
                    # 构建线段的顶点序列
                    line_vertices = np.array([start_vertex, end_vertex], dtype=np.float32)
                    # 创建几何体
                    geometry = gfx.Geometry(positions=line_vertices)
                    # 创建线材质
                    material = gfx.LineMaterial(color="black", thickness=2)
                    # 创建 Line 对象
                    line = gfx.Line(geometry, material)
                    # 将 Line 对象添加到编辑器
                    obj.add(line)

                self.editor.add(obj)

        dlg.connect('close_request', do_close_request)
        dlg.set_modal(True)  # 设置为模态窗口
        dlg.set_transient_for(self)  # 设置父窗口
        dlg.present()

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