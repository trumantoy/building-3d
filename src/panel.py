import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, GObject, Gio, Gdk

import pygfx as gfx
from simtoy import *

@Gtk.Template(filename='ui/panel.ui')
class Panel (Gtk.Paned):
    __gtype_name__ = "Panel"
    provider = Gtk.CssProvider.new()

    geoms = Gtk.Template.Child('geoms')
    roofcolor = Gtk.Template.Child('roofcolor')
    roomcolor = Gtk.Template.Child('roomcolor')
    expander_position = Gtk.Template.Child('position')
    expander_pointcloud = Gtk.Template.Child('pointcloud')
    expander_mesh = Gtk.Template.Child('mesh')

    spin_x = Gtk.Template.Child('x')
    spin_y = Gtk.Template.Child('y')
    spin_z = Gtk.Template.Child('z')
        
    def __init__(self):
        Gtk.StyleContext.add_provider_for_display(self.get_display(),self.provider,Gtk.STYLE_PROVIDER_PRIORITY_USER)

        self.model = Gio.ListStore.new(GObject.Object)
        self.selection_model = Gtk.SingleSelection.new(self.model)
        self.selection_model.set_autoselect(False)
        self.selection_model.set_can_unselect(True)
        self.selection_model.connect("selection-changed", self.item_selection_changed)

        factory = Gtk.SignalListItemFactory()

        factory.connect("setup", self.setup_listitem)
        factory.connect("bind", self.bind_listitem)
        
        self.geoms.set_model(self.selection_model)
        self.geoms.set_factory(factory)
        
        self.roofcolor.set_dialog(Gtk.ColorDialog())
        self.roomcolor.set_dialog(Gtk.ColorDialog())

    def setup_listitem(self, factory, listviewitem):
        # 创建一个水平排列的容器
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        
        # 创建图标（使用默认的文件夹图标）
        icon = Gtk.ToggleButton()
        icon.set_icon_name("display-brightness-symbolic")
        icon.set_active(True)
        icon.set_has_frame(False)
        css = """
            .borderless-toggle-button {
                background: none;
            }
            """
        self.provider.load_from_data(css)
        icon.get_style_context().add_class("borderless-toggle-button")
        icon.connect("toggled", self.item_visible_toggled, listviewitem)

        # 创建标签
        label = Gtk.Label()
        label.set_halign(Gtk.Align.START)
        
        # 将图标和标签添加到容器中
        box.append(icon)
        box.append(label)
        
        # 设置列表项的显示内容
        listviewitem.set_child(box)

    def bind_listitem(self, factory, listviewitem):
        # 获取容器和其中的子部件
        box = listviewitem.get_child()
        icon = box.get_first_child()
        label = box.get_last_child()

        item = listviewitem.get_item()
        label.set_label(item.name)
        

    def add(self, name, obj):
        item = GObject.Object()
        item.name = name
        item.obj = obj
        self.model.append(item)
    
    def remove(self, name):
        for i,item in enumerate(self.model):
            if item.name == name:
                self.model.remove(i)
                break
        
    def item_visible_toggled(self,sender,listviewitem):
        item = listviewitem.get_item()
        
        if sender.get_active():
            sender.set_icon_name("display-brightness-symbolic")
            item.obj.visible = True
        else:
            sender.set_icon_name("")
            item.obj.visible = False

    def item_selection_changed(self, selection_model, position, n_items):
        selected_index = selection_model.get_selected()
        item = self.model.get_item(selected_index)
        self.spin_x.set_value(item.obj.local.x)
        self.spin_y.set_value(item.obj.local.y)
        self.spin_z.set_value(item.obj.local.z)

        if type(item.obj) == PointCloud:
            self.expander_position.set_visible(True)
            self.expander_pointcloud.set_visible(True)
            self.expander_mesh.set_visible(False)
        elif type(item.obj) == Building:
            self.expander_position.set_visible(True)
            self.expander_pointcloud.set_visible(False)
            self.expander_mesh.set_visible(True)
        else:
            self.expander_position.set_visible(False)
            self.expander_pointcloud.set_visible(False)
            self.expander_mesh.set_visible(False)
    
    @Gtk.Template.Callback()
    def assessment_value_changed(self, spin_button):
        assessment = spin_button.get_value()

        for item in self.model:
            if type(item.obj) != Building:
                continue
            
            if item.obj.assessment >= assessment:
                item.obj.visible = True
            else:
                item.obj.visible = False

    @Gtk.Template.Callback()
    def x_value_changed(self, spin_button):
        value = spin_button.get_value()
        selected_index = self.selection_model.get_selected()
        item = self.model.get_item(selected_index)
        item.obj.local.x = value

    @Gtk.Template.Callback()
    def y_value_changed(self, spin_button):
        value = spin_button.get_value()
        selected_index = self.selection_model.get_selected()
        item = self.model.get_item(selected_index)
        item.obj.local.y = value

    @Gtk.Template.Callback()
    def z_value_changed(self, spin_button):
        value = spin_button.get_value()
        selected_index = self.selection_model.get_selected()
        item = self.model.get_item(selected_index)
        item.obj.local.z = value

    @Gtk.Template.Callback()
    def point_size_value_changed(self,spin_button):
        value = spin_button.get_value()
        selected_index = self.selection_model.get_selected()
        item = self.model.get_item(selected_index)
        item.obj.material.size = value

    @Gtk.Template.Callback()
    def roofcolor_activated(self,sender,*args):
        print('11')

    @Gtk.Template.Callback()
    def roomcolor_activated(self,sender,*args):
        selected_index = self.selection_model.get_selected()
        item = self.model.get_item(selected_index)
        color = self.roomcolor.get_color()
        item.obj.material.color = (color.red, color.green, color.blue)