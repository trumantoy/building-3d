import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, GObject, Gio, Gdk

@Gtk.Template(filename='ui/panel.ui')
class Panel (Gtk.ScrolledWindow):
    __gtype_name__ = "Panel"
    provider = Gtk.CssProvider.new()

    geoms = Gtk.Template.Child('geoms')
    
    def __init__(self):
        self.model = Gtk.StringList()
        selection_model = Gtk.SingleSelection.new(self.model)
        factory = Gtk.SignalListItemFactory()

        factory.connect("setup", self.setup_listitem)
        factory.connect("bind", self.bind_listitem)
        
        self.geoms.set_model(selection_model)
        self.geoms.set_factory(factory)
    
    def add(self, text):
        self.model.append(text)

    def setup_listitem(self, factory, list_item):
        # 创建一个水平排列的容器
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        
        # 创建图标（使用默认的文件夹图标）
        icon = Gtk.Image.new_from_icon_name("line-symbolic")
        icon.set_icon_size(Gtk.IconSize.LARGE)
        
        # 创建标签
        label = Gtk.Label()
        label.set_halign(Gtk.Align.START)
        
        # 将图标和标签添加到容器中
        box.append(icon)
        box.append(label)
        
        # 设置列表项的显示内容
        list_item.set_child(box)

    def bind_listitem(self, factory, list_item):
        # 获取容器和其中的子部件
        box = list_item.get_child()
        icon = box.get_first_child()
        label = box.get_last_child()

        string_obj = list_item.get_item()
        text = string_obj.get_string()
        label.set_label(text)
        
        # 根据文本设置不同的图标
        if "点" in text:
            icon.set_from_icon_name("content-loading-symbolic")
        elif '线' in text:
            icon.set_from_icon_name("list-remove-symbolic")
        else:
            icon.set_from_icon_name("media-playback-stop")
