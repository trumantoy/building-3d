import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, GObject, Gio, Gdk

@Gtk.Template(filename='ui/panel.ui')
class Panel (Gtk.Paned):
    __gtype_name__ = "Panel"
    provider = Gtk.CssProvider.new()

    geoms = Gtk.Template.Child('geoms')
    roofcolor = Gtk.Template.Child('roofcolor')
    roomcolor = Gtk.Template.Child('roomcolor')
    
    def __init__(self):
        Gtk.StyleContext.add_provider_for_display(self.get_display(),self.provider,Gtk.STYLE_PROVIDER_PRIORITY_USER)

        self.model = Gtk.StringList()
        selection_model = Gtk.SingleSelection.new(self.model)
        factory = Gtk.SignalListItemFactory()

        factory.connect("setup", self.setup_listitem)
        factory.connect("bind", self.bind_listitem)
        
        self.geoms.set_model(selection_model)
        self.geoms.set_factory(factory)

        self.roofcolor.set_dialog(Gtk.ColorDialog())
        self.roomcolor.set_dialog(Gtk.ColorDialog())

    @Gtk.Template.Callback()
    def geom_activated(self,sender, i):
        item = sender.get_model().get_item(i)
        print(item)
        
    @Gtk.Template.Callback()
    def roofcolor_activated(self,sender,*args):
        print('11')

    @Gtk.Template.Callback()
    def roomcolor_activated(self,sender,*args):
        print('2')

    def add(self, text):
        self.model.append(text)

    def setup_listitem(self, factory, list_item):
        # 创建一个水平排列的容器
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        
        # 创建图标（使用默认的文件夹图标）
        icon = Gtk.ToggleButton()
        icon.set_icon_name("display-brightness-symbolic")
        icon.set_has_frame(False)
        css = """
            .borderless-toggle-button {
                background: none;
            }
            """
        self.provider.load_from_data(css)
        icon.get_style_context().add_class("borderless-toggle-button")
        icon.connect("toggled", self.on_toggle_active)

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

    def on_toggle_active(self,sender):
        if sender.get_active():
            # 激活状态时的图标
            sender.set_icon_name("")
        else:
            # 非激活状态时的图标
            sender.set_icon_name("display-brightness-symbolic")
    