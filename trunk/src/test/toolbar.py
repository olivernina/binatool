from cvtypes import *
import cvtools
import pygtk
pygtk.require('2.0')
import gtk


def make_menu_item(name, callback, data=None):
    item = gtk.MenuItem(name)
    item.connect("activate", callback, data)
    item.show()
    return item

class ThreshToolbar:
    
    def cb_pos_menu_select(self, item, pos):
        self.alg_selected = pos   
    
    def set_model(self,newmodel):
        self.model = newmodel
    def run_action(self,item):
        filename  = self.model.get_current_file_path()
#        print filename
        
        if self.alg_selected==0:
           print "run Otsu algorithm"
           image = cv.LoadImage(filename,0)
           otsu = cvtools.otsu_thresholding(image)
           self.model.curr_iplimage = otsu
#           self.model.set_curr_iplimage(otsu)
           cvtools.display("Otsu", otsu)
           
        if self.alg_selected==1:
           print "run Nibalck algorithm"
           image = cv.LoadImage(filename,0)
           niblack = cvtools.Niblack(image, 7)
           self.model.curr_iplimage = niblack
           cvtools.display("Niblack", niblack)
           
        if self.alg_selected==2:
           print "run Sauvola algorithm"
           image = cv.LoadImage(filename,0)
           sauvola = cvtools.Sauvola(image, 7)
           self.model.curr_iplimage = sauvola
           cvtools.display("Sauvola", sauvola)
           
        if self.alg_selected==3:
           print "run Kitler algorithm"
           image = cv.LoadImage(filename,0)
           kittler = cvtools.Kittler(image)
           self.model.curr_iplimage = kittler
           cvtools.display("Kittler", kittler)
        
#        print self.alg_selected
    def __init__(self,model=None):
        
        
        self.model = model
        self.window = gtk.Window (gtk.WINDOW_TOPLEVEL)
        self.window.connect("destroy", lambda w: gtk.main_quit())
        self.window.set_title("Thresholding Algorithms")
        self.alg_selected=0
        box1 = gtk.VBox(False, 0)
        self.window.add(box1)
        box1.show()

        box2 = gtk.HBox(False, 10)
        box2.set_border_width(10)
        box1.pack_start(box2, True, True, 0)
        box2.show()

        box2 = gtk.HBox(False, 10)
        box2.set_border_width(10)

        label = gtk.Label("Algorithm: ")
        box2.pack_start(label, False, False, 0)
        label.show()
  
        opt = gtk.OptionMenu()
        menu = gtk.Menu()

        item = make_menu_item ("Otsu", self.cb_pos_menu_select, 0)  # An option menu to change the position of the value
        menu.append(item)
  
        item = make_menu_item ("Niblack", self.cb_pos_menu_select,1)
        menu.append(item)
  
        item = make_menu_item ("Sauvola", self.cb_pos_menu_select, 2)
        menu.append(item)
  
        item = make_menu_item ("Kittler", self.cb_pos_menu_select, 3)
        menu.append(item)
  
        opt.set_menu(menu)
        box2.pack_start(opt, True, True, 0)
        opt.show()

        box1.pack_start(box2, True, True, 0)
        box2.show()

        box2 = gtk.HBox(False, 10)
        box2.set_border_width(10)


        separator = gtk.HSeparator()
        box1.pack_start(separator, False, True, 0)
        separator.show()

        box2 = gtk.VBox(False, 10)
        box2.set_border_width(10)
        box1.pack_start(box2, False, True, 0)
        box2.show()

        button = gtk.Button("Run")
        button.connect("clicked", self.run_action)
        box2.pack_start(button, True, True, 0)
        button.set_flags(gtk.CAN_DEFAULT)
        button.grab_default()
        button.show()
        
        button = gtk.Button("Quit")
        button.connect("clicked", lambda w: gtk.main_quit())
        box2.pack_start(button, True, True, 0)
        button.set_flags(gtk.CAN_DEFAULT)
        button.grab_default()
        button.show()
        
        self.window.show()

def main():
    gtk.main()
    return 0            

if __name__ == "__main__":
    ThreshToolbar(None)
    main()
