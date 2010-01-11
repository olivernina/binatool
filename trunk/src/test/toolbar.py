#!/usr/bin/env python

# example rangewidgets.py

import pygtk
pygtk.require('2.0')
import gtk

# Convenience functions

def make_menu_item(name, callback, data=None):
    item = gtk.MenuItem(name)
    item.connect("activate", callback, data)
    item.show()
    return item

def scale_set_default_values(scale):
    scale.set_update_policy(gtk.UPDATE_CONTINUOUS)
    scale.set_digits(1)
    scale.set_value_pos(gtk.POS_TOP)
    scale.set_draw_value(True)

class ThreshToolbar:
    def cb_pos_menu_select(self, item, pos):
        # Set the value position on both scale widgets
#        self.hscale.set_value_pos(pos)
#        self.vscale.set_value_pos(pos)
        if pos is 2:
            print pos
        
    def otsu_menu_select(self,item,pos):
        print item
        print pos
    def run_action(self,item):
        print "here"
    def __init__(self):
        # Standard window-creating stuff
        self.window = gtk.Window (gtk.WINDOW_TOPLEVEL)
        self.window.connect("destroy", lambda w: gtk.main_quit())
        self.window.set_title("Thresholding Algorithms")

        box1 = gtk.VBox(False, 0)
        self.window.add(box1)
        box1.show()

        box2 = gtk.HBox(False, 10)
        box2.set_border_width(10)
        box1.pack_start(box2, True, True, 0)
        box2.show()

        box2 = gtk.HBox(False, 10)
        box2.set_border_width(10)

        # An option menu to change the position of the value
        label = gtk.Label("Algorithm: ")
        box2.pack_start(label, False, False, 0)
        label.show()
  
        opt = gtk.OptionMenu()
        menu = gtk.Menu()

        item = make_menu_item ("Otsu", self.cb_pos_menu_select, 0)
        menu.append(item)
  
        item = make_menu_item ("Niblack", self.cb_pos_menu_select,1)
        menu.append(item)
  
        item = make_menu_item ("Sauvola", self.cb_pos_menu_select, 2)
        menu.append(item)
  
        item = make_menu_item ("Kitler", self.cb_pos_menu_select, 3)
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
    ThreshToolbar()
    main()
