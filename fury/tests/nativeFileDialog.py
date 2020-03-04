import numpy as np
from os.path import join as pjoin

from fury import actor, window

import textwrapper;
import ui;
scene = window.Scene()
scene.background((1, 1, 1))

open_static = ui.TextBlock2D()
open_static.message = "Select template"

use_dir = ui.TextBlock2D()
use_dir.message = "Use directory"

out_dir = ui.TextBlock2D()
out_dir.message = "Output directory"

from pathlib import Path
home = str(Path.home())

file_menu = ui.FileMenu2D(home, size=(450, 600))

panel = ui.Panel2D(size=(800,900))


class ProcessingObject(object):

    ftemplate = None
    dname = None
    out_dname = None


po = ProcessingObject()


def open_static_callback(obj, event):
    po.ftemplate = pjoin(file_menu.current_directory,
                       file_menu.listbox.selected[0])
    open_static.message +=  ' ' + file_menu.listbox.selected[0]
    showm.render()

    print(po.ftemplate)


open_static.actor.AddObserver('LeftButtonPressEvent',
                              open_static_callback,
                              1.0)

def use_dir_callback(obj, event):
    po.dname = file_menu.current_directory
    use_dir.message += ' selected!'
    showm.render()
    print(po.dname)


use_dir.actor.AddObserver('LeftButtonPressEvent',
                          use_dir_callback,
                          1.0)


def out_dir_callback(obj, event):
    po.out_dname = file_menu.current_directory
    out_dir.message += ' selected!'
    showm.render()
    print(po.out_dname)


out_dir.actor.AddObserver('LeftButtonPressEvent',
                          out_dir_callback,
                          1.0)


showm = window.ShowManager(scene, size=(1200, 1000))

showm.initialize()
panel.add_element(file_menu, coords=(250,20))
panel.add_element(open_static, coords=(20, 200))
panel.add_element(use_dir, coords=(20, 110))
panel.add_element(out_dir, coords=(20, 10))


scene.add(panel)
#scene.add(actor.axes())

showm.render()
showm.start()