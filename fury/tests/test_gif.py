"""
In this File, I am testing the way of Saving Frames as GIF on the Tutorial named `Using a timer` 
"""

import os
import itertools
import numpy as np
from PIL import Image, ImageDraw
from fury import window, actor, ui


xyz = 10 * np.random.rand(100, 3)
colors = np.random.rand(100, 4)
radii = np.random.rand(100) + 0.5


os.mkdir("temp")


# This function does the Execution of the Code with the Threshold for the Counter provided.
def execute(Threshold):

    scene = window.Scene()

    sphere_actor = actor.sphere(centers=xyz,
                                colors=colors,
                                radii=radii)

    scene.add(sphere_actor)

    showm = window.ShowManager(scene,
                               size=(900, 768), reset_camera=False,
                               order_transparent=True)

    showm.initialize()

    tb = ui.TextBlock2D(bold=True)

    # use itertools to avoid global variables
    counter = itertools.count()

    def timer_callback(_obj, _event):
        cnt = next(counter)
        tb.message = "Let's count up to 100 and exit :" + str(cnt)
        showm.scene.azimuth(0.05 * cnt)
        sphere_actor.GetProperty().SetOpacity(cnt/100.)
        showm.render()

        # We will stop when counter becomes equal to provided `Threshold`
        if cnt == Threshold:
            showm.exit()

    scene.add(tb)

    # Run every 200 milliseconds
    showm.add_timer_callback(True, 200, timer_callback)

    showm.start()

    # We are saving the Image everytime
    window.record(scene, out_path="temp/" + str(Threshold) +
                  "bubbles.png", size=(900, 768))


# Saving the Series of Images into a List
images = []

# Assuming that `100` Iterations are sufficient for observing complete Simulation
# `10` Indicates that we are interested in taking the picture after every `10` Iterations

for t in range(1, 101, 10):
    execute(t)
    image_name = "temp/" + str(t) + "bubbles.png"

    # Using `Image` from `PIL` to open the saved Images
    im_sys = Image.open(image_name)

    images.append(im_sys)


# Deleting the `temp` Folder made for storing Images. It requires Permission.
# os.remove("temp")


# Using `save` from PIL to convert the Series of Frames into a GIF Image.
images[0].save('Bubbles.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=0.005, loop=0)
