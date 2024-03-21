"""
=======================================================
Play video on a Cube
=======================================================

The goal of this demo is to show how to visualize a video
on a cube by updating its textures.
"""

from fury import actor, window
import numpy as np
import cv2

#########################################################################
# The 6 sources for the video, can be URL or directory paths on your machine.
# There'll be a significant delay if your internet connectivity is poor,
# use local directory paths for fast rendering.
sources = [
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/'
    + 'sample/BigBuckBunny.mp4',
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/'
    + 'sample/BigBuckBunny.mp4',
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/'
    + 'sample/BigBuckBunny.mp4',
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/'
    + 'sample/BigBuckBunny.mp4',
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/'
    + 'sample/BigBuckBunny.mp4',
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/'
    + 'sample/BigBuckBunny.mp4'
]

#########################################################################
# We are creating ``OpenCV videoCapture`` objects to capture frames from
# sources.
video_captures = [cv2.VideoCapture(source) for source in sources]

# rgb_images will store the RGB values of the frames.
rgb_images = []
for video_capture in video_captures:
    _, bgr_image = video_capture.read()

    # OpenCV reads in BGR, we are converting it to RGB.
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb_images.append(rgb_image)


########################################################################
# ``timer_callback`` gets called repeatedly to change texture.

def timer_callback(_caller, _timer_event):
    rgb_images = []
    for video_capture in video_captures:
        # Taking the new frames
        _, bgr_image = video_capture.read()

        # Condition used to stop rendering when the smallest video is over.
        if isinstance(bgr_image, np.ndarray):
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb_image)
        else:
            show_manager.exit()
            return

    for actor_, image in zip(cube, rgb_images):
        # texture_update is a function to update the texture of an actor
        actor.texture_update(actor_, image)

    # you've to re-render the pipeline again to display the results
    show_manager.render()

#######################################################################
# ``texture_on_cube`` is the function we use, the images are assigned in
# cubemap order.


"""
     |----|
     | +Y |
|----|----|----|----|
| -X | +Z | +X | -Z |
|----|----|----|----|
     | -Y |
     |----|
"""
######################################################################


cube = actor.texture_on_cube(*rgb_images, centers=(0, 0, 0))

# adding the returned Actors to scene
scene = window.Scene()
scene.add(*cube)

######################################################################
# ``ShowManager`` controls the frequency of changing textures.
# The video is rendered by changing textures very frequently.
show_manager = window.ShowManager(scene, size=(1280, 720), reset_camera=False)
show_manager.add_timer_callback(True, int(1/60), timer_callback)


######################################################################
# Flip it to ``True`` for video.
interactive = False
if interactive:
    show_manager.start()

window.record(scene, size=(1280, 720), out_path='viz_play_cube.png')
