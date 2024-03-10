"""
=======================================================
Play a video in the 3D world
=======================================================

The goal of this demo is to show how to visualize a video
on a cube by updating a texture.
"""

from fury import window, actor
import numpy as np
import cv2

def timer_callback(_caller, _timer_event):
    rgb_images = []
    for video_capture in video_captures:
        _, bgr_image = video_capture.read()

        # This condition is used to stop the code when the smallest code is over.
        if isinstance(bgr_image, np.ndarray):
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb_image)
        else:
            show_manager.exit()
            return

    cube.texture_update(
        show_manager,
        *rgb_images
    )

# the sources for the video, can be URL or directory links on your machine.
sources = [
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
    'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'
]

video_captures = [cv2.VideoCapture(source) for source in sources]
rgb_images = []
for video_capture in video_captures:
    _, bgr_image = video_capture.read()
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb_images.append(rgb_image)

# calling the TexturedCube class to create a TexturedCube with different textures on all 6 sides.
cube = actor.TexturedCube(*rgb_images)
scene = cube.get_scene()
show_manager = window.ShowManager(scene, size=(1280, 720), reset_camera=False)
show_manager.add_timer_callback(True, int(1/60), timer_callback)
show_manager.start()
