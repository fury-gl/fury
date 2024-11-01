import numpy as np
import pygfx as gfx


class Renderer(gfx.WgpuRenderer):

    def __init__(self, *, scene=None, camera=None):
        super().__init__()

        if scene is None:
            scene = Scene()
        if camera is None:
            camera = gfx.PerspectiveCamera()

        self.scene = scene
        self.camera = camera

    def reset_camera(self, pos=(0, 0, 0)):
        self.camera.show_pos(pos)

    def reset_camera_tight(self, *, margin_factor=1.02):
        wobjects_bb = []

        self.scene.traverse(
            lambda ob: wobjects_bb.append(ob.get_bounding_box()), True
        )

        min_x = wobjects_bb[0][0][0]
        max_x = wobjects_bb[0][1][0]
        min_y = wobjects_bb[0][0][1]
        max_y = wobjects_bb[0][1][1]

        for bb in wobjects_bb:
            min_x = min(min_x, bb[0][0], bb[1][0])
            max_x = max(max_x, bb[0][0], bb[1][0])
            min_y = min(min_y, bb[0][1], bb[1][1])
            max_y = max(max_y, bb[0][1], bb[1][1])

        width, height = max_x - min_x, max_y - min_y
        center = np.array((min_x + width / 2.0, min_y + height / 2.0, 0))

        angle = np.pi * self.camera.fov / 180.0
        dist = max(width / self.camera.width, height) / np.sin(angle / 2.0) / 2.0
        position = center + np.array((0, 0, dist * margin_factor))

        self.reset_camera((0, 1, 0))
        self.camera.set_state({ 'position': position })

    def camera_info(self):
        print("# Active Camera")
        print(self.camera.get_state())


    def zoom(self, value):
        self.camera.zoom(value)



class Scene(gfx.Scene):

    def __init__(self, *, background=(0, 0, 0), skybox=None):
        super().__init__()
