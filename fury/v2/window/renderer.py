from pygfx import (
    DirectionalLight,
    OrbitController,
    PerspectiveCamera,
    Viewport as GfxViewport,
    WgpuRenderer,
)

from fury.decorators import warn_on_args_to_kwargs
from fury.v2.window import Scene


class Screen(GfxViewport):
    @warn_on_args_to_kwargs()
    def __init__(
        self, renderer, *, rect=None, scene=None, camera=None, camera_light=True
    ):
        super().__init__(renderer, rect)
        if scene is None:
            scene = Scene()
        if camera is None:
            camera = PerspectiveCamera(50)
            if camera_light:
                light = DirectionalLight()
                camera.add(light)
            scene.add(camera)
            camera.show_object(scene)

        controller = OrbitController(camera, register_events=self)
        controller.add_camera(camera, include_state={"position", "rotation"})

        self.scene = scene
        self.camera = camera

    def add(self, *actors):
        self.scene.add(*actors)

    def remove(self, *actors):
        self.scene.remove(*actors)

    def update_camera(self):
        self.camera.width = self.rect[2]
        self.camera.height = self.rect[3]
        self.camera.show_object(self.scene)

    def render(self, flush=False):
        super().render(self.scene, self.camera, flush=flush)

    def reset_camera(self):
        self.camera.local.position = (0, 0, 100)
        self.camera.look_at((0, 0, 0))


class Renderer(WgpuRenderer):
    @warn_on_args_to_kwargs()
    def __init__(
        self,
        target,
        *args,
        pixel_ratio=None,
        pixel_filter=None,
        show_fps=False,
        blend_mode="default",
        sort_objects=False,
        enable_events=True,
        gamma_correction=1,
        **kwargs,
    ):
        super().__init__(
            target,
            *args,
            pixel_ratio=pixel_ratio,
            pixel_filter=pixel_filter,
            show_fps=show_fps,
            blend_mode=blend_mode,
            sort_objects=sort_objects,
            enable_events=enable_events,
            gamma_correction=gamma_correction,
            **kwargs,
        )
        self.screens = []

    def update_screens(self, screen_bbs):
        for screen, screen_bb in zip(self.screens, screen_bbs):
            screen.rect = screen_bb
            screen.update_camera()

    def render_screens(self, screen=None):
        if screen is None:
            for s in self.screens:
                s.render(flush=False)
        else:
            self.screens[screen].render(flush=False)

        self.flush()
