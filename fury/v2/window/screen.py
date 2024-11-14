from dataclasses import dataclass

from pygfx import (
    AmbientLight,
    Background,
    BackgroundSkyboxMaterial,
    Camera,
    Controller,
    DirectionalLight,
    OrbitController,
    PerspectiveCamera,
    Scene as GfxScene,
    Viewport,
)


class Scene(GfxScene):
    def __init__(
        self,
        *,
        background=(0, 0, 0, 1),
        skybox=None,
        lights=None,
    ):
        super().__init__()

        self._bg_color = background
        self._bg_actor = None

        if skybox:
            self._bg_actor = self._skybox(skybox)
        else:
            self._bg_actor = Background.from_color(background)

        self.add(self._bg_actor)

        self.lights = lights
        if self.lights is None:
            self.lights = []
            self.lights.append(AmbientLight())

        self.add(*self.lights)

    def _skybox(self, cube_map):
        return Background(
            geometry=None, material=BackgroundSkyboxMaterial(map=cube_map)
        )

    @property
    def background(self):
        return self._background_color

    @background.setter
    def background(self, value):
        self.remove(self._bg_actor)
        self._bg_actor = Background.from_color(value)
        self.add(self._bg_actor)

    def set_skybox(self, cube_map):
        self.remove(self._bg_actor)
        self._bg_actor = self._skybox(cube_map)
        self.add(self._bg_actor)

    def clear(self):
        self.remove(*self.children)


@dataclass
class Screen:
    viewport: Viewport
    scene: Scene
    camera: Camera
    controller: Controller

    @property
    def size(self):
        return self.viewport.rect[2:]

    @property
    def position(self):
        return self.viewport.rect[:2]

    @property
    def bounding_box(self):
        return self.viewport.rect

    @bounding_box.setter
    def bounding_box(self, value):
        self.viewport.rect = value


def create_screen(
    renderer, *, rect=None, scene=None, camera=None, controller=None, camera_light=True
):
    vp = Viewport(renderer, rect)
    if scene is None:
        scene = Scene()
    if camera is None:
        camera = PerspectiveCamera(50)
        if camera_light:
            light = DirectionalLight()
            camera.add(light)
            scene.add(camera)

    if controller is None:
        controller = OrbitController(camera, register_events=vp)

    screen = Screen(vp, scene, camera, controller)
    update_camera(camera, screen.size, scene)
    return screen


def update_camera(camera, size, target):
    camera.width = size[0]
    camera.height = size[1]

    if (isinstance(target, Scene) and len(target.children) > 3) or not isinstance(
        target, Scene
    ):
        camera.show_object(target)


def update_viewports(screens, screen_bbs):
    for screen, screen_bb in zip(screens, screen_bbs):
        screen.bounding_box = screen_bb
        update_camera(screen.camera, screen.size, screen.scene)


def render_screens(renderer, screens):
    for screen in screens:
        screen.viewport.render(screen.scene, screen.camera, flush=False)

    renderer.flush()


def calculate_screen_sizes(screens, size):
    if screens is None:
        return [(0, 0, *size)]

    screen_bbs = []

    v_sections = len(screens)
    width = (1 / v_sections) * size[0]
    x = 0

    for h_section in screens:
        if h_section == 0:
            continue

        height = (1 / h_section) * size[1]
        y = 0

        for _ in range(h_section):
            screen_bbs.append((x, y, width, height))
            y += height

        x += width

    return screen_bbs
