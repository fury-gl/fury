from pygfx import (
    AmbientLight,
    Background,
    BackgroundSkyboxMaterial,
    DirectionalLight,
    OrbitController,
    PerspectiveCamera,
    Scene as GfxScene,
    Viewport as GfxViewport,
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


def create_screen(
    renderer, *, rect=None, scene=None, camera=None, controller=None, camera_light=True
):
    screen = GfxViewport(renderer, rect)
    if scene is None:
        scene = Scene()
    if camera is None:
        camera = PerspectiveCamera(50)
        if camera_light:
            light = DirectionalLight()
            camera.add(light)
            scene.add(camera)

    camera.show_object(scene)

    if controller is None:
        OrbitController(camera, register_events=screen)

    return screen


def reset_camera(screen):
    screen.camera.local.position = (0, 0, 100)
    screen.camera.look_at((0, 0, 0))


def update_camera(screen):
    screen.camera.width = screen.rect[2]
    screen.camera.height = screen.rect[3]
    screen.camera.show_object(screen.scene)


def update_screens(screens, screen_bbs):
    for screen, screen_bb in zip(screens, screen_bbs):
        screen.rect = screen_bb
        update_camera(screen)


def render_screens(renderer, screens):
    for s in screens:
        s.render(s.scene, s.camera, flush=False)

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
