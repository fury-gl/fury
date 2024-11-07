from pygfx import AmbientLight, Background, BackgroundSkyboxMaterial, Scene as GfxScene

from fury.decorators import warn_on_args_to_kwargs


class Scene(GfxScene):
    @warn_on_args_to_kwargs()
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
