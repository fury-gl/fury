from functools import partial
from fury.window import Scene, ShowManager


class EffectManager():
    """Class that manages the application of post-processing effects on actors.

    Parameters
    ----------
    manager : ShowManager
        Target manager that will render post processed actors."""

    def __init__(self, manager : ShowManager):
        manager.initialize()
        self.scene = Scene()
        cam_params = manager.scene.get_camera()
        self.scene.set_camera(*cam_params)
        self.on_manager = manager
        self.off_manager = ShowManager(self.scene,
                                       size=manager.size)
        self.off_manager.window.SetOffScreenRendering(True)
        self.off_manager.initialize()
        self._n_active_effects = 0
        self._active_effects = {}

    def add(self, effect : callable):
        """Add an effect to the EffectManager. The effect must have a callable property,
        that will act as the callback for the interactor. Check the KDE effect for reference.

        Parameters
        ----------
        effect : callable
            Effect to be added to the `EffectManager`.
        """
        callback = partial(
            effect,
            off_manager=self.off_manager,
            on_manager=self.on_manager
        )
        if hasattr(effect, "apply"):
            effect.apply(self)

        callback()
        callback_id = self.on_manager.add_iren_callback(callback, "RenderEvent")

        self._active_effects[effect] = (callback_id, effect._offscreen_actor)
        self._n_active_effects += 1
        self.on_manager.scene.add(effect._onscreen_actor)

    def remove_effect(self, effect):
        """
        Remove an existing effect from the effect manager.

        Parameters
        ----------
        effect_actor : callable
            Effect to be removed.
        """
        if self._n_active_effects > 0:
            self.on_manager.iren.RemoveObserver(self._active_effects[effect][0])
            self.off_manager.scene.RemoveActor(self._active_effects[effect][1])
            self.on_manager.scene.RemoveActor(effect._onscreen_actor)
            self._active_effects.pop(effect)
            self._n_active_effects -= 1
        else:
            raise IndexError("Manager has no active effects.")
