from functools import partial
from fury.window import Scene, ShowManager


class EffectManager():
    """Class that manages the application of post-processing effects on actors.

    Parameters
    ----------
    manager : ShowManager
        Target manager that will render post processed actors."""

    def __init__(self, manager : ShowManager):
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

    def add(self, effect):
        """Add an effect to the EffectManager.
        
        Parameters
        ----------
        effect : callable
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

        self._active_effects[effect._onscreen_actor] = (callback_id, effect._offscreen_actor)
        self._n_active_effects += 1
        self.on_manager.scene.add(effect._onscreen_actor)

    def remove_effect(self, effect_actor):
        """Remove an existing effect from the effects manager.
        Beware that the effect and the actor will be removed from the rendering pipeline
        and shall not work after this action.

        Parameters
        ----------
        effect_actor : actor.Actor
            Actor of effect to be removed.
        """
        if self._n_active_effects > 0:
            self.on_manager.iren.RemoveObserver(self._active_effects[effect_actor][0])
            self.off_manager.scene.RemoveActor(self._active_effects[effect_actor][1])
            self.on_manager.scene.RemoveActor(effect_actor)
            self._active_effects.pop(effect_actor)
            self._n_active_effects -= 1
        else:
            raise IndexError("Manager has no active effects.")
