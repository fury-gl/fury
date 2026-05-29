"""
================
PlaybackPanel UI
================
"""

import numpy as np
from fury import actor, window
from fury.ui import PlaybackPanel

###############################################################################
# 1. Create the Scene and a target actor (Rotating Cube)
scene = window.Scene()

centers = np.array([[0, 0, 0]])
colors = np.array([[1, 0.5, 0]])
cube = actor.box(centers, directions=(0, 1, 0), colors=colors, scales=(2, 2, 2))
scene.add(cube)

###############################################################################
# 2. Initialize the PlaybackPanel

playback_ui = PlaybackPanel(position=(50, 50), width=700, loop=True)
playback_ui.final_time = 60.0
scene.add(playback_ui)

state = {"current_time": 0.0, "rotation_speed": 1.0}


def on_progress_changed(t):
    state["current_time"] = t
    cube.local.rotation = (0, 0, t * 0.01, 1)


def on_speed_changed(s):
    state["rotation_speed"] = s


playback_ui.on_progress_bar_changed = on_progress_changed
playback_ui.on_speed_changed = on_speed_changed

###############################################################################
# 5. Define the Callback


def update_playback_logic(target_actor):
    """Callback to sync animation state with UI."""
    if playback_ui._playing:
        step = 0.05 * state["rotation_speed"]
        state["current_time"] += step

        if state["current_time"] > playback_ui.final_time:
            if playback_ui._loop:
                state["current_time"] = 0
            else:
                state["current_time"] = playback_ui.final_time
                playback_ui.pause()

        playback_ui.current_time = state["current_time"]

        target_actor.local.rotation = (0, 0, state["current_time"] * 0.01, 1)


###############################################################################
# 6. Start the ShowManager and Register the Callback

show_m = window.ShowManager(
    scene=scene, size=(800, 700), title="FURY Playback Panel UI"
)

show_m.register_callback(update_playback_logic, 0.05, True, "PlaybackSync", cube)

show_m.start()
