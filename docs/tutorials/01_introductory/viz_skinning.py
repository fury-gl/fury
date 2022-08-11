import numpy as np
from fury import window, utils, actor
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf
from fury.lib import Transform

from fury.animation.timeline import Timeline
from fury.animation import helpers

# scene = window.Scene()

# fetch_gltf('RiggedSimple', 'glTF')
# filename = read_viz_gltf('SimpleSkin')

# gltf_obj = glTF(filename, apply_normals=True)
# actors = gltf_obj.actors()

# scene.add(*actors)

# window.show(scene)
axes = actor.axes()

sphere = actor.sphere(centers=np.random.rand(1, 3), colors=np.random.rand(1, 3))
# vertices = utils.vertices_from_actor(sphere)
tranform = np.array([[2, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
# vertices[:] = np.dot(vertices, tranform)

timeline = Timeline(playback_panel=True)
timeline.add_actor(sphere)
timeline.set_keyframe('shape', 0.0, np.identity(3))
timeline.set_keyframe('shape', 5.0, tranform)
timeline.set_position(0, np.identity(3))

scene = window.Scene()
showm = window.ShowManager(scene, size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

scene.add(timeline)


def timer_callback(_obj, _event):
    timeline.update_animation()
    if timeline.is_interpolatable('shape'):
        print(timeline.get_value('shape', timeline.current_timestamp))
    # print(timeline._data)
    # print(timeline.get_value('position', timeline.current_timestamp))
    showm.render()


showm.add_timer_callback(True, 10, timer_callback)

showm.start()
