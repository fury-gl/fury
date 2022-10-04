from fury import window
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf

scene = window.Scene()

fetch_gltf('AnimatedMorphCube', 'glTF')
filename = read_viz_gltf('AnimatedMorphCube')

gltf_obj = glTF(filename, apply_normals=False)

timeline = gltf_obj.morph_timeline()['Square']

scene = window.Scene()
showm = window.ShowManager(scene, size=(900, 768), reset_camera=True,
                           order_transparent=True)


def timer_callback(_obj, _event):
    gltf_obj.update_morph(timeline)
    showm.render()


gltf_obj.update_morph(timeline)
showm.initialize()
scene.add(timeline)
showm.add_timer_callback(True, 20, timer_callback)
showm.start()
