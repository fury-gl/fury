from fury import window
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf

scene = window.Scene()

fetch_gltf('RiggedFigure', 'glTF')
filename = read_viz_gltf('RiggedSimple')

gltf_obj = glTF(filename, apply_normals=False)

timeline = gltf_obj.skin_timeline()['anim_0']

scene = window.Scene()
showm = window.ShowManager(scene, size=(900, 768), reset_camera=True,
                           order_transparent=True)

gltf_obj.initialise_skin(timeline, bones=False)
showm.initialize()
scene.add(timeline)


def timer_callback(_obj, _event):
    gltf_obj.update_skin(timeline)
    showm.render()


showm.add_timer_callback(True, 20, timer_callback)
scene.reset_camera()
showm.start()
