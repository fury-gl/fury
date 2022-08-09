from fury import window, utils
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf

scene = window.Scene()

fetch_gltf('RiggedSimple', 'glTF')
filename = read_viz_gltf('RiggedSimple')

gltf_obj = glTF(filename, apply_normals=True)
actors = gltf_obj.actors()

scene.add(*actors)

window.show(scene)
