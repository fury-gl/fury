from fury import window, material, utils
from fury.gltf import glTF
from fury.io import load_cubemap_texture, load_image
from fury.data import fetch_gltf, read_viz_gltf, read_viz_cubemap
from fury.lib import Texture, ImageFlip

# rgb_array = load_image(
#     '/home/shivam/Downloads/skybox.jpg')

# grid = utils.rgb_to_vtk(rgb_array)
# cubemap = Texture()
# flip = ImageFlip()
# flip.SetInputDataObject(grid)
# flip.SetFilteredAxis(1)
# cubemap.InterpolateOn()
# cubemap.MipmapOn()
# cubemap.SetInputConnection(0, flip.GetOutputPort(0))
# cubemap.UseSRGBColorSpaceOn()

scene = window.Scene(skybox=None)
# scene.SetBackground(0.5, 0.3, 0.3)

fetch_gltf('DamagedHelmet')
filename = read_viz_gltf('DamagedHelmet')

gltf_obj = glTF(filename, apply_normals=True)
actors = gltf_obj.actors()

scene.add(*actors)
scene.UseImageBasedLightingOn()

cameras = gltf_obj.cameras
if cameras:
    scene.SetActiveCamera(cameras[0])

interactive = True

if interactive:
    window.show(scene, size=(1280, 720))

window.record(scene, out_path='viz_gltf.png', size=(1280, 720))
