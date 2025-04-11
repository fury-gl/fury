import numpy as np
from fury import actor, window
from dipy.data import read_mni_template

###############################################################################
# Let's read the 3D data from dipy

nifti = read_mni_template()
data = np.asarray(nifti.dataobj)
print(data.shape)

###############################################################################
# Create slice actor to visualize the 3D data as XY, YZ and XZ slices.

slicer_actor = actor.slicer(data)
scene = window.Scene()
scene.add(slicer_actor)


def handle_pick_event(event):
    info = event.pick_info
    intensity = np.asarray(info["rgba"].rgb).mean()
    print(f"Voxel {info['index']}: {intensity:.2f}")


def handle_wheel_event(event):
    position = slicer_actor.get_slices()
    position += event.dy // 20
    position = np.maximum(np.zeros((3,)), position)
    position = np.minimum(np.asarray(data.shape), position)
    slicer_actor.show_slices(position)


slicer_actor.add_event_handler(handle_pick_event, "pointer_down")
slicer_actor.add_event_handler(handle_wheel_event, "wheel")

if __name__ == "__main__":
    show_m = window.ShowManager(scene=scene, title="FURY 2.0: Slicer Example")
    show_m.start()
