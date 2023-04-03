import numpy as np
from fury import window, actor

centers = np.array([[0,0.,0]])
directions = np.array([[0.,1,1]])
colors = np.array([[0,0,1.]])
height = np.array([1])
dual_arrow = actor.dualpoint_arrow(centers, directions, colors, heights=height,
          tip_length=0.35, tip_radius=0.1, shaft_radius=0.03, scales=1, resolution=10,
          vertices=None, faces=None, repeat_primitive=True)

# from fury.material import wireframe
# wireframe(dual_arrow, True)

# from fury.material import culling
# culling(dual_arrow, front=True, back=False)

# dual_arrow.GetProperty().SetRepresentationToWireframe()
# dual_arrow.GetProperty().BackfaceCullingOff()
# dual_arrow.GetProperty().FrontfaceCullingOff()

# def create_mesh(actor):
#     actor.GetProperty().SetRepresentationToWireframe()
#     actor.GetProperty().BackfaceCullingOff()
#     actor.GetProperty().FrontfaceCullingOff()
scene = window.Scene()
# scene.background((0.8,0.8,0.9))
scene.add(dual_arrow)
# scene.add(actor.axes())

interactive = True

# if interactive:
#     window.show(scene, size=(1280, 790))

show = window.ShowManager(scene, size=(1280, 790))
show.initialize()

show.start()