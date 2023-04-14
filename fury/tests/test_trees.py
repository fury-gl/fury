from fury import window, actor, lib, trees
import numpy as np


# TESTING A QUADTREE OF RANDOM 2D POINTS
br = trees.Branch2d(4, 0.0, 1.0, 0.0, 1.0)

npoints = 200

points = np.random.rand(npoints, 2)

tree = trees.Tree2d(br)
for i in range(npoints):
    tree.AddPoint(trees.Point2d(points[i, 0], points[i, 1]))

print(tree)


actorslist = trees.GetActorFromBranch2d(tree.GetRoot(), (1.0, 1.0, 1.0), 1.0)
zeros = np.zeros((npoints, 1))
actorPoints = actor.dots(np.hstack((points, zeros)), (1.0, 0.5, 0.4), 1, 5)

scene = window.Scene()
scene.set_camera(position=(-6, 5, -10),
                 focal_point=(tree.GetRoot().GetXMiddlePoint(),
                              tree.GetRoot().GetYMiddlePoint(),
                              0.0),
                 view_up=(0.0, 0.0, 0.0))
showmanager = window.ShowManager(
    scene,
    "trees demo",
    (1080,
     1080),
    reset_camera=True,
    order_transparent=True)

scene.add(actorPoints)

if tree.GetRoot().IsDivided():
    for i in range(len(actorslist)):
        scene.add(actorslist[i])
else:
    scene.add(actorslist)


interactor = lib.RenderWindowInteractor()
interactor.SetRenderWindow(showmanager.window)

interactor.Initialize()
showmanager.render()
interactor.Start()




xl = 1.0
yl = 2.0
zl = 3.0


# TESTING AN OCTREE WITH RANDOM 3D POINTS
br = trees.Branch3d(4, 0.0, xl, 0.0, yl, 0.0, zl)
npoints = 200

data = np.random.rand(npoints, 3)
data[:, 0] = xl*data[:, 0]
data[:, 1] = yl*data[:, 1]
data[:, 2] = zl*data[:, 2]
np.random.seed()

tree = trees.Tree3d(br)
for i in range(npoints):
    tree.AddPoint(trees.Point3d(data[i, 0], data[i, 1], data[i, 2]))

print(tree)


# BELOW, THE ABSTRACT PROCESSING METHOD
def abstract(branch : trees.Branch3d, number):
    print(number + branch.GetPointsNumber())


tree.GetRoot().ProcessBranch(abstract, 10)


# FOR THIS EXAMPLE, LET'S RENDER THE OCTREE WITH THE PROVIDED FUNCTIONS
scene = window.Scene()
scene.set_camera(position=(-6, 5, -10),
                 focal_point=(tree.GetRoot().GetXMiddlePoint(),
                              tree.GetRoot().GetYMiddlePoint(),
                              tree.GetRoot().GetZMiddlePoint()),
                 view_up=(0.0, 0.0, 0.0))
showmanager = window.ShowManager(
    scene,
    "trees demo",
    (1080,
     1080),
    reset_camera=True,
    order_transparent=True)


actorPoints = actor.dots(data, (1.0, 0.5, 0.4), 1, 5)
scene.add(actorPoints)


actorslist = trees.GetActorFromBranch3d(tree.GetRoot(), (1.0, 1.0, 1.0), 1.0)


if tree.GetRoot().IsDivided():
    for i in range(len(actorslist)):
        scene.add(actorslist[i])
else:
    scene.add(actorslist)


interactor = lib.RenderWindowInteractor()
interactor.SetRenderWindow(showmanager.window)

interactor.Initialize()
showmanager.render()
interactor.Start()
