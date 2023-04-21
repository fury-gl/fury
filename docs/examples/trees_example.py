import trees
from fury import window, actor, lib
import numpy as np


# TESTING A QUADTREE OF RANDOM 2D POINTS
br = trees.Branch2d(4, 0.0, 1.0, 0.0, 1.0)

npoints = 5

points = np.random.rand(npoints, 2)

tree = trees.Tree2d(br)
for i in range(npoints):
    tree.add_point(trees.Point2d(points[i, 0], points[i, 1]))

actorslist = trees.actor_from_branch_2d(tree.root(), (1.0, 1.0, 1.0), 1.0)
zeros = np.zeros((npoints, 1))
actorPoints = actor.dots(np.hstack((points, zeros)), (1.0, 0.5, 0.4), 1, 5)

scene = window.Scene()
scene.set_camera(position=(-6, 5, -10),
                 focal_point=(tree.root().x_mid_point(),
                              tree.root().y_mid_point(),
                              0.0),
                 view_up=(0.0, 0.0, 0.0))
showmanager = window.ShowManager(
    scene,
    "trees demo",
    (1920,
     1080),
    reset_camera=True,
    order_transparent=True)

scene.add(actorPoints)

if tree.root().is_divided():
    for i in range(len(actorslist)):
        scene.add(actorslist[i])
else:
    scene.add(actorslist)


interactor = lib.RenderWindowInteractor()
interactor.SetRenderWindow(showmanager.window)

interactor.Initialize()
showmanager.render()
interactor.Start()


# TESTING THE UPDATE FUNCTION
def divide(branch, div : float):
    for i in range(branch.points_list().shape[0]):
        update_coord = branch.points_list()[i]()/div
        branch.points_list()[i].set_coord(update_coord[0], update_coord[1])

div = 2.0
tree.root().process_branch(divide, div)

print("Before update")
print(tree)
tree.root().update()
print("After update")
print(tree)
actorslist = trees.actor_from_branch_2d(tree.root(), (1.0, 1.0, 1.0), 1.0)
zeros = np.zeros((npoints, 1))
actorPoints = actor.dots(np.hstack((points, zeros))/div, (1.0, 0.5, 0.4), 1, 5)

scene = window.Scene()
scene.set_camera(position=(-6, 5, -10),
                 focal_point=(tree.root().x_mid_point(),
                              tree.root().y_mid_point(),
                              0.0),
                 view_up=(0.0, 0.0, 0.0))
showmanager = window.ShowManager(
    scene,
    "trees demo",
    (1920,
     1080),
    reset_camera=True,
    order_transparent=True)

scene.add(actorPoints)

if tree.root().is_divided():
    for i in range(len(actorslist)):
        scene.add(actorslist[i])
else:
    scene.add(actorslist)


interactor = lib.RenderWindowInteractor()
interactor.SetRenderWindow(showmanager.window)

interactor.Initialize()
showmanager.render()
interactor.Start()



# TESTING AN OCTREE WITH RANDOM 3D POINTS
xl = 1.0
yl = 2.0
zl = 3.0
br = trees.Branch3d(4, 0.0, xl, 0.0, yl, 0.0, zl)
npoints = 200

np.random.seed(101)
data = np.random.rand(npoints, 3)
data[:, 0] = xl*data[:, 0]
data[:, 1] = yl*data[:, 1]
data[:, 2] = zl*data[:, 2]

tree = trees.Tree3d(br)
for i in range(npoints):
    tree.add_point(trees.Point3d(data[i, 0], data[i, 1], data[i, 2]))

print(tree)


# BELOW, THE ABSTRACT PROCESSING METHOD
def abstract(branch : trees.Branch3d, number):
    return number + branch.n_points()


print(tree.root().process_branch(abstract, 0))


# FOR THIS EXAMPLE, LET'S RENDER THE OCTREE WITH THE PROVIDED FUNCTIONS
scene = window.Scene()
scene.set_camera(position=(-6, 5, -10),
                 focal_point=(tree.root().x_mid_point(),
                              tree.root().y_mid_point(),
                              tree.root().z_mid_point()),
                 view_up=(0.0, 0.0, 0.0))
showmanager = window.ShowManager(
    scene,
    "trees demo",
    (1920,
     1080),
    reset_camera=True,
    order_transparent=True)


actorPoints = actor.dots(data, (1.0, 0.5, 0.4), 1, 5)
scene.add(actorPoints)


actorslist = trees.actor_from_branch_3d(tree.root(), (1.0, 1.0, 1.0), 1.0)


if tree.root().is_divided():
    for i in range(len(actorslist)):
        scene.add(actorslist[i])
else:
    scene.add(actorslist)


interactor = lib.RenderWindowInteractor()
interactor.SetRenderWindow(showmanager.window)

interactor.Initialize()
showmanager.render()
interactor.Start()


def divide(branch, div : float):
    for i in range(branch.points_list().shape[0]):
        update_coord = branch.points_list()[i]()/div
        branch.points_list()[i].set_coord(update_coord[0], update_coord[1], update_coord[2])

        
div = 2.0
tree.root().process_branch(divide, div)

tree.root().update()

actorslist = trees.actor_from_branch_3d(tree.root(), (1.0, 1.0, 1.0), 1.0)
zeros = np.zeros((npoints, 1))
actorPoints = actor.dots(np.hstack((data, zeros))/div, (1.0, 0.5, 0.4), 1, 5)

scene = window.Scene()
scene.set_camera(position=(-6, 5, -10),
                 focal_point=(tree.root().x_mid_point(),
                              tree.root().y_mid_point(),
                              tree.root().z_mid_point()),
                 view_up=(0.0, 0.0, 0.0))
showmanager = window.ShowManager(
    scene,
    "trees demo",
    (1920,
     1080),
    reset_camera=True,
    order_transparent=True)


actorPoints = actor.dots(data/div, (1.0, 0.5, 0.4), 1, 5)
scene.add(actorPoints)


actorslist = trees.actor_from_branch_3d(tree.root(), (1.0, 1.0, 1.0), 1.0)


if tree.root().is_divided():
    for i in range(len(actorslist)):
        scene.add(actorslist[i])
else:
    scene.add(actorslist)


interactor = lib.RenderWindowInteractor()
interactor.SetRenderWindow(showmanager.window)

interactor.Initialize()
showmanager.render()
interactor.Start()
