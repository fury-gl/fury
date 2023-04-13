import sys
sys.path.append('C:/Users/Lampada/Desktop/GSoC/fury/fury/')

from trees.octree import *
from fury.lib import Points, CellArray, PolyData, PolyDataMapper, Actor, Renderer, RenderWindow, RenderWindowInteractor
from fury import window, actor, utils, lib

    
# TESTING A QUADTREE OF RANDOM 2D POINTS
br = branch2d(4, 0.0, 1.0, 0.0, 1.0)

npoints = 200

points = np.random.rand(npoints, 2)

tree = Tree2d(br)
for i in range(npoints):
    tree.AddPoint(point2d(points[i, 0], points[i, 1]))

print(tree)  



# TESTING AN OCTREE WITH RANDOM 3D POINTS
br = branch3d(4, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
npoints = 200

data = np.random.rand(npoints, 3)
np.random.seed()

tree = Tree3d(br)
for i in range(npoints):
    tree.AddPoint(point3d(data[i, 0], data[i, 1], data[i, 2]))

print(tree)


# BELOW, THE ABSTRACT PROCESSING METHOD 
def abstract(branch : branch3d, number):
    print(number + branch.GetPointsNumber())

tree.GetRoot().ProcessBranch(abstract, 10)



# FOR THIS EXAMPLE, LET'S RENDER THE OCTREE WITH THE PROVIDED FUNCTIONS
scene = window.Scene()
scene.set_camera(position=(-6, 5, -10), 
                 focal_point=(tree.GetRoot().GetXMiddlePoint(), 
                              tree.GetRoot().GetYMiddlePoint(), 
                              tree.GetRoot().GetZMiddlePoint()),
                 view_up=(0.0, 0.0, 0.0))
showmanager = window.ShowManager(scene, "trees demo", (1080, 1080), reset_camera = True, order_transparent=True)



actorPoints = actor.dots(data, (1.0, 0.5, 0.4), 1, 5)
scene.add(actorPoints)


actorslist = GetActorFromBranch(tree.GetRoot())

for i in range(len(actorslist)):
    scene.add(actorslist[i])


interactor = lib.RenderWindowInteractor()
interactor.SetRenderWindow(showmanager.window)

interactor.Initialize()
showmanager.render()
interactor.Start()
