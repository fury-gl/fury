# Beads rendered in Pybullet

import time

import numpy as np
import pybullet as p

p.connect(p.GUI)
plane = p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0, plane)

orientation = [0, 0, 0, -1]
radius = 0.091
mass = 0.1

linkCol = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
linkVis = p.createVisualShape(
    p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1], specularColor=[1, 1, 1, 1]
)

n = 50

link_Masses = []
linkCollisionShapeIndices = []
linkVisualShapeIndices = []
linkPos = np.random.rand(n, 3)

linkPos[:][2] += 15
linkOrientations = []
linkInertialFramePositions = []
linkInertialFrameOrientations = []
indices = []
jointTypes = []
axis = []

for i in range(len(linkPos)):
    link_Masses.append(mass)
    linkCollisionShapeIndices.append(linkCol)
    linkVisualShapeIndices.append(linkVis)
    linkOrientations.append(orientation)
    linkInertialFramePositions.append([0, 0, 0])
    linkInertialFrameOrientations.append(orientation)
    indices.append(i)
    jointTypes.append(p.JOINT_FIXED)
    axis.append([0, 0, 0])

link_Masses = link_Masses[:n]
linkCollisionShapeIndices = linkCollisionShapeIndices[:n]
linkVisualShapeIndices = linkVisualShapeIndices[:n]
linkPos = linkPos[:n]
linkOrientations = linkOrientations[:n]
linkInertialFramePositions = linkInertialFramePositions[:n]
linkInertialFrameOrientations = linkInertialFrameOrientations[:n]
indices = indices[:n]
jointTypes = jointTypes[:n]
axis = axis[:n]

p.createMultiBody(
    baseMass=mass,
    baseCollisionShapeIndex=linkCol,
    baseVisualShapeIndex=linkVis,
    basePosition=[-5.444793, 17.301618, 0],
    baseOrientation=orientation,
    linkMasses=link_Masses,
    linkCollisionShapeIndices=linkCollisionShapeIndices,
    linkVisualShapeIndices=linkVisualShapeIndices,
    linkPositions=linkPos,
    linkOrientations=linkOrientations,
    linkInertialFramePositions=linkInertialFramePositions,
    linkInertialFrameOrientations=linkInertialFrameOrientations,
    linkParentIndices=indices,
    linkJointTypes=jointTypes,
    linkJointAxis=axis,
    useMaximalCoordinates=True,
)

p.setGravity(0, 0, -9.8)
dt = 1.0 / 120.0
p.setRealTimeSimulation(1)

for i in range(1000000):
    p.stepSimulation()
    time.sleep(dt)

exit()
