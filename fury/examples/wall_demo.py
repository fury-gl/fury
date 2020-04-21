import numpy as np
from fury import window, actor, ui
import itertools
import pybullet as p
import time
import math

client = p.connect(p.GUI)
p.setGravity(0, 0, -10, physicsClientId=client)
xyz = np.array([[0, 0, 0]])
enableCol = 1
f = 1

class storage:
    f = 1
    orn_prev = 0

#BALL
ball = actor.sphere(centers = np.array([[2, 3, 3]]), 
                    colors=np.array([1,1,1]), 
                    radii=0.6)
ball_coll = p.createCollisionShape(p.GEOM_SPHERE,
                                    radius=0.3)
ball_vis = p.createVisualShape(p.GEOM_SPHERE,
                                radius=0.3)
ball_ = p.createMultiBody(baseMass=3,
                          baseCollisionShapeIndex=ball_coll,
                          baseVisualShapeIndex=ball_vis,
                          basePosition=[2, 3, 3],
                          baseOrientation=[ 0, 0, 0, 1 ])
p.changeDynamics(ball_, -1, lateralFriction=0.3, restitution=0.5)
                  
#BASE
base = actor.box(centers=np.array([[-4, 3, -0.1]]),
                         directions=[1.57, 0,0],
                         size=(20, 15, 0.2) ,
                         colors=(0, 1, 0))
base_coll = p.createCollisionShape(p.GEOM_BOX,
                                   halfExtents=[10, 7.5, 0.1])
base_vis = p.createVisualShape(p.GEOM_BOX,
                                   halfExtents=[10, 7.5, 0.1])
base_ = p.createMultiBody(baseVisualShapeIndex=base_vis,
                          baseCollisionShapeIndex=base_coll,
                          basePosition=[-4, 3, -0.1],
                          baseOrientation=[ 0, 0, 0, 1 ])
p.changeDynamics(base_, -1, lateralFriction=0.3, restitution=0.5)


height = 20
base_length = 20


brick_Ids = []
brick_actors = []

scene = window.Scene()
scene.add(ball)
scene.add(base)

print("working....")
for i in range(height):
    temp = []
    temp_actors=[]
    for j in range(base_length):
        pos = np.array([[-1, (0.2+j*0.4), (0.1 + 0.2*i)]])
        brick = actor.box(centers=pos,
                         directions=np.array([1.57, 0,0]),
                         size=(0.2, 0.4, 0.2) ,
                         colors=np.random.rand(1,3))
        scene.add(brick)
        temp_actors.append(brick)
        #physics of the brick

        brick_coll = p.createCollisionShape(p.GEOM_BOX,
                                            halfExtents=[0.1, 0.2, 0.1])
        brick_vis = p.createVisualShape(p.GEOM_BOX,
                                            halfExtents=[0.1, 0.2, 0.1],
                                            rgbaColor=[j*0.1, j*0.2, j*0.4,1])
        brick_ = p.createMultiBody(baseMass=0.5,
                                   baseCollisionShapeIndex=brick_coll,
                                   baseVisualShapeIndex=brick_vis,
                                   basePosition=[-1, (0.2+j*0.4), (0.1 + 0.2*i)],
                                   baseOrientation=[ 0, 0, 0, 1 ])
        p.changeDynamics(brick_, -1, lateralFriction=0.1, restitution=0.1)
        temp.append(brick_)
    brick_Ids.append(temp)
    brick_actors.append(temp_actors)


# showm = window.ShowManager(scene,
#                            size=(900, 768), reset_camera=False,
#                            order_transparent=True)
# showm.initialize()
# counter = itertools.count()
# tb = ui.TextBlock2D(bold=True)


for i in range(100000):
    if storage.f:
        print("entered")
        for j in range(5):
            pos, orn = p.getBasePositionAndOrientation(ball_)
            p.applyExternalForce(ball_, -1,
                                  forceObj=[-5000, 0, 0],
                                  posObj=pos,
                                  flags=p.WORLD_FRAME)
        storage.f = 0
    p.stepSimulation()    

# def timer_callback(_obj, _event):
#     showm.render()
#     tb.message = "simulation"
#     if storage.f:
#         print("entered")
#         for j in range(5):
#             pos, orn = p.getBasePositionAndOrientation(ball_)
#             p.applyExternalForce(ball_, -1,
#                                  forceObj=[-5000, 0, 0],
#                                  posObj=pos,
#                                  flags=p.WORLD_FRAME)
#             storage.f = 0
    
#     for i in range(height):
#         for j in range(base_length):
#             curr_id = brick_Ids[i][j]
#             curr_brick = brick_actors[i][j]
#             pos, orn = p.getBasePositionAndOrientation(curr_id)
            
#             curr_brick.SetPosition(pos[0], pos[1], pos[2])
#             orn_deg = np.degrees(p.getEulerFromQuaternion(orn))
#             curr_brick.SetOrientation(orn_deg[0], orn_deg[1], orn_deg[2])
    
#     pos, orn = p.getBasePositionAndOrientation(ball_)
#     ball.SetPosition(pos[0], pos[1], pos[2])

#     # cuboid_actor.RotateWXYZ(orn[1], orn[2], orn[3], orn[0])
#     p.stepSimulation()
#     # print(orn)
    
#     p.stepSimulation()
#     if counter == 2000:
#         showm.exit()

# showm.add_timer_callback(True, 10, timer_callback)
# showm.start()
# window.record(showm.scene, size=(900, 768), out_path="viz_timer.png")