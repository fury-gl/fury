"""
=========================
Ball Collision Simulation
=========================

This example simulation shows how to use pybullet to render physics simulations
in fury. In this example we render the collision between a blue ball and red
ball and also display a message by confirming the collision.

First some imports.
"""
import numpy as np
from fury import window, actor, ui
import itertools
import pybullet as p

client = p.connect(p.DIRECT)

###############################################################################
# Parameters and definition of red and blue balls.

red_radius = 0.5
blue_radius = 0.5
duration = 50

# Red Ball
red_ball_actor = actor.sphere(centers=np.array([[0, 0, 0]]),
                              colors=np.array([[1, 0, 0]]),
                              radii=red_radius)

red_ball_coll = p.createCollisionShape(p.GEOM_SPHERE, radius=red_radius)

red_ball = p.createMultiBody(baseMass=0.5,
                             baseCollisionShapeIndex=red_ball_coll,
                             basePosition=[10, 0, 0],
                             baseOrientation=[0, 0, 0, 1])

# Blue ball
blue_ball_actor = actor.sphere(centers=np.array([[0, 0, 0]]),
                               colors=np.array([[0, 0, 1]]),
                               radii=blue_radius)

blue_ball_coll = p.createCollisionShape(p.GEOM_SPHERE, radius=blue_radius)

blue_ball = p.createMultiBody(baseMass=0.5,
                              baseCollisionShapeIndex=blue_ball_coll,
                              basePosition=[-10, 0, 0],
                              baseOrientation=[0, 0, 0, 1])

###############################################################################
# We set the coefficient of restitution of both the balls to `0.6`.

p.changeDynamics(red_ball, -1, restitution=0.6)
p.changeDynamics(blue_ball, -1, restitution=0.6)

###############################################################################
# We add all the actors to the scene.

scene = window.Scene()
scene.add(actor.axes())
scene.add(red_ball_actor)
scene.add(blue_ball_actor)

showm = window.ShowManager(scene, size=(900, 700), reset_camera=False,
                           order_transparent=True)

showm.initialize()
counter = itertools.count()

###############################################################################
# Method to sync objects.


def sync_actor(actor, multibody):
    pos, orn = p.getBasePositionAndOrientation(multibody)
    actor.SetPosition(*pos)
    orn_deg = np.degrees(p.getEulerFromQuaternion(orn))
    actor.SetOrientation(*orn_deg)


apply_force = True
tb = ui.TextBlock2D(position=(0, 600), font_size=30, color=(1, 0.5, 0),
                    text="")
scene.add(tb)
scene.set_camera(position=(0.30, -18.78, 0.89),
                 focal_point=(0.15, 0.25, 0.40),
                 view_up=(0, 0, 1.00))


###############################################################################
# Timer callback to sync and step simulation every second.

def timer_callback(_obj, _event):
    global apply_force
    cnt = next(counter)
    showm.render()
    red_pos, red_orn = p.getBasePositionAndOrientation(red_ball)
    blue_pos, blue_orn = p.getBasePositionAndOrientation(blue_ball)

    # Apply force for the first step of the simulation.
    if apply_force:
        p.applyExternalForce(red_ball, -1,
                             forceObj=[-40000, 0, 0],
                             posObj=red_pos,
                             flags=p.WORLD_FRAME)

        p.applyExternalForce(blue_ball, -1,
                             forceObj=[40000, 0, 0],
                             posObj=blue_pos,
                             flags=p.WORLD_FRAME)

        apply_force = 0

    sync_actor(blue_ball_actor, blue_ball)
    sync_actor(red_ball_actor, red_ball)

    # Get various collision information using `p.getContactPoints`.
    contact = p.getContactPoints(red_ball, blue_ball, -1, -1)
    if len(contact) != 0:
        tb.message = "Collision!!"

    p.stepSimulation()

    if cnt == 50:
        showm.exit()


showm.add_timer_callback(True, duration, timer_callback)

interactive = False

if interactive:
    showm.start()

window.record(scene, size=(900, 700), out_path="viz_ball_collide.png")
