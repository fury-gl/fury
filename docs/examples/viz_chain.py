"""
=====================
Chain Simulation
=====================

This example simulation shows how to use pybullet to render physics simulations
in fury. In this example we specifically render a Chain oscillating to and from.

First some imports.
"""
import itertools

import numpy as np
import pybullet as p

from fury import actor, ui, utils, window

###############################################################################
# Setup pybullet and add gravity.

p.connect(p.DIRECT)

# Apply gravity to the scene.
p.setGravity(0, 0, -10)

###############################################################################
# Now we render the Chain using the following parameters and definitions.

# Parameters
n_links = 20
dx_link = 0.1  # Size of segments
link_mass = 0.5
base_mass = 0.1
radii = 0.5

joint_friction = 0.0005  # rotational joint friction [N/(rad/s)]

link_shape = p.createCollisionShape(
    p.GEOM_CYLINDER,
    radius=radii,
    height=dx_link,
    collisionFramePosition=[0, 0, -dx_link / 2],
)

base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.01])

visualShapeId = -1

link_Masses = np.zeros(n_links)
link_Masses[:] = link_mass

linkCollisionShapeIndices = np.zeros(n_links)
linkCollisionShapeIndices[:] = np.array(link_shape)
linkVisualShapeIndices = -1 * np.ones(n_links)
linkPositions = np.zeros((n_links, 3))
linkPositions[:] = np.array([0, 0, -dx_link])
linkOrientations = np.zeros((n_links, 4))
linkOrientations[:] = np.array([0, 0, 0, 1])
linkInertialFramePositions = np.zeros((n_links, 3))
linkInertialFrameOrns = np.zeros((n_links, 4))
linkInertialFrameOrns[:] = np.array([0, 0, 0, 1])
indices = np.arange(n_links)
jointTypes = np.zeros(n_links)
jointTypes[:] = np.array(p.JOINT_SPHERICAL)
axis = np.zeros((n_links, 3))
axis[:] = np.array([1, 0, 0])

linkDirections = np.zeros((n_links, 3))
linkDirections[:] = np.array([1, 1, 1])

link_radii = np.zeros(n_links)
link_radii[:] = radii

link_heights = np.zeros(n_links)
link_heights[:] = dx_link

rope_actor = actor.cylinder(
    centers=linkPositions,
    directions=linkDirections,
    colors=np.random.rand(n_links, 3),
    radius=radii,
    heights=link_heights,
    capped=True,
)

basePosition = [0, 0, 2]
baseOrientation = [0, 0, 0, 1]
rope = p.createMultiBody(
    base_mass,
    base_shape,
    visualShapeId,
    basePosition,
    baseOrientation,
    linkMasses=link_Masses,
    linkCollisionShapeIndices=linkCollisionShapeIndices.astype(int),
    linkVisualShapeIndices=linkVisualShapeIndices.astype(int),
    linkPositions=linkPositions,
    linkOrientations=linkOrientations,
    linkInertialFramePositions=linkInertialFramePositions,
    linkInertialFrameOrientations=linkInertialFrameOrns,
    linkParentIndices=indices.astype(int),
    linkJointTypes=jointTypes.astype(int),
    linkJointAxis=axis.astype(int),
)

###############################################################################
# We remove stiffness among the joints by adding friction to them.

friction_vec = [joint_friction] * 3  # same all axis
control_mode = p.POSITION_CONTROL  # set pos control mode
for j in range(p.getNumJoints(rope)):
    p.setJointMotorControlMultiDof(
        rope,
        j,
        control_mode,
        targetPosition=[0, 0, 0, 1],
        targetVelocity=[0, 0, 0],
        positionGain=0,
        velocityGain=1,
        force=friction_vec,
    )

###############################################################################
# Next, we define a constraint base that will help us in the oscillation of the
# chain.

root_robe_c = p.createConstraint(
    rope, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 2]
)

# some traj to inject motion
amplitude_x = 0.3
amplitude_y = 0.0
freq = 0.6

base_actor = actor.box(
    centers=np.array([[0, 0, 0]]),
    directions=np.array([[0, 0, 0]]),
    scales=(0.02, 0.02, 0.02),
    colors=np.array([[1, 0, 0]]),
)

###############################################################################
# We add the necessary actors to the scene.

scene = window.Scene()
scene.background((1, 1, 1))
scene.set_camera((2.2, -3.0, 3.0), (-0.3, 0.6, 0.7), (-0.2, 0.2, 1.0))
scene.add(actor.axes(scale=(0.1, 0.1, 0.1)))
scene.add(rope_actor)
scene.add(base_actor)

# Create show manager.
showm = window.ShowManager(
    scene, size=(900, 768), reset_camera=False, order_transparent=True
)


# Counter iterator for tracking simulation steps.
counter = itertools.count()


###############################################################################
# We define a couple of syncing methods for the base and chain.

# Function for syncing actors with multi-bodies.
def sync_actor(actor, multibody):
    pos, orn = p.getBasePositionAndOrientation(multibody)
    actor.SetPosition(*pos)
    orn_deg = np.degrees(p.getEulerFromQuaternion(orn))
    actor.SetOrientation(*orn_deg)


vertices = utils.vertices_from_actor(rope_actor)
num_vertices = vertices.shape[0]
num_objects = linkPositions.shape[0]
sec = int(num_vertices / num_objects)


def sync_joints(actor_list, multibody):
    for joint in range(p.getNumJoints(multibody)):
        # `p.getLinkState` offers various information about the joints
        # as a list and the values in 4th and 5th index refer to the joint's
        # position and orientation respectively.
        pos, orn = p.getLinkState(multibody, joint)[4:6]

        rot_mat = np.reshape(
            p.getMatrixFromQuaternion(
                p.getDifferenceQuaternion(orn, linkOrientations[joint])
            ),
            (3, 3),
        )

        vertices[joint * sec : joint * sec + sec] = (
            vertices[joint * sec : joint * sec + sec] - linkPositions[joint]
        ) @ rot_mat + pos

        linkPositions[joint] = pos
        linkOrientations[joint] = orn


###############################################################################
# We define a TextBlock to display the Avg. FPS and Simulation steps.

fpss = np.array([])
tb = ui.TextBlock2D(
    position=(0, 680), font_size=30, color=(1, 0.5, 0), text='Avg. FPS: \nSim Steps: '
)
scene.add(tb)

t = 0.0
freq_sim = 240


###############################################################################
# Timer callback to sync objects, simulate steps and oscillate the base.


def timer_callback(_obj, _event):
    cnt = next(counter)
    global t, fpss
    showm.render()

    t += 1.0 / freq_sim

    if cnt % 1 == 0:
        fps = showm.frame_rate
        fpss = np.append(fpss, fps)
        tb.message = (
            'Avg. FPS: ' + str(np.round(np.mean(fpss), 0)) + '\nSim Steps: ' + str(cnt)
        )

    # some trajectory
    ux = amplitude_x * np.sin(2 * np.pi * freq * t)
    uy = amplitude_y * np.cos(2 * np.pi * freq * t)

    # move base around
    pivot = [3 * ux, uy, 2]
    orn = p.getQuaternionFromEuler([0, 0, 0])
    p.changeConstraint(root_robe_c, pivot, jointChildFrameOrientation=orn, maxForce=500)

    # Sync base and chain.
    sync_actor(base_actor, rope)
    sync_joints(rope_actor, rope)
    utils.update_actor(rope_actor)

    # Simulate a step.
    p.stepSimulation()

    # Exit after 2000 steps of simulation.
    if cnt == 130:
        showm.exit()


# Add the timer callback to showmanager.
# Increasing the duration value will slow down the simulation.
showm.add_timer_callback(True, 1, timer_callback)

interactive = False

# start simulation
if interactive:
    showm.start()

window.record(scene, size=(900, 768), out_path='viz_chain.png')
