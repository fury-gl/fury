"""
========================
Wrecking Ball Simulation
========================

This example simulation shows how to use pybullet to render physics simulations
in fury. In this example we specifically render a brick wall being destroyed by
a wrecking ball.

First some imports.
"""
import itertools

import numpy as np
import pybullet as p

from fury import actor, ui, utils, window

###############################################################################
# Initiate pybullet and enable gravity.

p.connect(p.DIRECT)
p.setGravity(0, 0, -10)

###############################################################################
# Define some handy parameters to customize simulation.

# Parameters
wall_length = 5
wall_breadth = 5
wall_height = 5

brick_size = np.array([0.2, 0.4, 0.2])

n_links = 15
# Size of segments
dx_link = 0.1
link_mass = 0.5
base_mass = 0.1
# radius of the cylindrical links or the rope
radii = 0.1

ball_mass = 10
# radius of the wrecking ball
ball_radius = 0.5
ball_color = np.array([[1, 0, 0]])

joint_friction = 0.0005

###############################################################################
# Creating the base plane actor.

# Base
base_actor = actor.box(
    centers=np.array([[0, 0, 0]]),
    directions=[0, 0, 0],
    scales=(5, 5, 0.2),
    colors=(1, 1, 1),
)
base_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2.5, 2.5, 0.1])
base = p.createMultiBody(
    baseCollisionShapeIndex=base_coll,
    basePosition=[0, 0, -0.1],
    baseOrientation=[0, 0, 0, 1],
)
p.changeDynamics(base, -1, lateralFriction=0.3, restitution=0.5)

###############################################################################
# The following definitions are made to render a NxNxN brick wall.

# Generate bricks.
nb_bricks = wall_length * wall_breadth * wall_height
brick_centers = np.zeros((nb_bricks, 3))

brick_directions = np.zeros((nb_bricks, 3))
brick_directions[:] = np.array([1.57, 0, 0])

brick_orns = np.zeros((nb_bricks, 4))

brick_sizes = np.zeros((nb_bricks, 3))
brick_sizes[:] = brick_size

brick_colors = np.random.rand(nb_bricks, 3)

brick_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=brick_size / 2)

bricks = np.zeros(nb_bricks, dtype=np.int16)

###############################################################################
# The following is the logic to position the bricks in our desired location and
# generate the actor.

idx = 0
# Setting up wall
for i in range(wall_length):
    for k in range(wall_height):
        for j in range(wall_breadth):
            center_pos = np.array([(i * 0.2) - 1.8, (j * 0.4) - 0.9, (0.2 * k) + 0.1])
            brick_centers[idx] = center_pos
            brick_orns[idx] = np.array([0, 0, 0, 1])
            bricks[idx] = p.createMultiBody(
                baseMass=0.5,
                baseCollisionShapeIndex=brick_coll,
                basePosition=center_pos,
                baseOrientation=brick_orns[i],
            )
            p.changeDynamics(bricks[idx], -1, lateralFriction=0.1, restitution=0.1)
            idx += 1

brick_actor = actor.box(
    centers=brick_centers,
    directions=brick_directions,
    scales=brick_sizes,
    colors=brick_colors,
)

###############################################################################
# Now we render the wrecking ball consisting of a fixed hinge, a ball and rope.

# Generate wrecking ball
link_shape = p.createCollisionShape(
    p.GEOM_CYLINDER,
    radius=radii,
    height=dx_link,
    collisionFramePosition=[0, 0, -dx_link / 2],
)

base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.01])
ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)

visualShapeId = -1

link_Masses = np.zeros(n_links)
link_Masses[:] = link_mass
link_Masses[-1] = 5
linkCollisionShapeIndices = np.zeros(n_links)
linkCollisionShapeIndices[:] = np.array(link_shape)
linkCollisionShapeIndices[-1] = ball_shape
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
    linkPositions=linkPositions.astype(int),
    linkOrientations=linkOrientations.astype(int),
    linkInertialFramePositions=linkInertialFramePositions.astype(int),
    linkInertialFrameOrientations=linkInertialFrameOrns.astype(int),
    linkParentIndices=indices.astype(int),
    linkJointTypes=jointTypes.astype(int),
    linkJointAxis=axis.astype(int),
)

###############################################################################
# Next we define the frictional force between the joints of wrecking ball.

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
# We add the following constraint to keep the cubical hinge fixed.

root_robe_c = p.createConstraint(
    rope, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 2]
)

box_actor = actor.box(
    centers=np.array([[0, 0, 0]]),
    directions=np.array([[0, 0, 0]]),
    scales=(0.02, 0.02, 0.02),
    colors=np.array([[1, 0, 0]]),
)

ball_actor = actor.sphere(
    centers=np.array([[0, 0, 0]]), radii=ball_radius, colors=np.array([1, 0, 1])
)

###############################################################################
# Now we add the necessary actors to the scene and set the camera for better
# visualization.

scene = window.Scene()
scene.set_camera((10.28, -7.10, 6.39), (0.0, 0.0, 0.4), (-0.35, 0.26, 1.0))
scene.add(actor.axes(scale=(0.5, 0.5, 0.5)), base_actor, brick_actor)
scene.add(rope_actor, box_actor, ball_actor)

showm = window.ShowManager(
    scene, size=(900, 768), reset_camera=False, order_transparent=True
)


###############################################################################
# Position the base correctly.

base_pos, base_orn = p.getBasePositionAndOrientation(base)
base_actor.SetPosition(*base_pos)

###############################################################################
# Calculate the vertices of the bricks.

brick_vertices = utils.vertices_from_actor(brick_actor)
num_vertices = brick_vertices.shape[0]
num_objects = brick_centers.shape[0]
brick_sec = int(num_vertices / num_objects)

###############################################################################
# Calculate the vertices of the wrecking ball.

chain_vertices = utils.vertices_from_actor(rope_actor)
num_vertices = chain_vertices.shape[0]
num_objects = brick_centers.shape[0]
chain_sec = int(num_vertices / num_objects)


###############################################################################
# We define methods to sync bricks and wrecking ball.

# Function for syncing actors with multibodies.
def sync_brick(object_index, multibody):
    pos, orn = p.getBasePositionAndOrientation(multibody)

    rot_mat = np.reshape(
        p.getMatrixFromQuaternion(
            p.getDifferenceQuaternion(orn, brick_orns[object_index])
        ),
        (3, 3),
    )

    sec = brick_sec

    brick_vertices[object_index * sec : object_index * sec + sec] = (
        brick_vertices[object_index * sec : object_index * sec + sec]
        - brick_centers[object_index]
    ) @ rot_mat + pos

    brick_centers[object_index] = pos
    brick_orns[object_index] = orn


def sync_chain(actor_list, multibody):
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

        sec = chain_sec

        chain_vertices[joint * sec : joint * sec + sec] = (
            chain_vertices[joint * sec : joint * sec + sec] - linkPositions[joint]
        ) @ rot_mat + pos

        linkPositions[joint] = pos
        linkOrientations[joint] = orn


###############################################################################
# Some helper tools to keep track of avg. FPS and simulation steps.

counter = itertools.count()
fpss = np.array([])
tb = ui.TextBlock2D(
    position=(0, 680), font_size=30, color=(1, 0.5, 0), text='Avg. FPS: \nSim Steps: '
)
scene.add(tb)

###############################################################################
# Timer callback to sync objects, simulate steps and apply force.

apply_force = True


# Create timer callback which will execute at each step of simulation.
def timer_callback(_obj, _event):
    global apply_force, fpss
    cnt = next(counter)
    showm.render()

    if cnt % 1 == 0:
        fps = showm.frame_rate
        fpss = np.append(fpss, fps)
        tb.message = (
            'Avg. FPS: ' + str(np.round(np.mean(fpss), 0)) + '\nSim Steps: ' + str(cnt)
        )

    # Updating the position and orientation of each individual brick.
    for idx, brick in enumerate(bricks):
        sync_brick(idx, brick)

    pos, _ = p.getBasePositionAndOrientation(rope)

    if apply_force:
        p.applyExternalForce(
            rope, -1, forceObj=[-500, 0, 0], posObj=pos, flags=p.WORLD_FRAME
        )
        apply_force = False

    pos = p.getLinkState(rope, p.getNumJoints(rope) - 1)[4]
    ball_actor.SetPosition(*pos)
    sync_chain(rope_actor, rope)
    utils.update_actor(brick_actor)
    utils.update_actor(rope_actor)

    # Simulate a step.
    p.stepSimulation()

    if cnt == 130:
        showm.exit()


# Add the timer callback to showmanager.
# Increasing the duration value will slow down the simulation.
showm.add_timer_callback(True, 1, timer_callback)

interactive = False

# start simulation
if interactive:
    showm.start()

window.record(scene, size=(900, 768), out_path='viz_wrecking_ball.png')
