# rope render in pybullet

import math
import time

import pybullet as p

p.connect(p.GUI)
p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0, 0)

# PARAMETERS
n_links = 30          # num links
dx_link = 0.02        # length of link segment
link_mass = 0.005     # 5g
base_mass = 0.1       # 100g

joint_friction = 0.0005  # rotational joint friction [N/(rad/s)]

# setup shapes
link_shape = p.createCollisionShape(
    p.GEOM_CYLINDER,
    radius=0.005,
    height=dx_link,
    collisionFramePosition=[0, 0, -dx_link / 2],
)
base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.01])

linkMasses = [link_mass] * n_links
linkCollisionShapeIndices = [link_shape] * n_links
linkVisualShapeIndices = [-1] * n_links

# relative position
linkPositions = []
for i in range(n_links):
    linkPositions.append([0, 0, -dx_link])

# cm positions
linkOrientations = [[0, 0, 0, 1]] * n_links
linkInertialFramePositions = [[0, 0, 0]] * n_links
linkInertialFrameOrns = [[0, 0, 0, 1]] * n_links

# connection graph
indices = range(n_links)

# use spherical joints
jointTypes = [p.JOINT_SPHERICAL] * n_links
jointTypes[1] = p.JOINT_FIXED

# rotational axis (dosnt't matter, spherical)
axis = [[1, 0, 0]] * n_links

# create rope body
visualShapeId = -1
basePosition = [0, 0, 2]
baseOrientation = [0, 0, 0, 1]
rope = p.createMultiBody(
    base_mass,
    base_shape,
    visualShapeId,
    basePosition,
    baseOrientation,
    linkMasses=linkMasses,
    linkCollisionShapeIndices=linkCollisionShapeIndices,
    linkVisualShapeIndices=linkVisualShapeIndices,
    linkPositions=linkPositions,
    linkOrientations=linkOrientations,
    linkInertialFramePositions=linkInertialFramePositions,
    linkInertialFrameOrientations=linkInertialFrameOrns,
    linkParentIndices=indices,
    linkJointTypes=jointTypes,
    linkJointAxis=axis,
)
# flags=p.URDF_USE_SELF_COLLISION)


n_joints = p.getNumJoints(rope)

# remove stiffness in motors, add friction force

friction_vec = [joint_friction] * 3   # same all axis
control_mode = p.POSITION_CONTROL   # set pos control mode
for j in range(n_joints):
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


p.setGravity(0, 0, -9.81)

# fixed constrain to keep root cube in place
root_robe_c = p.createConstraint(
    rope, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 2]
)

# some traj to inject motion
amplitude_x = 0.3
amplitude_y = 0.0
freq = 0.6

# manually simulate joint damping
Damping = 0.001

t = 0.0
freq_sim = 240
while 1:
    time.sleep(1.0 / freq_sim)
    t += 1.0 / freq_sim

    # some trajectory
    ux = amplitude_x * math.sin(2 * math.pi * freq * t)
    uy = amplitude_y * math.cos(2 * math.pi * freq * t)

    # move base around
    pivot = [ux, uy, 2]
    orn = p.getQuaternionFromEuler([0, 0, 0])
    p.changeConstraint(root_robe_c, pivot, jointChildFrameOrientation=orn, maxForce=500)

    # manually apply viscous friction: f_damping = -Damping*omega
    """
    for j in range(n_joints):
    q_state = p.getJointStateMultiDof(rope, j)
    omega = q_state[1]
    f_dampling = [-Damping*v for v in omega]
    p.setJointMotorControlMultiDof(rope, j, p.TORQUE_CONTROL, force=f_dampling)
    """

    # update
    p.stepSimulation()
