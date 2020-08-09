import numpy as np
from fury import window, actor, ui, utils
import itertools
import pybullet as p

# Instantiate Pybullet client.
client = p.connect(p.GUI)
# Apply gravity to the scene.
p.setGravity(0, 0, -10, physicsClientId=client)

###### Creating ceiling Plane
ceil_actor = actor.box(centers=np.array([[0, 0, 0]]),
                         directions=[0,0,0],
                         scale=(5, 5, 0.2) ,
                         colors=(255, 255, 255))
ceil_coll = p.createCollisionShape(p.GEOM_BOX,
                                   halfExtents=[2.5, 2.5, 0.1]) # half of the actual size.
ceil = p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=ceil_coll,
                          basePosition=[0, 0, -0.2],
                          baseOrientation=[ 0, 0, 0, 1 ])

############ Creating String

# Parameters
n_links = 30
dx_link = 0.02
link_mass = 0.005
base_mass = 0.1
radii = 0.005

joint_friction = 0.0005

link_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                    radius=radii,
                                    height=dx_link,
                                    collisionFramePosition=[0, 0, -dx_link/2])

base_shape = p.createCollisionShape(p.GEOM_BOX,
                                    halfExtents=[0.01, 0.01, 0.01])

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
linkInertialFrameOrientations = np.zeros((n_links, 4))
linkInertialFrameOrientations[:] = np.array([0, 0, 0, 1])
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

rope_actor = actor.cylinder(centers=linkPositions,
                            directions=linkDirections,
                            colors=np.random.rand(n_links, 3),
                            radius=radii,
                            heights=link_heights)

basePosition = [0, 0, 2]
baseOrientation = [0, 0, 0, 1]
rope = p.createMultiBody(base_mass,
                              base_shape,
                              visualShapeId,
                              basePosition,
                              baseOrientation,
                              linkMasses=link_Masses,
                              linkCollisionShapeIndices=linkCollisionShapeIndices,
                              linkVisualShapeIndices=linkVisualShapeIndices,
                              linkPositions=linkPositions,
                              linkOrientations=linkOrientations,
                              linkInertialFramePositions=linkInertialFramePositions,
                              linkInertialFrameOrientations=linkInertialFrameOrientations,
                              linkParentIndices=indices,
                              linkJointTypes=jointTypes,
                              linkJointAxis=axis)
                            #   useMaximalCoordinates=useMaximalCoordinates)

# remove stiffness in motors, add friction force

friction_vec = [joint_friction]*3   # same all axis
control_mode = p.POSITION_CONTROL   # set pos control mode
for j in range(p.getNumJoints(rope)):
    p.setJointMotorControlMultiDof(rope,j,control_mode,
                                    targetPosition=[0,0,0,1],
                                    targetVelocity=[0,0,0],
                                    positionGain=0,
                                    velocityGain=1,
                                    force=friction_vec)

# fixed constrain to keep root cube in place
root_robe_c = p.createConstraint(rope, -1, -1, -1,
                                 p.JOINT_FIXED, [0, 0, 0],
                                 [0, 0, 0], [0, 0, 2])

# some traj to inject motion
amplitude_x = 0.3
amplitude_y = 0.0
freq = 0.6

# manually simulate joint damping
Damping = 0.001


base_actor = actor.box(centers=np.array([[0, 0, 0]]),
                       directions=np.array([[0, 0, 0]]),
                       scale=(0.02, 0.02, 0.02),
                       colors=np.array([[1, 0, 0]]))

scene = window.Scene()
scene.background((1, 1, 1))
scene.add(actor.axes())
scene.add(rope_actor)
scene.add(ceil_actor)
scene.add(base_actor)


# Create show manager.
showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)

showm.initialize()

# Counter interator for tracking simulation steps.
counter = itertools.count()

# Function for syncing actors with multibodies.
def sync_actor(actor, multibody):
    pos, orn = p.getBasePositionAndOrientation(multibody)
    actor.SetPosition(*pos)
    orn_deg = np.degrees(p.getEulerFromQuaternion(orn))
    actor.SetOrientation(*orn_deg)

sync_actor(ceil_actor, ceil)

vertices = utils.vertices_from_actor(rope_actor)
num_vertices = vertices.shape[0]
num_objects = linkPositions.shape[0]
sec = np.int(num_vertices / num_objects)

def sync_joints(actor_list, multibody):
    for joint in range(p.getNumJoints(multibody)):
        pos, orn = p.getLinkState(multibody, joint)[4:6]

        rot_mat = np.reshape(
            p.getMatrixFromQuaternion(
                p.getDifferenceQuaternion(orn, linkOrientations[joint])),
            (3, 3))

        vertices[joint * sec: joint * sec + sec] = \
            (vertices[joint * sec: joint * sec + sec] -
            linkPositions[joint])@rot_mat + pos

        linkPositions[joint] = pos
        linkOrientations[joint] = orn


t = 0.0
freq_sim = 240
def timer_callback(_obj, _event):
    cnt = next(counter)
    global t
    showm.render()

    t += 1./freq_sim

    # some trajectory
    ux = amplitude_x*np.sin(2*np.pi*freq*t)
    uy = amplitude_y*np.cos(2*np.pi*freq*t)

    # move base arround
    pivot = [ux, uy, 2]
    orn = p.getQuaternionFromEuler([0, 0, 0])
    p.changeConstraint(root_robe_c, pivot, jointChildFrameOrientation=orn, maxForce=500)


    # Set position and orientation of the ball.
    # sync_actor(ceil_actor, ceil)
    for i in range(p.getNumJoints(rope)):
        # sync_actor(ball_actor, sphere)
        sync_actor(base_actor, rope)
        sync_joints(rope_actor, rope)
        utils.update_actor(rope_actor)

    # Simulate a step.
    p.stepSimulation()

    # Exit after 2000 steps of simulation.
    # if cnt == 2000:
    #     showm.exit()

# Add the timer callback to showmanager.
# Increasing the duration value will slow down the simulation.
showm.add_timer_callback(True, 1, timer_callback)

interactive = True

# start simulation
if interactive: showm.start()
