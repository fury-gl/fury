from fury import actor, window, utils
import numpy as np
import pybullet as p

p.connect(p.GUI)
p.setGravity(0, 0, -10)

# Parameters
wall_length = 5
wall_breadth = 5
wall_height = 5

brick_size = np.array([0.2, 0.4, 0.2])

chain_segments = 10
segment_length = 0.1
segment_radius = 0.5
segment_mass = 0.1
segment_color = np.array([1, 0, 0])

ball_mass = 10
ball_radius = 0.5
ball_color = np.array([[1, 0, 0]])

joint_friction = 0.0005

# Base
base_actor = actor.box(centers=np.array([[0, 0, 0]]),
                       directions=[0, 0, 0],
                       scale=(5, 5, 0.2),
                       colors=(1, 1, 1))
base_coll = p.createCollisionShape(p.GEOM_BOX,
                                   halfExtents=[2.5, 2.5, 0.1])
base = p.createMultiBody(
                          baseCollisionShapeIndex=base_coll,
                          basePosition=[0, 0, -0.1],
                          baseOrientation=[0, 0, 0, 1])
p.changeDynamics(base, -1, lateralFriction=0.3, restitution=0.5)


# Generate bricks.
nb_bricks = wall_length*wall_breadth*wall_height
brick_centers = np.zeros((nb_bricks, 3))

brick_directions = np.zeros((nb_bricks, 3))
brick_directions[:] = np.array([1.57, 0, 0])

brick_orns = np.zeros((nb_bricks, 4))

brick_sizes = np.zeros((nb_bricks, 3))
brick_sizes[:] = brick_size

brick_colors = np.random.rand(nb_bricks, 3)

brick_coll = p.createCollisionShape(p.GEOM_BOX,
                                    halfExtents=brick_size/2)

bricks = np.zeros(nb_bricks, dtype=np.int16)

idx = 0
# Setting up wall
for i in range(wall_length):
    for k in range(wall_height):
        for j in range(wall_breadth):
            center_pos = np.array([(i*0.2)-1.8, (j*0.4)-0.9, (0.2*k)+0.1])
            brick_centers[idx] = center_pos
            brick_orns[idx] = np.array([0, 0, 0, 1])
            bricks[idx] = p.createMultiBody(baseMass=0.5,
                                            baseCollisionShapeIndex=brick_coll,
                                            basePosition=center_pos,
                                            baseOrientation=brick_orns[i])
            p.changeDynamics(bricks[idx], -1, lateralFriction=0.1,
                             restitution=0.1)
            idx += 1

brick_actor = actor.box(centers=brick_centers,
                        directions=brick_directions,
                        scale=brick_sizes,
                        colors=brick_colors)


# Generate wrecking ball
link_shape = p.createCollisionShape(p.GEOM_CYLINDER,
                                    radius=segment_radius,
                                    height=segment_length,
                                    collisionFramePosition=[0, 0,
                                                            -segment_length/2])

ball_shape = p.createCollisionShape(p.GEOM_SPHERE,
                                    radius=ball_radius)

visualShapeId = -1

link_masses = np.zeros(chain_segments)
link_masses[:] = segment_mass

linkCollisionShapeIndices = np.zeros(chain_segments)
linkCollisionShapeIndices[:] = np.array(link_shape)
linkVisualShapeIndices = -1 * np.ones(chain_segments)
linkPositions = np.zeros((chain_segments, 3))
linkPositions[:] = np.array([0, 0, segment_length])
linkOrientations = np.zeros((chain_segments, 4))
linkOrientations[:] = np.array([0, 0, 0, 1])
linkInertialFramePos = np.zeros((chain_segments, 3))
linkInertialFrameOrns = np.zeros((chain_segments, 4))
linkInertialFrameOrns[:] = np.array([0, 0, 0, 1])
indices = np.arange(chain_segments)
jointTypes = np.zeros(chain_segments)
jointTypes[:] = np.array(p.JOINT_SPHERICAL)
axis = np.zeros((chain_segments, 3))
axis[:] = np.array([1, 0, 0])

linkDirections = np.zeros((chain_segments, 3))
linkDirections[:] = np.array([1, 1, 1])

link_radii = np.zeros(chain_segments)
link_radii[:] = segment_radius

link_heights = np.zeros(chain_segments)
link_heights[:] = segment_length

link_colors = np.zeros((chain_segments, 3))
link_colors[:] = segment_color

chain_actor = actor.cylinder(centers=linkPositions,
                             directions=linkDirections,
                             colors=link_colors,
                             radius=segment_radius,
                             heights=link_heights, capped=True)

basePosition = [0, 0, 2]
baseOrientation = [0, 0, 0, 1]
chain = p.createMultiBody(ball_mass, ball_shape, visualShapeId,
                          basePosition, baseOrientation,
                          linkMasses=link_masses,
                          linkCollisionShapeIndices=linkCollisionShapeIndices,
                          linkVisualShapeIndices=linkVisualShapeIndices,
                          linkPositions=linkPositions,
                          linkOrientations=linkOrientations,
                          linkInertialFramePositions=linkInertialFramePos,
                          linkInertialFrameOrientations=linkInertialFrameOrns,
                          linkParentIndices=indices,
                          linkJointTypes=jointTypes,
                          linkJointAxis=axis)

friction_vec = [joint_friction]*3   # same all axis
control_mode = p.POSITION_CONTROL   # set pos control mode
for j in range(p.getNumJoints(chain)):
    p.setJointMotorControlMultiDof(chain, j, control_mode,
                                   targetPosition=[0, 0, 0, 1],
                                   targetVelocity=[0, 0, 0],
                                   positionGain=0,
                                   velocityGain=1,
                                   force=friction_vec)

root_hinge = p.createConstraint(chain, indices[-1], -1, -1,
                                p.JOINT_FIXED, [0, 0, 0],
                                [0, 0, 0], [0, 0, 2])

ball_actor = actor.sphere(centers=np.array([[0, 0, 0]]),
                          radii=ball_radius,
                          colors=ball_color)

scene = window.Scene()
scene.add(actor.axes(scale=(0.5, 0.5, 0.5)), base_actor, brick_actor)
scene.add(chain_actor, ball_actor)

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)

showm.initialize()

base_pos, base_orn = p.getBasePositionAndOrientation(base)
base_actor.SetPosition(*base_pos)

brick_vertices = utils.vertices_from_actor(brick_actor)
num_vertices = brick_vertices.shape[0]
num_objects = brick_centers.shape[0]
brick_sec = np.int(num_vertices / num_objects)

chain_vertices = utils.vertices_from_actor(chain_actor)
num_vertices = chain_vertices.shape[0]
num_objects = brick_centers.shape[0]
chain_sec = np.int(num_vertices / num_objects)

# Function for syncing actors with multibodies.
def sync_brick(object_index, multibody):
    pos, orn = p.getBasePositionAndOrientation(multibody)

    rot_mat = np.reshape(
        p.getMatrixFromQuaternion(
            p.getDifferenceQuaternion(orn, brick_orns[object_index])),
        (3, 3))

    sec = brick_sec

    brick_vertices[object_index * sec: object_index * sec + sec] = \
        (brick_vertices[object_index * sec: object_index * sec + sec] -
         brick_centers[object_index])@rot_mat + pos

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
                p.getDifferenceQuaternion(orn, linkOrientations[joint])),
            (3, 3))

        sec = chain_sec

        chain_vertices[joint * sec: joint * sec + sec] =\
            (chain_vertices[joint * sec: joint * sec + sec] -
             linkPositions[joint])@rot_mat + pos

        linkPositions[joint] = pos
        linkOrientations[joint] = orn


# Create timer callback which will execute at each step of simulation.
def timer_callback(_obj, _event):
    showm.render()

    # Updating the position and orientation of each individual brick.
    for idx, brick in enumerate(bricks):
        sync_brick(idx, brick)

    pos, _ = p.getBasePositionAndOrientation(chain)
    ball_actor.SetPosition(*pos)
    sync_chain(chain_actor, chain)
    utils.update_actor(brick_actor)
    utils.update_actor(chain_actor)

    # Simulate a step.
    p.stepSimulation()


# Add the timer callback to showmanager.
# Increasing the duration value will slow down the simulation.
showm.add_timer_callback(True, 1, timer_callback)

interactive = True

# start simulation
if interactive:
    showm.start()
