from fury import actor, window, utils
import numpy as np
import pybullet as p

client = p.connect(p.DIRECT)
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

ball_mass = 10
ball_radius = 5

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





scene = window.Scene()
scene.add(actor.axes(scale=(0.5, 0.5, 0.5)), base_actor, brick_actor)

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)

showm.initialize()

base_pos, base_orn = p.getBasePositionAndOrientation(base)
base_actor.SetPosition(*base_pos)

vertices = utils.vertices_from_actor(brick_actor)
num_vertices = vertices.shape[0]
num_objects = brick_centers.shape[0]
sec = np.int(num_vertices / num_objects)


# Function for syncing actors with multibodies.
def sync_brick(object_index, multibody):
    pos, orn = p.getBasePositionAndOrientation(multibody)

    rot_mat = np.reshape(
        p.getMatrixFromQuaternion(
            p.getDifferenceQuaternion(orn, brick_orns[object_index])),
        (3, 3))

    vertices[object_index * sec: object_index * sec + sec] = \
        (vertices[object_index * sec: object_index * sec + sec] -
         brick_centers[object_index])@rot_mat + pos

    brick_centers[object_index] = pos
    brick_orns[object_index] = orn


# Create timer callback which will execute at each step of simulation.
def timer_callback(_obj, _event):
    showm.render()

    # Updating the position and orientation of each individual brick.
    for idx, brick in enumerate(bricks):
        sync_brick(idx, brick)
    utils.update_actor(brick_actor)

    # Simulate a step.
    p.stepSimulation()


# Add the timer callback to showmanager.
# Increasing the duration value will slow down the simulation.
showm.add_timer_callback(True, 1, timer_callback)

interactive = True

# start simulation
if interactive:
    showm.start()
