# To be removed, 1 brick = 1 actor, inefficient.

import numpy as np
from fury import window, actor, ui
import itertools
import pybullet as p

# Instantiate Pybullet client.
client = p.connect(p.DIRECT)
# Apply gravity to the scene.
p.setGravity(0, 0, -10, physicsClientId=client)

# Creating BALL
# Ball actor
ball_actor = actor.sphere(centers=np.array([[0, 0, 0]]),
                          colors=np.array([1, 0, 0]),
                          radii=0.3)

# Collision shape for the ball.
ball_coll = p.createCollisionShape(p.GEOM_SPHERE,
                                   radius=0.3)

# Creating a Multibody which will be tracked by pybullet.
ball = p.createMultiBody(baseMass=3,
                         baseCollisionShapeIndex=ball_coll,
                         basePosition=[2, 0, 1.5],
                         baseOrientation=[0, 0, 0, 1])

# Change the dynamics of the ball by adding friction and restitution.
p.changeDynamics(ball, -1, lateralFriction=0.3, restitution=0.5)

# Creating BASE Plane
base_actor = actor.box(centers=np.array([[0, 0, 0]]),
                       directions=[0, 0, 0],
                       scales=(5, 5, 0.2),
                       colors=(1, 1, 1))
base_coll = p.createCollisionShape(p.GEOM_BOX,
                                   halfExtents=[2.5, 2.5, 0.1])
# half of the actual size.
base = p.createMultiBody(
                          baseCollisionShapeIndex=base_coll,
                          basePosition=[0, 0, -0.1],
                          baseOrientation=[0, 0, 0, 1])
p.changeDynamics(base, -1, lateralFriction=0.3, restitution=0.5)

# defining the height and width of the wall.
wall_height = 10
wall_width = 10

# Lists for keeping track of bricks.
brick_Ids = []
brick_actors = []

# Add the actors to the scene.
scene = window.Scene()
scene.add(actor.axes())
scene.add(ball_actor)
scene.add(base_actor)

# Generate bricks.
for i in range(wall_height):
    temp = []
    temp_actors = []
    for j in range(wall_width):
        pos = np.array([[-1, (0.2+j*0.4), (0.1 + 0.2*i)]])

        # brick defination
        brick_actor = actor.box(centers=np.array([[0, 0, 0]]),
                                directions=np.array([1.57, 0, 0]),
                                scales=(0.2, 0.4, 0.2),
                                colors=np.random.rand(1, 3))
        brick_coll = p.createCollisionShape(p.GEOM_BOX,
                                            halfExtents=[0.1, 0.2, 0.1])
        brick = p.createMultiBody(baseMass=0.5,
                                  baseCollisionShapeIndex=brick_coll,
                                  basePosition=[-1, (j*0.4)-1.8, (0.2*i)+0.1],
                                  baseOrientation=[0, 0, 0, 1])
        p.changeDynamics(brick, -1, lateralFriction=0.1, restitution=0.1)

        # Get the position of brick from pybullet
        pos, _ = p.getBasePositionAndOrientation(brick)
        brick_actor.SetPosition(*pos)

        # NOTE: Collision is enabled by default for dynamic bodies.
        # For example, we can set brick collision with the ball and base
        # explicitly.
        enableCol = 1
        p.setCollisionFilterPair(ball, brick, -1, -1, enableCol)
        p.setCollisionFilterPair(base, brick, -1, -1, enableCol)

        # add the bricks to the scene.
        scene.add(brick_actor)

        temp_actors.append(brick_actor)
        temp.append(brick)

    brick_Ids.append(temp)
    brick_actors.append(temp_actors)

# Create show manager.
showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)



# Counter interator for tracking simulation steps.
counter = itertools.count()

# Variable for tracking applied force.
apply_force = True

# Get the position and orientation of base and set it.
base_pos, base_orn = p.getBasePositionAndOrientation(base)
base_actor.SetPosition(*base_pos)


# Function for syncing actors with multibodies.
def sync_actor(actor, multibody):
    pos, orn = p.getBasePositionAndOrientation(multibody)
    actor.SetPosition(*pos)
    orn_deg = np.degrees(p.getEulerFromQuaternion(orn))
    actor.SetOrientation(*orn_deg)


fpss = np.array([])
tb = ui.TextBlock2D(text="",
                    position=(0, 680), font_size=30, color=(1, 0.5, 0))
scene.add(tb)
scene.set_camera(position=(10.46, -8.13, 6.18),
                 focal_point=(0.0, 0.0, 0.79),
                 view_up=(-0.27, 0.26, 0.90))


# Create timer callback which will execute at each step of simulation.
def timer_callback(_obj, _event):
    global apply_force, fpss
    cnt = next(counter)
    showm.render()

    if cnt % 1 == 0:
        fps = showm.frame_rate
        fpss = np.append(fpss, fps)
        tb.message = "Avg. FPS: " + str(np.round(np.mean(fpss), 0)) +\
            "\nSim Steps: " + str(cnt)

    # Get the position and orientation of the ball.
    ball_pos, ball_orn = p.getBasePositionAndOrientation(ball)

    # Apply force for 5 times for the first step of simulation.
    if apply_force:
        # Apply the force.
        p.applyExternalForce(ball, -1,
                             forceObj=[-10000, 0, 0],
                             posObj=ball_pos,
                             flags=p.WORLD_FRAME)
        apply_force = False

    # Set position and orientation of the ball.
    sync_actor(ball_actor, ball)

    # Updating the position and orientation of each individual brick.
    for i, brick_row in enumerate(brick_actors):
        for j, brick_actor in enumerate(brick_row):
            sync_actor(brick_actor, brick_Ids[i][j])

    # Simulate a step.
    p.stepSimulation()

    # Exit after 2000 steps of simulation.
    if cnt == 2000:
        showm.exit()


# Add the timer callback to showmanager.
# Increasing the duration value will slow down the simulation.
showm.add_timer_callback(True, 10, timer_callback)

interactive = True

# start simulation
if interactive:
    showm.start()
