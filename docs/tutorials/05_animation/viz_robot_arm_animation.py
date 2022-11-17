"""
=====================
Keyframe animation
=====================

Tutorial on making a robot arm animation in FURY.

"""
import numpy as np
from fury import actor, window
from fury.animation.timeline import Timeline
from fury.utils import set_actor_origin

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

###############################################################################
# Creating robot arm components

base = actor.cylinder(np.array([[0, 0, 0]]), np.array([[0, 1, 0]]),
                      colors=(0, 1, 0), radius=1)
main_arm = actor.box(np.array([[0, 0, 0]]), colors=(1, 0.5, 0),
                     scales=(12, 1, 1))

sub_arm = actor.box(np.array([[0, 0, 0]]), colors=(0, 0.5, 0.8),
                    scales=(8, 0.7, 0.7))
joint_1 = actor.sphere(np.array([[0, 0, 0]]), colors=np.array([1, 0, 1]),
                       radii=1.2)
joint_2 = actor.sphere(np.array([[0, 0, 0]]), colors=np.array([1, 0, 1]))

end = actor.cone(np.array([[0, 0, 0]]), np.array([[1, 0, 0]]),
                 np.array([[1, 0, 0]]), heights=2.2, resolution=6)

###############################################################################
# Setting the origin or rotation of both shafts to the beginning.
# Length of main arm is 12 so the beginning would be at -6 in x direction.
set_actor_origin(main_arm, np.array([-6, 0, 0]))
# Length of the sub arm is 8 so the beginning would be at -4 in x direction.
set_actor_origin(sub_arm, np.array([-4, 0, 0]))

###############################################################################
# Creating a timeline to animate the actor
tl_main = Timeline([main_arm, joint_1], playback_panel=True, length=2 * np.pi)
tl_child = Timeline([sub_arm, joint_2])
tl_grand_child = Timeline(end)

###############################################################################
# Adding Timelines in hierarchical order
tl_main.add_child_timeline(tl_child)
tl_child.add_child_timeline(tl_grand_child)


###############################################################################
# Creating Arm joints time dependent animation functions.

def rot_main_arm(t):
    return np.array([np.sin(t/2) * 180, np.cos(t/2) * 180, 0])


def rot_sub_arm(t):
    return np.array([np.sin(t) * 180, np.cos(t) * 70, np.cos(t) * 40])


def rot_drill(t):
    return np.array([t * 1000, 0, 0])


###############################################################################
# Setting timelines (joints) relative position
# 1- Placing the main arm on the cube static base.
tl_main.set_position(0, np.array([0, 1.3, 0]))

###############################################################################
# 2- Translating the timeline containing the sub arm to the end of the first
# arm.
tl_child.set_position(0, np.array([12, 0, 0]))

###############################################################################
# 3- Translating the timeline containing the drill to the end of the sub arm.
tl_grand_child.set_position(0, np.array([8, 0, 0]))

###############################################################################
# Setting rotation time-based evaluators
tl_main.set_rotation_interpolator(rot_main_arm, is_evaluator=True)
tl_child.set_rotation_interpolator(rot_sub_arm, is_evaluator=True)
tl_grand_child.set_rotation_interpolator(rot_drill, is_evaluator=True)

###############################################################################
# Setting camera position to observe the robot arm.
scene.camera().SetPosition(0, 0, 90)

###############################################################################
# Adding timelines to the main Timeline.
scene.add(tl_main, base)


###############################################################################
# making a function to update the animation and render the scene.
def timer_callback(_obj, _event):
    tl_main.update_animation()
    showm.render()


###############################################################################
# Adding the callback function that updates the animation.
showm.add_timer_callback(True, 10, timer_callback)

interactive = False

if interactive:
    showm.start()

window.record(scene, out_path='viz_robot_arm.png',
              size=(900, 768))
