"""
===================
Arm Robot Animation
===================

Tutorial on making a robot arm animation in FURY.
"""

import numpy as np

import fury

scene = fury.window.Scene()

showm = fury.window.ShowManager(
    scene, size=(900, 768), reset_camera=False, order_transparent=True
)
showm.initialize()


###############################################################################
# Creating robot arm components

base = fury.actor.cylinder(
    np.array([[0, 0, 0]]), np.array([[0, 1, 0]]), colors=(0, 1, 0), radius=1
)
main_arm = fury.actor.box(np.array([[0, 0, 0]]), colors=(1, 0.5, 0), scales=(12, 1, 1))

sub_arm = fury.actor.box(
    np.array([[0, 0, 0]]), colors=(0, 0.5, 0.8), scales=(8, 0.7, 0.7)
)
joint_1 = fury.actor.sphere(
    np.array([[0, 0, 0]]), colors=np.array([1, 0, 1]), radii=1.2
)
joint_2 = fury.actor.sphere(np.array([[0, 0, 0]]), colors=np.array([1, 0, 1]))

end = fury.actor.cone(
    np.array([[0, 0, 0]]),
    np.array([[1, 0, 0]]),
    np.array([[1, 0, 0]]),
    heights=2.2,
    resolution=6,
)

###############################################################################
# Setting the center of both shafts to the beginning.
fury.utils.set_actor_origin(main_arm, np.array([-6, 0, 0]))
fury.utils.set_actor_origin(sub_arm, np.array([-4, 0, 0]))

###############################################################################
# Creating a timeline
timeline = fury.animation.Timeline(playback_panel=True)

###############################################################################
# Creating animations
main_arm_animation = fury.animation.Animation([main_arm, joint_1], length=2 * np.pi)
child_arm_animation = fury.animation.Animation([sub_arm, joint_2])
drill_animation = fury.animation.Animation(end)


###############################################################################
# Adding other Animations in hierarchical order
main_arm_animation.add_child_animation(child_arm_animation)
child_arm_animation.add_child_animation(drill_animation)


###############################################################################
# Creating Arm joints time dependent animation functions.


def rot_main_arm(t):
    return np.array([np.sin(t / 2) * 180, np.cos(t / 2) * 180, 0])


def rot_sub_arm(t):
    return np.array([np.sin(t) * 180, np.cos(t) * 70, np.cos(t) * 40])


def rot_drill(t):
    return np.array([t * 1000, 0, 0])


###############################################################################
# Setting timelines (joints) relative position
# 1- Placing the main arm on the cube static base.
main_arm_animation.set_position(0, np.array([0, 1.3, 0]))

###############################################################################
# 2- Translating the timeline containing the sub arm to the end of the first
# arm.
child_arm_animation.set_position(0, np.array([12, 0, 0]))

###############################################################################
# 3- Translating the timeline containing the drill to the end of the sub arm.
drill_animation.set_position(0, np.array([8, 0, 0]))

###############################################################################
# Setting rotation time-based evaluators
main_arm_animation.set_rotation_interpolator(rot_main_arm, is_evaluator=True)
child_arm_animation.set_rotation_interpolator(rot_sub_arm, is_evaluator=True)
drill_animation.set_rotation_interpolator(rot_drill, is_evaluator=True)

###############################################################################
# Setting camera position to observe the robot arm.
scene.camera().SetPosition(0, 0, 90)

###############################################################################
# Adding the base actor to the scene
scene.add(base)

###############################################################################
# Adding the main parent animation to the Timeline.
timeline.add_animation(main_arm_animation)

###############################################################################
# Now we add the timeline to the ShowManager
showm.add_animation(timeline)

interactive = False

if interactive:
    showm.start()

fury.window.record(scene, out_path="viz_robot_arm.png", size=(900, 768))
