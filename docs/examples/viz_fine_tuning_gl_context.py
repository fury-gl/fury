"""
=======================================================
Fine-tuning the OpenGL state using shader callbacks
=======================================================

VTK it’s powerfully, but sometimes we need to get more control about how
OpenGL will render the actors.  For example, enforcing that deep-test keep
disabled during the draw call of a specific actor.  This can be useful when
we need to fine-tuning the performance or create specific visualization
effects in order to understand a certain data, like to enhance the
visualization of clusters in a network.
"""

###############################################################################
# First, let's import some functions
import numpy as np

from fury.shaders import shader_apply_effects

from fury import window, actor
import itertools


###############################################################################
# We just proceeds as usual: creating the actors and initializing a scene in
# FURY

centers = 1*np.array([
    [0, 0, 0],
    [-1, 0, 0],
    [1, 0, 0]
])
centers_no_depth_test = centers - np.array([[0, -1, 0]])
centers_normal_blending = centers_no_depth_test - np.array([[0, -1, 0]])
colors = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

actors = actor.sphere(
    centers, opacity=.8, radii=.4, colors=colors)
actors_no_depth_test = actor.sphere(
    centers_no_depth_test, opacity=.8, radii=.4, colors=colors)
actor_normal_blending = actor.sphere(
    centers_normal_blending, opacity=.8, radii=.4, colors=colors)

renderer = window.Scene()
scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=False)

###############################################################################
# All actors must be added  in the scene

scene.add(actors)
scene.add(actor_normal_blending)
scene.add(actors_no_depth_test)
###############################################################################
# Now, we will enter in the topic of this example. First, we need to create
# (or use one of the pre-built gl_function of FURY) to
# change the OpenGL state of a given fury window instance (showm.window).
#
# Here we're using the pre-build FURY window functions which has already a
# set of  specific behaviors to  be applied in the OpenGL context

shader_apply_effects(
    showm, actors,
    effects=[window.gl_enable_blend, window.gl_enable_depth])

id_observer = shader_apply_effects(
    showm, actor_normal_blending,
    effect=window.gl_set_normal_blending)

###############################################################################
# It's also possible to pass a list of effects. The final opengl state it'll
# be the composition of each effect that each function has in the opengl state
shader_apply_effects(
    showm, actors_no_depth_test,
    effects=[
        window.gl_reset_blend, window.gl_disable_blend,
        window.gl_disable_depth, window.gl_set_additive_blending])


###############################################################################
# Finaly, just render and see the results

showm.initialize()
# window.gl_set_additive_blending(showm.window)
counter = itertools.count()

# After one hundred of steps we will remove the additive blending effect
# from actor_normal_blending object


def timer_callback(obj, event):
    cnt = next(counter)
    showm.render()
    showm.scene.GetActiveCamera().Azimuth(1)
    if cnt == 100:
        actor_normal_blending.GetMapper().RemoveObserver(id_observer)
    if cnt == 1000:
        showm.exit()


showm.add_timer_callback(True, 5, timer_callback)
showm.start()
