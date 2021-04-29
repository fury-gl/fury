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

from fury.shaders import add_shader_callback

from fury import window, actor
import itertools

from functools import partial


###############################################################################
# We just proceeds as usual: creating the actors and initializing a scene in
# FURY 

centers = 1*np.array([
    [0, 0, 0],
    [-1, 0, 0],
    [1, 0, 0]
])
centers_no_depth_test = centers - np.array([[0, -1, 0]])
centers_additive = centers_no_depth_test - np.array([[0, -1, 0]])
centers_mul = centers_additive - np.array([[0, -1, 0]])
centers_sub = centers_mul - np.array([[0, -1, 0]])
colors = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

actors = actor.sdf(
    centers, primitives='sphere', colors=colors, scales=2)
actorNoDepthTest = actor.sdf(
    centers_no_depth_test, primitives='sphere', colors=colors, scales=2)
actorAdd = actor.sdf(
    centers_additive, primitives='sphere', colors=colors, scales=2)

renderer = window.Scene()
scene = window.Scene()
interactive = True


showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)

###############################################################################
# All actors must be added  in the scene

scene.add(actorAdd)
scene.add(actors)
scene.add(actorNoDepthTest)

###############################################################################
# Now, we will enter in the topic of this example. First, we need to create
# (or use one of the pre-built gl_function of FURY) to
# change the OpenGL state of a given fury window instance (showm.window).
#
# The function bellow it's a simple example and can be used to disable the
# GL_DEPTH_STATE  of the opengl context used by FURY. VTK allway's change
# this state to True before the draw call, therefore we need to set them
# inside of a shader callback, even if you have just one actor.


def gl_disable_depth(window):
    '''this functions it's allways accessible through
    fury.window.gl_disable_depth
    '''
    GL_DEPTH_TEST = 2929
    GL_BLEND = 3042
    glState = window.GetState()
    glState.vtkglDisable(GL_DEPTH_TEST)
    glState.vtkglDisable(GL_BLEND)

###############################################################################
# Next, we write a standard callback function.


def callback(
        _caller, _event, calldata=None,
        gl_set_func=None, window_obj=None):
    program = calldata
    if program is not None:
        gl_set_func(window_obj)


###############################################################################
# Then we use that callback function  as argument to the add_shader_callback
# method from FURY. The callback function will be called in every draw call
id_observer_depth = add_shader_callback(
        actorNoDepthTest, partial(
            callback,
            gl_set_func=gl_disable_depth, window_obj=showm.window))


###############################################################################
# Here we're using the pre-build FURY window functions which add specific
# behaviors to the OpenGL context

id_observer_normal = add_shader_callback(
        actors, partial(
            callback, gl_set_func=window.gl_set_normal_blending,
            window_obj=showm.window))


id_observer_additive = add_shader_callback(
        actorAdd, partial(
            callback, gl_set_func=window.gl_set_additive_blending,
            window_obj=showm.window))


###############################################################################
# Finaly, just render and see the results

showm.initialize()
counter = itertools.count()


def timer_callback(obj, event):
    cnt = next(counter)
    showm.render()
    if cnt == 10000:
        showm.exit()


showm.add_timer_callback(True, 200, timer_callback)
showm.start()
