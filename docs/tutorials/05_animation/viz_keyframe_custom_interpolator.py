"""
=====================
Keyframe animation
=====================

Keyframe animation using custom interpolator.

"""
import numpy as np
from fury import actor, window
from fury.animation.timeline import Timeline


###############################################################################
# Implementing a custom interpolator
# ===============================================
#
# A keyframe interpolator must have a method `interpolate` which take the time
# as an argument and returns a value.
# It's recommended to inherit from `Interpolator` class, which has some useful
# methods dealing with timestamps.

###############################################################################
# In glTF, animations using cubic spline interpolator needs at least two
# points, and each point has two tangent vectors.
# The interpolation equation for such data is in the glTF tutorials below:
# https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_007_Animations.html#cubic-spline-interpolation
#
# Tangent based cubic spline interpolation function:
#
# >>> def cubicSpline(previousPoint, previousTangent, nextPoint, nextTangent,
# >>>                   interpolationValue):
# >>>     t = interpolationValue
# >>>     t2 = t * t
# >>>     t3 = t2 * t
# >>>     return (2 * t3 - 3 * t2 + 1) * previousPoint +
# >>>            (t3 - 2 * t2 + t) * previousTangent +
# >>>            (-2 * t3 + 3 * t2) * nextPoint +
# >>>            (t3 - t2) * nextTangent
#
# First we create a class that must take a dict object that contains the
# animation keyframes when initialized as follows:
#
# >>> class cubic_spline_interpolator(Interpolator):
# >>>     def __init__(self, keyframes):
# >>>         super(cubic_spline_interpolator, self).__init__(keyframes)
#
# Note: Also any other additional arguments are ok, see `spline_interpolator`
# Second step is to implement the `interpolate` method (must have the name
# `interpolate`) that only takes the current time as input.

def tan_cubic_spline_interpolator(keyframes):
    # keyframes should be on the following form:
    # {
    # 1: {'value': ndarray, 'in_tangent': ndarray, 'out_tangent': ndarray},
    # 2: {'value': np.array([1, 2, 3], 'in_tangent': ndarray},
    # }
    # See here, we might get incomplete data (out_tangent) in the second
    # keyframe. In this case we need to have a default behaviour dealing
    # with these missing data.
    # Setting the tangent to a zero vector in this case is the best choice
    for time in keyframes:
        data = keyframes.get(time)
        value = data.get('value')
        if data.get('in_tangent') is None:
            data['in_tangent'] = np.zeros_like(value)
        if data.get('in_tangent') is None:
            data['in_tangent'] = np.zeros_like(value)

    def interpolate(t):
        # `get_neighbour_timestamps` method takes time as input and returns
        # the surrounding timestamps.
        t0, t1 = self.get_neighbour_timestamps(t)

        # `get_time_tau` static method takes current time and surrounding
        # timestamps and returns a value from 0 to 1
        dt = self.get_time_tau(t, t0, t1)

        time_delta = t1 - t0

        # to get a keyframe data at a specific timestamp, use
        # `self.keyframes.get(t0)`. This keyframe data contains `value` and any
        # other data set as a custom argument using keyframe setters.
        # for example:
        # >>> timeline = Timeline()
        # >>> timeline.set_position(0, np.array([1, 1, 1]),
        # >>>                       custom_field=np.array([2, 3, 1]))
        # In this case `self.keyframes.get(0)` would return:
        # {'value': array(1, 1, 1), 'custom_field': array(2, 3, 1)}
        #
        # now we continue with the cubic spline equation.
        p0 = self.keyframes.get(t0).get('value')
        tan_0 = self.keyframes.get(t0).get('out_tangent') * time_delta
        p1 = self.keyframes.get(t1).get('value')
        tan_1 = self.keyframes.get(t1).get('in_tangent') * time_delta
        # cubic spline equation using tangents
        t2 = dt * dt
        t3 = t2 * dt
        return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + dt) * tan_0 + (
                -2 * t3 + 3 * t2) * p1 + (t3 - t2) * tan_1


scene = window.Scene()
showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

###############################################################################
# Cubic spline keyframes data same as the one you get from glTF file.
# ===================================================================

#              t   in tangent     position                   out tangent
translation = [[0, [0., 0., 0.],  [3.3051798, 6.640117, 0.], [1., 0., 0.]],
               [1, [0., 0., 0.],  [3.3051798, 8., 0.],       [-1., 0., 0.]],
               [2, [-1., 0., 0.], [3.3051798, 6., 0.],       [1., 0., 0.]],
               [3, [0., 0., 0.],  [3.3051798, 8., 0.],       [-1., 0., 0.]],
               [4, [0, -1., 0.],  [3.3051798, 6., 0.],       [0., 0., 0.]]]

###############################################################################
# Initializing a ``Timeline`` and adding sphere actor to it.
timeline = Timeline(playback_panel=True,  motion_path_res=100)

sphere = actor.sphere(np.array([[0, 0, 0]]), (1, 0, 1), radii=0.1)

timeline.add_actor(sphere)

###############################################################################
# Setting position keyframes
# ==========================
for keyframe_data in translation:
    t, in_tan, pos, out_tan = keyframe_data
    # Since we used the name 'in_tangent' and 'out_tangent' in the interpolator
    # We must use the same name as an argument to set it in the keyframe data.
    timeline.set_position(t, pos, in_tangent=in_tan, out_tangent=out_tan)

###############################################################################
# Set the new interpolator to interpolate position keyframes
timeline.set_position_interpolator(tan_cubic_spline_interpolator)

###############################################################################
# adding the timeline and the static actors to the scene.
scene.add(timeline)


###############################################################################
# making a function to update the animation
def timer_callback(_obj, _event):
    timeline.update_animation()
    showm.render()


###############################################################################
# Adding the callback function that updates the animation
showm.add_timer_callback(True, 10, timer_callback)

interactive = False

if interactive:
    showm.start()

window.record(scene, out_path='viz_keyframe_custom_interpolator.png',
              size=(900, 768))
