import numpy as np
from fury import window
from fury.animation.timeline import Timeline
from fury.animation.interpolator import (linear_interpolator, 
                                         step_interpolator, slerp)
from fury.animation import helpers
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf

scene = window.Scene()

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()


def tan_cubic_spline_interpolator(keyframes):
    # First we must get ordered timestamps array:
    timestamps = helpers.get_timestamps_from_keyframes(keyframes)

    for time in keyframes:
        data = keyframes.get(time)
        value = data.get('value')
        if data.get('in_tangent') is None:
            data['in_tangent'] = np.zeros_like(value)
        if data.get('in_tangent') is None:
            data['in_tangent'] = np.zeros_like(value)

    def interpolate(t):
        # `get_previous_timestamp`and `get_next_timestamp` functions take
        # timestamps array and current time as inputs and returns the
        # surrounding timestamps.
        t0 = helpers.get_previous_timestamp(timestamps, t)
        t1 = helpers.get_next_timestamp(timestamps, t)

        # `get_time_tau` function takes current time and surrounding
        # timestamps and returns a value from 0 to 1
        dt = helpers.get_time_tau(t, t0, t1)

        time_delta = t1 - t0

        p0 = keyframes.get(t0).get('value')
        tan_0 = keyframes.get(t0).get('out_tangent') * time_delta
        p1 = keyframes.get(t1).get('value')
        tan_1 = keyframes.get(t1).get('in_tangent') * time_delta
        # cubic spline equation using tangents
        t2 = dt * dt
        t3 = t2 * dt
        return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + dt) * tan_0 + (
                -2 * t3 + 3 * t2) * p1 + (t3 - t2) * tan_1
    return interpolate


fetch_gltf('InterpolationTest', 'glTF')
filename = read_viz_gltf('InterpolationTest')

gltf_obj = glTF(filename)
actors = gltf_obj.actors()

print(len(actors))

# simplyfy the example, add the followingcode to a function indide the glTF
# object.
transforms = gltf_obj.node_transform
nodes = gltf_obj.nodes

print(nodes)

interpolator = {
    'LINEAR': linear_interpolator,
    'STEP': step_interpolator,
    'CUBICSPLINE': tan_cubic_spline_interpolator
}

main_timeline = Timeline(playback_panel=True)

for transform in transforms:
    target_node = transform['node']
    for i, node_list in enumerate(nodes):
        if target_node in node_list:
            timeline = Timeline()
            timeline.add_actor(actors[i])

            timeframes = transform['input']
            transforms = transform['output']
            prop = transform['property']
            interp = interpolator.get(transform['interpolation'])
            timeshape = timeframes.shape
            transhape = transforms.shape
            if transform['interpolation'] == 'CUBICSPLINE':
                transforms = transforms.reshape((timeshape[0], -1, transhape[1]))

            for time, node_tran in zip(timeframes, transforms):

                in_tan, out_tan = None, None
                if node_tran.ndim == 2:
                    cubicspline = node_tran
                    in_tan = cubicspline[0]
                    node_tran = cubicspline[1]
                    out_tan = cubicspline[2]

                if prop == 'rotation':
                    timeline.set_rotation(time[0], node_tran,
                                          in_tangent=in_tan,
                                          out_tangent=out_tan)

                    timeline.set_rotation_interpolator(slerp)
                if prop == 'translation':
                    timeline.set_position(time[0], node_tran,
                                          in_tangent=in_tan,
                                          out_tangent=out_tan)

                    timeline.set_position_interpolator(interp)
                if prop == 'scale':
                    timeline.set_scale(time[0], node_tran,
                                       in_tangent=in_tan,
                                       out_tangent=out_tan)
                    timeline.set_scale_interpolator(interp)
            main_timeline.add_timeline(timeline)
        else:
            print('Adding static actor')
            main_timeline.add_static_actor(actors[i])

scene.add(main_timeline)


def timer_callback(_obj, _event):
    main_timeline.update_animation()
    showm.render()


# Adding the callback function that updates the animation
showm.add_timer_callback(True, 10, timer_callback)

showm.start()
