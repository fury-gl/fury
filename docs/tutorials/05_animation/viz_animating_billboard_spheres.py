from fury.animation.interpolator import cubic_bezier_interpolator
from fury.animation.timeline import Timeline
from fury.shaders import import_fury_shader, compose_shader
import numpy as np
import random
from fury import window, actor, ui

scene = window.Scene()
showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()


no_components = 200_000

centers = np.random.random([no_components, 3]) * no_components
colors = np.random.random([no_components, 3])
scales = np.random.random(no_components) * no_components ** 0.5 * 4


fs_dec = compose_shader(
    [import_fury_shader('lighting/blinn_phong_model.frag'),
     import_fury_shader('sdf/sph_intersect.frag')])

fs_impl = compose_shader(
    [import_fury_shader('gs_billboard_sphere_impl.frag')])


geom_squares = actor.billboard(centers, colors=colors, scales=scales,
                               fs_dec=fs_dec, fs_impl=fs_impl,
                               gs_prog='default')

timeline = Timeline(geom_squares, playback_panel=True)

###############################################################################
# Generating random position keyframes
indices = np.r_[0:30_000, 50_000:100_000]

for t in range(0, 30, 15):
    ###########################################################################
    # Generating random position values
    positions = np.random.random([indices.size, 3]) * no_components

    ###########################################################################
    # Generating bezier control points.
    cp_dir = np.random.uniform(-1, 1, [indices.size, 3])
    pre_cps = positions + cp_dir * random.randrange(0, 3 * no_components)
    post_cps = positions - cp_dir * random.randrange(0, 3 * no_components)

    ###########################################################################
    # Adding custom keyframe. Here I called it `centers`.
    timeline.set_keyframe('centers', t, positions, pre_cp=pre_cps,
                          post_cp=post_cps)

###############################################################################
# Setting the interpolator to cubic bezier interpolator for `centers`.
timeline.set_interpolator('centers', cubic_bezier_interpolator)


###############################################################################
# Init vars needed for FPS calculation
tb = ui.TextBlock2D(bold=True, position=(0, 730))
fpss = [0] * 20

scene.add(timeline, tb)


def timer_callback(_obj, _event):
    global timer_id, fpss

    ###########################################################################
    # Calculating current approximate FPS
    fpss.append(showm.frame_rate)
    fpss.pop(0)
    fps = sum(fpss) / len(fpss)
    tb.message = 'FPS: ' + str(int(np.round(fps)))

    ###########################################################################
    # Updating the timeline (to handle animation time and animation state)
    timeline.update_animation()

    ###########################################################################
    # setting centers from timeline
    geom_squares.centers[~indices] = timeline.get_current_value('centers')

    showm.render()


timer_id = showm.add_timer_callback(True, 1, timer_callback)

interactive = True

if interactive:
    showm.start()

window.record(showm.scene, size=(900, 768),
              out_path="viz_geometry_billboards_animation.png")
