from fury.shaders import import_fury_shader, compose_shader
import numpy as np
from fury import window, actor, ui, utils
import itertools

centers = np.random.random([3000, 3]) * 12
colors = np.random.random([3000, 3])
scales = np.random.random(3000)

scene = window.Scene()

showm = window.ShowManager(scene, size=(1000, 768))

fs_dec = compose_shader([import_fury_shader('lighting/blinn_phong_model.frag'),
                         import_fury_shader('sdf/sph_intersect.frag')])

fs_impl = compose_shader([import_fury_shader('gs_billboard_sphere_impl.frag')])

geom_squares = actor.billboard(centers, colors=colors, scales=scales,
                               using_gs=True, fs_dec=fs_dec, fs_impl=fs_impl)

scene.add(geom_squares)

interactive = True
if interactive:
    showm.start()

window.record(scene, size=(600, 600), out_path="viz_billboard_sphere_gs.png")

###############################################################################
# Animating a large number of geometry-shader-billboard spheres and comparing
# them to normal billboards
###############################################################################

scene = window.Scene()
no_components = 200_000

centers = np.random.random([no_components, 3]) * no_components
colors = np.random.random([no_components, 3])
scales = np.random.random(no_components) * no_components ** 0.5

using_geometry_shader = True


fs_dec = ""
fs_impl = ""
if using_geometry_shader:
    fs_dec = compose_shader(
        [import_fury_shader('lighting/blinn_phong_model.frag'),
         import_fury_shader('sdf/sph_intersect.frag')])

    fs_impl = compose_shader(
        [import_fury_shader('gs_billboard_sphere_impl.frag')])


geom_squares = actor.billboard(centers, colors=colors, scales=scales,
                               using_gs=using_geometry_shader, fs_dec=fs_dec,
                               fs_impl=fs_impl)


vcolors = utils.colors_from_actor(geom_squares)
vscales = utils.array_from_actor(geom_squares, 'scale')

scene.add(geom_squares)

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)

showm.initialize()

tb = ui.TextBlock2D(bold=True)

counter = itertools.count()

fps_len = 20
fpss = [0] * fps_len

prev_color = vcolors
next_color = np.random.rand(*vcolors.shape) * 255

if using_geometry_shader:
    prev_scale = vscales
    next_scale = np.random.rand(*vscales.shape) * no_components ** 0.5


def timer_callback(_obj, _event):
    global timer_id, fpss, prev_color, next_color, next_scale, prev_scale
    every = 15
    cnt = next(counter)

    fpss.append(showm.frame_rate)
    fpss.pop(0)
    fps = sum(fpss) / fps_len
    tb.message = 'FPS: ' + str(int(np.round(fps)))
    if cnt % every == 0:
        prev_color = next_color
        next_color = np.random.rand(*vcolors.shape) * 255
        if using_geometry_shader:
            prev_scale = next_scale
            next_scale = np.random.rand(*vscales.shape) * no_components ** 0.5
    dt = (cnt % every) / every
    vcolors[:] = dt * next_color + (1 - dt) * prev_color
    if using_geometry_shader:
        vscales[:] = dt * next_scale + (1 - dt) * prev_scale
    utils.update_actor(geom_squares)
    showm.render()


scene.add(tb)

timer_id = showm.add_timer_callback(True, 1, timer_callback)

if interactive:
    showm.start()

window.record(showm.scene, size=(900, 768),
              out_path="viz_geometry_billboards_animation.png")
