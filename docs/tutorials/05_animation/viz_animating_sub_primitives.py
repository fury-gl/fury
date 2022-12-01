from fury.animation.interpolator import cubic_bezier_interpolator
from fury.animation.timeline import Timeline
from fury.shaders import import_fury_shader, compose_shader
import numpy as np
import random
from fury import window, actor, ui
from fury.utils import vertices_from_actor, array_from_actor, \
    primitives_count_from_actor, update_actor

scene = window.Scene()
showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

centers = np.random.random([50, 3]) * 20
colors = np.random.random([50, 3])
directions = np.random.random([50, 3])
scales = (np.random.random(50) + 1) * 2

spheres = actor.cube(centers, directions, colors=colors, scales=scales)


class PartialActor:
    def __init__(self, act, indices):
        if not hasattr(act, 'vertices'):
            act.vertices = vertices_from_actor(act)
        if not hasattr(act, 'colors'):
            act.scales = array_from_actor(act, 'scale')
        self.actor = act
        self.indices = indices
        self.no_primitives = len(indices)
        no_all_vertices = len(act.vertices)
        no_all_primitives = primitives_count_from_actor(act)
        self.vertices_per_prim = no_all_vertices // no_all_primitives
        self.no_vertices = self.no_primitives * self.vertices_per_prim
        self.mask = np.repeat(
            indices, self.vertices_per_prim
        ) * self.vertices_per_prim + np.tile(
            np.arange(self.vertices_per_prim, dtype=int), self.no_primitives)

        self.orig_vert = np.copy(act.vertices[self.mask])
        vertices = self.orig_vert.reshape([self.no_vertices, 3])
        self.center = np.mean(vertices, axis=0)
        self.position = np.array([0, 0, 0])
        self.orig_vert_centered = self.orig_vert - self.center

    def SetPositions(self, positions):
        self.actor.vertices[self.mask] = self.orig_vert + np.repeat(
            positions, self.vertices_per_prim, axis=0) + self.position

    def SetPosition(self, position):
        self.position = position
        self.actor.vertices[self.mask] = self.orig_vert + np.tile(
            position, (self.no_vertices, 1))

    def add_to_scene(self, scene):
        scene.add(self.actor)


sub_spheres = PartialActor(spheres, [1, 4, 5, 8, 16, 7, 4])
timeline = Timeline(sub_spheres, playback_panel=True)

sub_spheres_2 = PartialActor(spheres, [0, 2, 3, 6])
timeline_2 = Timeline(sub_spheres_2)

timeline.add(timeline_2)

###############################################################################
# Generating random position keyframes
for t in range(0, 30, 1):
    ###########################################################################
    # Generating random position values
    positions = np.random.uniform(-1, 1, [sub_spheres.no_primitives, 3]) * 30

    ###########################################################################
    # Generating bezier control points.
    cp_dir = np.random.uniform(-1, 1, [sub_spheres.no_primitives, 3])
    pre_cps = positions + cp_dir * random.randrange(0, 15)
    post_cps = positions - cp_dir * random.randrange(0, 15)

    ###########################################################################
    # Adding custom keyframe. Here I called it `centers`.
    timeline.set_keyframe('centers', t, positions, pre_cp=pre_cps,
                          post_cp=post_cps)
    timeline_2.set_position(t, np.random.uniform(-100, 100, 3))

###############################################################################
# Setting the interpolator to cubic bezier interpolator for `centers`.
timeline.set_interpolator('centers', cubic_bezier_interpolator)

###############################################################################
# Adding timeline to scene
scene.add(timeline)


def timer_callback(_obj, _event):
    ###########################################################################
    # Updating the timeline (to handle animation time and animation state)
    timeline.update_animation()

    ###########################################################################
    # setting centers from timeline
    c = timeline.get_current_value('centers')
    sub_spheres.SetPositions(c)
    update_actor(spheres)

    showm.render()


timer_id = showm.add_timer_callback(True, 1, timer_callback)

interactive = False

if interactive:
    showm.start()

window.record(showm.scene, size=(900, 768),
              out_path="viz_geometry_billboards_animation.png")
