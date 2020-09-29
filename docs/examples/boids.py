"""
================================
Flock simulation in a box
================================

This is an example of boids in a box using FURY.
"""

##############################################################################
# Explanation:

import numpy as np
from fury import window, actor, ui, utils, disable_warnings, pick, swarm
import itertools

disable_warnings()


def left_click_callback(obj, event):

    # Get the event position on display and pick

    event_pos = pickm.event_position(showm.iren)
    picked_info = pickm.pick(event_pos, showm.scene)

    vertex_index = picked_info['vertex']

    # Calculate the objects index
    vertices = utils.vertices_from_actor(cone_actor)
    no_vertices_per_cone = vertices.shape[0]
    # sec = np.int(no_vertices_per_cone / gm.num_particles)
    object_index = np.int(np.floor((vertex_index / no_vertices_per_cone) *
                          gm.num_particles))

    # Find how many vertices correspond to each object
    # sec = np.int(num_vertices / num_objects)
    selected = np.zeros(gm.num_particles, dtype=np.bool)

    if not selected[object_index]:
        scale = 1
        color_add = np.array([30, 30, 30], dtype='uint8')
        selected[object_index] = True
        # gm.num_attractors = 1
        # gm.pos_attractors = gm.pos[object_index][None, :]
        # gm.vel_attractors = gm.vel[object_index][None, :]
        # gm.attractors_indices.append(object_index)
    else:
        scale = 1
        color_add = np.array([-30, -30, -30], dtype='uint8')
        selected[object_index] = False

    # Update vertices positions
    vertices[object_index * sec: object_index * sec + sec] = scale * \
        (vertices[object_index * sec: object_index * sec + sec] -
         gm.pos[object_index]) + gm.pos[object_index]

    # Update colors
    vcolors[object_index * sec: object_index * sec + sec] += color_add

    # Tell actor that memory is modified
    utils.update_actor(cone_actor)

    face_index = picked_info['face']

    # Show some info
    text = 'Object ' + str(object_index) + '\n'
    text += 'Vertex ID ' + str(vertex_index) + '\n'
    text += 'Face ID ' + str(face_index) + '\n'
    text += 'World pos ' + str(np.round(picked_info['xyz'], 2)) + '\n'
    text += 'Actor ID ' + str(id(picked_info['actor']))
    text_block.message = text
    showm.render()


gm = swarm.GlobalMemory()
test_rules = False
specify_rand = True
if specify_rand:
    np.random.seed(42)
if test_rules is True:
    gm.vel = np.array([[-np.sqrt(2)/2, np.sqrt(2)/2, 0], [0, 1., 0],
                      [np.sqrt(2)/2, np.sqrt(2)/2, 0], [np.sqrt(2)/2,
                      np.sqrt(2)/2, 0], [np.sqrt(2)/2, np.sqrt(2)/2, 0]])
    gm.pos = .5 * np.array([[-5, 0., 0], [0, 0., 0], [5, 0., 0], [10, 0., 0],
                           [15, 0., 0]])
    directions = gm.vel.copy()
else:
    gm.vel = (-0.5 + (np.random.rand(gm.num_particles, 3)))*5
    # gm.vel[:, 0] = 0.
    # gm.vel[:, 2] = 0.
    # vel = gm.vel / np.linalg.norm(vel, axis=1).reshape((gm.num_particles, 1))

    gm.pos = np.array([gm.box_lx, gm.box_ly, gm.box_lz]) * (np.random.rand(
                      gm.num_particles, 3) - 0.5) * 0.6
    # gm.pos[:, 0] = 0
    # gm.pos[:, 2] = 0
    directions = gm.vel.copy()

scene = window.Scene()
box_centers = np.array([[0, 0, 0]])
box_directions = np.array([[0, 1, 0]])
box_colors = np.array([[255, 255, 255]])
box_actor = actor.box(box_centers, box_directions, box_colors,
                      scales=(gm.box_lx, gm.box_ly, gm.box_lz))
utils.opacity(box_actor, 0.)
scene.add(box_actor)

lines = swarm.box_edges((gm.box_lx, gm.box_ly, gm.box_lz))
line_actor = actor.streamtube(lines, colors=(1, 0.5, 0), linewidth=0.1)
scene.add(line_actor)
obstacle_actor = actor.sphere(centers=gm.pos_obstacles,
                              colors=gm.color_obstacles,
                              radii=gm.radii_obstacles)
scene.add(obstacle_actor)
leader_actor = False #True
gm.vel_leaders = np.random.rand(gm.num_leaders, 3) * 10
directions_leader = gm.vel_leaders.copy()
if leader_actor:
    leader_actor = actor.cone(centers=gm.pos_leaders,
                                directions=directions_leader, colors=gm.color_leaders,
                                heights=gm.radii_leaders,
                                resolution=10, vertices=None, faces=None)
else:
    leader_actor = actor.sphere(centers=gm.pos_leaders,
                            colors=gm.color_leaders,
                            radii=gm.radii_leaders)
scene.add(leader_actor)

cone_actor = actor.cone(centers=gm.pos,
                        directions=directions, colors=gm.colors,
                        heights=gm.height_cones,
                        resolution=10, vertices=None, faces=None)
scene.add(cone_actor)

axes_actor = actor.axes(scale=(1, 1, 1), colorx=(1, 0, 0), colory=(0, 1, 0),
                        colorz=(0, 0, 1), opacity=1)
scene.add(axes_actor)
showm = window.ShowManager(scene,
                           size=(3000, 2000), reset_camera=True,
                           order_transparent=True)
showm.initialize()
tb = ui.TextBlock2D(bold=True)
counter = itertools.count()
vertices = utils.vertices_from_actor(cone_actor)
vcolors = utils.colors_from_actor(cone_actor, 'colors')
no_vertices_per_cone = len(vertices)/gm.num_particles
initial_vertices = vertices.copy() - \
    np.repeat(gm.pos, no_vertices_per_cone, axis=0)

if gm.num_leaders > 0:
    vertices_leader = utils.vertices_from_actor(leader_actor)
    no_vertices_per_leader = len(vertices_leader)/gm.num_leaders
    initial_vertices_leader = vertices_leader.copy() - \
        np.repeat(gm.pos_leaders, no_vertices_per_leader, axis=0)
    sec_leader = np.int(no_vertices_per_leader / gm.num_leaders)

if gm.num_obstacles > 0:
    vertices_obstacle = utils.vertices_from_actor(obstacle_actor)
    no_vertices_per_obstacle = len(vertices_obstacle)/gm.num_obstacles
    initial_vertices_obstacle = vertices_obstacle.copy() - \
        np.repeat(gm.pos_obstacles, no_vertices_per_obstacle, axis=0)
scene.zoom(1.2)
pickm = pick.PickingManager()
panel = ui.Panel2D(size=(400, 200), color=(1, .5, .0), align="right")
panel.center = (150, 200)

text_block = ui.TextBlock2D(text="Left click on object \n")
panel.add_element(text_block, (0.3, 0.3))
scene.add(panel)
# It rotates arrow at origin and then shifts to position;
num_vertices = vertices.shape[0]
sec = np.int(num_vertices / gm.num_particles)
cone_actor.AddObserver('LeftButtonPressEvent', left_click_callback, 1)


def timer_callback(_obj, _event):
    gm.cnt += 1
    tb.message = "Let's count up to 1000 and exit :" + str(gm.cnt)
    ################
    turnfraction = 0.01
    cnt = next(counter)
    dst = 10
    angle = 2 * np.pi * turnfraction * cnt
    x2 = dst * np.cos(angle)
    y2 = dst * np.sin(angle)
    angle_1 = 2 * np.pi * turnfraction * (cnt+1)
    x2_1 = dst * np.cos(angle_1)
    y2_1 = dst * np.sin(angle_1)
    # xyz_leader = np.array([[x2, y2, 0.]])
    # gm.pos_leaders = np.array([[x2_1, y2_1, 0.]])
    # gm.vel_leaders = np.array((gm.pos_leaders - xyz_leader)/np.linalg.norm(gm.pos_leaders- xyz_leader))
    ###############
    gm.pos_leaders = gm.pos_leaders + gm.vel_leaders
    gm.pos_obstacles = gm.pos_obstacles + gm.vel_obstacles

    swarm.boids_rules(gm, vertices, vcolors)
    swarm.collision_particle_walls(gm, True)
    swarm.collision_obstacle_leader_walls(gm)
    gm.pos = gm.pos + gm.vel
    # swarm.collision_obstacle_leader_walls(gm)
    for i in range(gm.num_particles):
        # directions and velocities normalization
        dnorm = directions[i]/np.linalg.norm(directions[i])
        vnorm = gm.vel[i]/np.linalg.norm(gm.vel[i])
        R_followers = swarm.vec2vec_rotmat(vnorm, dnorm)
        vertices[i * sec: i * sec + sec] = np.dot(initial_vertices[i * sec: i *
                                                  sec + sec], R_followers) + \
            np.repeat(gm.pos[i: i+1], no_vertices_per_cone, axis=0)
    utils.update_actor(cone_actor)

    for i in range(gm.num_leaders):
        if gm.num_leaders > 0:
            # if leader_actor is True:
            dnorm_leaders = directions_leader[i]/np.linalg.norm(directions_leader[i])
            vnorm_leaders = gm.vel_leaders[i]/np.linalg.norm(gm.vel_leaders[i])
            R_leaders = swarm.vec2vec_rotmat(vnorm_leaders, dnorm_leaders)
            vertices_leader[i * sec_leader: i * sec_leader + sec_leader] = np.dot(initial_vertices_leader[i * sec_leader: i *
                                                    sec_leader + sec_leader], R_leaders) + \
                np.repeat(gm.pos_leaders[i: i+1], no_vertices_per_leader, axis=0)
            # else:
            #         vertices_leader[i * sec_leader: i * sec_leader + sec_leader] = initial_vertices_leader[i * sec_leader: i *
            #                                                 sec_leader + sec_leader] + \
            #             np.repeat(gm.pos_leaders[i: i+1], no_vertices_per_leader, axis=0)

    utils.update_actor(leader_actor)

    if gm.num_obstacles > 0:
        vertices_obstacle[:] = initial_vertices_obstacle + \
            np.repeat(gm.pos_obstacles, no_vertices_per_obstacle, axis=0)
        utils.update_actor(obstacle_actor)

    scene.reset_clipping_range()
    showm.render()
    if gm.cnt == gm.steps:
        showm.exit()


scene.add(tb)
showm.add_timer_callback(True, 1, timer_callback)

showm.start()