from os.path import join
import numpy as np
from fury import actor, window, ui, pick
from fury.testing import assert_greater
import numpy.testing as npt
import itertools
from fury.data import DATA_DIR


def test_picking_manager():

    xyz = 10 * np.random.rand(100, 3)
    colors = np.random.rand(100, 4)
    radii = np.random.rand(100) + 0.5

    scene = window.Scene()

    sphere_actor = actor.sphere(centers=xyz,
                                colors=colors,
                                radii=radii)

    scene.add(sphere_actor)

    showm = window.ShowManager(scene,
                               size=(900, 768), reset_camera=False,
                               order_transparent=True)

    showm.initialize()

    tb = ui.TextBlock2D(bold=True)

    # use itertools to avoid global variables
    counter = itertools.count()

    pickm = pick.PickingManager()

    record_indices = {'vertex_indices': [],
                      'face_indices': [],
                      'xyz': [],
                      'actor': []}

    def timer_callback(_obj, _event):
        cnt = next(counter)
        tb.message = "Let's count up to 15 and exit :" + str(cnt)
        showm.scene.azimuth(0.05 * cnt)
        # sphere_actor.GetProperty().SetOpacity(cnt/100.)
        if cnt % 10 == 0:
            # pick at position
            info = pickm.pick((900/2, 768/2), scene)
            record_indices['vertex_indices'].append(info['vertex'])
            record_indices['face_indices'].append(info['face'])
            record_indices['xyz'].append(info['xyz'])
            record_indices['actor'].append(info['actor'])

        showm.render()
        if cnt == 15:
            showm.exit()

    scene.add(tb)

    # Run every 200 milliseconds
    showm.add_timer_callback(True, 200, timer_callback)
    showm.start()

    assert_greater(np.sum(np.array(record_indices['vertex_indices'])), 1)
    assert_greater(np.sum(np.array(record_indices['face_indices'])), 1)

    for ac in record_indices['actor']:
        if ac is not None:
            npt.assert_equal(ac is sphere_actor, True)

    assert_greater(np.sum(np.abs(np.diff(np.array(record_indices['xyz']),
                                         axis=0))), 0)


def _get_three_cubes():
    centers = 0.5 * np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0.]])
    colors = np.array([[0.8, 0, 0], [0, 0.8, 0], [0, 0, 0.8]])
    radii = 0.1 * np.array([50, 100, 150.])
    return centers, colors, radii


def test_selector_manager():

    centers, colors, radii = _get_three_cubes()

    scene = window.Scene()

    cube_actor = actor.cube(centers, directions=(1, 0, 2),
                            colors=colors, scales=radii)

    pts = 100 * (np.random.rand(100, 3) - 0.5) + np.array([20, 0, 0.])
    pts_actor = actor.dots(pts, dot_size=10)

    rgb = 255 * np.ones((400, 400, 3), dtype=np.uint8)
    tex_actor = actor.texture(rgb)

    scene.add(cube_actor)
    scene.add(pts_actor)
    scene.add(tex_actor)

    showm = window.ShowManager(scene,
                               size=(900, 768), reset_camera=False,
                               order_transparent=True)

    showm.initialize()

    tb = ui.TextBlock2D(bold=True)

    # use itertools to avoid global variables
    counter = itertools.count()

    selm = pick.SelectionManager(select='faces')

    selm.selectable_off([tex_actor])
    selm.selectable_on([tex_actor])
    selm.selectable_off([tex_actor])

    def timer_callback(_obj, _event):
        cnt = next(counter)
        tb.message = "Let's count up to 15 and exit :" + str(cnt)
        if cnt % 10 == 0:
            # select large area
            info_plus = selm.select((900//2, 768//2), scene, (30, 30))
            for info in info_plus.keys():
                if info_plus[info]['actor'] in [cube_actor, pts_actor]:
                    npt.assert_(True)
                else:
                    npt.assert_(False)
            # select single pixel
            info_ = selm.pick((900//2, 768//2), scene)
            if info_['actor'] in [cube_actor, pts_actor]:
                npt.assert_(True)
            else:
                npt.assert_(False)

        showm.render()
        if cnt == 15:
            showm.exit()
            pass

    scene.add(tb)

    # Run every 200 milliseconds
    showm.add_timer_callback(True, 200, timer_callback)
    showm.start()


def test_hover_selection_faces(recording=False):
    # simply hover going through blue, green, red

    recording_filename = join(DATA_DIR, 'selector_faces.log.gz')

    centers, colors, radii = _get_three_cubes()

    scene = window.Scene()

    cube_actor = actor.cube(centers, directions=(1, 0, 0),
                            colors=colors, scales=radii)

    scene.add(cube_actor)

    selm = pick.SelectionManager(select='faces')

    showm = window.ShowManager(scene,
                               size=(900, 768), reset_camera=False,
                               order_transparent=True)

    showm.initialize()
    global track_objects
    track_objects = []

    def hover_callback(_obj, _event):
        global track_objects
        event_pos = selm.event_position(showm.iren)
        info = selm.select(event_pos, showm.scene, (10, 10))
        selected_faces = info[0]['face']
        if selected_faces is not None:
            track_objects.append(selected_faces[0]//12)
        showm.render()

    showm.add_iren_callback(hover_callback)

    if recording:
        showm.record_events_to_file(recording_filename)

    else:
        showm.play_events_from_file(recording_filename)

    track_objects = set(track_objects)

    npt.assert_({0, 1, 2}.issubset(track_objects))
    del track_objects


def test_hover_selection_vertices(recording=False):
    # simply hover through blue, green, red cubes
    # close to any vertices of each of the cubes

    recording_filename = join(DATA_DIR, 'selector_vertices.log.gz')

    centers, colors, radii = _get_three_cubes()

    scene = window.Scene()

    cube_actor = actor.cube(centers, directions=(1, 0, 0),
                            colors=colors, scales=radii)

    scene.add(cube_actor)

    selm = pick.SelectionManager(select='vertices')

    showm = window.ShowManager(scene,
                               size=(900, 768), reset_camera=False,
                               order_transparent=True)

    showm.initialize()

    global track_objects2
    track_objects2 = []

    def hover_callback(_obj, _event):
        global track_objects2
        event_pos = selm.event_position(showm.iren)
        info = selm.select(event_pos, showm.scene, (100, 100))
        selected_triangles = info[0]['vertex']
        if selected_triangles is not None:
            track_objects2.append(selected_triangles[0]//8)
        showm.render()

    showm.add_iren_callback(hover_callback)

    if recording:
        showm.record_events_to_file(recording_filename)

    else:
        showm.play_events_from_file(recording_filename)

    track_obj = set(track_objects2)

    npt.assert_(track_obj.issubset({0, 1, 2}))
    del track_objects2


def test_hover_selection_actors_only(recording=False):
    # simply hover going through blue, green, red cubes

    recording_filename = join(DATA_DIR, 'selector_actors.log.gz')

    centers, colors, radii = _get_three_cubes()

    scene = window.Scene()

    cube_actor = actor.cube(centers, directions=(1, 0, 0),
                            colors=colors, scales=radii)

    scene.add(cube_actor)

    selm = pick.SelectionManager(select='actors')

    showm = window.ShowManager(scene,
                               size=(900, 768), reset_camera=False,
                               order_transparent=True)

    showm.initialize()

    def hover_callback(_obj, _event):
        event_pos = selm.event_position(showm.iren)
        info = selm.pick(event_pos, showm.scene)
        selected_actor = info['actor']
        # print(id(selected_actor), id(cube_actor))
        if selected_actor is not None:
            npt.assert_equal(id(cube_actor), id(selected_actor))
        showm.render()

    showm.add_iren_callback(hover_callback)

    if recording:
        showm.record_events_to_file(recording_filename)

    else:
        showm.play_events_from_file(recording_filename)


if __name__ == "__main__":
    npt.run_module_suite()
