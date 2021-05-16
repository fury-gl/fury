import numpy as np
from fury import actor, window, ui, pick
from fury.testing import assert_greater
import numpy.testing as npt
import itertools


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
        tb.message = "Let's count up to 100 and exit :" + str(cnt)
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


def test_selector_manager():

    centers = 0.5 * np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0.]])
    colors = np.array([[0.8, 0, 0], [0, 0.8, 0], [0, 0, 0.8]])
    radii = 0.1 * np.array([50, 100, 150.])

    scene = window.Scene()

    cube_actor = actor.cube(centers, directions=(1, 0, 2),
                            colors=colors, scales=radii)
    # sphere_actor.GetProperty().SetRepresentationToWireframe()
    print('Sphere actor', id(cube_actor))

    pts = 100 * (np.random.rand(100, 3) - 0.5) + np.array([20, 0, 0.])
    pts_actor = actor.dots(pts, dot_size=10)
    print('Points actor', id(pts_actor))

    scene.add(cube_actor)
    scene.add(pts_actor)

    showm = window.ShowManager(scene,
                               size=(900, 768), reset_camera=False,
                               order_transparent=True)

    showm.initialize()

    tb = ui.TextBlock2D(bold=True)

    # use itertools to avoid global variables
    counter = itertools.count()

    selm = pick.SelectionManager(select='faces')

    def timer_callback(_obj, _event):
        cnt = next(counter)
        tb.message = "Let's count up to 100 and exit :" + str(cnt)
        # showm.scene.azimuth(0.05 * cnt)
        # sphere_actor.GetProperty().SetOpacity(cnt/100.)
        if cnt % 10 == 0:
            # select large area
            info_plus = selm.select((900//2, 768//2), scene, (30, 30))
            print(info_plus)
            for info in info_plus.keys():
                if info_plus[info]['actor'] in [cube_actor, pts_actor]:
                    npt.assert_(True)
                else:
                    npt.assert_(False)
            # print(id(info['actor']) == id(cube_actor))
            # print(id(info['actor']) == id(pts_actor))

        showm.render()
        if cnt == 15:
            # showm.exit()
            pass

    def hover_callback(_obj, _event):
        event_pos = selm.event_position(showm.iren)
        #pick a single pixel
        info = selm.pick(event_pos, showm.scene)
        print(info)
        # if info['face'] is not None:
        #     npt.assert_equal(len(info['face']), 1)
        showm.render()


    scene.add(tb)

    # Run every 200 milliseconds
    showm.add_timer_callback(True, 200, timer_callback)
    showm.add_iren_callback(hover_callback)
    showm.start()

if __name__ == "__main__":

    # test_picking_manager()
    test_selector_manager()
