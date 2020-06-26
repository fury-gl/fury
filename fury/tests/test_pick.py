from tempfile import TemporaryDirectory as InTemporaryDirectory
import numpy as np
from fury import actor, window, io, ui, pick
from fury.testing import captured_output, assert_less_equal
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

    def timer_callback(_obj, _event):
        cnt = next(counter)
        tb.message = "Let's count up to 100 and exit :" + str(cnt)
        showm.scene.azimuth(0.05 * cnt)
        sphere_actor.GetProperty().SetOpacity(cnt/100.)
        if cnt % 10 == 0:
            info = pickm.pick(900/2, 768/2, 0, scene)
            print(info)

        showm.render()
        if cnt == 100:
            showm.exit()

    scene.add(tb)

    # Run every 200 milliseconds
    showm.add_timer_callback(True, 200, timer_callback)

    showm.start()


if __name__ == "__main__":

    test_picking_manager()