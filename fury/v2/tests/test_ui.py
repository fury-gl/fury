import numpy.testing as npt

from fury.v2.ui import Panel2D


def test_panel_2d():
    panel = Panel2D((200, 50))

    npt.assert_equal(panel.size, (200, 50))
    npt.assert_equal(panel.obj.local.position, (0, 0, 0))
    npt.assert_equal(panel.obj.material.color, (255, 255, 0, 0.6))
