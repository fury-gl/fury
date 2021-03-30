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




def test_selector_manager_tmp():
    import vtk 
    hsel = vtk.vtkHardwareSelector()
    hsel.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
    hsel.SetRenderer(scene)

    event_pos = showm.iren.GetEventPosition()
    pass

import vtk
from fury.utils import numpy_support as nps

class SelectorManager(object):

    def __init__(self):
        self.hsel = vtk.vtkHardwareSelector()
        self.hsel.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)

    def select(self, disp_xy, sc, area=1):
        self.hsel.SetRenderer(sc)
        picking_area = area
        res = self.hsel.Select()
        event_pos = disp_xy
        self.hsel.SetArea(event_pos[0]-picking_area, event_pos[1]-picking_area,
                          event_pos[0]+picking_area, event_pos[1]+picking_area)
        res = self.hsel.Select()

        num_nodes = res.GetNumberOfNodes()
        if (num_nodes < 1):
            selected_node = None
        else:
            sel_node = res.GetNode(0)
            selected_nodes = set(np.floor(nps.vtk_to_numpy(
                sel_node.GetSelectionList())/2).astype(int))

            selected_node = list(selected_nodes)[0]

        print(selected_node)
        # selected_actor.text.SetText(str(selected_node))
        # if(selected_node is not None):
        #     if(labels is not None):
        #         selected_actor.text.SetText(labels[selected_node])
        #     else:
        #         selected_actor.text.SetText("#%d" % selected_node)
        #     selected_actor.SetPosition(positions[selected_node])

        # else:
        #     selected_actor.text.SetText("")

    def event_position(self, iren):
        """ Returns event display position from interactor

        Parameters
        ----------
        iren : interactor
            The interactor object can be retrieved for example
            using providing ShowManager's iren attribute.
        """
        return iren.GetEventPosition()


def test_selector_manager():

    xyz = 10 * np.random.rand(100, 3)
    colors = np.random.rand(100, 4)
    radii = np.random.rand(100, 3) + 0.5

    centers = 0.5 * np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0.]])
    colors2 = np.array([[0.8, 0, 0], [0, 0.8, 0], [0, 0, 0.8]])
    radii2 = 0.1 * np.array([50, 100, 150.])

    scene = window.Scene()

    # sphere_actor = actor.sphere(centers=xyz,
    #                             colors=colors,
    #                             radii=radii)

    directions = np.array([[np.sqrt(2)/2, 0, np.sqrt(2)/2],
                       [np.sqrt(2)/2, np.sqrt(2)/2, 0],
                       [0, np.sqrt(2)/2, np.sqrt(2)/2]])
    sphere_actor = actor.cube(centers, directions, colors2, scales=radii2)

    scene.add(sphere_actor)

    showm = window.ShowManager(scene,
                               size=(900, 768), reset_camera=False,
                               order_transparent=True)

    showm.initialize()

    tb = ui.TextBlock2D(bold=True)

    # use itertools to avoid global variables
    counter = itertools.count()

    pickm = SelectorManager()

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
            info = pickm.select((900//2, 768//2), scene, 1)
            record_indices['vertex_indices'].append(info['vertex'])
            record_indices['face_indices'].append(info['face'])
            record_indices['xyz'].append(info['xyz'])
            record_indices['actor'].append(info['actor'])

        showm.render()
        if cnt == 15:
            # showm.exit()
            pass

    def hover_callback(_obj, _event):
        event_pos = pickm.event_position(showm.iren)
        info = pickm.select(event_pos, showm.scene, 1)
        print(info)
        showm.render()
        

    scene.add(tb)

    # Run every 200 milliseconds
    # showm.add_timer_callback(True, 200, timer_callback)
    showm.add_iren_callback(hover_callback)
    showm.start()

    assert_greater(np.sum(np.array(record_indices['vertex_indices'])), 1)
    assert_greater(np.sum(np.array(record_indices['face_indices'])), 1)

    for ac in record_indices['actor']:
        if ac is not None:
            npt.assert_equal(ac is sphere_actor, True)

    assert_greater(np.sum(np.abs(np.diff(np.array(record_indices['xyz']),
                                         axis=0))), 0)



if __name__ == "__main__":

    # test_picking_manager()
    test_selector_manager()
