from collections.abc import Sequence
import vtk
import numpy as np
from fury.utils import numpy_support


class PickingManager:
    def __init__(self, vertices=True, faces=True, actors=True,
                 world_coords=True):
        """ Picking Manager helps with picking 3D objects

        Parameters
        -----------
        vertices : bool
            If True allows to pick vertex indices.
        faces : bool
            If True allows to pick face indices.
        actors : bool
            If True allows to pick actor indices.
        world_coords : bool
            If True allows to pick xyz position in world coordinates.
        """

        self.pickers = {}
        if vertices:
            self.pickers['vertices'] = vtk.vtkPointPicker()
        if faces:
            self.pickers['faces'] = vtk.vtkCellPicker()
        if actors:
            self.pickers['actors'] = vtk.vtkPropPicker()
        if world_coords:
            self.pickers['world_coords'] = vtk.vtkWorldPointPicker()

    def pick(self, disp_xy, sc):
        """ Pick on display coordinates

        Parameters
        ----------
        disp_xy : tuple
            Display coordinates x, y.

        sc : Scene
        """

        x, y = disp_xy
        z = 0
        info = {'vertex': None, 'face': None, 'actor': None, 'xyz': None}
        keys = self.pickers.keys()

        if 'vertices' in keys:
            self.pickers['vertices'].Pick(x, y, z, sc)
            info['vertex'] = self.pickers['vertices'].GetPointId()

        if 'faces' in keys:
            self.pickers['faces'].Pick(x, y, z, sc)
            info['vertex'] = self.pickers['faces'].GetPointId()
            info['face'] = self.pickers['faces'].GetCellId()

        if 'actors' in keys:
            self.pickers['actors'].Pick(x, y, z, sc)
            info['actor'] = self.pickers['actors'].GetViewProp()

        if 'world_coords' in keys:
            self.pickers['world_coords'].Pick(x, y, z, sc)
            info['xyz'] = self.pickers['world_coords'].GetPickPosition()

        return info

    def event_position(self, iren):
        """ Returns event display position from interactor

        Parameters
        ----------
        iren : interactor
            The interactor object can be retrieved for example
            using providing ShowManager's iren attribute.
        """
        return iren.GetEventPosition()

    def pickable_on(self, actors):
        """ Choose which actors can be picked

        Parameters
        ----------
        actors : actor or sequence of actors
        """

        if isinstance(actors, Sequence):
            for a in actors:
                a.PickableOn()
        else:
            actors.PickableOn()

    def pickable_off(self, actors):
        """ Choose which actors cannot be picked

        Parameters
        ----------
        actors : actor or sequence of actors
        """

        if isinstance(actors, Sequence):
            for a in actors:
                a.PickableOff()
        else:
            actors.PickableOff()


class SelectionManager:

    def __init__(self, select='faces'):
        """ Selection Manager helps with picking many
        objects simultaneously

        Parameters
        -----------
        select : 'faces'
            Options are 'faces', 'vertices' or 'actors'.
            Default 'faces'.

        Methods
        -------
        select()
        pick()

        """

        self.hsel = vtk.vtkHardwareSelector()
        self.update_selection_type(select)

    def update_selection_type(self, select):
        """ Update selection type

        Parameters
        -----------
        select : 'faces'
            Options are 'faces', 'vertices' or 'actors'.
            Default 'faces'.
        """

        self.selected_type = select.lower()
        if select == 'faces' or select == 'edges':
            self.hsel.SetFieldAssociation(
                vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
        elif select == 'points' or select == 'vertices':
            self.hsel.SetFieldAssociation(
                vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS)
        elif select == 'actors':
            self.hsel.SetActorPassOnly(True)
        else:
            raise ValueError('Unkown parameter select')

    def pick(self, disp_xy, sc):
        """ Pick on display coordinates returns a single object

        Parameters
        ----------
        disp_xy : tuple
            Display coordinates x, y.

        sc : Scene
        """
        return self.select(disp_xy, sc, area=0)[0]

    def select(self, disp_xy, sc, area=0):
        """ Select multiple objects using display coordinates

        Parameters
        ----------
        disp_xy : tuple
            Display coordinates x, y.

        sc : Scene

        area : int or 2d tuple of ints
            Selection area around x, y coords.
        """
        info_plus = {}
        self.hsel.SetRenderer(sc)
        if isinstance(area, int):
            picking_area = area, area
        else:
            picking_area = area

        try:
            self.hsel.SetArea(disp_xy[0] - picking_area[0],
                              disp_xy[1] - picking_area[1],
                              disp_xy[0] + picking_area[0],
                              disp_xy[1] + picking_area[1])
            res = self.hsel.Select()

        except OverflowError:
            return {0: {'node': None, 'vertex': None,
                        'face': None, 'actor': None}}

        num_nodes = res.GetNumberOfNodes()
        if num_nodes < 1:
            sel_node = None
            return {0: {'node': None, 'vertex': None,
                        'face': None, 'actor': None}}
        else:

            for i in range(num_nodes):

                sel_node = res.GetNode(i)
                info = {'node': None, 'vertex': None,
                        'face': None, 'actor': None}

                if sel_node is not None:
                    selected_nodes = set(
                        np.floor(numpy_support.vtk_to_numpy(
                                    sel_node.GetSelectionList())).astype(int))

                    info['node'] = sel_node
                    info['actor'] = \
                        sel_node.GetProperties().Get(sel_node.PROP())
                    if self.selected_type == 'faces':
                        info['face'] = list(selected_nodes)
                    if self.selected_type == 'vertex':
                        info['vertex'] = list(selected_nodes)
                info_plus[i] = info

        return info_plus

    def event_position(self, iren):
        """ Returns event display position from interactor

        Parameters
        ----------
        iren : interactor
            The interactor object can be retrieved for example
            using ShowManager's iren attribute.
        """
        return iren.GetEventPosition()

    def selectable_on(self, actors):
        """ Choose which actors can be selected

        Parameters
        ----------
        actors : actor or sequence of actors
        """

        if isinstance(actors, Sequence):
            for a in actors:
                a.PickableOn()
        else:
            actors.PickableOn()

    def selectable_off(self, actors):
        """ Choose which actors cannot be selected

        Parameters
        ----------
        actors : actor or sequence of actors
        """

        if isinstance(actors, Sequence):
            for a in actors:
                a.PickableOff()
        else:
            actors.PickableOff()
