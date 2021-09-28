import vtk


class PickingManager(object):
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
        disp_xyz : tuple
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
