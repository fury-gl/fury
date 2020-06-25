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

        # if self.mode == 'face':
        #     self.picker = vtk.vtkCellPicker()
        # elif mode == 'vertex':
        #     self.picker = vtk.vtkPointPicker()
        # elif self.mode == 'actor':
        #     self.picker = vtk.vtkPropPicker()
        # elif mode == 'world':
        #     self.picker = vtk.vtkWorldPointPicker()
        # else:
        #     raise ValueError('Unknown picking option')

    def pick(self, x, y, z, sc):

        info = {'vertex': None, 'face': None, 'actor': None, 'xyz': None}

        if 'vertices' in self.pickers.keys():
            self.pickers['vertices'].Pick(x, y, z, sc)
            info['vertex'] = self.pickers['vertices'].GetPointId()

        if 'faces' in self.pickers.keys():
            self.pickers['faces'].Pick(x, y, z, sc)
            info['vertex'] = self.pickers['faces'].GetPointId()
            info['face'] = self.pickers['faces'].GetCellId()

        if 'actors' in self.pickers.keys():
            self.pickers['actors'].Pick(x, y, z, sc)
            info['actor'] = self.pickers['actors'].GetViewProp()

        if 'world_coords' in self.pickers.keys():
            self.pickers['world_coords'].Pick(x, y, z, sc)
            info['xyz'] = self.pickers['world_coords'].GetPickPosition()

        return info

    def event_position(self, iren):
        return iren.GetEventPosition()
