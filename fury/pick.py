import vtk


class PickingManager(object):
    def __init__(self, mode='face'):
        """ Picking Manager helps with picking 3D objects

        Parameters
        -----------
        mode : str
            If ``vertex`` allows to pick vertex indices.
            If ``face`` allows to pick face indices and vertices indices.
            If ``actor`` allows to pick actor indices.
            If ``world`` allows to pick xyz position in world coords.
        """

        self.mode = mode
        if self.mode == 'face':
            self.picker = vtk.vtkCellPicker()
        elif mode == 'vertex':
            self.picker = vtk.vtkPointPicker()
        elif self.mode == 'actor':
            self.picker = vtk.vtkPropPicker()
        elif mode == 'world':
            self.picker = vtk.vtkWorldPointPicker()
        else:
            raise ValueError('Unknown picking option')

    def pick(self, x, y, z, sc):
        if self.mode == 'selector':
            pass
        else:
            self.picker.Pick(x, y, z, sc)
        info = {'vertex': None, 'face': None, 'actor': None, 'xyz': None}
        if self.mode == 'face':
            info['vertex'] = self.picker.GetPointId()
            info['face'] = self.picker.GetCellId()
        elif self.mode == 'vertex':
            info['vertex'] = self.picker.GetPointId()
        elif self.mode == 'actor':
            info['actor'] = self.picker.GetViewProp()
        elif self.mode == 'world':
            info['xyz'] = self.picker.GetPickPosition()
        else:
            raise ValueError('Unknown picking mode')
        return info

    def event_position(self, iren):
        return iren.GetEventPosition()
