import vtk
# from vtkmodules import vtkCommonCore

class PickingManager(object):
    def __init__(self, mode='face', selector_vertex=True):
        """ Picking Manager helps with picking 3D objects

        Parameters
        -----------
        mode : str
            If ``vertex`` allows to pick vertices indices.
            If ``face`` allows to pick face indices and vertices indices.
            If ``actor`` allows to pick actor index.
            If ``world`` allows to pick xyz position in world coords.
            If ``selector`` allows to pick faces and vertices but should
            be faster in hovering.

        selector_vertex : bool
            Used only in ``selector`` mode.
        """
        self.mode = mode
        self.selector_vertex = selector_vertex
        if self.mode == 'face':
            self.picker = vtk.vtkCellPicker()
        elif mode == 'vertex':
            self.picker = vtk.vtkPointPicker()
        elif self.mode == 'actor':
            self.picker = vtk.vtkPropPicker()
        elif mode == 'world':
            self.picker = vtk.vtkWorldPointPicker()
        elif self.mode == 'selector':
            self.picker = vtk.vtkScenePicker()
            self.picker.SetEnableVertexPicking(self.selector_vertex)
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
        elif self.mode == 'selector':
            event_pos = (int(x), int(y))
            info['vertex'] = self.picker.GetVertexId(event_pos)
            info['face'] = self.picker.GetCellId(event_pos)
        else:
            raise ValueError('Unknown picking mode')
        return info

    def event_position(self, iren):
        return iren.GetEventPosition()
