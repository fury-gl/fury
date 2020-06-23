from fury import window


class PickingManager(object):
    def __init__(self, mode='face', selector_vertex=True):
        self.mode = mode
        if self.mode == 'face':
            self.picker = window.vtk.vtkCellPicker()
        elif mode == 'vertex':
            self.picker = window.vtk.vtkPointPicker()
        elif self.mode == 'actor':
            self.picker = window.vtk.vtkPropPicker()
        elif mode == 'world':
            self.picker = window.vtk.vtkWorldPointPicker()
        elif self.mode == 'selector':
            self.picker = window.vtk.vtkScenePicker()
            self.picker.SetEnableVertexPicking(selector_vertex)
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
            event_pos = (x, y, z)
            info['vertex'] = self.picker.GetVertexId(event_pos)
            info['face'] = self.picker.GetCellId(event_pos)
        else:
            raise ValueError('Unknown picking mode')
        return info

    def event_position(self, iren):
        return iren.GetEventPosition()
