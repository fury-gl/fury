from fury import window


class PickingManager(object):
    def __init__(self, mode='face'):
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
        else:
            raise ValueError('Unknown picking option')

    def pick(self, x, y, z, sc):
        self.picker.Pick(x, y, z, sc)
        if self.mode == 'face':
            return {'vertex': self.picker.GetPointId(),
                    'face': self.picker.GetCellId()}
        elif self.mode == 'vertex':
            return {'vertex': self.picker.GetPointId(),
                    'face': None}
        elif self.mode == 'actor':
            return {'actor': self.picker.GetViewProp()}
        elif self.mode == 'world':
            # TODO need to understand Start/End selection
            return {'xyz': self.picker.GetPickPosition()}
        elif self.mode == 'selector':
            pass
        else:
            raise ValueError('Uknown picking mode')

    def event_position(self, iren):
        return iren.GetEventPosition()