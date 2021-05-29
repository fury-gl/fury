import os
from vtk.util.numpy_support import vtk_to_numpy
import vtk
import imagezmq


class FuryStreamClient:
    def __init__(
        self, showm, write_in_stdout=True,
            image_buffer=None, info_buffer=None,
            broker_url=None):
        '''

        Parameters
        ----------
            showm: fury showm manager
        '''
        self.showm = showm
        self.window2image_filter = vtk.vtkWindowToImageFilter()
        self.window2image_filter.SetInput(self.showm.window)
        self.write_in_stdout = write_in_stdout
        self.image_buffer = image_buffer
        self.info_buffer = info_buffer
        self._id_timer = None
        self._id_observer = None
        self.sender = None
        self._in_request = False
        if broker_url is not None:
            self.sender = imagezmq.ImageSender(connect_to=broker_url)

    def init(self, ms=16):
        if self.write_in_stdout:
            os.system('cls' if os.name == 'nt' else 'clear')

        window2image_filter = self.window2image_filter

        def callback(caller, timerevent):
            if not self._in_request:
                self._in_request = True
                self.window2image_filter.Update()
                self.window2image_filter.Modified()
                vtk_image = window2image_filter.GetOutput()
                vtk_array = vtk_image.GetPointData().GetScalars()
                np_arr = vtk_to_numpy(vtk_array).astype('uint8')
                
                h, w, _ = vtk_image.GetDimensions()
                num_components = vtk_array.GetNumberOfComponents()
            
                if self.image_buffer is not None:
                    if self.info_buffer is not None:
                        self.info_buffer[0] = h
                        self.info_buffer[1] = w
                        self.info_buffer[2] = num_components
                    np_arr = np_arr.flatten()
                    self.image_buffer[:] = np_arr
                self._in_request = False

        if ms > 0:
            id_timer = self.showm.add_timer_callback(
                True, ms, callback)
            self._id_timer = id_timer
        else:
            id_observer = self.showm.iren.AddObserver(
                'RenderEvent', callback)
            self._id_observer = id_observer

    def stop(self):
        if self._id_timer is not None:
            self.showm.destroy_timer(self._id_timer)

        if self._id_observer is not None:
            self.showm.iren.RemoveObserver(self._id_observer)
