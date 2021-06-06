from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vtk
import multiprocessing
import sys
if sys.version_info.minor >= 8:
    from multiprocessing import shared_memory
    PY_VERSION_8 = False
else:
    shared_memory = None
    PY_VERSION_8 = False

import numpy as np

from fury.stream.tools import CircularQueue
# import imagezmq


class FuryStreamClient:
    def __init__(
            self, showm,
            window_size=(200, 200),
            max_window_size=None,
            buffer_count=2,
            image_buffers=None,
            info_buffer=None,
            use_raw_array=None):

        '''

        Parameters
        ----------
            showm: fury showm manager
        '''
        self.showm = showm
        self.window2image_filter = vtk.vtkWindowToImageFilter()
        self.window2image_filter.SetInput(self.showm.window)
        # self.write_in_stdout = write_in_stdout
        self.image_buffers = []
        self.image_buffer_names = []
        self.image_reprs = []
        self.buffer_count = buffer_count
        if max_window_size is None:
            max_window_size = window_size
        self.max_size = max_window_size[0]*max_window_size[1]
        self.max_window_size = max_window_size
        if self.max_size < window_size[0]*window_size[1]:
            raise ValueError('max_window_size must be greater than window_size')

        # se use_raw_array não é setado escolhe a abordagem
        # automaticamente
        if use_raw_array is None:
            use_raw_array = not PY_VERSION_8

        if not PY_VERSION_8 and not use_raw_array:
            raise ValueError("""
                In order to use the SharedMemory approach
                you should have to use python 3.8 or higher""")
        if info_buffer is None:
            # 0 number of components
            # 1 id buffer
            # 2, 3, width first buffer, height first buffer
            # 4, 5, width second buffer , height second buffer
            info_list = [3, 0]
            for _ in range(self.buffer_count):
                info_list += [self.max_window_size[0]]
                info_list += [self.max_window_size[1]]
            self.info_buffer = multiprocessing.RawArray(
                    'I', info_list
            )
        if use_raw_array:
            self.image_buffer_names = None
            if image_buffers is None:
                for _ in range(self.buffer_count):
                    self.image_buffers.append(multiprocessing.RawArray(
                        'B', np.random.randint(
                            0, 255,
                            size=max_window_size[0]*max_window_size[1]*3)
                        .astype('uint8')))
            else:
                self.info_buffer = info_buffer
                self.image_buffers = image_buffers
                # do not work anymore with shared memory
        else:
            for _ in range(self.buffer_count):
                bufferSize = max_window_size[0]*max_window_size[1]*3
                buffer = shared_memory.SharedMemory(
                    create=True, size=bufferSize)
                self.image_buffers.append(buffer)
                self.image_reprs.append(
                    np.ndarray(
                        bufferSize, dtype=np.uint8, buffer=buffer.buf))
                self.image_buffer_names.append(buffer.name)

        self._id_timer = None
        self._id_observer = None
        self.sender = None
        self._in_request = False
        self.update = True
        self.use_raw_array = use_raw_array
        # if broker_url is not None:
        #    pass
        # self.sender = imagezmq.ImageSender(connect_to=broker_url)

    def init(self, ms=16,):
        # if self.write_in_stdout:
        # os.system('cls' if os.name == 'nt' else 'clear')

        window2image_filter = self.window2image_filter

        def callback(caller, timerevent):
            if not self._in_request:
                if not self.update:
                    return
                self._in_request = True
                self.window2image_filter.Update()
                self.window2image_filter.Modified()
                vtk_image = window2image_filter.GetOutput()
                vtk_array = vtk_image.GetPointData().GetScalars()
                num_components = vtk_array.GetNumberOfComponents()
                if self.use_raw_array:
                    h, w, _ = vtk_image.GetDimensions()
                    np_arr = vtk_to_numpy(vtk_array).astype('uint8')
                    np_arr = np_arr.flatten()
                else:
                    w, h, _ = vtk_image.GetDimensions()
                    np_arr = np.frombuffer(vtk_array, dtype=np.uint8)

                if self.image_buffers is not None:
                    buffer_size = int(h*w)

                    self.info_buffer[0] = num_components

                    # N-Buffering
                    next_buffer_index = (self.info_buffer[1]+1) \
                        % self.buffer_count
                    # print(next_buffer_index)
                    # 2, 4, 6
                    if buffer_size == self.max_size:
                        if self.use_raw_array:
                            self.image_buffers[next_buffer_index][:] = np_arr
                        else:
                            self.image_reprs[next_buffer_index][:] = np_arr
                    elif buffer_size < self.max_size:
                        if self.use_raw_array:
                            self.image_buffers[
                                next_buffer_index][0:buffer_size*3] = np_arr
                        else:
                            self.image_reprs[
                                next_buffer_index][0:buffer_size*3] = np_arr
                    else:
                        rand_img = np.random.randint(
                            0, 255, size=self.max_size*3,
                            dtype='uint8')
                        if self.use_raw_array:
                            self.image_buffers[
                                next_buffer_index][:] = rand_img
                        else:
                            self.image_reprs[
                                next_buffer_index][:] = rand_img
                        w = self.max_window_size[0]
                        h = self.max_window_size[1]
                    self.info_buffer[2+next_buffer_index*2] = w
                    self.info_buffer[2+next_buffer_index*2+1] = h
                    self.info_buffer[1] = next_buffer_index
                self._in_request = False

        if ms > 0:
            id_timer = self.showm.add_timer_callback(
                True, ms, callback)
            self._id_timer = id_timer
        else:
            id_observer = self.showm.iren.AddObserver(
                'RenderEvent', callback)
            self._id_observer = id_observer
        self.showm.render()
        callback(None, None)

    def cleanup(self):
        if self.image_buffer_names is not None:
            for buffer in self.image_buffers:
                buffer.close()
                buffer.unlink()

    def stop(self):
        if self._id_timer is not None:
            self.showm.destroy_timer(self._id_timer)

        if self._id_observer is not None:
            self.showm.iren.RemoveObserver(self._id_observer)


class FuryStreamInteraction:
    def __init__(
            self, showm,  max_queue_size=50,
            queue_head_tail_buffer=None,
            queue_buffers_list=None, fury_client=None,):

        self.showm = showm
        self.iren = self.showm.iren
        self.fury_client = fury_client
        self.circular_queue = CircularQueue(
            max_size=max_queue_size, dimension=6,
            head_tail_buffer=queue_head_tail_buffer,
            buffers_list=queue_buffers_list)
        self._id_timer = None

    def start(self, ms=16):

        def callback(caller, timerevent):
            # self.showm.scene.GetActiveCamera().Azimuth(2)

            data = self.circular_queue.dequeue()
            if data is not None:
                event_id = data[0]
                newX = int(self.showm.size[0]*data[2])
                newY = int(self.showm.size[1]*data[3])
                ctrl_key = int(data[4])
                shift_key = int(data[5])
                newY = self.showm.size[1] - newY
                if event_id == 1:
                    zoomFactor = 1.0 - data[1] / 1000.0
                    camera = self.showm.scene.GetActiveCamera()
                    fp = camera.GetFocalPoint()
                    pos = camera.GetPosition()
                    delta = [fp[i] - pos[i] for i in range(3)]
                    camera.Zoom(zoomFactor)

                    pos2 = camera.GetPosition()
                    camera.SetFocalPoint(
                        [pos2[i] + delta[i] for i in range(3)])
                    if data[1] < 0:
                        self.iren.MouseWheelForwardEvent()
                    else:
                        self.iren.MouseWheelBackwardEvent()

                    self.showm.window.Modified()

                elif event_id == 2:
                    self.iren.SetEventInformation(
                        newX, newY, ctrl_key, shift_key, chr(0), 0, None)

                    self.iren.MouseMoveEvent()

                elif event_id in [3, 4, 5, 6, 7, 8]:
                    self.iren.SetEventInformation(
                        newX, newY, ctrl_key, shift_key,
                        chr(0), 0, None)

                    mouse_actions = {
                        3: self.showm.iren.LeftButtonPressEvent,
                        4: self.showm.iren.LeftButtonReleaseEvent,
                        5: self.showm.iren.MiddleButtonPressEvent,
                        6: self.showm.iren.MiddleButtonReleaseEvent,
                        7: self.showm.iren.RightButtonPressEvent,
                        8: self.showm.iren.RightButtonReleaseEvent,
                    }
                    mouse_actions[event_id]()

                    # self.iren.SetLastEventPosition(newX, newY)
                    self.showm.window.Modified()
                # maybe when the fury host rendering is disabled
                # self.fury_client.window2image_filter.Update()
                # self.fury_client.window2image_filter.Modified()
                # self.showm.render()

        self._id_timer = self.showm.add_timer_callback(True, ms, callback)

    def stop(self):
        if self._id_timer is not None:
            self.showm.destroy_timer(self._id_timer)
