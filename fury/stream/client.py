import vtk
import multiprocessing
import sys
if sys.version_info.minor >= 8:
    from multiprocessing import shared_memory
    PY_VERSION_8 = True
else:
    shared_memory = None
    PY_VERSION_8 = False

import numpy as np

from fury.stream.tools import CircularQueue, IntervalTimer

import logging
import time


class FuryStreamClient:
    def __init__(
            self, showm,
            window_size=(200, 200),
            max_window_size=None,
            whithout_iren_start=False,
            use_raw_array=True,
            buffer_count=2,
    ):
        '''

        Parameters
        ----------
            showm: fury showm manager
        '''

        self._whithout_iren_start = whithout_iren_start
        self.showm = showm
        self.window2image_filter = vtk.vtkWindowToImageFilter()
        self.window2image_filter.SetInput(self.showm.window)
        self.image_buffers = []
        self.image_buffer_names = []
        self.info_buffer_name = None
        self.image_reprs = []
        self.buffer_count = buffer_count
        if max_window_size is None:
            max_window_size = window_size

        self.max_size = max_window_size[0]*max_window_size[1]
        self.max_window_size = max_window_size
        if self.max_size < window_size[0]*window_size[1]:
            raise ValueError(
                'max_window_size must be greater than window_size')

        if not PY_VERSION_8 and not use_raw_array:
            raise ValueError("""
                In order to use the SharedMemory approach
                you should have to use python 3.8 or higher""")

        # 0 number of components
        # 1 id buffer
        # 2, 3, width first buffer, height first buffer
        # 4, 5, width second buffer , height second buffer
        info_list = [3, 0]
        for _ in range(self.buffer_count):
            info_list += [self.max_window_size[0]]
            info_list += [self.max_window_size[1]]
        info_list = np.array(
            info_list, dtype='uint64'
        )
        if use_raw_array:
            self.info_buffer = multiprocessing.RawArray(
                    'I', info_list
            )
            self.info_buffer_repr = np.ctypeslib.as_array(self.info_buffer)
        else:
            self.info_buffer = shared_memory.SharedMemory(
                create=True, size=info_list.nbytes)
            self.info_buffer_repr = np.ndarray(
                    info_list.shape[0],
                    dtype='uint64', buffer=self.info_buffer.buf)
            self.info_buffer_name = self.info_buffer.name
        if use_raw_array:
            self.image_buffer_names = None

            self.image_reprs = []
            for _ in range(self.buffer_count):
                buffer = multiprocessing.RawArray(
                    'B', np.random.randint(
                        0, 255,
                        size=max_window_size[0]*max_window_size[1]*3)
                    .astype('uint8'))
                self.image_buffers.append(buffer)
                self.image_reprs.append(
                    np.ctypeslib.as_array(buffer))
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
        self._interval_timer = None
        self._in_request = False
        self.update = True
        self.use_raw_array = use_raw_array
        self._started = False

    def start(self, ms=16,):
        if self._started:
            self.stop()

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
                # num_components = vtk_array.GetNumberOfComponents()

                w, h, _ = vtk_image.GetDimensions()
                np_arr = np.frombuffer(vtk_array, dtype='uint8')

                if self.image_buffers is not None:
                    buffer_size = int(h*w)
                    # self.info_buffer[0] = num_components

                    # N-Buffering
                    next_buffer_index = int(
                        (self.info_buffer_repr[1]+1) % self.buffer_count)
                    if buffer_size == self.max_size:
                        # if self.use_raw_array:
                        # throws a type error due uint8
                        # memoryview(
                        #     self.image_buffers[next_buffer_index]
                        # )[:] = np_arr
                        self.image_reprs[
                            next_buffer_index][:] = np_arr
                    elif buffer_size < self.max_size:
                        self.image_reprs[
                                next_buffer_index][0:buffer_size*3] = np_arr
                    else:
                        rand_img = np.random.randint(
                            0, 255, size=self.max_size*3,
                            dtype='uint8')

                        self.image_reprs[
                            next_buffer_index][:] = rand_img

                        w = self.max_window_size[0]
                        h = self.max_window_size[1]
                    self.info_buffer_repr[2+next_buffer_index*2] = w
                    self.info_buffer_repr[2+next_buffer_index*2+1] = h
                    self.info_buffer_repr[1] = next_buffer_index
                self._in_request = False

        if ms > 0:
            if self._whithout_iren_start:
                self._interval_timer = IntervalTimer(
                    ms/1000,
                    callback,
                    None,
                    None)
            else:
                id_timer = self.showm.add_timer_callback(
                    True, ms, callback)
                # self.showm.window.AddObserver("TimerEvent", callback)
                # id_timer = self.showm.window.CreateRepeatingTimer(ms)
                self._id_timer = id_timer

        else:
            # id_observer = self.showm.iren.AddObserver(
            id_observer = self.showm.iren.AddObserver(
                'RenderEvent', callback)
            self._id_observer = id_observer
        self.showm.render()
        self._started = True
        callback(None, None)

    def stop(self):
        if not self._started:
            return

        if self._interval_timer is not None:
            self._interval_timer.stop()
        if self._id_timer is not None:
            # self.showm.destroy_timer(self._id_timer)
            self.showm.iren.DestroyTimer(self._id_timer)
            self._id_timer = None
        if self._id_observer is not None:
            self.showm.iren.RemoveObserver(self._id_observer)
            self._id_observer = None

        self._started = False

    def cleanup(self):
        if not self.use_raw_array:
            self.info_buffer.close()
            # this it's due the python core issues
            # https://bugs.python.org/issue38119
            # https://bugs.python.org/issue39959
            # https://github.com/luizalabs/shared-memory-dict/issues/13
            try:
                self.info_buffer.unlink()
            except FileNotFoundError:
                print(f'Shared Memory {self.info_buffer_name}\
                        (info_buffer) File not found')
            for buffer, name in zip(
                    self.image_buffers, self.image_buffer_names):
                buffer.close()
                try:
                    buffer.unlink()
                except FileNotFoundError:
                    print(f'Shared Memory {name}(buffer image) File not found')


def interaction_callback(
        circular_queue, showm, iren, render_after=False):
    ts = time.time()*1000
    data = circular_queue.dequeue()
    if data is not None:
        event_id = data[0]
        user_timestamp = data[6]
        logging.info(
            'Interaction: time to dequeue ' +
            f'{ts-user_timestamp:.2f} ms')

        ts = time.time()*1000
        newX = int(showm.size[0]*data[2])
        newY = int(showm.size[1]*data[3])
        ctrl_key = int(data[4])
        shift_key = int(data[5])
        newY = showm.size[1] - newY
        if event_id == 1:
            zoomFactor = 1.0 - data[1] / 1000.0
            camera = showm.scene.GetActiveCamera()
            fp = camera.GetFocalPoint()
            pos = camera.GetPosition()
            delta = [fp[i] - pos[i] for i in range(3)]
            camera.Zoom(zoomFactor)

            pos2 = camera.GetPosition()
            camera.SetFocalPoint(
                [pos2[i] + delta[i] for i in range(3)])
            if data[1] < 0:
                iren.MouseWheelForwardEvent()
            else:
                iren.MouseWheelBackwardEvent()

            showm.window.Modified()

        elif event_id == 2:
            iren.SetEventInformation(
                newX, newY, ctrl_key, shift_key, chr(0), 0, None)

            iren.MouseMoveEvent()

        elif event_id in [3, 4, 5, 6, 7, 8]:
            iren.SetEventInformation(
                newX, newY, ctrl_key, shift_key,
                chr(0), 0, None)

            mouse_actions = {
                3: showm.iren.LeftButtonPressEvent,
                4: showm.iren.LeftButtonReleaseEvent,
                5: showm.iren.MiddleButtonPressEvent,
                6: showm.iren.MiddleButtonReleaseEvent,
                7: showm.iren.RightButtonPressEvent,
                8: showm.iren.RightButtonReleaseEvent,
            }
            mouse_actions[event_id]()
            showm.window.Modified()
        logging.info(
            'Interaction: time to peform event ' +
            f'{ts-user_timestamp:.2f} ms')
        # maybe when the fury host rendering is disabled
        # fury_client.window2image_filter.Update()
        # fury_client.window2image_filter.Modified()
        # this should be called if we are using
        # renderevent attached to a vtkwindow instance
        if render_after:
            showm.render()


class FuryStreamInteraction:
    def __init__(
            self, showm,  max_queue_size=50,
            use_raw_array=True, whithout_iren_start=False):

        self.showm = showm
        self.iren = self.showm.iren
        self.circular_queue = CircularQueue(
            max_size=max_queue_size, dimension=8,
            use_raw_array=use_raw_array)
        self._id_timer = None
        self._id_observer = None
        self._interval_timer = None
        self._whithout_iren_start = whithout_iren_start
        self._started = False

    def start(self, ms=16):
        if self._started:
            self.stop()

        def callback(caller, timerevent):
            interaction_callback(
                self.circular_queue, self.showm, self.iren, False)

        if self._whithout_iren_start:
            self._interval_timer = IntervalTimer(
                ms/1000,
                interaction_callback,
                self.circular_queue,
                self.showm,
                self.iren,
                True
            )
        else:
            self._id_observer = self.showm.iren.AddObserver(
                "TimerEvent", callback)
            self._id_timer = self.showm.iren.CreateRepeatingTimer(ms)

        self._started = True

    def stop(self):
        if not self._started:
            return

        if self._id_timer is not None:
            self.showm.window.DestroyTimer(self._id_timer)
        else:
            if self._interval_timer is not None:
                self._interval_timer.stop()
                del self._interval_timer
                self._interval_timer = None

        self._started = False

    def cleanup(self):
        self.circular_queue.cleanup()
