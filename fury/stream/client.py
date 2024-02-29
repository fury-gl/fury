import logging
import platform
import time
from functools import partial

import numpy as np
import vtk

from fury.stream.constants import _CQUEUE, PY_VERSION_8
from fury.stream.tools import (
    ArrayCircularQueue,
    IntervalTimer,
    IntervalTimerThreading,
    RawArrayImageBufferManager,
    SharedMemCircularQueue,
    SharedMemImageBufferManager,
)


def callback_stream_client(stream_client):
    """This callback is used to update the image inside of
    the ImageManager instance

    Parameters
    ----------
    stream_client : StreamClient

    """
    if stream_client.in_request:
        return
    stream_client.in_request = True
    stream_client.window2image_filter.Update()
    stream_client.window2image_filter.Modified()
    vtk_image = stream_client.window2image_filter.GetOutput()
    vtk_array = vtk_image.GetPointData().GetScalars()

    w, h, _ = vtk_image.GetDimensions()
    np_arr = np.frombuffer(vtk_array, dtype='uint8')
    if np_arr is None:
        stream_client.in_request = False
        return

    stream_client.img_manager.write_into(w, h, np_arr)
    stream_client.in_request = False


class FuryStreamClient:
    """This obj is responsible to create a StreamClient."""

    def __init__(
        self,
        showm,
        max_window_size=None,
        use_raw_array=True,
        whithout_iren_start=False,
        num_buffers=2,
    ):
        """A StreamClient extracts a framebuffer from the OpenGL context
        and writes into a shared memory resource.

        Parameters
        ----------
        showm : ShowManager
        max_window_size : tuple of ints, optional
            This allows resize events inside of the FURY window instance.
            Should be greater than the window size.
        use_raw_array : bool, optional
            If False then FuryStreamClient will use SharedMemory
            instead of RawArrays. Notice that Python >=3.8 it's a
            requirement
            to use SharedMemory)
        whithout_iren_start : bool, optional
            Sometimes you can't initiate the vtkInteractor instance.
        num_buffers : int, optional
            Number of buffers to be used in the n-buffering
            technique.

        """
        self._whithout_iren_start = whithout_iren_start
        self.showm = showm
        self.window2image_filter = vtk.vtkWindowToImageFilter()
        self.window2image_filter.SetInput(self.showm.window)
        self.image_buffers = []
        self.image_buffer_names = []
        self.info_buffer_name = None
        self.image_reprs = []
        self.num_buffers = num_buffers
        if max_window_size is None:
            max_window_size = (
                int(self.showm.size[0] * (1 + 0.1)),
                int(self.showm.size[1] * (1 + 0.1)),
            )

        self.max_size = max_window_size[0] * max_window_size[1]
        self.max_window_size = max_window_size
        if self.max_size < self.showm.size[0] * self.showm.size[1]:
            raise ValueError('max_window_size must be greater than window_size')

        if not PY_VERSION_8 and not use_raw_array:
            raise ValueError(
                """
               SharedMemory works only in python 3.8 or higher"""
            )

        if use_raw_array:
            self.img_manager = RawArrayImageBufferManager(
                max_window_size=max_window_size, num_buffers=num_buffers
            )
        else:
            self.img_manager = SharedMemImageBufferManager(
                max_window_size=max_window_size, num_buffers=num_buffers
            )

        self._id_timer = None
        self._id_observer = None
        self._interval_timer = None
        self.in_request = False
        self.update = True
        self.use_raw_array = use_raw_array
        self._started = False

    def start(self, ms=0, use_asyncio=False):
        """Start the stream client.

        Parameters
        ----------
        ms : float, optional
            positive number. This update the image buffer using a interval
            of ms milliseconds. If ms is 0 then the stream client
            will update the buffer after every Render event.
        use_asyncio : bool, optional
            If False then the stream client will update the image
            using a threading technique.

        """

        def callback_for_vtk(caller, event, *args, **kwargs):
            callback_stream_client(**{'stream_client': kwargs['stream_client']})

        use_asyncio = platform.system() == 'Windows' or use_asyncio
        if self._started:
            self.stop()
        if ms > 0:
            if self._whithout_iren_start:

                Interval = IntervalTimer if use_asyncio else IntervalTimerThreading
                self._interval_timer = Interval(
                    ms / 1000, callback_stream_client, **{'stream_client': self}
                )
            else:
                self._id_observer = self.showm.iren.AddObserver(
                    'TimerEvent', partial(callback_for_vtk, **{'stream_client': self})
                )
                self._id_timer = self.showm.iren.CreateRepeatingTimer(ms)
        else:
            self._id_observer = self.showm.iren.AddObserver(
                'RenderEvent', partial(callback_for_vtk, **{'stream_client': self})
            )
        self.showm.window.Render()
        self.showm.iren.Render()
        self._started = True
        callback_stream_client(**{'stream_client': self})

    def stop(self):
        """Stop the stream client."""
        if not self._started:
            return False

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
        """Release the shared memory resources if necessary."""
        if self.use_raw_array:
            return

        self.img_manager.info_buffer.close()
        # this it's due the python core issues
        # https://bugs.python.org/issue38119
        # https://bugs.python.org/issue39959
        # https://github.com/luizalabs/shared-memory-dict/issues/13
        try:
            self.img_manager.info_buffer.unlink()
        except FileNotFoundError:
            print(
                f'Shared Memory {self.img_manager.info_buffer_name}\
                    (info_buffer) File not found'
            )
        for buffer, name in zip(
            self.img_manager.image_buffers, self.img_manager.image_buffer_names
        ):
            buffer.close()
            try:
                buffer.unlink()
            except FileNotFoundError:
                print(f'Shared Memory {name}(buffer image) File not found')


def interaction_callback(circular_queue, showm, iren, render_after):
    """This callback is used to invoke vtk interaction events
    reading those events from the provided circular_queue instance

    Parameters
    ----------
    circular_queue : CircularQueue
    showm : ShowmManager
    iren : vtkInteractor
    render_after : bool, optional
        If the render method should be called after an
        dequeue

    """
    ts = time.time() * 1000
    data = circular_queue.dequeue()
    if data is None:
        return

    user_event_id = data[0]
    user_timestamp = data[_CQUEUE.index_info.user_timestamp]

    ts = time.time() * 1000
    newX = int(showm.size[0] * data[_CQUEUE.index_info.x])
    newY = int(showm.size[1] * data[_CQUEUE.index_info.y])
    ctrl_key = int(data[_CQUEUE.index_info.ctrl])
    shift_key = int(data[_CQUEUE.index_info.shift])
    newY = showm.size[1] - newY
    event_ids = _CQUEUE.event_ids
    if user_event_id == event_ids.mouse_weel:
        zoomFactor = 1.0 - data[_CQUEUE.index_info.weel] / 1000.0
        camera = showm.scene.GetActiveCamera()
        fp = camera.GetFocalPoint()
        pos = camera.GetPosition()
        delta = [fp[i] - pos[i] for i in range(3)]

        pos2 = camera.GetPosition()
        camera.SetFocalPoint([pos2[i] + delta[i] for i in range(3)])
        camera.Zoom(zoomFactor)
    elif user_event_id == event_ids.mouse_move:
        iren.SetEventInformation(newX, newY, ctrl_key, shift_key, chr(0), 0, None)

        iren.MouseMoveEvent()

    elif event_ids.mouse_ids:
        iren.SetEventInformation(newX, newY, ctrl_key, shift_key, chr(0), 0, None)
        mouse_actions = {
            event_ids.left_btn_press: iren.LeftButtonPressEvent,
            event_ids.left_btn_release: iren.LeftButtonReleaseEvent,
            event_ids.middle_btn_press: iren.MiddleButtonPressEvent,
            event_ids.middle_btn_release: iren.MiddleButtonReleaseEvent,
            event_ids.right_btn_press: iren.RightButtonPressEvent,
            event_ids.right_btn_release: iren.RightButtonReleaseEvent,
        }
        mouse_actions[user_event_id]()
    logging.info('Interaction: time to perform event ' + f'{ts-user_timestamp:.2f} ms')
    if render_after:
        showm.window.Render()
        showm.iren.Render()


class FuryStreamInteraction:
    """This obj. is responsible to manage the user interaction"""

    def __init__(
        self, showm, max_queue_size=50, use_raw_array=True, whithout_iren_start=False
    ):
        """Initialize the StreamInteraction obj.

        Parameters
        ----------
        showm : ShowmManager
        max_queue_size : int, optional
            maximum number of events to be stored.
        use_raw_array : bool, optional
            If False then a CircularQueue will be created using
            SharedMemory instead of RawArrays. Notice that
            Python >=3.8 it's requirement to use SharedMemory.
        whithout_iren_start : bool, optional
            Set that to True if you can't initiate the vtkInteractor
            instance.

        """
        self.showm = showm
        self.iren = self.showm.iren
        if use_raw_array:
            self.circular_queue = ArrayCircularQueue(
                max_size=max_queue_size, dimension=_CQUEUE.dimension
            )
        else:
            self.circular_queue = SharedMemCircularQueue(
                max_size=max_queue_size, dimension=_CQUEUE.dimension
            )

        self._id_timer = None
        self._id_observer = None
        self._interval_timer = None
        self._whithout_iren_start = whithout_iren_start
        self._started = False

    def start(self, ms=3, use_asyncio=False):
        """Start the stream interaction client.

        Parameters
        ----------
        ms : float, optional
            positive number greater than zero.
        use_asyncio : bool, optional
            If False then the interaction will be performed in a
            separate thread.

        """
        use_asyncio = platform.system() == 'Windows' or use_asyncio
        if ms <= 0:
            raise ValueError('ms must be greater than zero')

        if self._started:
            self.stop()
        if self._whithout_iren_start:
            Interval = IntervalTimer if use_asyncio else IntervalTimerThreading
            self._interval_timer = Interval(
                ms / 1000,
                interaction_callback,
                *[self.circular_queue, self.showm, self.iren, True],
            )
        else:

            def callback(caller, event, *args, **kwargs):
                interaction_callback(self.circular_queue, self.showm, self.iren, True)

            self._id_observer = self.showm.iren.AddObserver('TimerEvent', callback)
            self._id_timer = self.showm.iren.CreateRepeatingTimer(ms)

        self._started = True

    def stop(self):
        """Stop the stream interaction client."""
        if not self._started:
            return False

        if self._id_timer is not None:
            # self.showm.destroy_timer(self._id_timer)
            self.showm.iren.DestroyTimer(self._id_timer)
            self._id_timer = None

        if self._id_observer is not None:
            self.showm.iren.RemoveObserver(self._id_observer)
            self._id_observer = None

        if self._interval_timer is not None:
            self._interval_timer.stop()
            self._interval_timer = None

        self._started = False

    def cleanup(self):
        """Release the shared memory resources if necessary."""
        self.circular_queue.cleanup()
