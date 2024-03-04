# import os
# os.environ['PYTHONASYNCIODEBUG'] = '1'
# import logging
import numpy as np
from aiohttp import web

from fury.stream.constants import _CQUEUE, PY_VERSION_8
from fury.stream.server.async_app import get_app
from fury.stream.tools import (
    ArrayCircularQueue,
    RawArrayImageBufferManager,
    SharedMemCircularQueue,
    SharedMemImageBufferManager,
)

if PY_VERSION_8:
    from fury.stream.tools import remove_shm_from_resource_tracker


try:
    from aiortc import VideoStreamTrack
    from av import VideoFrame

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    VideoStreamTrack = object

CYTHON_AVAILABLE = False
if WEBRTC_AVAILABLE:
    try:
        import pyximport

        pyximport.install()
        from fury.stream.server.FuryVideoFrame import FuryVideoFrame

        CYTHON_AVAILABLE = True
    except ImportError:
        pass


class RTCServer(VideoStreamTrack):
    """This Obj it's responsible to create the VideoStream for
    the WebRTCServer
    """

    def __init__(
        self,
        image_buffer_manager,
    ):
        """Initialize the RTCServer

        Parameters
        ----------
        image_buffer_manager : ImageBufferManager

        """
        super().__init__()

        self.frame = None
        self.buffer_manager = image_buffer_manager

    async def recv(self):
        """Return a VideoFrame to be used in the WebRTC Server

        The frame will be created using the image stored in the
        shared memory

        Returns
        -------
        frame : VideoFrame

        """
        pts, time_base = await self.next_timestamp()

        width, height, image = self.buffer_manager.get_current_frame()

        if (
            self.frame is None
            or self.frame.planes[0].width != width
            or self.frame.planes[0].height != height
        ):
            if CYTHON_AVAILABLE:
                self.frame = FuryVideoFrame(width, height, 'rgb24')
        self.image = image

        if not CYTHON_AVAILABLE:
            # if the buffer it's already flipped
            # self.frame.planes[0].update(self.image)
            self.image = np.frombuffer(self.image, 'uint8')[
                0 : width * height * 3
            ].reshape((height, width, 3))
            self.image = np.flipud(self.image)
            self.frame = VideoFrame.from_ndarray(self.image)
        else:
            self.frame.update_from_buffer(self.image)

        self.frame.pts = pts
        self.frame.time_base = time_base

        return self.frame

    def release(self):
        """Release the RTCServer"""
        try:
            if self.stream is None:
                return
            self.stream.release()
            self.stream = None
        except AttributeError:
            pass


def web_server_raw_array(
    image_buffers=None,
    info_buffer=None,
    queue_head_tail_buffer=None,
    queue_buffer=None,
    port=8000,
    host='localhost',
    provides_mjpeg=True,
    provides_webrtc=True,
    ms_jpeg=16,
    run_app=True,
):
    """This will create a streaming webserver running on the
    given port and host using RawArrays.

    Parameters
    ----------
    image_buffers : list of buffers
        A list of buffers with each one containing a frame.
    info_buffer : buffer
        A buffer with the information about the current
        frame to be streamed and the respective sizes
    queue_head_tail_buffer : buffer
        If buffer is passed than this Obj will read a
        a already created RawArray.
    queue_buffer : buffer
        If queue_buffer is passed than this Obj will read a
        a already created RawArray containing the user interactions
        events stored in the queue_buffer.
    port : int, optional
        Port to be used by the aiohttp server
    host : str, optional, default localhost
        host to be used by the aiohttp server
    provides_mjpeg : bool, default True
        If a MJPEG streaming should be available.
        If True you can consume that through
        host:port/video/mjpeg
        or if you want to interact you can consume that
        through your browser
        http://host:port?encoding=mjpeg
    provides_webrtc : bool, default True
        If a WebRTC streaming should be available.
        http://host:port
    ms_jpeg : float, optional
        This it's used  only if the MJPEG will be used. The
        ms_jpeg represents the amount of milliseconds between to
        consecutive calls of the jpeg encoding.
    run_app : bool, default True
        This will run the aiohttp application. The False condition
        is used just to be able to test the server.

    """
    image_buffer_manager = RawArrayImageBufferManager(
        image_buffers=image_buffers, info_buffer=info_buffer
    )

    rtc_server = None
    create_webrtc = provides_webrtc and WEBRTC_AVAILABLE
    if create_webrtc:
        rtc_server = RTCServer(image_buffer_manager)
    else:
        provides_mjpeg = True

    circular_queue = None
    if queue_buffer is not None:
        circular_queue = ArrayCircularQueue(
            dimension=_CQUEUE.dimension,
            head_tail_buffer=queue_head_tail_buffer,
            buffer=queue_buffer,
        )

    app_fury = get_app(
        rtc_server,
        circular_queue=circular_queue,
        image_buffer_manager=image_buffer_manager,
        provides_mjpeg=provides_mjpeg,
    )

    if run_app:
        web.run_app(app_fury, host=host, port=port, ssl_context=None)

    if rtc_server is not None:
        rtc_server.release()

    if circular_queue is not None:
        circular_queue.cleanup()

    image_buffer_manager.cleanup()


def web_server(
    image_buffer_names=None,
    info_buffer_name=None,
    queue_head_tail_buffer_name=None,
    queue_buffer_name=None,
    port=8000,
    host='localhost',
    provides_mjpeg=True,
    provides_webrtc=True,
    avoid_unlink_shared_mem=True,
    ms_jpeg=16,
    run_app=True,
):
    """This will create a streaming webserver running on the given port
    and host using SharedMemory.

    Parameters
    ----------
    image_buffers_name : list of str
        A list of buffers with each one containing a frame.
    info_buffer_name : str
        A buffer with the information about the current
        frame to be streamed and the respective sizes
    queue_head_tail_buffer_name : str, optional
        If buffer is passed than this Obj will read a
        a already created RawArray.
    buffer_name : str, optional
        If queue_buffer is passed than this Obj will read a
        a already created RawArray containing the user interactions
        events stored in the queue_buffer.
    port : int, optional
        Port to be used by the aiohttp server
    host : str, optional, default localhost
        host to be used by the aiohttp server
    provides_mjpeg : bool, default True
        If a MJPEG streaming should be available.
        If True you can consume that through
        host:port/video/mjpeg
        or if you want to interact you can consume that
        through your browser
        http://host:port?encoding=mjpeg
    provides_webrtc : bool, default True
        If a WebRTC streaming should be available.
        http://host:port
    avoid_unlink_shared_mem : bool, default False
        If True, then this will apply a monkey-patch solution to
        a python>=3.8 core bug
    ms_jpeg : float, optional
        This it's used  only if the MJPEG will be used. The
        ms_jpeg represents the amount of milliseconds between to
        consecutive calls of the jpeg encoding.
    run_app : bool, default True
        This will run the aiohttp application. The False condition
        is used just to be able to test the server.

    """
    if avoid_unlink_shared_mem and PY_VERSION_8:
        remove_shm_from_resource_tracker()

    image_buffer_manager = SharedMemImageBufferManager(
        image_buffer_names=image_buffer_names, info_buffer_name=info_buffer_name
    )

    rtc_server = None
    create_webrtc = provides_webrtc and WEBRTC_AVAILABLE
    if create_webrtc:
        rtc_server = RTCServer(image_buffer_manager)
    else:
        provides_mjpeg = True

    circular_queue = None
    if queue_buffer_name is not None:
        circular_queue = SharedMemCircularQueue(
            dimension=_CQUEUE.dimension,
            buffer_name=queue_buffer_name,
            head_tail_buffer_name=queue_head_tail_buffer_name,
        )

    app_fury = get_app(
        rtc_server,
        circular_queue=circular_queue,
        image_buffer_manager=image_buffer_manager,
        provides_mjpeg=provides_mjpeg,
    )

    if run_app:
        web.run_app(app_fury, host=host, port=port, ssl_context=None)

    if rtc_server is not None:
        rtc_server.release()

    if circular_queue is not None:
        circular_queue.cleanup()

    image_buffer_manager.cleanup()
