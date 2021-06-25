# import os
# os.environ['PYTHONASYNCIODEBUG'] = '1'
import logging
import sys
if sys.version_info.minor >= 8:
    from fury.stream.tools import remove_shm_from_resource_tracker
    PY_VERSION_8 = True
else:
    PY_VERSION_8 = False

from aiohttp import web
from av import VideoFrame
from aiortc import VideoStreamTrack
import numpy as np

from fury.stream.server.async_app import get_app
from fury.stream.tools import SharedMemCircularQueue, ArrayCircularQueue
from fury.stream.tools import (
    SharedMemImageBufferManager, RawArrayImageBufferManager)
from fury.stream.constants import _CQUEUE

try:
    import pyximport
    pyximport.install()
    from fury.stream.server.FuryVideoFrame import FuryVideoFrame
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


class RTCServer(VideoStreamTrack):
    def __init__(
            self, image_buffer_manager,
    ):
        """This Obj it's responsible to create the VideoStream for
        the WebRTCServer

        Parameters:
        -----------
        image_buffer_manager : ImageBufferManager
        """
        super().__init__()

        self.frame = None
        self.buffer_manager = image_buffer_manager

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        width, height, image = self.buffer_manager.get_current_frame()

        if self.frame is None \
            or self.frame.planes[0].width != width \
                or self.frame.planes[0].height != height:
            if CYTHON_AVAILABLE:
                self.frame = FuryVideoFrame(width, height, "rgb24")
            # else:
            #    self.frame = VideoFrame(width, height, "rgb24")
        self.image = image

        if not CYTHON_AVAILABLE:
            # if the buffer it's already flipped
            # self.frame.planes[0].update(self.image)
            self.image = np.frombuffer(
                        self.image,
                        'uint8'
                    )[0:width*height*3].reshape((height, width, 3))
            self.image = np.flipud(self.image)
            self.frame = VideoFrame.from_ndarray(self.image)
        else:
            self.frame.update_from_buffer(self.image)

        self.frame.pts = pts
        self.frame.time_base = time_base

        return self.frame

    def release(self):
        logging.info("Release Server")
        try:
            if not (self.stream is None):
                self.stream.release()
                self.stream = None
        except AttributeError:
            pass


def web_server(
        image_buffers=None,
        image_buffer_names=None,
        info_buffer=None,
        info_buffer_name=None,
        queue_head_tail_buffer=None,
        queue_head_tail_buffer_name=None,
        queue_buffer=None,
        queue_buffer_name=None,
        port=8000, host='localhost',
        provides_mjpeg=True,
        provides_webrtc=True,
        avoid_unlink_shared_mem=False,
        ms_jpeg=16):
    """

        Parameters
        ----------
        image_buffers : list of buffers, optional
            A list of buffers with each one containing a frame.
            If image_buffers was not passed then it's mandatory
            pass the image_buffer_names argument.
        image_buffer_names : list of str, optional
        info_buffer : buffer, optional
            A buffer with the information about the current
            frame to be streamed and the respective sizes
            If info_buffer was not passed then it's mandatory
            pass the info_buffer_name argument.
        info_buffer_name : str, optional
        queue_head_tail_buffer : buffer, optional
            If buffer is passed than this Obj will read a
            a already created RawArray.
            You must pass queue_head_tail_buffer or
            queue_head_tail_buffer_name.
        queue_head_tail_buffer_name : str, optional
        queue_buffer : buffer, optional
            If queue_buffer is passed than this Obj will read a
            a already created RawArray containing the user interactions
            events stored in the queue_buffer.
            You must pass queue_buffer or a queue_buffer_name.
        buffer_name : str, optional
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
            ms_jpeg represents the amount of miliseconds between to
            consecutive calls of the jpeg enconding.

        """

    use_raw_array = info_buffer_name is None

    if avoid_unlink_shared_mem and PY_VERSION_8 and not use_raw_array:
        remove_shm_from_resource_tracker()

    if not use_raw_array:
        image_buffer_manager = SharedMemImageBufferManager(
            image_buffer_names=image_buffer_names,
            info_buffer_name=info_buffer_name
        )
    else:
        image_buffer_manager = RawArrayImageBufferManager(
            image_buffers=image_buffers, info_buffer=info_buffer
        )

    if provides_webrtc:
        rtc_server = RTCServer(
            image_buffer_manager)
    else:
        rtc_server = None

    if queue_buffer is not None:
        circular_queue = ArrayCircularQueue(
            dimension=_CQUEUE.dimension,
            head_tail_buffer=queue_head_tail_buffer,
            buffer=queue_buffer
        )
    elif queue_buffer_name is not None:
        circular_queue = SharedMemCircularQueue(
            dimension=_CQUEUE.dimension,
            buffer_name=queue_buffer_name,
            head_tail_buffer_name=queue_head_tail_buffer_name)
    else:
        circular_queue = None

    app_fury = get_app(
       rtc_server, circular_queue=circular_queue,
       image_buffer_manager=image_buffer_manager,
       provides_mjpeg=provides_mjpeg
    )

    web.run_app(
        app_fury, host=host, port=port, ssl_context=None)

    if circular_queue is not None:
        circular_queue.cleanup()

    if rtc_server is not None:
        rtc_server.release()

    image_buffer_manager.cleanup()
