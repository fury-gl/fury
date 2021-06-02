from aiohttp import web
from av import VideoFrame
from aiortc import VideoStreamTrack

import numpy as np

from fury.stream.servers.webrtc.async_app import get_app
from fury.stream.tools import CircularQueue


def webrtc_server(
        stream_client=None,
        image_buffers=None, info_buffer=None,
        circular_queue=None,
        queue_head_tail_buffer=None,
        queue_buffers_list=None,
        port=8000, host='localhost', flip_img=True,
        www_folder=None, use_vidgear=False):

    if stream_client is not None:
        image_buffers = stream_client.image_buffers
        info_buffer = stream_client.info_buffer

    class RTCServer(VideoStreamTrack):
        def __init__(self,):
            super().__init__()

            # starts with a random image

            image_info = np.frombuffer(
                info_buffer, 'uint32')
            # self.image = np.random.randint(
            #      0, 255, (image_info[1], image_info[0], 3),
            #      dtype='uint8')
            self.image = np.zeros(
                (image_info[1], image_info[0], 3), dtype='uint8')

        async def recv(self):
            pts, time_base = await self.next_timestamp()

            image_info = np.frombuffer(
                info_buffer, 'uint32')

            self.image = np.frombuffer(
                image_buffers[info_buffer[3]], 'uint8').reshape(
                (image_info[1], image_info[0], 3))

            if flip_img:
                self.image = np.flipud(self.image)
            av_frame = VideoFrame.from_ndarray(self.image)
            av_frame.pts = pts
            av_frame.time_base = time_base
            return av_frame

        def terminate(self):
            try:
                if not (self.stream is None):
                    self.stream.release()
                    self.stream = None
            except AttributeError:
                pass
    if circular_queue is None and queue_buffers_list is not None:
        circular_queue = CircularQueue(
            head_tail_buffer=queue_head_tail_buffer,
            buffers_list=queue_buffers_list)

    app_fury = get_app(RTCServer(), circular_queue=circular_queue)
    web.run_app(
        app_fury, host=host, port=port, ssl_context=None)
