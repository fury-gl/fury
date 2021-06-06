# import os
# os.environ['PYTHONASYNCIODEBUG'] = '1'
# import logging
# logging.basicConfig(level=logging.ERROR)
import time

from aiohttp import web
from av import VideoFrame
from aiortc import VideoStreamTrack
import pyximport; pyximport.install()
from .FuryVideoFrame import FuryVideoFrame
from multiprocessing import shared_memory

import numpy as np

from fury.stream.servers.webrtc.async_app import get_app
from fury.stream.tools import CircularQueue


def webrtc_server(
        image_buffer_names=None, info_buffer=None,
        circular_queue=None,
        queue_head_tail_buffer=None,
        queue_buffers_list=None,
        port=8000, host='localhost', flip_img=True,
        www_folder=None):

    # if stream_client is not None:
    #     image_buffers = stream_client.image_buffers
    #     info_buffer = stream_client.info_buffer
    
    class RTCServer(VideoStreamTrack):
        def __init__(self,):
            super().__init__()
        
            # starts with a random image

            image_info = np.frombuffer(
                info_buffer, 'uint32')
            self.image = np.random.randint(
                 0, 255, (image_info[1], image_info[0], 3),
                 dtype='uint8')
            self.frame = None
            self.image_buffers = []
            self.image_reprs = []
            self.image_buffer_names = image_buffer_names
            for buffer_name in self.image_buffer_names:
                buffer = shared_memory.SharedMemory(buffer_name)
                self.image_buffers.append(buffer)
                self.image_reprs.append(np.ndarray(len(buffer.buf), dtype=np.uint8, buffer=buffer.buf))
        
        async def recv(self):
            pts, time_base = await self.next_timestamp()
            image_info = np.frombuffer(
                info_buffer, 'uint32')
            buffer_index = image_info[1]
            width = image_info[2+buffer_index*2]
            height = image_info[2+buffer_index*2+1]

            self.image = self.image_reprs[buffer_index]

            # if flip_img:
            #     self.image = np.flipud(self.image)
            if(self.frame is None 
                or self.frame.planes[0].width!=width
                or self.frame.planes[0].height!=height):
                print("creating frame with size: %d x %d"%(width,height))
                self.frame = FuryVideoFrame(width, height, "rgb24")
            # print(self.image[0:3])
            self.frame.update_from_ndarray(self.image)
            self.frame.pts = pts
            self.frame.time_base = time_base

            # time.sleep(0.1)
            return self.frame

        def terminate(self):
            try:
                if not (self.stream is None):
                    self.stream.release()
                    self.stream = None
            except AttributeError:
                pass

        def __del__(self):
            for buffer in self.image_buffers:
                buffer.close()
                buffer.unlink()
            # print("Freeing buffer from RTC Server")
            # super().__del__()
    
    if circular_queue is None and queue_buffers_list is not None:
        circular_queue = CircularQueue(
            head_tail_buffer=queue_head_tail_buffer,
            buffers_list=queue_buffers_list)

    # if use_vidgear:
    #     import uvicorn, asyncio, cv2
    #     from vidgear.gears.asyncio import WebGear_RTC

    #     web = WebGear_RTC(logging=True)
    #     web.config["server"] = RTCServer()

    #     # run this app on Uvicorn server at address http://localhost:8000/
    #     uvicorn.run(web(), host="localhost", port=8000)

    #     # close app safely
    #     web.shutdown()
    # else:
    app_fury = get_app(
        RTCServer(), circular_queue=circular_queue,
    )
    web.run_app(
        app_fury, host=host, port=port, ssl_context=None)


def interaction_server(
        circular_queue=None,
        queue_head_tail_buffer=None,
        queue_buffers_list=None,
        port=8080, host='localhost',
        www_folder=None):

    if circular_queue is None and queue_buffers_list is not None:
        circular_queue = CircularQueue(
            head_tail_buffer=queue_head_tail_buffer,
            buffers_list=queue_buffers_list)

    app_fury = get_app(
        None, circular_queue=circular_queue)
    web.run_app(
        app_fury, host=host, port=port, ssl_context=None)
