import subprocess
import numpy as np
import sys
if sys.version_info.minor >= 8:
    PY_VERSION_8 = True
else:
    PY_VERSION_8 = False

from IPython.display import IFrame
from IPython.core.display import display

import time

from fury.stream.client import FuryStreamClient, FuryStreamInteraction

class Widget:
    def __init__(
            self, showm, max_window_size=None, ms_stream=0,
            domain='localhost', port=None, ms_interaction=1, queue_size=20,
            encoding='mjpeg'):
        if not PY_VERSION_8:
            raise ImportError('Python 3.8 or greater is required to use the\
                widget class')
        self.showm = showm
        self.window_size = self.showm.size
        if max_window_size is None:
            max_window_size = (
                int(self.window_size[0]*(1+0.25)),
                int(self.window_size[1]*(1+0.25))
            )
        self.max_window_size = max_window_size
        self.ms_stream = ms_stream
        self.ms_interaction = ms_interaction
        self.domain = domain
        if port is None:
            port = np.random.randint(7000, 8888)
        self.port = port
        self.queue_size = queue_size
        self._server_started = False
        self.pserver = None
        self.encoding = encoding
    @property
    def command_string(self):
        s = 'from fury.stream.server import web_server;'
        s += f"web_server(image_buffer_names={self.stream.image_buffer_names}"
        s += f",info_buffer_name='{self.stream.info_buffer_name}',"
        s += "queue_head_tail_buffer_name='"
        s += f"{self.stream_interaction.circular_queue.head_tail_buffer_name}'"
        s += f",queue_buffer_name='"
        s += f"{self.stream_interaction.circular_queue.buffer.buffer_name}'"
        if self.encoding == 'mjpeg':
            s += ",provides_mjpeg=True"
            s += ",provides_webrtc=False"
        s += f",port={self.port},host='{self.domain}')"
        return s

    def start_server(self):
        if self._server_started:
            return

        self.showm.window.SetOffScreenRendering(1)
        #self.showm.iren.EnableRenderOff()
        # self.showm.initialize()
        self.stream = FuryStreamClient(
            self.showm, self.window_size,
            max_window_size=self.max_window_size,
            use_raw_array=False,
            whithout_iren_start=True
        )
        self.stream_interaction = FuryStreamInteraction(
            self.showm, max_queue_size=self.queue_size,
            fury_client=self.stream,
            whithout_iren_start=True,
            use_raw_array=False)

        self.stream_interaction.start(ms=self.ms_interaction)
        self.stream.init(16)
        self._server_started = True

       
    def run_command(self):
        if self._server_started:
            args = [
                sys.executable, '-c',  
                self.command_string
            ]
            self.pserver = subprocess.Popen(
                args,
                #f'python -c "{self.command_string}"',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)

    @property
    def url(self):
        return f'http://{self.domain}:{self.port}?iframe=1&encoding={self.encoding}'

    def return_iframe(self, height=200):
        display(IFrame(
            self.url,
            '100%', f'{int(height)}px')
        )
    
    def start(self):
        self.start_server()
        self.run_command()
        print(f'url: {self.url}')

    def display(self, height=150):
        self.start_server()
        self.run_command()
        time.sleep(1)
        self.return_iframe(height) 

    def stop(self):
        self.stream.stop()
        self.stream_interaction.stop()
        if self.pserver is not None:
            #x = requests.get(f'http://{self.domain}:{self.port}/shutdown')
            self.pserver.kill()
            self.pserver.wait()
            self._server_started = False
            self.pserver = None
            self.cleanup()
            # del self.stream
            # del self.stream_interaction

    def cleanup(self):
        self.stream.cleanup()
        self.stream_interaction.cleanup()

    def __del__(self):
        self.stop()

