import numpy as np
import multiprocessing
import time
import logging
from threading import Timer

import sys
if sys.version_info.minor >= 8:
    from multiprocessing import shared_memory
    PY_VERSION_8 = True
else:
    shared_memory = None
    PY_VERSION_8 = False


class MultiDimensionalBuffer:
    def __init__(
            self, max_size=None, dimension=8, buffer=None,
            buffer_name=None,
            use_raw_array=True):

        use_raw_array = use_raw_array and buffer_name is None
        if not PY_VERSION_8 and not use_raw_array:
            raise ValueError("""
                In order to use the SharedMemory approach
                you should have to use python 3.8 or higher""")

        if buffer is None and buffer_name is None:
            if use_raw_array:

                buffer_arr = np.zeros(dimension*(max_size+1), dtype='float64')
                buffer = multiprocessing.RawArray(
                        'd', buffer_arr)
                self._buffer = buffer
                self._buffer_repr = np.ctypeslib.as_array(self._buffer)
            else:
                buffer_arr = np.zeros(dimension*(max_size+1), dtype='float64')
                buffer = shared_memory.SharedMemory(
                    create=True, size=buffer_arr.nbytes)
                self._buffer_repr = np.ndarray(
                        buffer_arr.shape[0],
                        dtype='float64', buffer=buffer.buf)
                self._buffer = buffer
                buffer_name = buffer.name
                self._unlink_shared_mem = True
                print('created', max_size, dimension,  len(buffer.buf))
        else:
            if buffer_name is None:
                max_size = int(len(buffer)//dimension)
                max_size -= 1
                self._buffer = buffer
                self._buffer_repr = np.ctypeslib.as_array(self._buffer)
            else:
                buffer = shared_memory.SharedMemory(buffer_name)
                # 8 represents 8 bytes of float64
                max_size = int(len(buffer.buf)//dimension/8)
                max_size -= 1
                self._buffer = buffer
                self._buffer_repr = np.ndarray(
                        len(buffer.buf)//8,
                        dtype='float64', buffer=buffer.buf)
                self._buffer = buffer
                self._unlink_shared_mem = False 
                print('read', max_size, dimension,  len(buffer.buf))

        self.buffer_name = buffer_name
        self.dimension = dimension
        self.max_size = max_size
        self.use_raw_array = use_raw_array

    @property
    def buffer(self):
        return self._buffer

    @buffer.setter
    def buffer(self, data):
        if isinstance(data, (np.ndarray, np.generic)):
            if data.dtype == 'float64':
                self._buffer_repr[:] = data

    def get_start_end(self, idx):
        dim = self.dimension
        start = idx*dim
        end = dim*(idx+1)
        return start, end

    def __getitem__(self, idx):
        start, end = self.get_start_end(idx)
        logging.info(f'dequeue start {int(time.time()*1000)}')
        ts = time.time()*1000
        if self.use_raw_array:
            itens = np.frombuffer(self._buffer, 'float64')[start:end]
        else:
            itens = self._buffer_repr[start:end]
        te = time.time()*1000
        logging.info(f'dequeue frombuffer cost {te-ts:.2f}')
        return itens

    def __setitem__(self, idx, data):
        start, end = self.get_start_end(idx)
        if isinstance(data, (np.ndarray, np.generic)):
            if data.dtype == 'float64':
                # if end - start == self.dimension and start >= 0 and end >= 0:
                self._buffer_repr[start:end] = data

    def cleanup(self):
        if not self.use_raw_array:
            self._buffer.close()
            if self._unlink_shared_mem:
                # this it's due the python core issues
                # https://bugs.python.org/issue38119
                # https://bugs.python.org/issue39959
                # https://github.com/luizalabs/shared-memory-dict/issues/13
                try:
                    self._buffer.unlink()
                except FileNotFoundError:
                    print(f'Shared Memory {self.buffer_name}(queue_event_buffer) File not found')



class CircularQueue:
    def __init__(
        self, max_size=10, dimension=8,
            head_tail_buffer=None,  buffer=None,
            head_tail_buffer_name=None, buffer_name=None,
            use_raw_array=True):

        use_raw_array = use_raw_array and buffer_name is None
        if not PY_VERSION_8 and not use_raw_array:
            raise ValueError("""
                In order to use the SharedMemory approach
                you should have to use python 3.8 or higher""")
        buffer = MultiDimensionalBuffer(
            max_size, dimension, buffer,
            buffer_name, use_raw_array
        )

        head_tail_arr = np.array([-1, -1, 0], dtype='int64')
        if head_tail_buffer is None and head_tail_buffer_name is None:
            if use_raw_array:
                head_tail_buffer = multiprocessing.Array(
                    'l', head_tail_arr,
                )
                head_tail_buffer_name = None
            else:
                head_tail_buffer = shared_memory.SharedMemory(
                    create=True, size=head_tail_arr.nbytes)
        
                head_tail_buffer_name = head_tail_buffer.name
                self._unlink_shared_mem = True 
        else:
            if not use_raw_array:
                head_tail_buffer = shared_memory.SharedMemory(
                    head_tail_buffer_name)
                self._unlink_shared_mem = False 

        self.use_raw_array = use_raw_array
        self.dimension = buffer.dimension
        self.head_tail_buffer = head_tail_buffer
        self.head_tail_buffer_name = head_tail_buffer_name
        self.max_size = buffer.max_size
        self.buffer = buffer

        if use_raw_array:
            self.head_tail_buffer_repr = self.head_tail_buffer
        else:
            self.head_tail_buffer_repr = np.ndarray(
                head_tail_arr.shape[0],
                dtype='int64', buffer=self.head_tail_buffer.buf)

    @property
    def head(self):
        if self.use_raw_array:
            return np.frombuffer(self.head_tail_buffer.get_obj(), 'int64')[0]
        else:
            return self.head_tail_buffer_repr[0]

    @head.setter
    def head(self, value):
        self.head_tail_buffer_repr[0] = value

    @property
    def tail(self):
        if self.use_raw_array:
            return np.frombuffer(self.head_tail_buffer.get_obj(), 'int64')[1]
        else:
            return self.head_tail_buffer_repr[1]

    @tail.setter
    def tail(self, value):
        self.head_tail_buffer_repr[1] = value

    def set_head_tail(self, head, tail, lock=1):
        self.head_tail_buffer_repr[:] = np.array([head, tail, lock]).astype('int64')

    def is_unlocked(self):
        return self.head_tail_buffer_repr[2] == 0

    def lock(self):
        self.head_tail_buffer_repr[2] = 1

    def unlock(self):
        self.head_tail_buffer_repr[2] = 0

    @property
    def queue(self):
        if self.use_raw_array:
            return np.frombuffer(
                self.buffer._buffer.get_obj(), 'float64').reshape(
                    (self.max_size, self.dimension))
        else:
            return self.buffer._buffer_repr

    def enqueue(self, data):
        # with self.head_tail_buffer.get_lock():
        ok = False
        if self.is_unlocked():
            self.lock()
            if ((self.tail + 1) % self.max_size == self.head):
                ok = False
            else:
                if (self.head == -1):
                    self.set_head_tail(0, 0, 1)
                else:
                    self.tail = (self.tail + 1) % self.max_size
                self.buffer[self.tail] = data

                ok = True
            self.unlock()
        return ok

    def dequeue(self):
        # with self.buffers._buffer_repr.get_lock():
        # with self.head_tail_buffer.get_lock():
        if self.is_unlocked():
            self.lock()
            if self.head == -1:
                interactions = None
            else:
                if self.head != self.tail:
                    interactions = self.buffer[self.head]
                    self.head = (self.head + 1) % self.max_size
                else:
                    interactions = self.buffer[self.head]
                    self.set_head_tail(-1, -1, 1)

            self.unlock()
            return interactions

    def cleanup(self):
        if not self.use_raw_array:
            self.buffer.cleanup()
            self.head_tail_buffer.close()
            if self._unlink_shared_mem:
                # this it's due the python core issues
                # https://bugs.python.org/issue38119
                # https://bugs.python.org/issue39959
                # https://github.com/luizalabs/shared-memory-dict/issues/13
                try:
                    self.head_tail_buffer.unlink()
                except FileNotFoundError:
                    print(f'Shared Memory {self.head_tail_buffer_name}(head_tail) File not found')


class IntervalTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        """
        Implements a object with the same behavior of setInterval from Js
        I got this from
        https://stackoverflow.com/questions/3393612/run-certain-code-every-n-seconds
        """
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        # self.next_call = time.time()
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)

            # self.next_call += self.interval
            # self._timer = Timer(self.next_call - time.time(), self._run)
            self._timer.daemon = True
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False
        # self._timer.join()
