import numpy as np
import multiprocessing
import time
import logging
import sys
if sys.version_info.minor >= 8:
    from multiprocessing import shared_memory
    PY_VERSION_8 = True
else:
    shared_memory = None
    PY_VERSION_8 = False


class MultiDimensionalBuffer:
    def __init__(
            self, max_size=None, dimension=8, buffers_list=None,
            buffer_name=None,
            use_raw_array=True):
        
        use_raw_array = use_raw_array and buffer_name is None
        if not PY_VERSION_8 and not use_raw_array:
            raise ValueError("""
                In order to use the SharedMemory approach
                you should have to use python 3.8 or higher""")

        if buffers_list is None and buffer_name is None:
            buffer_arr = np.zeros(dimension*(max_size+1), dtype='float64')
            if use_raw_array:
                buffers_list = multiprocessing.RawArray(
                        'd', buffer_arr)
                self._buffers = buffers_list
                self._buffer_repr = np.ctypeslib.as_array(self._buffers)
            else:
                buffers_list = shared_memory.SharedMemory(
                    create=True, size=buffer_arr.nbytes)
                self._buffer_repr = np.ndarray(
                        buffer_arr.shape[0],
                        dtype='float64', buffer=buffers_list.buf)
                self._buffers = buffers_list
                buffer_name = buffers_list.name
        else:
            if buffer_name is None:
                max_size = int(len(buffers_list)//dimension)
                max_size -= 1
                
                self._buffers = buffers_list
                self._buffer_repr = np.ctypeslib.as_array(self._buffers)
            else:
                buffer = shared_memory.SharedMemory(buffer_name)
                # 8 represents 8 bytes of float64
                max_size = int(len(buffer.buf)//dimension/8)
                max_size -= 1

                self._buffers = buffer
                self._buffer_repr = np.ndarray(
                        len(buffer.buf)//8,
                        dtype='float64', buffer=buffer.buf)
                self._buffers = buffer

        self.buffer_name = buffer_name
        self.dimension = dimension
        self.max_size = max_size
        self.use_raw_array = use_raw_array

    @property
    def buffers(self):
        return self._buffers

    @buffers.setter
    def buffers(self, data):
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
            itens = np.frombuffer(self._buffers, 'float64')[start:end]
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


class CircularQueue:
    def __init__(
        self, max_size=10, dimension=8,
            head_tail_buffer=None,  buffers_list=None,
            head_tail_buffer_name=None, buffer_name=None,
            use_raw_array=True):

        use_raw_array = use_raw_array and buffer_name is None
        if not PY_VERSION_8 and not use_raw_array:
            raise ValueError("""
                In order to use the SharedMemory approach
                you should have to use python 3.8 or higher""")
        buffers = MultiDimensionalBuffer(
            max_size, dimension, buffers_list,
            buffer_name, use_raw_array
        )

        head_tail_arr = np.array([-1, -1], dtype='int64')
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
        else:
            if not use_raw_array:
                head_tail_buffer = shared_memory.SharedMemory(
                    head_tail_buffer_name)

        self.use_raw_array = use_raw_array
        self.dimension = buffers.dimension
        self.head_tail_buffer = head_tail_buffer
        self.head_tail_buffer_name = head_tail_buffer_name
        self.max_size = buffers.max_size
        self.buffers = buffers

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

    def set_head_tail(self, head, tail):
        self.head_tail_buffer_repr[:] = np.array([head, tail]).astype('int64')

    @property
    def queue(self):
        if self.use_raw_array:
            return np.frombuffer(
                self.buffers._buffers.get_obj(), 'float64').reshape(
                    (self.max_size, self.dimension))
        else:
            return self.buffers._buffer_repr

    def enqueue(self, data):
        # with self.head_tail_buffer.get_lock():
        if ((self.tail + 1) % self.max_size == self.head):
            return False

        elif (self.head == -1):
            self.set_head_tail(0, 0)
        else:
            self.tail = (self.tail + 1) % self.max_size

        self.buffers[self.tail] = data
        return True

    def dequeue(self):
        # with self.buffers._buffer_repr.get_lock():
        # with self.head_tail_buffer.get_lock():
        if self.head == -1:
            return None

        if self.head != self.tail:
            interactions = self.buffers[self.head]
            self.head = (self.head + 1) % self.max_size
        else:
            interactions = self.buffers[self.head]
            self.set_head_tail(-1, -1)

        return interactions
