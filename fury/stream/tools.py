import numpy as np
import multiprocessing
import time
import logging

logging.basicConfig(level=logging.INFO)


class MultiDimensionalBufferList:
    def __init__(self, max_size=None, dimension=None, buffers_list=None):

        if buffers_list is None:
            buffers_list = [
                multiprocessing.RawArray(
                    'd', np.zeros(max_size, dtype='float64'))
                for i in range(dimension)
            ]
        else:
            dimension = len(buffers_list)
            max_size = np.frombuffer(buffers_list[0], 'float64').shape[0]

        self._buffers = buffers_list
        self._memory_views = [
            np.ctypeslib.as_array(buffer)
            for buffer in buffers_list]
        self.dimension = dimension
        self.max_size = max_size

    @property
    def buffers(self):
        return self._buffers

    @buffers.setter
    def buffers(self, data):
        self._memory_views[:] = data

    def __getitem__(self, idx):
        return [
            np.frombuffer(self._buffers[i], 'float64')[idx]
            for i in range(self.dimension)
        ]

    def __setitem__(self, idx, data):
        for i in range(self.dimension):
            self._memory_views[i][idx] = data[i]


class MultiDimensionalBuffer:
    def __init__(self, max_size=None, dimension=7, buffers_list=None):

        if dimension is not None and max_size is None:
            max_size = np.frombuffer(
                buffers_list, 'float64').shape[0]//dimension

        if buffers_list is None:
            buffers_list = multiprocessing.RawArray(
                    'd', np.zeros(dimension*(max_size+1), dtype='float64'))

        self._buffers = buffers_list
        self._memory_views = np.ctypeslib.as_array(self._buffers)
        # self._memory_views = self._buffers
        self.dimension = dimension
        self.max_size = max_size
        self.full_size = int(max_size*dimension)

    @property
    def buffers(self):
        return self._buffers

    @buffers.setter
    def buffers(self, data):
        if isinstance(data, (np.ndarray, np.generic)):
            if data.dtype == 'float64':
                self._memory_views[:] = data

    def get_start_end(self, idx):
        dim = self.dimension
        start = idx*dim
        end = dim*(idx+1)
        return start, end

    def __getitem__(self, idx):
        start, end = self.get_start_end(idx)
        logging.info(f'dequeue start {int(time.time()*1000)}')
        ts = time.time()*1000
        itens = np.frombuffer(self._buffers, 'float64')[start:end]
        te = time.time()*1000
        logging.info(f'dequeue frombuffer cost {te-ts:.2f}')
        return itens

    def __setitem__(self, idx, data):
        start, end = self.get_start_end(idx)
        if isinstance(data, (np.ndarray, np.generic)):
            if data.dtype == 'float64':
                self._memory_views[start:end] = data


class CircularQueue:
    def __init__(
        self, max_size=10, dimension=8,
            head_tail_buffer=None,  buffers_list=None):

        buffers = MultiDimensionalBuffer(max_size, dimension, buffers_list)
        if head_tail_buffer is None or buffers_list is None:
            head_tail_buffer = multiprocessing.Array(
                'l', np.array([-1, -1], dtype='int64'),
            )

        self.dimension = buffers.dimension
        self.head_tail_buffer = head_tail_buffer

        self.max_size = buffers.max_size
        self.buffers = buffers

    @property
    def head(self):
        return np.frombuffer(self.head_tail_buffer.get_obj(), 'int64')[0]

    @head.setter
    def head(self, value):
        self.head_tail_buffer[0] = value

    @property
    def tail(self):
        return np.frombuffer(self.head_tail_buffer.get_obj(), 'int64')[1]

    @tail.setter
    def tail(self, value):
        self.head_tail_buffer[1] = value

    def set_head_tail(self, head, tail):
        self.head_tail_buffer[:] = np.array([head, tail]).astype('int64')

    @property
    def queue(self):
        return np.frombuffer(
            self.buffers._buffers.get_obj(), 'float64').reshape((self.max_size, self.dimension))

    def enqueue(self, data):
        # with self.buffers._memory_views.get_lock():
        with self.head_tail_buffer.get_lock():
            if ((self.tail + 1) % self.max_size == self.head):
                # self.head = 0
                # self.tail = 0
                # self.buffers[self.tail] = data
                # return True
                return False

            elif (self.head == -1):
                # self.head = 0
                # self.tail = 0
                self.set_head_tail(0, 0)
                self.buffers[self.tail] = data
                return True
            else:
                self.tail = (self.tail + 1) % self.max_size
                # head = self.head
                # self.set_head_tail(head, tail)
            self.buffers[self.tail] = data
            return True

    def dequeue(self):
        # with self.buffers._memory_views.get_lock():
        with self.head_tail_buffer.get_lock():
            if self.head == -1:
                return None

            if self.head != self.tail:
                interactions = self.buffers[self.head]
                # tail = self.tail
                self.head = (self.head + 1) % self.max_size
                # self.set_head_tail(head, tail)
            else:
                interactions = self.buffers[self.head]
                self.set_head_tail(-1, -1)

            return interactions
