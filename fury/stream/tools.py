import numpy as np
import multiprocessing


class MultiDimensionalBuffer:
    def __init__(self, max_size=10, dimension=4):

        self._buffers = [
            multiprocessing.RawArray(
                'd', np.zeros(max_size, dtype='float64'))
            for i in range(dimension)
        ]
        self.dimension = dimension
        self.max_size = max_size

    @property
    def buffers(self):
        return self._buffers

    @buffers.setter
    def buffers(self, data):
        self._buffers[:] = data

    def __getitem__(self, idx):
        return [
            np.frombuffer(self._buffers[i], 'float64')[idx]
            for i in range(self.dimension)
        ]

    def __setitem__(self, idx, data):
        for i in range(self.dimension):
            self._buffers[i][idx] = data[i]


class CircularQueue:

    def __init__(self, max_size=50, dimension=4):
        head_tail_buffer = multiprocessing.RawArray(
            'i', np.array([-1, -1], dtype='int32'),
        )
        self.dimension = dimension
        self.head_tail_buffer = head_tail_buffer
        self.max_size = max_size
        self.buffers = MultiDimensionalBuffer(max_size, dimension)

    @property
    def head(self):
        return np.frombuffer(self.head_tail_buffer, 'int32')[0]

    @head.setter
    def head(self, value):
        self.head_tail_buffer[0] = value

    @property
    def tail(self):
        return np.frombuffer(self.head_tail_buffer, 'int32')[1]

    @tail.setter
    def tail(self, value):
        self.head_tail_buffer[1] = value

    @property
    def queue(self):
        return [
            np.frombuffer(self.buffers[i], 'float64')
            for i in range(self.dimension)
        ]

    def enqueue(self, data):
        if ((self.tail + 1) % self.max_size == self.head):
            return False

        elif (self.head == -1):
            self.head = 0
            self.tail = 0
            self.buffers[self.tail] = data
            return True
        else:
            self.tail = (self.tail + 1) % self.max_size
            self.buffers[self.tail] = data
            return True

    def dequeue(self):
        if self.head == -1:
            return None

        if self.head != self.tail:
            interactions = self.buffers[self.head]
            self.head = (self.head + 1) % self.max_size
        else:
            interactions = self.buffers[self.head]
            self.head = -1
            self.tail = -1

        return interactions
