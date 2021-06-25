import numpy as np
import multiprocessing
import time
import logging
from threading import Timer
from abc import ABC, abstractmethod

import sys
if sys.version_info.minor >= 8:
    from multiprocessing import shared_memory
    from multiprocessing import resource_tracker

    PY_VERSION_8 = True
else:
    shared_memory = None
    PY_VERSION_8 = False


def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't 
    be tracked
    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(
            self, name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(
            self, name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


class GenericMultiDimensionalBuffer(ABC):
    def __init__(
            self, max_size=None, dimension=8):
        """This implements a abstract (generic) multidimensional buffer.

        Parameters
        ----------
        max_size : int, optional
            If buffer_name or buffer was not passed then max_size
            it's mandatory
        dimension : int, default 8
        """
        self.max_size = max_size
        self.dimension = dimension
        self.buffer_name = None
        self._buffer = None
        self._buffer_repr = None
        self._created = False

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

    @abstractmethod
    def load_mem_resource(self):
        pass

    @abstractmethod
    def create_mem_resource(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass


class RawArrayMultiDimensionalBuffer(GenericMultiDimensionalBuffer):
    def __init__(self, max_size, dimension=4, buffer=None):
        """This implements a  multidimensional buffer with RawArray.
        Stream system uses that to implemenet the CircularQueue
        with shared memory resources.

        Parameters
        ----------
        max_size : int, optional
            If buffer_name or buffer was not passed then max_size
            it's mandatory
        dimension : int, default 8
        buffer : buffer, optional
            If buffer is not passed to __init__
            then the multidimensional buffer obj will create a new
            RawArray object to store the data
            If buffer is passed than this Obj will read a
            a already created RawArray
        """
        super().__init__(max_size, dimension)
        if buffer is None:
            self.create_mem_resource()
        else:
            self._buffer = buffer
            self.load_mem_resource()

    def create_mem_resource(self):
        buffer_arr = np.zeros(
            self.dimension*(self.max_size+1), dtype='float64')
        buffer = multiprocessing.RawArray(
                'd', buffer_arr)
        self._buffer = buffer
        self._buffer_repr = np.ctypeslib.as_array(self._buffer)

    def load_mem_resource(self):
        self.max_size = int(len(self._buffer)//self.dimension)
        self.max_size -= 1
        self._buffer_repr = np.ctypeslib.as_array(self._buffer)

    def cleanup(self):
        pass


class SharedMemMultiDimensionalBuffer(GenericMultiDimensionalBuffer):
    def __init__(self, max_size, dimension=4, buffer_name=None):
        """This implements a generic multidimensional buffer
        with SharedMemory. Stream system uses that to implemenet the
        CircularQueue with shared memory resources.

        Parameters
        ----------
        max_size : int, optional
            If buffer_name or buffer was not passed then max_size
            it's mandatory
        dimension : int, default 8
        buffer_name : str, optional
            if buffer_name is passed than this Obj will read a
            a already created SharedMemory

        """
        super().__init__(max_size, dimension)
        if buffer_name is None:
            self.create_mem_resource()
            self._created = True
        else:
            self.buffer_name = buffer_name
            self.load_mem_resource()
            self._created = False

    def create_mem_resource(self):

        buffer_arr = np.zeros(
            self.dimension*(self.max_size+1), dtype='float64')
        buffer = shared_memory.SharedMemory(
                    create=True, size=buffer_arr.nbytes)
        self._buffer_repr = np.ndarray(
                buffer_arr.shape[0],
                dtype='float64', buffer=buffer.buf)
        self._buffer = buffer
        self.buffer_name = buffer.name

    def load_mem_resource(self):
        self._buffer = shared_memory.SharedMemory(self.buffer_name)
        # 8 represents 8 bytes of float64
        self.max_size = int(len(self._buffer.buf)//self.dimension/8)
        self.max_size -= 1
        self._buffer_repr = np.ndarray(
            len(self._buffer.buf)//8,
            dtype='float64', buffer=self._buffer.buf)

    def cleanup(self):
        self._buffer.close()
        if self._created:
            # this it's due the python core issues
            # https://bugs.python.org/issue38119
            # https://bugs.python.org/issue39959
            # https://github.com/luizalabs/shared-memory-dict/issues/13
            try:
                self._buffer.unlink()
            except FileNotFoundError:
                print(
                    f'Shared Memory {self.buffer_name}(queue_event_buffer)\
                    File not found')


class GenericCircularQueue(ABC):
    def __init__(
            self, max_size=None, dimension=8,
            use_shared_mem=False, buffer=None, buffer_name=None):
        """This implements a generic circular queue which works with
        shared memory resources.

        Parameters
        ----------
        max_size : int, optional
            If buffer_name or buffer was not passed then max_size
            it's mandatory. This will be used to construct the
            multidimensional buffer
        dimension : int, default 8
            This will be used to construct the multidimensional buffer
        use_shared_mem : bool, default False
            If the multidimensional memory resource should create or read
            using SharedMemory or RawArrays
        buffer : RawArray, optional
        buffer_name: str, optional

        """
        self._created = False
        self.head_tail_buffer_name = None
        self.head_tail_buffer_repr = None
        self.head_tail_buffer = None
        if use_shared_mem:
            self.buffer = SharedMemMultiDimensionalBuffer(
                max_size=max_size, dimension=dimension, buffer_name=buffer_name
            )
        else:
            self.buffer = RawArrayMultiDimensionalBuffer(
                max_size=max_size, dimension=dimension, buffer=buffer
            )

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
        self.head_tail_buffer_repr[:] = np.array(
            [head, tail, lock]).astype('int64')

    def _enqueue(self, data):
        ok = False
        if ((self.tail + 1) % self.buffer.max_size == self.head):
            ok = False
        else:
            if (self.head == -1):
                self.set_head_tail(0, 0, 1)
            else:
                self.tail = (self.tail + 1) % self.buffer.max_size
            self.buffer[self.tail] = data

            ok = True
        return ok

    def _dequeue(self):
        if self.head == -1:
            interactions = None
        else:
            if self.head != self.tail:
                interactions = self.buffer[self.head]
                self.head = (self.head + 1) % self.buffer.max_size
            else:
                interactions = self.buffer[self.head]
                self.set_head_tail(-1, -1, 1)
        return interactions

    @abstractmethod
    def enqueue(self):
        pass

    @abstractmethod
    def dequeue(self):
        pass

    @abstractmethod
    def load_mem_resource(self):
        pass

    @abstractmethod
    def create_mem_resource(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass


class ArrayCircularQueue(GenericCircularQueue):
    def __init__(
        self, max_size=10, dimension=6,
            head_tail_buffer=None, buffer=None):
        """This implements a MultiDimensional Queue which works with
        Arrays and RawArrays. Stream system uses that to implement
        user interactions

        Parameters
        ----------
        max_size : int, optional
            If buffer_name or buffer was not passed then max_size
            it's mandatory. This will be used to construct the
            multidimensional buffer
        dimension : int, default 8
            This will be used to construct the multidimensional buffer
        head_tail_buffer : buffer, optional
            If buffer is not passed to __init__
            then this obj will create a new
            RawArray to store head and tail position.
        buffer : buffer, optional
            If buffer  is not passed to __init__
            then the multidimensional buffer obj will create a new
            RawArray to store the data

        """

        super().__init__(
            max_size, dimension, use_shared_mem=False, buffer=buffer)

        if head_tail_buffer is None:
            self.create_mem_resource()
            self._created = True
        else:
            self.head_tail_buffer = head_tail_buffer
            self._created = False

        self.head_tail_buffer_name = None
        self.head_tail_buffer_repr = self.head_tail_buffer
        self.use_raw_array = True
        if self._created:
            self.set_head_tail(-1, -1, 0)

    def load_mem_resource(self):
        pass

    def create_mem_resource(self):
        # head_tail_arr[0] int; head position
        # head_tail_arr[1] int; tail position
        head_tail_arr = np.array([-1, -1, 0], dtype='int64')
        self.head_tail_buffer = multiprocessing.Array(
                    'l', head_tail_arr,
        )

    def enqueue(self, data):
        ok = False
        with self.head_tail_buffer.get_lock():
            ok = self._enqueue(data)
        return ok

    def dequeue(self):
        with self.head_tail_buffer.get_lock():
            interactions = self._dequeue()
        return interactions

    def cleanup(self):
        pass


class SharedMemCircularQueue(GenericCircularQueue):
    def __init__(
        self, max_size=10, dimension=6,
            head_tail_buffer_name=None, buffer_name=None):
        """This implements a MultiDimensional Queue which works with
        SharedMemory.
        Stream system uses that to implemenet user interactions

        Parameters
        ----------
        max_size : int, optional
            If buffer_name or buffer was not passed then max_size
            it's mandatory. This will be used to construct the
            multidimensional buffer
        dimension : int, default 8
            This will be used to construct the multidimensional buffer
        head_tail_buffer_name : str, optional
            if buffer_name is passed than this Obj will read a
            a already created SharedMemory with the head and tail
            informations
        buffer_name : str, optional
            if buffer_name is passed than this Obj will read a
            a already created SharedMemory to create the MultiDimensionalBuffer
        """
        super().__init__(
            max_size, dimension, use_shared_mem=True, buffer_name=buffer_name)

        if head_tail_buffer_name is None:
            self.create_mem_resource()
            self._created = True
        else:
            self.head_tail_buffer_name = head_tail_buffer_name
            self.load_mem_resource()
            self._created = False

        self.head_tail_buffer_repr = np.ndarray(
                3,
                dtype='int64', buffer=self.head_tail_buffer.buf)
        self.use_raw_array = False
        if self._created:
            self.set_head_tail(-1, -1, 0)

    def load_mem_resource(self):
        self.head_tail_buffer = shared_memory.SharedMemory(
                    self.head_tail_buffer_name)

    def create_mem_resource(self):
        # head_tail_arr[0] int; head position
        # head_tail_arr[1] int; tail position
        head_tail_arr = np.array([-1, -1, 0], dtype='int64')
        self.head_tail_buffer = shared_memory.SharedMemory(
                create=True, size=head_tail_arr.nbytes)
        self.head_tail_buffer_name = self.head_tail_buffer.name

    def is_unlocked(self):
        return self.head_tail_buffer_repr[2] == 0

    def lock(self):
        self.head_tail_buffer_repr[2] = 1

    def unlock(self):
        self.head_tail_buffer_repr[2] = 0

    def enqueue(self, data):
        ok = False
        if self.is_unlocked():
            self.lock()
            ok = self._enqueue(data)
            self.unlock()
        return ok

    def dequeue(self):
        if self.is_unlocked():
            self.lock()
            interactions = self._dequeue()
            self.unlock()
        return interactions

    def cleanup(self):
        self.buffer.cleanup()
        self.head_tail_buffer.close()
        if self._created:
            print('unlink')
            # this it's due the python core issues
            # https://bugs.python.org/issue38119
            # https://bugs.python.org/issue39959
            # https://github.com/luizalabs/shared-memory-dict/issues/13
            try:
                self.head_tail_buffer.unlink()
            except FileNotFoundError:
                print(
                    f'Shared Memory {self.head_tail_buffer_name}(head_tail)\
                     File not found')


class IntervalTimer(object):
    def __init__(self, seconds, callback, *args, **kwargs):
        """Implements a object with the same behavior of setInterval from Js

        Parameters
        ----------
        seconds : float
            A postive float number. Represents the total amount of
            seconds between each call
        callback : function
            The function to be called
        *args : args
            args to be passed to callback
        **kwargs : kwargs
            kwargs to be passed to callback

        Examples
        -------
        >>> def callback(arr):
        >>>    arr += [len(arr)]
        >>> arr = []
        >>> interval_timer = tools.IntervalTimer(1, callback, arr)
        >>> interval_timer.start()
        >>> time.sleep(5)
        >>> interval_timer.stop()
        >>> # len(arr) == 5

        Refs
        ----
        I got this from
        https://stackoverflow.com/questions/3393612/run-certain-code-every-n-seconds

        """
        self._timer = None
        self.seconds = seconds
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        # self.next_call = time.time()
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.callback(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.seconds, self._run)

            # self.next_call += selfseconds.
            # self._timer = Timer(self.next_call - time.time(), self._run)
            self._timer.daemon = True
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False
        # self._timer.join()
