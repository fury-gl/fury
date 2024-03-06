import asyncio
import io
import logging
import multiprocessing
import time
from abc import ABC, abstractmethod
from threading import Timer

import numpy as np
from PIL import Image, ImageDraw

from fury.stream.constants import PY_VERSION_8

if PY_VERSION_8:
    from multiprocessing import resource_tracker, shared_memory
else:
    shared_memory = None   # type: ignore


_FLOAT_ShM_TYPE = 'd'
_INT_ShM_TYPE = 'i'
_UINT_ShM_TYPE = 'I'
_BYTE_ShM_TYPE = 'B'

_FLOAT_SIZE = np.dtype(_FLOAT_ShM_TYPE).itemsize
_INT_SIZE = np.dtype(_INT_ShM_TYPE).itemsize
_UINT_SIZE = np.dtype(_UINT_ShM_TYPE).itemsize
_BYTE_SIZE = np.dtype(_BYTE_ShM_TYPE).itemsize


def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't
    be tracked

    Notes
    -----
        More details at: https://bugs.python.org/issue38119

    """

    def fix_register(name, rtype):
        if rtype == 'shared_memory':
            return
        try:
            return resource_tracker._resource_tracker.register(self, name, rtype)
        except NameError:
            return None

    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == 'shared_memory':
            return
        try:
            return resource_tracker._resource_tracker.unregister(self, name, rtype)
        except NameError:
            return None

    resource_tracker.unregister = fix_unregister

    if 'shared_memory' in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS['shared_memory']


class GenericMultiDimensionalBuffer(ABC):
    """This implements a abstract (generic) multidimensional buffer."""

    def __init__(self, max_size=None, dimension=8):
        """Initialize the multidimensional buffer.

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
            if data.dtype == _FLOAT_ShM_TYPE:
                self._buffer_repr[:] = data

    def get_start_end(self, idx):
        dim = self.dimension
        start = idx * dim
        end = dim * (idx + 1)
        return start, end

    def __getitem__(self, idx):
        start, end = self.get_start_end(idx)
        logging.info(f'dequeue start {int(time.time()*1000)}')
        ts = time.time() * 1000

        items = self._buffer_repr[start:end]
        te = time.time() * 1000
        logging.info(f'dequeue frombuffer cost {te-ts:.2f}')
        return items

    def __setitem__(self, idx, data):
        start, end = self.get_start_end(idx)
        if isinstance(data, (np.ndarray, np.generic)):
            if data.dtype == _FLOAT_ShM_TYPE:
                # if end - start == self.dimension and start >= 0 and end >= 0:
                self._buffer_repr[start:end] = data

    @abstractmethod
    def load_mem_resource(self):
        ...  # pragma: no cover

    @abstractmethod
    def create_mem_resource(self):
        ...  # pragma: no cover

    @abstractmethod
    def cleanup(self):
        ...  # pragma: no cover


class RawArrayMultiDimensionalBuffer(GenericMultiDimensionalBuffer):
    """This implements a  multidimensional buffer with RawArray."""

    def __init__(self, max_size, dimension=4, buffer=None):
        """Stream system uses that to implement the CircularQueue
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
            self.dimension * (self.max_size + 1), dtype=_FLOAT_ShM_TYPE
        )
        buffer = multiprocessing.RawArray(
            _FLOAT_ShM_TYPE, np.ctypeslib.as_ctypes(buffer_arr)
        )
        self._buffer = buffer
        self._buffer_repr = np.ctypeslib.as_array(self._buffer)

    def load_mem_resource(self):
        self.max_size = int(len(self._buffer) // self.dimension)
        self.max_size -= 1
        self._buffer_repr = np.ctypeslib.as_array(self._buffer)

    def cleanup(self):
        pass


class SharedMemMultiDimensionalBuffer(GenericMultiDimensionalBuffer):
    """This implements a generic multidimensional buffer
    with SharedMemory.
    """

    def __init__(self, max_size, dimension=4, buffer_name=None):
        """Stream system uses that to implement the
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

        self._create_repr()

    def create_mem_resource(self):
        self._num_el = self.dimension * (self.max_size + 1)
        buffer_arr = np.zeros(self._num_el + 2, dtype=_FLOAT_ShM_TYPE)
        self._buffer = shared_memory.SharedMemory(create=True, size=buffer_arr.nbytes)
        sizes = np.ndarray(
            2, dtype=_FLOAT_ShM_TYPE, buffer=self._buffer.buf[0 : _FLOAT_SIZE * 2]
        )
        sizes[0] = self.max_size
        sizes[1] = self.dimension
        self.buffer_name = self._buffer.name
        logging.info(
            [
                'create repr multidimensional buffer ',
            ]
        )

    def load_mem_resource(self):
        self._buffer = shared_memory.SharedMemory(self.buffer_name)
        sizes = np.ndarray(2, dtype='d', buffer=self._buffer.buf[0 : _FLOAT_SIZE * 2])
        self.max_size = int(sizes[0])
        self.dimension = int(sizes[1])
        num_el = int((sizes[0] + 1) * sizes[1])
        self._num_el = num_el
        logging.info(
            [
                'load repr multidimensional buffer',
            ]
        )

    def _create_repr(self):
        start = _FLOAT_SIZE * 2
        end = (self._num_el + 2) * _FLOAT_SIZE
        self._buffer_repr = np.ndarray(
            self._num_el, dtype=_FLOAT_ShM_TYPE, buffer=self._buffer.buf[start:end]
        )
        logging.info(
            [
                'create repr multidimensional buffer',
                self._buffer_repr.shape,
                'max size',
                self.max_size,
                'dimension',
                self.dimension,
            ]
        )

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
                    File not found'
                )


class GenericCircularQueue(ABC):
    """This implements a generic circular queue which works with
    shared memory resources.
    """

    def __init__(
        self,
        max_size=None,
        dimension=8,
        use_shared_mem=False,
        buffer=None,
        buffer_name=None,
    ):
        """Initialize the circular queue.

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
        self._use_shared_mem = use_shared_mem
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
        if self._use_shared_mem:
            return self.head_tail_buffer_repr[0]
        else:
            return np.frombuffer(self.head_tail_buffer.get_obj(), _INT_ShM_TYPE)[0]

    @head.setter
    def head(self, value):
        self.head_tail_buffer_repr[0] = value

    @property
    def tail(self):
        if self._use_shared_mem:
            return self.head_tail_buffer_repr[1]
        else:
            return np.frombuffer(self.head_tail_buffer.get_obj(), _INT_ShM_TYPE)[1]

    @tail.setter
    def tail(self, value):
        self.head_tail_buffer_repr[1] = value

    def set_head_tail(self, head, tail, lock=1):
        self.head_tail_buffer_repr[0:3] = np.array([head, tail, lock]).astype(
            _INT_ShM_TYPE
        )

    def _enqueue(self, data):
        ok = False
        if (self.tail + 1) % self.buffer.max_size == self.head:
            ok = False
        else:
            if self.head == -1:
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
    def enqueue(self, data):
        pass  # pragma: no cover

    @abstractmethod
    def dequeue(self):
        pass  # pragma: no cover

    @abstractmethod
    def load_mem_resource(self):
        pass  # pragma: no cover

    @abstractmethod
    def create_mem_resource(self):
        pass  # pragma: no cover

    @abstractmethod
    def cleanup(self):
        pass  # pragma: no cover


class ArrayCircularQueue(GenericCircularQueue):
    """This implements a MultiDimensional Queue which works with
    Arrays and RawArrays.
    """

    def __init__(self, max_size=10, dimension=6, head_tail_buffer=None, buffer=None):
        """Stream system uses that to implement user interactions

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
        super().__init__(max_size, dimension, use_shared_mem=False, buffer=buffer)

        if head_tail_buffer is None:
            self.create_mem_resource()
            self._created = True
        else:
            self.head_tail_buffer = head_tail_buffer
            self._created = False

        self.head_tail_buffer_name = None
        self.head_tail_buffer_repr = self.head_tail_buffer
        if self._created:
            self.set_head_tail(-1, -1, 0)

    def load_mem_resource(self):
        pass  # pragma: no cover

    def create_mem_resource(self):
        # head_tail_arr[0] int; head position
        # head_tail_arr[1] int; tail position
        head_tail_arr = np.array([-1, -1, 0], dtype=_INT_ShM_TYPE)
        self.head_tail_buffer = multiprocessing.Array(
            _INT_ShM_TYPE,
            head_tail_arr,
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
    """This implements a MultiDimensional Queue which works with
    SharedMemory.
    """

    def __init__(
        self, max_size=10, dimension=6, head_tail_buffer_name=None, buffer_name=None
    ):
        """Stream system uses that to implement user interactions

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
            information
        buffer_name : str, optional
            if buffer_name is passed than this Obj will read a
            a already created SharedMemory to create the MultiDimensionalBuffer

        """
        super().__init__(
            max_size, dimension, use_shared_mem=True, buffer_name=buffer_name
        )

        if head_tail_buffer_name is None:
            self.create_mem_resource()
            self._created = True
        else:
            self.head_tail_buffer_name = head_tail_buffer_name
            self.load_mem_resource()
            self._created = False

        self.head_tail_buffer_repr = np.ndarray(
            3, dtype=_INT_ShM_TYPE, buffer=self.head_tail_buffer.buf[0 : 3 * _INT_SIZE]
        )
        logging.info(
            [
                'create shared mem',
                'size repr',
                self.head_tail_buffer_repr.shape,
                'size buffer',
                self.head_tail_buffer.size / _INT_SIZE,
            ]
        )
        if self._created:
            self.set_head_tail(-1, -1, 0)

    def load_mem_resource(self):
        self.head_tail_buffer = shared_memory.SharedMemory(self.head_tail_buffer_name)

    def create_mem_resource(self):
        # head_tail_arr[0] int; head position
        # head_tail_arr[1] int; tail position
        head_tail_arr = np.array([-1, -1, 0], dtype=_INT_ShM_TYPE)
        self.head_tail_buffer = shared_memory.SharedMemory(
            create=True, size=head_tail_arr.nbytes
        )
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
        interactions = None
        if self.is_unlocked():
            self.lock()
            interactions = self._dequeue()
            self.unlock()
        return interactions

    def cleanup(self):
        self.buffer.cleanup()
        self.head_tail_buffer.close()
        if self._created:
            # this it's due the python core issues
            # https://bugs.python.org/issue38119
            # https://bugs.python.org/issue39959
            # https://github.com/luizalabs/shared-memory-dict/issues/13
            try:
                self.head_tail_buffer.unlink()
            except FileNotFoundError:
                print(
                    f'Shared Memory {self.head_tail_buffer_name}(head_tail)\
                     File not found'
                )


class GenericImageBufferManager(ABC):
    """This implements a abstract (generic) ImageBufferManager with
    the n-buffer technique.
    """

    def __init__(self, max_window_size=None, num_buffers=2, use_shared_mem=False):
        """Initialize the ImageBufferManager.

        Parameters
        ----------
        max_window_size : tuple of ints, optional
            This allows resize events inside of the FURY window instance.
            Should be greater than the window size.
        num_buffers : int, optional
            Number of buffers to be used in the n-buffering
            technique.
        use_shared_mem: bool, default False

        """
        self.max_window_size = np.array(max_window_size)
        self.num_buffers = num_buffers
        self.info_buffer_size = num_buffers * 2 + 2
        self._use_shared_mem = use_shared_mem
        self.max_size = None  # int
        self.num_components = 3
        self.image_reprs = []
        self.image_buffers = []
        self.image_buffer_names = []
        self.info_buffer_name = None
        self.info_buffer = None
        self.info_buffer_repr = None
        self._created = False

        size = (self.max_window_size[0], self.max_window_size[1])
        img = Image.new('RGB', size, color=(0, 0, 0))

        d = ImageDraw.Draw(img)
        pos_text = (12, size[1] // 2)
        d.text(
            pos_text, 'Image size have exceed the Buffer Max Size', fill=(255, 255, 0)
        )
        img = np.flipud(img)
        self.img_exceed = np.asarray(img).flatten()

    @property
    def next_buffer_index(self):
        index = int((self.info_buffer_repr[1] + 1) % self.num_buffers)
        return index

    @property
    def buffer_index(self):
        index = self.info_buffer_repr[1]
        return index

    def write_into(self, w, h, np_arr):
        buffer_size = buffer_size = int(h * w * 3)
        next_buffer_index = self.next_buffer_index

        if buffer_size == self.max_size:
            self.image_reprs[next_buffer_index][:] = np_arr
        elif buffer_size < self.max_size:
            self.image_reprs[next_buffer_index][0:buffer_size] = np_arr
        else:
            self.image_reprs[next_buffer_index][0 : self.max_size] = self.img_exceed
            w = self.max_window_size[0]
            h = self.max_window_size[1]

        self.info_buffer_repr[2 + next_buffer_index * 2] = w
        self.info_buffer_repr[2 + next_buffer_index * 2 + 1] = h
        self.info_buffer_repr[1] = next_buffer_index

    def get_current_frame(self):
        """Get the current frame from the buffer."""
        if not self._use_shared_mem:
            image_info = np.frombuffer(self.info_buffer, _UINT_ShM_TYPE)
        else:
            image_info = self.info_buffer_repr

        buffer_index = int(image_info[1])

        self.width = int(image_info[2 + buffer_index * 2])
        self.height = int(image_info[2 + buffer_index * 2 + 1])

        image = self.image_reprs[buffer_index]
        self.image_buffer_repr = image

        return self.width, self.height, image

    def get_jpeg(self):
        """Returns a jpeg image from the buffer.

        Returns
        -------
            bytes: jpeg image.

        """
        width, height, image = self.get_current_frame()

        if self._use_shared_mem:
            image = np.frombuffer(image, _BYTE_ShM_TYPE)

        image = image[0 : width * height * 3].reshape((height, width, 3))
        image = np.flipud(image)
        image_encoded = Image.fromarray(image, mode='RGB')
        bytes_img_data = io.BytesIO()
        image_encoded.save(bytes_img_data, format='jpeg')
        bytes_img = bytes_img_data.getvalue()

        return bytes_img

    async def async_get_jpeg(self, ms=33):
        jpeg = self.get_jpeg()
        await asyncio.sleep(ms / 1000)
        return jpeg

    @abstractmethod
    def load_mem_resource(self):
        pass  # pragma: no cover

    @abstractmethod
    def create_mem_resource(self):
        pass  # pragma: no cover

    @abstractmethod
    def cleanup(self):
        pass  # pragma: no cover


class RawArrayImageBufferManager(GenericImageBufferManager):
    """This implements an ImageBufferManager using RawArrays."""

    def __init__(
        self,
        max_window_size=(100, 100),
        num_buffers=2,
        image_buffers=None,
        info_buffer=None,
    ):
        """Initialize the ImageBufferManager.

        Parameters
        ----------
        max_window_size : tuple of ints, optional
                This allows resize events inside of the FURY window instance.
                Should be greater than the window size.
        num_buffers : int, optional
                Number of buffers to be used in the n-buffering
                technique.
        info_buffer : buffer, optional
            A buffer with the information about the current
            frame to be streamed and the respective sizes
        image_buffers : list of buffers, optional
            A list of buffers with each one containing a frame.

        """
        super().__init__(max_window_size, num_buffers, use_shared_mem=False)
        if image_buffers is None or info_buffer is None:
            self.create_mem_resource()
        else:
            self.image_buffers = image_buffers
            self.info_buffer = info_buffer
            self.load_mem_resource()

    def create_mem_resource(self):
        self.max_size = self.max_window_size[0] * self.max_window_size[1]
        self.max_size *= self.num_components

        for _ in range(self.num_buffers):
            buffer = multiprocessing.RawArray(
                _BYTE_ShM_TYPE,
                np.ctypeslib.as_ctypes(
                    np.random.randint(0, 255, size=self.max_size, dtype=_BYTE_ShM_TYPE)
                ),
            )
            self.image_buffers.append(buffer)
            self.image_reprs.append(np.ctypeslib.as_array(buffer))

        # info_list stores the information about the n frame buffers
        # as well the respectives sizes.
        # 0 number of components
        # 1 id buffer
        # 2, 3, width first buffer, height first buffer
        # 4, 5, width second buffer , height second buffer
        info_list = [3, 0]
        for _ in range(self.num_buffers):
            info_list += [self.max_window_size[0]]
            info_list += [self.max_window_size[1]]
        info_list = np.array(info_list, dtype=_UINT_ShM_TYPE)
        self.info_buffer = multiprocessing.RawArray(
            _UINT_ShM_TYPE, np.ctypeslib.as_ctypes(np.array(info_list))
        )
        self.info_buffer_repr = np.ctypeslib.as_array(self.info_buffer)

    def load_mem_resource(self):
        self.info_buffer = np.frombuffer(self.info_buffer, _UINT_ShM_TYPE)
        self.info_buffer_repr = np.ctypeslib.as_array(self.info_buffer)
        for img_buffer in self.image_buffers:
            self.image_reprs.append(np.ctypeslib.as_array(img_buffer))

    def cleanup(self):
        pass


class SharedMemImageBufferManager(GenericImageBufferManager):
    """This implements an ImageBufferManager using the
    SharedMemory approach.
    """

    def __init__(
        self,
        max_window_size=(100, 100),
        num_buffers=2,
        image_buffer_names=None,
        info_buffer_name=None,
    ):
        """Initialize the ImageBufferManager.

        Parameters
        ----------
        max_window_size : tuple of ints, optional
                This allows resize events inside of the FURY window instance.
                Should be greater than the window size.
        num_buffers : int, optional
                Number of buffers to be used in the n-buffering
                technique.
        info_buffer_name : str
            The name of a buffer with the information about the current
            frame to be streamed and the respective sizes
        image_buffer_names : list of str, optional
            a list of buffer names. Each buffer contains a frame

        Notes
        -----
        Python >=3.8 is a requirement to use this object.

        """
        super().__init__(max_window_size, num_buffers, use_shared_mem=True)
        if image_buffer_names is None or info_buffer_name is None:
            self.create_mem_resource()
            self._created = True
        else:
            self.image_buffer_names = image_buffer_names
            self.info_buffer_name = info_buffer_name
            self._created = False
            self.load_mem_resource()

    def create_mem_resource(self):
        self.max_size = self.max_window_size[0] * self.max_window_size[1]
        self.max_size *= self.num_components
        self.max_size = int(self.max_size)
        for _ in range(self.num_buffers):
            buffer = shared_memory.SharedMemory(create=True, size=self.max_size)
            self.image_buffers.append(buffer)
            self.image_reprs.append(
                np.ndarray(self.max_size, dtype=_BYTE_ShM_TYPE, buffer=buffer.buf)
            )
            self.image_buffer_names.append(buffer.name)

        info_list = [2 + self.num_buffers * 2, 1, 3, 0]
        for _ in range(self.num_buffers):
            info_list += [self.max_window_size[0]]
            info_list += [self.max_window_size[1]]
        info_list = np.array(info_list, dtype=_UINT_ShM_TYPE)

        self.info_buffer = shared_memory.SharedMemory(
            create=True, size=info_list.nbytes
        )
        sizes = np.ndarray(
            2, dtype=_UINT_ShM_TYPE, buffer=self.info_buffer.buf[0 : _UINT_SIZE * 2]
        )
        sizes[0] = info_list[0]
        sizes[1] = 1
        self.info_buffer_repr = np.ndarray(
            sizes[0],
            dtype=_UINT_ShM_TYPE,
            buffer=self.info_buffer.buf[2 * _UINT_SIZE :],
        )
        logging.info(
            [
                'info buffer create',
                'buffer size',
                sizes[0],
                'repr size',
                self.info_buffer_repr.shape,
            ]
        )
        self.info_buffer_name = self.info_buffer.name

    def load_mem_resource(self):
        self.info_buffer = shared_memory.SharedMemory(self.info_buffer_name)
        sizes = np.ndarray(
            2, dtype=_UINT_ShM_TYPE, buffer=self.info_buffer.buf[0 : _UINT_SIZE * 2]
        )
        self.info_buffer_repr = np.ndarray(
            sizes[0],
            dtype=_UINT_ShM_TYPE,
            buffer=self.info_buffer.buf[2 * _UINT_SIZE :],
        )
        logging.info(
            [
                'info buffer load',
                'buffer size',
                sizes[0],
                'repr size',
                self.info_buffer_repr.shape,
            ]
        )
        for buffer_name in self.image_buffer_names:
            buffer = shared_memory.SharedMemory(buffer_name)
            self.image_buffers.append(buffer)
            self.image_reprs.append(
                np.ndarray(
                    buffer.size // _BYTE_SIZE, dtype=_BYTE_ShM_TYPE, buffer=buffer.buf
                )
            )

    def cleanup(self):
        """Release the resources used by the Shared Memory Manager"""
        self.info_buffer.close()
        # this it's due the python core issues
        # https://bugs.python.org/issue38119
        # https://bugs.python.org/issue39959
        # https://github.com/luizalabs/shared-memory-dict/issues/13
        if self._created:
            try:
                self.info_buffer.unlink()
            except FileNotFoundError:
                print(
                    f'Shared Memory {self.info_buffer_name}\
                        (info_buffer) File not found'
                )
        for buffer, name in zip(self.image_buffers, self.image_buffer_names):
            buffer.close()
            if self._created:
                try:
                    buffer.unlink()
                except FileNotFoundError:
                    print(f'Shared Memory {name}(buffer image) File not found')


class IntervalTimerThreading:
    """Implements a object with the same behavior of setInterval from Js"""

    def __init__(self, seconds, callback, *args, **kwargs):
        """

        Parameters
        ----------
        seconds : float
            A positive float number. Represents the total amount of
            seconds between each call
        callback : function
            The function to be called
        *args : args
            args to be passed to callback
        **kwargs : kwargs
            kwargs to be passed to callback

        Examples
        --------

        .. code-block:: python

            def callback(arr):
                arr += [len(arr)]
            arr = []
            interval_timer = tools.IntervalTimer(1, callback, arr)
            interval_timer.start()
            time.sleep(5)
            interval_timer.stop()
            # len(arr) == 5

        References
        -----------
        [1] https://stackoverflow.com/questions/3393612/run-certain-code-every-n-seconds

        """  # noqa
        self._timer = None
        self.seconds = seconds
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.callback(*self.args, **self.kwargs)

    def start(self):
        """Start the timer"""
        if self.is_running:
            return

        self._timer = Timer(self.seconds, self._run)
        self._timer.daemon = True
        self._timer.start()
        self.is_running = True

    def stop(self):
        """Stop the timer"""
        if self._timer is None:
            return

        self._timer.cancel()
        if self._timer.is_alive():
            self._timer.join()
        self.is_running = False
        self._timer = None


class IntervalTimer:
    """A object that creates a timer that calls a function periodically."""

    def __init__(self, seconds, callback, *args, **kwargs):
        """Parameters
        ----------
        seconds : float
            A positive float number. Represents the total amount of
            seconds between each call
        callback : function
            The function to be called
        *args : args
            args to be passed to callback
        **kwargs : kwargs
            kwargs to be passed to callback

        """
        self._seconds = seconds
        self._callback = callback
        self.args = args
        self.kwargs = kwargs
        self._is_running = False
        self.start()

    async def _run(self):
        self._is_running = True
        while True:
            await asyncio.sleep(self._seconds)
            if self._is_running:
                self._callback(*self.args, **self.kwargs)

    def start(self):
        """Start the timer"""
        if self._is_running:
            return
        self._loop = asyncio.get_event_loop()
        self._task = self._loop.create_task(self._run())

    def stop(self):
        """Stop the timer"""
        self._task.cancel()
        self._is_running = False
