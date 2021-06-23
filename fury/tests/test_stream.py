import time
import numpy as np
import numpy.testing as npt
import sys
if sys.version_info.minor >= 8:
    PY_VERSION_8 = True
else:
    PY_VERSION_8 = False

from fury import actor, window
from fury.stream import tools
from fury.stream.server import ImageBufferManager
from fury.stream.client import FuryStreamClient


def test_client_and_buffer_manager():
    def test(use_raw_array, ms_stream=16):
        width_0 = 100
        height_0 = 200

        centers = np.array([
            [0, 0, 0],
            [-1, 0, 0],
            [1, 0, 0]
        ])
        colors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        actors = actor.sdf(
            centers, primitives='sphere', colors=colors, scales=2)

        scene = window.Scene()
        scene.add(actors)
        showm = window.ShowManager(scene, reset_camera=False, size=(
            width_0, height_0), order_transparent=False,
        )

        showm.initialize()

        stream = FuryStreamClient(
            showm, use_raw_array=use_raw_array,
            whithout_iren_start=False)
        img_buffer_manager = ImageBufferManager(
            stream.info_buffer, stream.info_buffer_name,
            stream.image_buffers, stream.image_buffer_names
        )
        showm.render()
        stream.start(ms_stream)
        showm.render()
        time.sleep(3)
        # arr = window.snapshot(scene, size=showm.size)
        width, height, frame = img_buffer_manager.get_infos()
        assert width == width_0 and height == height_0
        image = np.frombuffer(
                    frame,
                    'uint8')[0:width*height*3].reshape((height, width, 3))
        # image = np.flipud(image)

        # image = image[:, :, ::-1]
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()
        # npt.assert_allclose(arr, image)
        report = window.analyze_snapshot(image, find_objects=True)
        npt.assert_equal(report.objects, 3)
        img_buffer_manager.cleanup()
        stream.stop()
        stream.cleanup()
    test(True)
    if PY_VERSION_8:
        test(False)


def test_time_interval():
    def callback(arr):
        arr += [len(arr)]

    arr = []
    interval_timer = tools.IntervalTimer(.5, callback, arr)
    interval_timer.start()
    time.sleep(2)
    old_len = len(arr)
    assert len(arr) > 0
    interval_timer.stop()
    old_len = len(arr)
    time.sleep(2)
    # check if the stop method worked
    assert len(arr) == old_len


def test_multidimensional_buffer():
    def test(use_raw_array=False):
        # creates a raw_array
        max_size = 3
        dimension = 4
        m_buffer = tools.MultiDimensionalBuffer(
            max_size, dimension, use_raw_array=use_raw_array)
        m_buffer[1] = np.array([.2, .3, .4, .5])

        # check if the dimension it's correct
        assert len(m_buffer[0]) == dimension
        # check if the max_size it's correct
        assert len(m_buffer[max_size]) == 4 and len(m_buffer[max_size+1]) == 0
        # check values
        npt.assert_equal(np.array([.2, .3, .4, .5]), m_buffer[1])
        # cleanup
        m_buffer.cleanup()

    # rawarray
    test(True)
    if PY_VERSION_8:
        test(False)

    # test if the communication it's ok
    def test_comm(use_raw_array=True):
        # creates a raw_array
        max_size = 3
        dimension = 4
        m_buffer_org = tools.MultiDimensionalBuffer(
            max_size, dimension, use_raw_array=use_raw_array)
        if use_raw_array:
            m_buffer_0 = tools.MultiDimensionalBuffer(
                max_size, dimension, buffer=m_buffer_org.buffer)
        else:
            m_buffer_0 = tools.MultiDimensionalBuffer(
                max_size, dimension, buffer_name=m_buffer_org.buffer_name)

        m_buffer_0[1] = np.array([.2, .3, .4, .5])

        # check if the dimension it's correct
        assert len(m_buffer_org[0]) == len(m_buffer_0[0])
        # check if the max_size it's correct
        assert len(m_buffer_org[max_size]) == 4 and\
            len(m_buffer_org[max_size+1]) == 0
        # check values
        npt.assert_equal(np.array([.2, .3, .4, .5]), m_buffer_org[1])
        # check it the correct max_size was recovered
        assert m_buffer_org.max_size == m_buffer_0.max_size
        assert m_buffer_org.dimension == m_buffer_0.dimension
        # cleanup
        m_buffer_0.cleanup()
        m_buffer_org.cleanup()

    # rawarray
    test_comm(True)
    if PY_VERSION_8:
        test(False)


def test_circular_queue():
    def test(use_raw_array=True):
        max_size = 3
        dimension = 4
        queue = tools.CircularQueue(
            max_size, dimension, use_raw_array=use_raw_array)
        # init as empty queue
        assert queue.head == -1 and queue.tail == -1
        assert queue.dequeue() is None
        arr = np.array([1.0, 2, 3, 4])
        ok = queue.enqueue(arr)
        assert ok
        ok = queue.enqueue(arr+1)
        ok = queue.enqueue(arr+2)
        assert ok
        # the ciruclar queue must be full (size 3)
        ok = queue.enqueue(arr+3)
        assert not ok
        # remember, python arrays starts at 0
        assert queue.head == 0 and queue.tail == 2
        arr_recovered = queue.dequeue()
        # it's a FIFO
        npt.assert_equal(arr, arr_recovered)
        assert queue.head == 1 and queue.tail == 2
        arr_recovered = queue.dequeue()
        npt.assert_equal(arr+1, arr_recovered)
        assert queue.head == 2 and queue.tail == 2
        arr_recovered = queue.dequeue()
        npt.assert_equal(arr+2, arr_recovered)
        assert queue.head == -1 and queue.tail == -1
        arr_recovered = queue.dequeue()
        assert arr_recovered is None
        queue.cleanup()
    test(True)
    if PY_VERSION_8:
        test(False)

