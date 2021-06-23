import time
import numpy as np
import numpy.testing as npt
import sys
if sys.version_info.minor >= 8:
    PY_VERSION_8 = True
else:
    PY_VERSION_8 = False

from fury.stream import tools


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
    test(False)
    if PY_VERSION_8:
        test(True)


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

