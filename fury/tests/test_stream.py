import asyncio
from importlib import reload
import sys
import time
from unittest import mock

import numpy as np
import numpy.testing as npt
import pytest

if sys.version_info.minor >= 8:
    PY_VERSION_8 = True
else:
    PY_VERSION_8 = False

from fury import actor, window
from fury.stream import tools
from fury.stream.client import FuryStreamClient, FuryStreamInteraction
from fury.stream.constants import _CQUEUE
from fury.stream.server.async_app import (
    WEBRTC_AVAILABLE,
    set_mouse,
    set_mouse_click,
    set_weel,
)
from fury.stream.server.main import RTCServer, web_server, web_server_raw_array
from fury.stream.widget import Widget, check_port_is_available


@pytest.fixture
def loop():
    """Use this fixture to get the event loop.

    References
    ----------
     https://promity.com/2020/06/03/testing-asynchronous-code-in-python/
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def test_rtc_video_stream(loop: asyncio.AbstractEventLoop):
    if not WEBRTC_AVAILABLE:
        print('\n aiortc not available -> skipping test\n')
        return

    def test(use_raw_array, ms_stream=16):
        width_0 = 100
        height_0 = 200

        centers = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        actors = actor.sphere(centers, colors=colors, radii=0.1)

        scene = window.Scene()
        scene.add(actors)
        showm = window.ShowManager(
            scene,
            reset_camera=False,
            size=(width_0, height_0),
            order_transparent=False,
        )

        stream = FuryStreamClient(
            showm, use_raw_array=use_raw_array, whithout_iren_start=False
        )
        if use_raw_array:
            img_buffer_manager = tools.RawArrayImageBufferManager(
                info_buffer=stream.img_manager.info_buffer,
                image_buffers=stream.img_manager.image_buffers,
            )
        else:
            img_buffer_manager = tools.SharedMemImageBufferManager(
                info_buffer_name=stream.img_manager.info_buffer_name,
                image_buffer_names=stream.img_manager.image_buffer_names,
            )

        rtc_server = RTCServer(img_buffer_manager)
        showm.render()
        stream.start(ms_stream)
        showm.render()
        loop.run_until_complete(rtc_server.recv())
        # sassert frame.width == width_0 and frame.height == height_0
        rtc_server.release()
        img_buffer_manager.cleanup()
        stream.stop()
        stream.cleanup()

    test(True, 16)
    test(True, 0)
    if PY_VERSION_8:
        test(False, 0)
        test(False, 16)


def test_pillow():
    use_raw_array = True
    ms_stream = 0
    width_0 = 100
    height_0 = 200

    centers = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    actors = actor.sphere(centers, colors=colors, radii=0.1)

    scene = window.Scene()
    scene.add(actors)
    showm = window.ShowManager(
        scene,
        reset_camera=False,
        size=(width_0, height_0),
        order_transparent=False,
    )

    stream = FuryStreamClient(
        showm, use_raw_array=use_raw_array, whithout_iren_start=False
    )
    if use_raw_array:
        img_buffer_manager = tools.RawArrayImageBufferManager(
            info_buffer=stream.img_manager.info_buffer,
            image_buffers=stream.img_manager.image_buffers,
        )
    else:
        img_buffer_manager = tools.SharedMemImageBufferManager(
            info_buffer_name=stream.img_manager.info_buffer_name,
            image_buffer_names=stream.img_manager.image_buffer_names,
        )

    showm.render()
    stream.start(ms_stream)
    showm.render()
    # test jpeg method
    img_buffer_manager.get_jpeg()
    width, height, frame = img_buffer_manager.get_current_frame()

    image = np.frombuffer(frame, 'uint8')[0 : width * height * 3].reshape(
        (height, width, 3)
    )
    report = window.analyze_snapshot(image, find_objects=True)
    npt.assert_equal(report.objects, 3)
    img_buffer_manager.cleanup()
    stream.stop()
    stream.cleanup()


def test_rtc_video_stream_whitout_cython(loop: asyncio.AbstractEventLoop):
    if not WEBRTC_AVAILABLE:
        print('\n aiortc not available -> skipping test\n')
        return

    use_raw_array = True
    ms_stream = 0
    # creates a context without cython
    with mock.patch.dict(sys.modules, {'pyximport': None}):
        reload(sys.modules['fury.stream.server.main'])
        width_0 = 100
        height_0 = 200

        centers = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        actors = actor.sphere(centers, colors=colors, radii=0.1)

        scene = window.Scene()
        scene.add(actors)
        showm = window.ShowManager(
            scene,
            reset_camera=False,
            size=(width_0, height_0),
            order_transparent=False,
        )

        stream = FuryStreamClient(
            showm, use_raw_array=use_raw_array, whithout_iren_start=False
        )
        if use_raw_array:
            img_buffer_manager = tools.RawArrayImageBufferManager(
                info_buffer=stream.img_manager.info_buffer,
                image_buffers=stream.img_manager.image_buffers,
            )
        else:
            img_buffer_manager = tools.SharedMemImageBufferManager(
                info_buffer_name=stream.img_manager.info_buffer_name,
                image_buffer_names=stream.img_manager.image_buffer_names,
            )

        rtc_server = RTCServer(img_buffer_manager)
        showm.render()
        stream.start(ms_stream)
        showm.render()
        loop.run_until_complete(rtc_server.recv())
        # assert frame.width == showm.size[0] and frame.height == showm.size[1]
        rtc_server.release()
        img_buffer_manager.cleanup()
        stream.stop()
        stream.cleanup()

    reload(sys.modules['fury.stream.server.main'])


def test_client_and_buffer_manager():
    def test(use_raw_array, ms_stream=16):
        width_0 = 100
        height_0 = 200

        centers = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        actors = actor.sphere(centers, colors=colors, radii=0.1)

        scene = window.Scene()
        scene.add(actors)
        showm = window.ShowManager(
            scene,
            reset_camera=False,
            size=(width_0, height_0),
            order_transparent=False,
        )

        stream = FuryStreamClient(
            showm, use_raw_array=use_raw_array, whithout_iren_start=False
        )
        if use_raw_array:
            img_buffer_manager = tools.RawArrayImageBufferManager(
                info_buffer=stream.img_manager.info_buffer,
                image_buffers=stream.img_manager.image_buffers,
            )
        else:
            img_buffer_manager = tools.SharedMemImageBufferManager(
                info_buffer_name=stream.img_manager.info_buffer_name,
                image_buffer_names=stream.img_manager.image_buffer_names,
            )

        showm.render()
        stream.start(ms_stream)
        showm.render()
        # test jpeg method
        img_buffer_manager.get_jpeg()
        width, height, frame = img_buffer_manager.get_current_frame()

        # assert width == showm.size[0] and height == showm.size[1]
        image = np.frombuffer(frame, 'uint8')[0 : width * height * 3].reshape(
            (height, width, 3)
        )
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

    test(True, 16)
    test(True, 0)
    if PY_VERSION_8:
        test(False, 0)
        test(False, 16)


def test_stream_client_conditions():
    def test(use_raw_array, ms_stream=16, whithout_iren_start=False):
        width_0 = 100
        height_0 = 200

        centers = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        actors = actor.sphere(centers, colors=colors, radii=0.1)

        scene = window.Scene()
        scene.add(actors)
        showm = window.ShowManager(
            scene,
            reset_camera=False,
            size=(width_0, height_0),
            order_transparent=False,
        )

        stream = FuryStreamClient(
            showm, use_raw_array=use_raw_array, whithout_iren_start=whithout_iren_start
        )

        showm.render()
        stream.start(ms_stream)
        npt.assert_equal(stream._started, True)
        # test if stop method has been called
        stream.start(ms_stream)
        npt.assert_equal(stream._started, True)
        showm.render()
        stream.stop()
        # double stop test
        npt.assert_equal(stream.stop(), False)
        stream.cleanup()

    test(True, 16, False)
    test(True, 0, True)
    if PY_VERSION_8:
        test(False, 16, False)
        test(False, 0, True)


def test_stream_client_resize():
    width_0 = 100
    height_0 = 200
    centers = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    actors = actor.sphere(centers, colors=colors, radii=0.1)
    scene = window.Scene()
    scene.add(actors)
    showm = window.ShowManager(
        scene,
        reset_camera=False,
        size=(width_0, height_0),
        order_transparent=False,
    )

    with npt.assert_raises(ValueError):
        FuryStreamClient(
            showm,
            use_raw_array=True,
            max_window_size=(width_0 - 10, height_0),
            whithout_iren_start=False,
        )

    stream = FuryStreamClient(
        showm,
        use_raw_array=True,
        max_window_size=(width_0, height_0),
        whithout_iren_start=False,
    )

    showm.window.SetSize((width_0 + 210, height_0 + 210))

    npt.assert_equal(0, stream.img_manager.buffer_index)
    stream.start()
    showm.render()
    showm.scene.azimuth(1)
    showm.render()
    showm.render()

    stream.stop()
    stream.cleanup()


def test_stream_interaction(loop: asyncio.AbstractEventLoop):
    async def test(use_raw_array, ms_stream=16):
        width_0 = 300
        height_0 = 200

        centers = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        actors = actor.sphere(centers, colors=colors, radii=0.1)

        scene = window.Scene()
        scene.add(actors)
        showm = window.ShowManager(
            scene,
            reset_camera=False,
            size=(width_0, height_0),
            order_transparent=False,
        )

        stream = FuryStreamClient(
            showm, use_raw_array=use_raw_array, whithout_iren_start=True
        )
        stream_interaction = FuryStreamInteraction(
            max_queue_size=500,
            showm=showm,
            use_raw_array=use_raw_array,
            whithout_iren_start=True,
        )

        showm.render()
        # test jpeg method
        for _ in range(10):
            stream_interaction.circular_queue.enqueue(
                np.array(
                    [_CQUEUE.event_ids.mouse_weel, 1, 0, 0, 0, 0, 0.1, 0], dtype='d'
                )
            )
        for _ in range(10):
            stream_interaction.circular_queue.enqueue(
                np.array(
                    [_CQUEUE.event_ids.mouse_weel, -1, 0, 0, 0, 0, 0.1, 0], dtype='d'
                )
            )
        dxs = []
        for shift, ctrl in ((0, 1), (1, 0), (0, 0)):
            x = width_0 / 2
            y = height_0 / 2
            stream_interaction.circular_queue.enqueue(
                np.array(
                    [_CQUEUE.event_ids.left_btn_press, 0, x, y, ctrl, shift, 0.1, 0],
                    dtype='d',
                )
            )
            for i in range(50):
                if ctrl == 1:
                    x = x + i / 50 * width_0 / 4
                else:
                    if i < 25:
                        dx = +i / 50
                        dxs.append(dx)
                        x = x - dx
                    else:
                        x = x + dxs[::-1][i - 25]
                stream_interaction.circular_queue.enqueue(
                    np.array(
                        [_CQUEUE.event_ids.mouse_move, 0, x, y, ctrl, shift, 0.1, 0],
                        dtype='d',
                    )
                )
            stream_interaction.circular_queue.enqueue(
                np.array(
                    [_CQUEUE.event_ids.left_btn_release, 0, x, y, ctrl, shift, 0.1, 0],
                    dtype='d',
                )
            )

        stream_interaction.start(ms_stream, use_asyncio=True)
        while stream_interaction.circular_queue.head != -1:
            showm.render()
            await asyncio.sleep(0.01)
        stream_interaction.stop()
        stream.stop()
        stream.cleanup()
        stream_interaction.cleanup()

    loop.run_until_complete(test(True, 16))
    if PY_VERSION_8:
        loop.run_until_complete(test(False, 16))


def test_stream_interaction_conditions():
    def test(use_raw_array, ms_stream, whitouth_iren_start):
        width_0 = 300
        height_0 = 200

        centers = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        actors = actor.sphere(centers, colors=colors, radii=0.1)

        scene = window.Scene()
        scene.add(actors)
        showm = window.ShowManager(scene, size=(width_0, height_0))

        stream = FuryStreamClient(
            showm, use_raw_array=use_raw_array, whithout_iren_start=whitouth_iren_start
        )
        stream_interaction = FuryStreamInteraction(
            max_queue_size=500,
            showm=showm,
            use_raw_array=use_raw_array,
            whithout_iren_start=whitouth_iren_start,
        )

        showm.render()

        # ms should always be greater than 0
        with npt.assert_raises(ValueError):
            stream_interaction.start(-1)

        stream_interaction.start(ms_stream)
        # test double start
        stream_interaction.start(ms_stream)
        while stream_interaction.circular_queue.head != -1:
            showm.render()
            time.sleep(0.01)
        stream_interaction.stop()
        # double stop test
        npt.assert_equal(stream_interaction.stop(), False)
        stream.stop()
        stream.cleanup()
        stream_interaction.cleanup()

    test(True, 16, True)
    if PY_VERSION_8:
        test(False, 16, True)


def test_time_interval_threading():
    def callback(arr):
        arr += [len(arr)]

    arr = []
    interval_timer = tools.IntervalTimerThreading(0.5, callback, arr)
    interval_timer.start()
    time.sleep(2)
    old_len = len(arr)
    assert len(arr) > 0
    interval_timer.stop()
    time.sleep(1)
    old_len = len(arr)
    time.sleep(2)
    # check if the stop method worked
    assert len(arr) == old_len


def test_time_interval_async(loop):
    def callback(arr):
        arr += [len(arr)]

    async def main():
        arr = []
        interval_timer = tools.IntervalTimer(0.5, callback, arr)
        interval_timer.start()
        await asyncio.sleep(2)
        interval_timer.stop()
        assert len(arr) > 0
        old_len0 = len(arr)
        await asyncio.sleep(10)
        old_len1 = len(arr)
        await asyncio.sleep(2)
        print(old_len0, old_len1, len(arr))
        # check if the stop method worked
        assert len(arr) == old_len0

    loop.run_until_complete(main())


def test_multidimensional_buffer():
    def test(use_raw_array=False):
        # creates a raw_array
        max_size = 3
        dimension = 4
        if use_raw_array:
            m_buffer = tools.RawArrayMultiDimensionalBuffer(
                max_size=max_size, dimension=dimension
            )
        else:
            m_buffer = tools.SharedMemMultiDimensionalBuffer(
                max_size=max_size, dimension=dimension
            )
        m_buffer.buffer = np.arange((max_size + 1) * dimension).astype('d')
        m_buffer[1] = np.array([0.2, 0.3, 0.4, 0.5])
        assert len(m_buffer[0]) == dimension
        if not use_raw_array:
            # in OSx this number can change due the minimum shared memory
            #  block size
            max_size = m_buffer.max_size

        assert len(m_buffer[max_size]) == 4
        npt.assert_equal(np.array([0.2, 0.3, 0.4, 0.5]), m_buffer[1])
        npt.assert_equal(np.array([8.0, 9, 10, 11]), m_buffer[2])
        m_buffer.cleanup()

    test(True)
    if PY_VERSION_8:
        test(False)

    def test_comm(use_raw_array=True):
        # test the communication between two MultiDimensionalBuffers
        max_size = 3
        dimension = 4
        if use_raw_array:
            m_buffer_org = tools.RawArrayMultiDimensionalBuffer(
                max_size=max_size, dimension=dimension
            )
            m_buffer_0 = tools.RawArrayMultiDimensionalBuffer(
                max_size, dimension, buffer=m_buffer_org.buffer
            )
        else:
            m_buffer_org = tools.SharedMemMultiDimensionalBuffer(
                max_size=max_size, dimension=dimension
            )
            m_buffer_0 = tools.SharedMemMultiDimensionalBuffer(
                max_size, dimension, buffer_name=m_buffer_org.buffer_name
            )

        m_buffer_0[1] = np.array([0.2, 0.3, 0.4, 0.5])
        if not use_raw_array:
            # in OSx this number can change due the minimum shared memory
            #  block size
            max_size = m_buffer_org.max_size
        assert len(m_buffer_org[0]) == len(m_buffer_0[0])
        assert len(m_buffer_org[max_size]) == 4
        # check values
        npt.assert_equal(np.array([0.2, 0.3, 0.4, 0.5]), m_buffer_org[1])
        # check if the correct max_size was recovered
        assert m_buffer_org.max_size == m_buffer_0.max_size
        assert m_buffer_org.dimension == m_buffer_0.dimension
        m_buffer_0.cleanup()
        m_buffer_org.cleanup()

    test_comm(True)
    if PY_VERSION_8:
        test(False)


def test_circular_queue():
    def test(use_raw_array=True):
        max_size = 3
        dimension = 4
        if use_raw_array:
            queue = tools.ArrayCircularQueue(max_size=max_size, dimension=dimension)
        else:
            queue = tools.SharedMemCircularQueue(max_size=max_size, dimension=dimension)
        # init as empty queue
        assert queue.head == -1 and queue.tail == -1
        assert queue.dequeue() is None
        arr = np.array([1.0, 2, 3, 4])
        ok = queue.enqueue(arr)
        assert ok
        ok = queue.enqueue(arr + 1)
        ok = queue.enqueue(arr + 2)
        assert ok
        # the circular queue must be full (size 3)
        ok = queue.enqueue(arr + 3)
        assert not ok
        assert queue.head == 0 and queue.tail == 2
        arr_recovered = queue.dequeue()
        # it's a FIFO
        npt.assert_equal(arr, arr_recovered)
        assert queue.head == 1 and queue.tail == 2
        arr_recovered = queue.dequeue()
        npt.assert_equal(arr + 1, arr_recovered)
        assert queue.head == 2 and queue.tail == 2
        arr_recovered = queue.dequeue()
        npt.assert_equal(arr + 2, arr_recovered)
        assert queue.head == -1 and queue.tail == -1
        arr_recovered = queue.dequeue()
        assert arr_recovered is None
        queue.cleanup()

    def test_comm(use_raw_array=True):
        max_size = 3
        dimension = 4

        if use_raw_array:
            queue = tools.ArrayCircularQueue(max_size=max_size, dimension=dimension)
            queue_sh = tools.ArrayCircularQueue(
                max_size=max_size,
                dimension=dimension,
                head_tail_buffer=queue.head_tail_buffer,
                buffer=queue.buffer.buffer,
            )
        else:
            queue = tools.SharedMemCircularQueue(max_size=max_size, dimension=dimension)
            queue_sh = tools.SharedMemCircularQueue(
                max_size=max_size,
                dimension=dimension,
                head_tail_buffer_name=queue.head_tail_buffer_name,
                buffer_name=queue.buffer.buffer_name,
            )
        # init as empty queue
        assert queue_sh.buffer.max_size == queue.buffer.max_size
        assert queue.head == -1 and queue.tail == -1
        assert queue.dequeue() is None
        arr = np.array([1.0, 2, 3, 4])
        ok = queue.enqueue(arr)
        assert ok
        queue_sh.cleanup()
        queue.cleanup()

    test(True)
    test_comm(True)
    if PY_VERSION_8:
        test(False)
        test_comm(False)


def test_queue_and_webserver():
    """Check if the correct
    envent ids and the data are stored in the
    correct positions
    """
    max_size = 3
    dimension = _CQUEUE.dimension
    use_raw_array = True

    # if the weel info has been stored correctly in the circular queue
    if use_raw_array:
        queue = tools.ArrayCircularQueue(max_size=max_size, dimension=dimension)
    else:
        queue = tools.SharedMemCircularQueue(max_size=max_size, dimension=dimension)
    set_weel({'deltaY': 0.2, 'timestampInMs': 123}, queue)
    arr_queue = queue.dequeue()
    arr = np.zeros(dimension)
    arr[0] = _CQUEUE.event_ids.mouse_weel
    arr[_CQUEUE.index_info.weel] = 0.2
    arr[_CQUEUE.index_info.user_timestamp] = 123
    npt.assert_equal(arr, arr_queue)

    # if the mouse position has been stored correctly in the circular queue
    data = {'x': -3, 'y': 2.0, 'ctrlKey': 1, 'shiftKey': 0, 'timestampInMs': 123}
    set_mouse(data, queue)
    arr_queue = queue.dequeue()
    arr = np.zeros(dimension)
    arr[0] = _CQUEUE.event_ids.mouse_move
    arr[_CQUEUE.index_info.x] = data['x']
    arr[_CQUEUE.index_info.y] = data['y']
    arr[_CQUEUE.index_info.ctrl] = data['ctrlKey']
    arr[_CQUEUE.index_info.shift] = data['shiftKey']
    arr[_CQUEUE.index_info.user_timestamp] = data['timestampInMs']
    npt.assert_equal(arr, arr_queue)

    data = {
        'mouseButton': 0,
        'on': 1,
        'x': -3,
        'y': 2.0,
        'ctrlKey': 1,
        'shiftKey': 0,
        'timestampInMs': 123,
    }
    set_mouse_click(data, queue)
    arr_queue = queue.dequeue()
    arr = np.zeros(dimension)
    arr[0] = _CQUEUE.event_ids.left_btn_press
    arr[_CQUEUE.index_info.x] = data['x']
    arr[_CQUEUE.index_info.y] = data['y']
    arr[_CQUEUE.index_info.ctrl] = data['ctrlKey']
    arr[_CQUEUE.index_info.shift] = data['shiftKey']
    arr[_CQUEUE.index_info.user_timestamp] = data['timestampInMs']
    npt.assert_equal(arr, arr_queue)
    queue.cleanup()


def test_webserver():
    def test(use_raw_array):
        width_0 = 100
        height_0 = 200

        centers = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        actors = actor.sphere(centers, colors=colors, radii=0.1)

        scene = window.Scene()
        scene.add(actors)
        showm = window.ShowManager(
            scene,
            reset_camera=False,
            size=(width_0, height_0),
            order_transparent=False,
        )
        stream = FuryStreamClient(showm, use_raw_array=use_raw_array)
        stream_interaction = FuryStreamInteraction(showm, use_raw_array=use_raw_array)

        if use_raw_array:
            web_server_raw_array(
                stream.img_manager.image_buffers,
                stream.img_manager.info_buffer,
                stream_interaction.circular_queue.head_tail_buffer,
                stream_interaction.circular_queue.buffer._buffer,
                8000,
                'localhost',
                True,
                True,
                run_app=False,
            )
        else:
            web_server(
                stream.img_manager.image_buffer_names,
                stream.img_manager.info_buffer_name,
                stream_interaction.circular_queue.head_tail_buffer_name,
                stream_interaction.circular_queue.buffer.buffer_name,
                8000,
                'localhost',
                True,
                True,
                True,
                run_app=False,
            )

        stream.stop()
        stream_interaction.stop()
        stream.cleanup()

    test(True)
    if PY_VERSION_8:
        test(False)


@pytest.mark.skipif(True, reason='Infinite loop. Need to check this test.')
def test_widget():
    if not PY_VERSION_8:
        return
    width_0 = 100
    height_0 = 200

    centers = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    actors = actor.sphere(centers, colors=colors, radii=0.1)
    scene = window.Scene()
    scene.add(actors)
    showm = window.ShowManager(
        scene,
        reset_camera=False,
        size=(width_0, height_0),
        order_transparent=False,
    )
    widget = Widget(showm)
    widget.start()
    time.sleep(2)

    npt.assert_equal(False, check_port_is_available(widget._host, widget._port))

    widget2 = Widget(showm, port=widget._port)
    widget2.display()
    widget2.start()

    widget2.stop()
    widget.stop()
