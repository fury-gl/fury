import time
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
