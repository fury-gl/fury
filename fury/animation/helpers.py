import numpy as np


def get_previous_timestamp(timestamps, current_time, include_last=False):
    for timestamp in timestamps[::-1] if include_last else timestamps[-2::-1]:
        if timestamp <= current_time:
            return timestamp
    return timestamps[0]


def get_next_timestamp(timestamps, current_time, include_first=False):
    for timestamp in timestamps[::-1] if include_first else timestamps[1:]:
        if timestamp > current_time:
            return timestamp
    return timestamps[-1]


def get_timestamps_from_keyframes(keyframes):
    return np.sort(np.array(list(keyframes)), axis=None)


def get_values_from_keyframes(keyframes):
    return np.asarray([keyframes.get(t).get('value') for t in
                       sorted(keyframes.keys())])


def lerp(v1, v2, t1, t2, t):
    if t1 == t2:
        return v1
    v = v2 - v1
    dt = 0 if t <= t1 else 1 if t >= t2 else (t - t1) / (t2 - t1)
    return dt * v + v1


def get_time_tau(t, t1, t2):
    return 0 if t <= t1 else 1 if t >= t2 else (t - t1) / (t2 - t1)


def get_distances(points):
    distances = []
    for x, y in zip(points, points[1:]):
        distances.append(np.linalg.norm(x - y))
    return distances
