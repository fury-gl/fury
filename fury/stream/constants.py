from collections import namedtuple


# The first element of each list stored in the multidimensional buffer
# from the circular_queue gives the id of the associated vtkEvent.
# id| Event
# 1 | MouseWeelEvent
# 2 | MouseMoveEvent
# 3 | LeftButtonPressEvent
# 4 | LeftButtonReleaseEvent
# 5 | MiddleButtonPressEvent
# 6 | MiddleButtonReleaseEvent
# 7 | RightButtonPressEvent
# 8 | RightButtonReleaseEvent
_event_ids = {
    'mouse_weel': 1,
    'mouse_move': 2,
    'mouse_ids': (3, 4, 5, 6, 7, 8),
    'left_btn_press': 3,
    'left_btn_release': 4,
    'middle_btn_press': 5,
    'middle_btn_release': 6,
    'right_btn_press': 7,
    'right_btn_release': 8,
}
# This immutable object it's used to avoid any kind of
# mistake in assignment of event ids

_CQUEUE_EVENT_IDs = namedtuple(
    'CQUEUE_EVENT_IDS', list(_event_ids.keys())
)(**_event_ids)

# In each circular_queue element we have the following informations
# index info
# 0 | event_id (int)
# 1 | weel value (float)
# 2 | X position (float)
# 3 | Y position (float)
# 4 | ctrl_key state (1 pressed 0 otherwise)
# 5 | shift_key state (1 pressed 0 otherwise)
# 6 | js event timestamp in mileseconds (ufloat)
_index_info = {
    'weel': 1,
    'x': 2,
    'y': 3,
    'ctrl': 4,
    'shift': 5,
    'user_timestamp': 6,
}
_CQUEUE_INDEX_INFO = namedtuple(
    'CQUEUE_INDEX_INFO', list(_index_info.keys())
)(**_index_info)

# dimension it's also a important parameter
# A wrong value can cause a silent error or a segmentation fault

_CQUEUE = namedtuple(
    'CQUEUE', ['dimension', 'event_ids', 'index_info']
)(**{
    'event_ids': _CQUEUE_EVENT_IDs,
    'index_info': _CQUEUE_INDEX_INFO,
    'dimension': 8,
})
