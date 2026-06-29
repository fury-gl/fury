"""
Visibility snapshot tests for every renderable actor.

Each actor is built with real inputs (numpy arrays / values -- no mocks,
patches, or dummy classes), rendered offscreen, and analyzed with
:func:`fury.window.analyze_snapshot`. The helper asserts the actor produces
foreground pixels when visible and nothing when hidden, toggling through the
actor's real ``.visible`` attribute (which is exactly what
:func:`fury.actor.set_group_visibility` does for a group).
"""

import pytest

from fury.actor.tests._helpers import ACTOR_FACTORIES
from fury.testing import assert_visibility


@pytest.mark.parametrize("name", list(ACTOR_FACTORIES))
def test_actor_visibility(name):
    """Every actor renders when visible and renders nothing when hidden."""
    obj = ACTOR_FACTORIES[name]()
    assert_visibility(obj, toggle=lambda v: setattr(obj, "visible", v))
