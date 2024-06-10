"""Function for testing decorator module."""

import numpy.testing as npt

from fury.decorators import doctest_skip_parser, warn_on_args_to_kwargs
from fury.testing import assert_true

HAVE_AMODULE = False
HAVE_BMODULE = True


def test_skipper():
    def f():
        pass

    docstring = """ Header
        >>> something # skip if not HAVE_AMODULE
        >>> something + else
        >>> a = 1 # skip if not HAVE_BMODULE
        >>> something2   # skip if HAVE_AMODULE
        """
    f.__doc__ = docstring
    f2 = doctest_skip_parser(f)
    assert_true(f is f2)
    npt.assert_equal(
        f2.__doc__,
        """ Header
        >>> something # doctest: +SKIP
        >>> something + else
        >>> a = 1
        >>> something2
        """,
    )
    global HAVE_AMODULE, HAVE_BMODULE
    HAVE_AMODULE = True
    HAVE_BMODULE = False
    f.__doc__ = docstring
    f2 = doctest_skip_parser(f)
    assert_true(f is f2)
    npt.assert_equal(
        f2.__doc__,
        """ Header
        >>> something
        >>> something + else
        >>> a = 1 # doctest: +SKIP
        >>> something2   # doctest: +SKIP
        """,
    )
    del HAVE_AMODULE
    f.__doc__ = docstring
    npt.assert_raises(NameError, doctest_skip_parser, f)


def test_warn_on_args_to_kwargs():
    @warn_on_args_to_kwargs()
    def func(a, b, *, c, d=4, e=5):
        return a + b + c + d + e

    npt.assert_equal(func(1, 2, c=3, d=4, e=5), 15)
    npt.assert_equal(func(1, 2, c=3, d=5), 16)
    npt.assert_equal(func(1, 2, c=3), 15)
    with npt.assert_warns(UserWarning):
        npt.assert_equal(func(1, 2, 3, 4, 5), 15)
    with npt.assert_warns(UserWarning):
        npt.assert_equal(func(1, 2, 3, 6), 17)
    with npt.assert_warns(UserWarning):
        npt.assert_equal(func(1, 2, 10), 22)
    with npt.assert_warns(UserWarning):
        npt.assert_equal(func(1, 2, 10, d=10, e=19), 42)
    with npt.assert_warns(UserWarning):
        npt.assert_equal(func(1, 2, 10, e=10), 27)
    with npt.assert_warns(UserWarning):
        npt.assert_equal(func(1, 2, 10, d=10), 28)
    npt.assert_raises(TypeError, func, 1, 2, d=4, e=5)
    npt.assert_raises(TypeError, func, 1, 2, e=10)
    npt.assert_raises(TypeError, func, 1, 2, d=10)
    npt.assert_raises(TypeError, func, 1, 3)
    npt.assert_raises(TypeError, func, 1)


# def test_warn_on_args_to_kwargs_with_versions():
#     @warn_on_args_to_kwargs()
#     def func(a, b, *, c, d=4, e=5):
#         return a + b + c + d + e

#     npt.assert_equal(func(1, 2, c=3, d=4, e=5), 15)
#     npt.assert_equal(func(1, 2, c=3, d=5), 16)
#     npt.assert_equal(func(1, 2, c=3), 15)
#     with npt.assert_warns(UserWarning):
#         npt.assert_equal(func(1, 2, 3, 4, 5), 15)
#     with npt.assert_warns(UserWarning):
#         npt.assert_equal(func(1, 2, 3, 6), 17)
#     with npt.assert_warns(UserWarning):
#         npt.assert_equal(func(1, 2, 10), 22)
