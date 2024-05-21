"""Function for testing decorator module."""

import numpy.testing as npt

from fury.decorators import doctest_skip_parser, keyword_only
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


def test_keyword_only():
    @keyword_only
    def f(*, a, b):
        return a + b

    npt.assert_equal(f(a=1, b=2), 3)
    npt.assert_raises(TypeError, f, a=1, b=2, c=2)
    npt.assert_raises(TypeError, f, 1, 2)
    npt.assert_raises(TypeError, f, 1, b=2)
    npt.assert_raises(TypeError, f, a=1, b=2)
    npt.assert_raises(
        TypeError,
        f,
        a=1,
    )
