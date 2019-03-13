"""Testing file unittest."""

import numpy as np

import fury.testing as ft
import numpy.testing as npt


def test_captured_output():
    def foo():
        print('hello world!')

    with ft.captured_output() as (out, _):
        foo()

    npt.assert_equal(out.getvalue().strip(), "hello world!")


def test_assert():
    npt.assert_raises(AssertionError, ft.assert_false, True)
    npt.assert_raises(AssertionError, ft.assert_true, False)
    npt.assert_raises(AssertionError, ft.assert_less, 2, 1)
    npt.assert_raises(AssertionError, ft.assert_less_equal, 2, 1)
    npt.assert_raises(AssertionError, ft.assert_greater, 1, 2)
    npt.assert_raises(AssertionError, ft.assert_greater_equal, 1, 2)
    npt.assert_raises(AssertionError, ft.assert_not_equal, 5, 5)
    npt.assert_raises(AssertionError, ft.assert_operator, 2, 1)

    arr = [np.arange(k) for k in range(2, 12, 3)]
    arr2 = [np.arange(k) for k in range(2, 12, 4)]
    npt.assert_raises(AssertionError, ft.assert_arrays_equal, arr, arr2)


if __name__ == '__main__':
    npt.run_module_suite()
