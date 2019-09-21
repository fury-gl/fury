"""Function for testing optpkg module."""

import numpy.testing as npt
from fury import get_info
from fury.testing import assert_true, assert_false
from fury.optpkg import is_tripwire, TripWire, TripWireError, optional_package
from types import ModuleType


def test_get_info():
    expected_keys = ['fury_version', 'pkg_path', 'commit_hash', 'sys_version',
                     'sys_executable', 'sys_platform', 'numpy_version',
                     'scipy_version', 'vtk_version']
    info = get_info()
    current_keys = info.keys()
    for ek in expected_keys:
        assert_true(ek in current_keys)
        assert_true(info[ek] not in [None, ''])


def test_is_tripwire():
    assert_false(is_tripwire(object()))
    assert_true(is_tripwire(TripWire('some message')))
    assert_false(is_tripwire(ValueError('some message')))


def test_tripwire():
    # Test tripwire object
    silly_module_name = TripWire('We do not have silly_module_name')
    npt.assert_raises(TripWireError,
                      getattr,
                      silly_module_name,
                      'do_silly_thing')
    npt.assert_raises(TripWireError,
                      silly_module_name)
    # Check AttributeError can be checked too
    try:
        silly_module_name.__wrapped__
    except TripWireError as err:
        assert_true(isinstance(err, AttributeError))
    else:
        raise RuntimeError("No error raised, but expected")


def test_optional_package():
    pkg, have_pkg, _ = optional_package('fake_pkg')
    npt.assert_raises(TripWireError, pkg)
    assert_false(have_pkg)

    pkg, have_pkg, _ = optional_package('os')
    assert_true(isinstance(pkg, ModuleType))
    npt.assert_equal(pkg.__name__, 'os')
    assert_true(have_pkg)
