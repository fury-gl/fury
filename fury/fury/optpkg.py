"""Routines to support optional packages."""

import importlib
try:
    import pytest
except ImportError:
    have_pytest = False
else:
    have_pytest = True


class TripWireError(AttributeError):
    """Exception if trying to use TripWire object."""


def is_tripwire(obj):
    """Return True if `obj` appears to be a TripWire object.

    Examples
    --------
    >>> is_tripwire(object())
    False
    >>> is_tripwire(TripWire('some message'))
    True

    """
    try:
        obj.any_attribute
    except TripWireError:
        return True
    except Exception:
        return False
    return False


class TripWire(object):
    """Class raising error if used.

    Standard use is to proxy modules that we could not import

    Examples
    --------
    >>> try:
    ...     import silly_module_name
    ... except ImportError:
    ...    silly_module_name = TripWire('We do not have silly_module_name')
    >>> silly_module_name.do_silly_thing('with silly string') #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TripWireError: We do not have silly_module_name

    """

    def __init__(self, msg):
        self._msg = msg

    def __getattr__(self, attr_name):
        """Raise informative error accessing attributes."""
        raise TripWireError(self._msg)

    def __call__(self, *args, **kwargs):
        """Raise informative error while calling."""
        raise TripWireError(self._msg)


def optional_package(name, trip_msg=None):
    """Return package-like thing and module setup for package `name`.

    Parameters
    ----------
    name : str
        package name
    trip_msg : None or str
        message to give when someone tries to use the return package, but we
        could not import it, and have returned a TripWire object instead.
        Default message if None.

    Returns
    -------
    pkg_like : module or ``TripWire`` instance
        If we can import the package, return it.  Otherwise return an object
        raising an error when accessed
    have_pkg : bool
        True if import for package was successful, false otherwise
    module_setup : function
        callable usually set as ``setup_module`` in calling namespace, to allow
        skipping tests.

    Examples
    --------
    Typical use would be something like this at the top of a module using an
    optional package:
    >>> from fury.optpkg import optional_package
    >>> pkg, have_pkg, setup_module = optional_package('not_a_package')
    Of course in this case the package doesn't exist, and so, in the module:
    >>> have_pkg
    False
    and
    >>> pkg.some_function() #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    TripWireError: We need package not_a_package for these functions, but
    ``import not_a_package`` raised an ImportError
    If the module does exist - we get the module
    >>> pkg, _, _ = optional_package('os')
    >>> hasattr(pkg, 'path')
    True
    Or a submodule if that's what we asked for
    >>> subpkg, _, _ = optional_package('os.path')
    >>> hasattr(subpkg, 'dirname')
    True

    """
    try:
        pkg = importlib.import_module(name)
    except ImportError:
        pass
    else:  # import worked
        # top level module
        return pkg, True, lambda: None
    if trip_msg is None:
        trip_msg = ('We need package %s for these functions, but '
                    '``import %s`` raised an ImportError'
                    % (name, name))
    pkg = TripWire(trip_msg)

    def setup_module():
        if have_pytest:
            pytest.mark.skip('No {0} for these tests'.format(name))

    return pkg, False, setup_module
