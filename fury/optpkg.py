"""Routines to support optional packages."""

import importlib

from packaging.version import Version

try:
    import pytest
except ImportError:
    have_pytest = False
else:
    have_pytest = True


class TripWireError(AttributeError):
    """
    Exception if trying to use TripWire object.

    This exception is raised when attempting to use a TripWire object,
    which is typically a proxy for a module that could not be imported.
    """


def is_tripwire(obj):
    """
    Return True if `obj` appears to be a TripWire object.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool
        True if `obj` is a TripWire object, False otherwise.

    Examples
    --------
    >>> is_tripwire(object())
    False
    >>> is_tripwire(TripWire('some message'))
    True
    """
    try:
        _ = obj.any_attribute
    except TripWireError:
        return True
    except Exception:
        return False
    return False


class TripWire:
    """
    Class raising error if used.

    Standard use is to proxy modules that we could not import.

    Parameters
    ----------
    msg : str
        Error message to display when the TripWire is triggered.

    Examples
    --------
    >>> try:
    ...     import silly_module_name
    ... except ImportError:
    ...    silly_module_name = TripWire('We do not have silly_module_name')
    >>> msg = 'with silly string'
    >>> silly_module_name.do_silly_thing(msg) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TripWireError: We do not have silly_module_name
    """

    def __init__(self, msg):
        """
        Initialize TripWire with error message.

        Parameters
        ----------
        msg : str
            Error message to display when the TripWire is triggered.
        """
        self._msg = msg

    def __getattr__(self, attr_name):
        """
        Raise informative error accessing attributes.

        Parameters
        ----------
        attr_name : str
            Name of the attribute.

        Returns
        -------
        None
            This method does not return as it always raises an error.

        Raises
        ------
        TripWireError
            Always raises this error with the message provided during initialization.
        """
        raise TripWireError(self._msg)

    def __call__(self, *args, **kwargs):
        """
        Raise informative error while calling.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the call.
        **kwargs : dict
            Keyword arguments passed to the call.

        Returns
        -------
        None
            This method does not return as it always raises an error.

        Raises
        ------
        TripWireError
            Always raises this error with the message provided during initialization.
        """
        raise TripWireError(self._msg)


def optional_package(name, *, trip_msg=None, min_version=None):
    """
    Return package-like thing and module setup for package `name`.

    Parameters
    ----------
    name : str
        Package name.
    trip_msg : None or str
        Message to give when someone tries to use the return package, but we
        could not import it, and have returned a TripWire object instead.
    min_version : None or str
        If not None, require that the imported package be at least this
        version.  If the package has no ``__version__`` attribute, or if the
        version is not parseable, raise an error.

    Returns
    -------
    pkg_like : module or ``TripWire`` instance
        If we can import the package, return it.  Otherwise return an object
        raising an error when accessed.
    have_pkg : bool
        True if import for package was successful, false otherwise.
    module_setup : function
        Callable usually set as ``setup_module`` in calling namespace, to allow
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
    else:
        if not min_version:
            return pkg, True, lambda: None

        current_version = getattr(pkg, "__version__", "0.0.0")
        if Version(current_version) >= Version(min_version):
            return pkg, True, lambda: None

        if trip_msg is None:
            trip_msg = (
                f"We need at least version {min_version} of "
                f"package {name}, but ``import {name}`` "
                f"found version {current_version}."
            )
            if current_version == "0.0.0":
                trip_msg += "Your installation might be incomplete or corrupted."

    if trip_msg is None:
        trip_msg = (
            f"We need package {name} for these functions, but "
            f"``import {name}`` raised an ImportError"
        )
    pkg = TripWire(trip_msg)

    def setup_module():
        """Configure test module to skip tests if package is missing."""
        if have_pytest:
            pytest.mark.skip(f"No {name} for these tests")

    return pkg, False, setup_module
