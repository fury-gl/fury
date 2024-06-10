"""Decorators for FURY tests."""

from functools import wraps
from inspect import signature
import platform
import re
import sys
from warnings import warn

from packaging import version

skip_linux = is_linux = platform.system().lower() == "linux"
skip_osx = is_osx = platform.system().lower() == "darwin"
skip_win = is_win = platform.system().lower() == "windows"
is_py35 = sys.version_info.major == 3 and sys.version_info.minor == 5
SKIP_RE = re.compile(r"(\s*>>>.*?)(\s*)#\s*skip\s+if\s+(.*)$")


def doctest_skip_parser(func):
    """Decorator replaces custom skip test markup in doctests.

    Say a function has a docstring::
    >>> something # skip if not HAVE_AMODULE
    >>> something + else
    >>> something # skip if HAVE_BMODULE
    This decorator will evaluate the expression after ``skip if``.  If this
    evaluates to True, then the comment is replaced by ``# doctest: +SKIP``.
    If False, then the comment is just removed. The expression is evaluated in
    the ``globals`` scope of `func`.
    For example, if the module global ``HAVE_AMODULE`` is False, and module
    global ``HAVE_BMODULE`` is False, the returned function will have
    docstring::
    >>> something # doctest: +SKIP
    >>> something + else
    >>> something

    """
    lines = func.__doc__.split("\n")
    new_lines = []
    for line in lines:
        match = SKIP_RE.match(line)
        if match is None:
            new_lines.append(line)
            continue
        code, space, expr = match.groups()
        if eval(expr, func.__globals__):
            code = code + space + "# doctest: +SKIP"
        new_lines.append(code)
    func.__doc__ = "\n".join(new_lines)
    return func


def warn_on_args_to_kwargs(from_version=None, until_version=None):
    """Decorator to enforce keyword-only arguments.

    This decorator enforces that all arguments after the first one are
    keyword-only arguments. It also checks that all keyword arguments are

    expected by the function.

    Parameters:
    -----------
    from_version: str
        The version of fury from which the function was supported.
    until_version: str
        The version of fury until which the function was supported.

    Returns:
    --------
    decorator: function
        Decorator function.
    """

    def decorator(func):
        """Decorator

        Parameters:
        -----------
        func: function
            Function to be decorated.

        Returns:
        --------
        wrapper: function
            Decorated function.

        Examples:
        ---------
        >>> @warn_on_args_to_kwargs()
        ... def f(a, b, *, c, d=1, e=1):
        ...     return a + b + c + d + e
        >>> f(1, 2, 3, 4, 5)
        15
        >>> f(1, 2, c=3, d=4, e=5)
        15
        >>> f(1, 2, 2, 4, e=5)
        14
        >>> f(1, 2, c=3, d=4)
        11
        >>> f(1, 2, d=3, e=5)
        Traceback (most recent call last):
        ...
        TypeError: f() missing 1 required keyword-only argument: 'c'
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = signature(func)
            params = sig.parameters
            #
            KEYWORD_ONLY_ARGS = [
                arg.name for arg in params.values() if arg.kind == arg.KEYWORD_ONLY
            ]
            POSITIONAL_ARGS = [
                arg.name
                for arg in params.values()
                if arg.kind in (arg.POSITIONAL_OR_KEYWORD, arg.POSITIONAL_ONLY)
            ]

            # Keyword-only arguments that do not have default values and not in kwargs
            missing_kwargs = [
                arg
                for arg in KEYWORD_ONLY_ARGS
                if arg not in kwargs and params[arg].default == params[arg].empty
            ]

            # Keyword-only arguments that have default values
            ARG_DEFAULT = [
                arg
                for arg in KEYWORD_ONLY_ARGS
                if arg not in kwargs and params[arg].default != params[arg].empty
            ]
            func_params_sample = []

            # Create a sample of the function parameters
            for arg in params.values():
                if arg.kind in (arg.POSITIONAL_OR_KEYWORD, arg.POSITIONAL_ONLY):
                    func_params_sample.append(f"{arg.name}_value")
                elif arg.kind == arg.KEYWORD_ONLY:
                    func_params_sample.append(f"{arg.name}='value'")
            func_params_sample = ", ".join(func_params_sample)
            args_kwargs_len = len(args) + len(kwargs)
            params_len = len(params)
            try:
                return func(*args, **kwargs)
            except TypeError:
                # if the version of fury is greater than until_version, an error should
                # be displayed to indicate that this way of calling the function func
                # was supported by from_version until_version but not by the current
                # FURY_VERSION.
                if from_version is not None and until_version is not None:
                    from fury import __version__ as FURY_VERSION

                    if version.parse(FURY_VERSION) > version.parse(until_version):
                        raise RuntimeError(
                            f"Calling the {func.__name__} function in this way "
                            f"was supported from {from_version} up to {until_version}, "
                            f"but not in the current version of FURY {FURY_VERSION}. "
                            f"Here's how you must call the Function {func.__name__}: "
                            f"{func.__name__}({func_params_sample})"
                        ) from None

                if ARG_DEFAULT:
                    missing_kwargs += ARG_DEFAULT
                if missing_kwargs and params_len >= args_kwargs_len:
                    positional_args_len = len(POSITIONAL_ARGS)
                    args_k = list(args[positional_args_len:])
                    args = list(args[:positional_args_len])
                    kwargs.update(dict(zip(missing_kwargs, args_k)))
                    result = func(*args, **kwargs)
                    warn(
                        f"We'll no longer accept the way you call the {func.__name__} "
                        f"function in future versions of FURY.\n"
                        "Here's how to call the Function {}: {}({})".format(
                            func.__name__, func.__name__, func_params_sample
                        ),
                        UserWarning,
                        stacklevel=3,
                    )
                return result

        return wrapper

    return decorator
