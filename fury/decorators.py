"""Decorators for FURY tests."""

from functools import wraps
from inspect import signature
import platform
import re
import sys
from warnings import warn

from packaging import version

import fury

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


def warn_on_args_to_kwargs(
    from_version="0.11.0",
    until_version="0.14.0",
):
    """Decorator to enforce keyword-only arguments.

    This decorator enforces that all arguments after the first one are
    keyword-only arguments. It also checks that all keyword arguments are
    expected by the function.

    Parameters
    ----------
    from_version: str, optional
        The version of fury from which the function was supported.
    until_version: str, optional
        The version of fury until which the function was supported.

    Returns
    -------
    decorator: Callable
        Decorator function.

    Examples
    --------
    >>> from fury.decorators import warn_on_args_to_kwargs
    >>> import fury
    >>> @warn_on_args_to_kwargs()
    ... def f(a, b, *, c, d=1, e=1):
    ...     return a + b + c + d + e
    >>> CURRENT_VERSION = fury.__version__
    >>> fury.__version__ = "0.11.0"
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
    >>> fury.__version__ = "0.12.0"
    >>> f(1, 2, 3, 4, e=5)
    Traceback (most recent call last):
    ...
    TypeError: f() takes 2 positional arguments but 4 positional arguments (and 1 keyword-only argument) were given
    >>> fury.__version__ = CURRENT_VERSION
    """  # noqa: E501

    def decorator(func):
        """Decorator function. This function enforces that all arguments after
        the first one are keyword-only arguments. It also checks that all
        keyword arguments are expected by the function.

        Parameters
        ----------
        func: function
            Function to be decorated.

        Returns
        -------
        wrapper: Callable
            Decorated function.
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
            except TypeError as e:
                FURY_CURRENT_VERSION = fury.__version__

                if ARG_DEFAULT:
                    missing_kwargs += ARG_DEFAULT
                if missing_kwargs and params_len >= args_kwargs_len:
                    # if the version of fury is greater than until_version,
                    # an error should be displayed,
                    if version.parse(FURY_CURRENT_VERSION) > version.parse(
                        until_version
                    ):
                        raise TypeError(e) from e

                    positional_args_len = len(POSITIONAL_ARGS)
                    args_k = list(args[positional_args_len:])
                    args = list(args[:positional_args_len])
                    kwargs.update(dict(zip(missing_kwargs, args_k)))
                    result = func(*args, **kwargs)

                    # if from_version is less or equal to fury.__version__ and,
                    # less or equal to until_version
                    # a warning should be displayed.
                    if (
                        version.parse(from_version)
                        <= version.parse(FURY_CURRENT_VERSION)
                        <= version.parse(until_version)
                    ):
                        warn(
                            f"We'll no longer accept the way you call the "
                            f"{func.__name__} function in future versions of FURY.\n\n"
                            "Here's how to call the Function {}: {}({})\n".format(
                                func.__name__, func.__name__, func_params_sample
                            ),
                            UserWarning,
                            stacklevel=3,
                        )

                    # if the current version of fury is less than from_version,
                    # the function should be called without any changes.

                    return result

        return wrapper

    return decorator
