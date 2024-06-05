"""Decorators for FURY tests."""

from functools import wraps
from inspect import signature
from functools import wraps
from inspect import signature
import platform
import re
import sys
from warnings import warn

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


def keyword_only(func):
    """A decorator to enforce keyword-only arguments.

    This decorator is used to enforce that certain arguments of a function
    are passed as keyword arguments. This is useful to prevent users from
    passing arguments in the wrong order.

    Parameters
    ----------
    func : callable
        The function to decorate.

    Returns
    -------
    callable
        The decorated function.

    Examples
    --------
    >>> @keyword_only
    ... def add(*, a, b):
    ...     return a + b
    >>> add(a=1, b=2)
    3
    >>> add(b=2, a=1, c=3)
    Traceback (most recent call last):
    ...
    TypeError: add() got an unexpected keyword arguments: c
    Usage: add(a=[your_value], b=[your_value])
    Please Provide keyword-only arguments: a=[your_value], b=[your_value]
    >>> add(1, 2)
    Traceback (most recent call last):
    ...
    TypeError: add() takes 0 positional arguments but 2 were given
    Usage: add(a=[your_value], b=[your_value])
    Please Provide keyword-only arguments: a=[your_value], b=[your_value]
    >>> add(a=1)
    Traceback (most recent call last):
    ...
    TypeError: add() missing 1 required keyword-only arguments: b
    Usage: add(a=[your_value], b=[your_value])
    Please Provide keyword-only arguments: a=[your_value], b=[your_value]
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = signature(func)
        params = sig.parameters
        missing_params = [
            arg.name
            for arg in params.values()
            if arg.name not in kwargs and arg.kind == arg.KEYWORD_ONLY
        ]
        params_sample = [
            f"{arg}=[your_value]"
            for arg in params.values()
            if arg.kind == arg.KEYWORD_ONLY
        ]
        params_sample_str = ", ".join(params_sample)
        unexpected_params_list = [arg for arg in kwargs if arg not in params]
        unexpected_params = ", ".join(unexpected_params_list)
        if args:
            raise TypeError(
                (
                    "{}() takes 0 positional arguments but {} were given\n"
                    "Usage: {}({})\n"
                    "Please Provide keyword-only arguments: {}"
                ).format(
                    func.__name__,
                    len(args),
                    func.__name__,
                    params_sample_str,
                    params_sample_str,
                )
            )
        else:
            if unexpected_params:
                raise TypeError(
                    "{}() got an unexpected keyword arguments: {}\n"
                    "Usage: {}({})\n"
                    "Please Provide keyword-only arguments: {}".format(
                        func.__name__,
                        unexpected_params,
                        func.__name__,
                        params_sample_str,
                        params_sample_str,
                    )
                )

            elif missing_params:
                raise TypeError(
                    "{}() missing {} required keyword-only arguments: {}\n"
                    "Usage: {}({})\n"
                    "Please Provide keyword-only arguments: {}".format(
                        func.__name__,
                        len(missing_params),
                        ", ".join(missing_params),
                        func.__name__,
                        params_sample_str,
                        params_sample_str,
                    )
                )
            return func(*args, **kwargs)

    return wrapper
