"""Decorators for FURY tests."""
import sys
import re
import platform


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
    This decorator will evaluate the expresssion after ``skip if``.  If this
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
    lines = func.__doc__.split('\n')
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
