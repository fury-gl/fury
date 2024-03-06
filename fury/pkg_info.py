from __future__ import annotations

from subprocess import run

from packaging.version import Version

try:
    from ._version import __version__
except ImportError:
    __version__ = '0+unknown'

COMMIT_HASH = '$Format:%h$'


def pkg_commit_hash(pkg_path: str | None = None) -> tuple[str, str]:
    """Get short form of commit hash

    In this file is a variable called COMMIT_HASH. This contains a substitution
    pattern that may have been filled by the execution of ``git archive``.

    We get the commit hash from (in order of preference):

    * A substituted value in ``archive_subst_hash``
    * A truncated commit hash value that is part of the local portion of the
      version
    * git's output, if we are in a git repository

    If all these fail, we return a not-found placeholder tuple

    Parameters
    ----------
    pkg_path : str
       directory containing package

    Returns
    -------
    hash_from : str
       Where we got the hash from - description
    hash_str : str
       short form of hash

    """
    if not COMMIT_HASH.startswith('$Format'):  # it has been substituted
        return 'archive substitution', COMMIT_HASH
    ver = Version(__version__)
    if ver.local is not None and ver.local.startswith('g'):
        return 'installation', ver.local[1:8]
    # maybe we are in a repository
    proc = run(
        ('git', 'rev-parse', '--short', 'HEAD'),
        capture_output=True,
        cwd=pkg_path,
    )
    if proc.stdout:
        return 'repository', proc.stdout.decode().strip()
    return '(none found)', '<not found>'
