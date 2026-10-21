#!/usr/bin/env python
"""Publish an existing documentation build to the website's gh-pages branch."""

import argparse
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys

from packaging.version import Version

REPOSITORY = "https://github.com/fury-gl/fury-website.git"
MANIFEST_URL = "https://fury.gl/_static/versions_switcher.json"
RELEASE_ID = re.compile(
    r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:(?:a|b|rc)(0|[1-9]\d*))?"
)


def _read_build_version(html_dir) -> str:
    """
    Read and validate the built website's embedded publication identity.

    Parameters
    ----------
    html_dir : Path
        Directory containing the completed HTML build.

    Returns
    -------
    str
        Exact build ID, either dev or a canonical release ID.

    Raises
    ------
    OSError
        If the build cannot be read.
    ValueError
        If metadata is missing, ambiguous, unsafe, or targets a test manifest.
    """
    html_dir = Path(html_dir)
    if not html_dir.is_dir():
        raise ValueError(f"HTML build not found at {html_dir}; build the docs first.")
    html = (html_dir / "index.html").read_text(encoding="utf-8")
    metadata = {}
    for field in ("version_match", "json_url"):
        assignment = rf"\bDOCUMENTATION_OPTIONS\s*\.\s*theme_switcher_{field}\s*="
        assignments = list(re.finditer(assignment, html))
        if len(assignments) != 1:
            raise ValueError(f"Expected one theme_switcher_{field}; rebuild the docs.")
        value = re.match(
            r"\s*(['\"])([^'\"\r\n]*)\1[ \t]*(?:;|(?=\r?\n|</script\s*>|$))",
            html[assignments[0].end() :],
        )
        if value is None:
            raise ValueError(f"Invalid theme_switcher_{field}; rebuild the docs.")
        metadata[field] = value[2]
    if metadata["json_url"] != MANIFEST_URL:
        raise ValueError("Build uses a test switcher manifest; rebuild for fury.gl.")
    version = metadata["version_match"]
    if version != "dev" and (
        RELEASE_ID.fullmatch(version) is None or str(Version(version)) != version[1:]
    ):
        raise ValueError(
            f"Unsupported build ID {version!r}; rebuild with FURY_DOCS_VERSION=dev "
            "or an exact vMAJOR.MINOR.PATCH release ID."
        )
    return version


def _remove_readonly(function, path, exception) -> None:
    """
    Retry deletion once after making a read-only entry writable.

    Parameters
    ----------
    function : callable
        Deletion operation that failed.
    path : str
        Entry on which the operation failed.
    exception : Exception
        Exception raised by the operation.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the failure is not a permission error or the retry fails.
    """
    if not isinstance(exception, PermissionError):
        raise exception
    entry = Path(path)
    mode = entry.stat().st_mode
    entry.chmod(mode | stat.S_IWUSR)
    function(path)


def _remove_tree(path) -> None:
    """
    Remove an ordinary directory, including Windows read-only files.

    Parameters
    ----------
    path : Path
        Directory to remove, never a link or reparse point.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the path is a link, junction, or other reparse point.
    OSError
        If deletion fails.
    """
    path = Path(path)
    if (
        path.is_symlink()
        or path.is_junction()
        or getattr(path.lstat(), "st_file_attributes", 0)
        & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise ValueError(f"Refusing to remove link or reparse point: {path}")
    shutil.rmtree(path, onexc=_remove_readonly)


def _update_versions(entries, version) -> tuple[list[dict], str | None]:
    """
    Validate website history and update one build's switcher entry.

    Parameters
    ----------
    entries : list of dict
        Existing website manifest entries.
    version : str
        Validated build identity.

    Returns
    -------
    list of dict
        Updated entries, preserving unrelated fields.
    str or None
        Highest final release's series, or None to retain existing aliases.

    Raises
    ------
    ValueError
        If history is invalid or publication would replace newer stable content.
    """
    if not isinstance(entries, list) or not entries:
        raise ValueError("Website manifest must be a nonempty list.")
    if version != "dev" and (
        RELEASE_ID.fullmatch(version) is None or str(Version(version)) != version[1:]
    ):
        raise ValueError(f"Unsupported release ID: {version!r}")
    updated = []
    seen_versions = set()
    seen_urls = set()
    for entry in entries:
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(field), str) or not entry[field]
            for field in ("name", "version", "url")
        ):
            raise ValueError("Manifest entries require name, version, and URL strings.")
        if "preferred" in entry and not isinstance(entry["preferred"], bool):
            raise ValueError("Manifest preferred flags must be booleans.")
        identity = entry["version"]
        if identity == "dev":
            series = "dev"
        else:
            if (
                RELEASE_ID.fullmatch(identity) is None
                or str(Version(identity)) != identity[1:]
            ):
                raise ValueError(f"Unsupported manifest release ID: {identity!r}")
            parsed = Version(identity)
            series = f"v{parsed.major}.{parsed.minor}.x"
        if entry["url"] != f"https://fury.gl/{series}/":
            raise ValueError(f"Invalid manifest destination: {entry['url']!r}")
        if identity in seen_versions or entry["url"] in seen_urls:
            raise ValueError("Duplicate manifest version ID or series URL.")
        seen_versions.add(identity)
        seen_urls.add(entry["url"])
        updated.append(entry.copy())

    if version == "dev":
        development = next(
            (entry for entry in updated if entry["version"] == "dev"), {}
        )
        development.update(name="dev", version="dev", url="https://fury.gl/dev/")
        development.pop("preferred", None)
        return [development] + [
            entry for entry in updated if entry["version"] != "dev"
        ], None

    parsed = Version(version)
    url = f"https://fury.gl/v{parsed.major}.{parsed.minor}.x/"
    current = next((entry for entry in updated if entry["url"] == url), None)
    if current is not None:
        previous = Version(current["version"])
        if parsed < previous or (parsed.is_prerelease and not previous.is_prerelease):
            raise ValueError(
                f"Refusing to replace {current['version']} with {version} in {url}"
            )
        current.update(name=version, version=version, url=url)
    else:
        updated.append({"name": version, "version": version, "url": url})

    development = [entry for entry in updated if entry["version"] == "dev"]
    releases = sorted(
        (entry for entry in updated if entry["version"] != "dev"),
        key=lambda entry: Version(entry["version"]),
        reverse=True,
    )
    stable = next(
        (entry for entry in releases if not Version(entry["version"]).is_prerelease),
        None,
    )
    for entry in development + releases:
        entry.pop("preferred", None)
        entry["name"] = entry["version"]
        if entry is stable:
            entry["preferred"] = True
            entry["name"] += " (stable)"
    target = None
    if stable is not None:
        parsed_stable = Version(stable["version"])
        target = f"v{parsed_stable.major}.{parsed_stable.minor}.x"
    return development + releases, target


def publish(html_dir, pages_dir, repository, version=None) -> None:
    """
    Publish a completed HTML build and remove staging after a successful push.

    Parameters
    ----------
    html_dir : path-like
        Completed HTML build directory.
    pages_dir : path-like
        Disposable website checkout directory.
    repository : str
        Git URL or local repository with an existing gh-pages branch.
    version : str, optional
        Assertion of the identity already embedded in the HTML build.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the build, manifest, or publication paths are unsafe or invalid.
    OSError
        If file operations fail.
    subprocess.CalledProcessError
        If a Git operation fails.
    RuntimeError
        If publication succeeds but local cleanup fails.
    """
    html_dir = Path(html_dir).resolve()
    build_version = _read_build_version(html_dir)
    if version is not None and version != build_version:
        raise ValueError(
            f"Requested {version!r}, but build identifies {build_version!r}; "
            "rebuild with the intended FURY_DOCS_VERSION instead of relabeling it."
        )
    pages_dir = Path(pages_dir).absolute()
    if (
        pages_dir.is_symlink()
        or pages_dir.is_junction()
        or (
            pages_dir.exists()
            and getattr(pages_dir.lstat(), "st_file_attributes", 0)
            & stat.FILE_ATTRIBUTE_REPARSE_POINT
        )
    ):
        raise ValueError(f"Staging must not be a link or reparse point: {pages_dir}")
    pages_dir = pages_dir.resolve()
    if html_dir.is_relative_to(pages_dir) or pages_dir.is_relative_to(html_dir):
        raise ValueError("HTML input and staging must not contain one another.")
    if pages_dir.exists() and not pages_dir.is_dir():
        raise ValueError(f"Staging is not a directory: {pages_dir}")
    if pages_dir.exists():
        _remove_tree(pages_dir)
    subprocess.run(
        [
            "git",
            "clone",
            "--config",
            "core.symlinks=false",
            "--single-branch",
            "--branch",
            "gh-pages",
            str(repository),
            str(pages_dir),
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
    )
    manifest = pages_dir / "_static" / "versions_switcher.json"
    entries, stable_target = _update_versions(
        json.loads(manifest.read_text(encoding="utf-8")), build_version
    )
    series = "dev"
    if build_version != "dev":
        parsed = Version(build_version)
        series = f"v{parsed.major}.{parsed.minor}.x"
    destination = pages_dir / series
    if (
        destination.is_symlink()
        or destination.is_junction()
        or (
            destination.exists()
            and getattr(destination.lstat(), "st_file_attributes", 0)
            & stat.FILE_ATTRIBUTE_REPARSE_POINT
        )
    ):
        raise ValueError(f"Destination must not be a link: {destination}")
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Destination is not a directory: {destination}")
        _remove_tree(destination)
    shutil.copytree(html_dir, destination)
    manifest.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (pages_dir / ".nojekyll").touch()
    subprocess.run(
        [
            "git",
            "add",
            "--all",
            "--",
            series,
            "_static/versions_switcher.json",
            ".nojekyll",
        ],
        cwd=pages_dir,
        check=True,
        text=True,
    )
    if stable_target is not None:
        for name in ("latest", "stable"):
            alias = pages_dir / name
            if (
                alias.is_dir()
                or alias.is_symlink()
                or alias.is_junction()
                or (
                    alias.exists()
                    and getattr(alias.lstat(), "st_file_attributes", 0)
                    & stat.FILE_ATTRIBUTE_REPARSE_POINT
                )
            ):
                raise ValueError(f"Alias must be an ordinary file: {alias}")
            alias.write_bytes(stable_target.encode("utf-8"))
            blob = subprocess.run(
                ["git", "hash-object", "-w", "--stdin"],
                input=stable_target,
                stdout=subprocess.PIPE,
                cwd=pages_dir,
                check=True,
                text=True,
            ).stdout.strip()
            subprocess.run(
                ["git", "update-index", "--add", "--cacheinfo", "120000", blob, name],
                cwd=pages_dir,
                check=True,
                text=True,
            )
    difference = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=pages_dir,
        check=False,
        text=True,
    )
    if difference.returncode == 1:
        subprocess.run(
            ["git", "commit", "-m", f"Updated doc release: {build_version}"],
            cwd=pages_dir,
            check=True,
            text=True,
        )
    elif difference.returncode != 0:
        difference.check_returncode()
    subprocess.run(
        ["git", "push", "origin", "HEAD:gh-pages"],
        cwd=pages_dir,
        check=True,
        text=True,
    )
    try:
        _remove_tree(pages_dir)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Publication succeeded at https://fury.gl/{series}/, "
            f"but local cleanup failed at {pages_dir}: {error}"
        ) from error
    print(f"Published https://fury.gl/{series}/")
    print(f"Cleanup complete: removed {pages_dir}")


def main(argv=None) -> int:
    """
    Parse upload arguments and report actionable publication failures.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse, defaulting to the process arguments.

    Returns
    -------
    int
        Zero after publication and cleanup, nonzero on failure.
    """
    docs_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version",
        nargs="?",
        help="Assert the build's embedded version (not relabel it)",
    )
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=docs_dir / "build" / "html",
        help="Completed HTML build directory",
    )
    parser.add_argument(
        "--repository", default=REPOSITORY, help="Website Git URL or local repository"
    )
    args = parser.parse_args(argv)
    try:
        publish(args.html_dir, docs_dir / "gh-pages", args.repository, args.version)
    except (OSError, ValueError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f"Upload failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
