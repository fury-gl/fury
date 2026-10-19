"""Prepare GitHub release introductions and offline previews from release notes."""

import argparse
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import re
from shutil import which
import subprocess
import sys
from urllib.parse import quote

_TAG = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+(?P<suffix>(?:a|b|rc)[0-9]+)?")
_SOURCE_URL = "https://github.com/fury-gl/fury/blob"


def _validate_tag(tag: str) -> bool:
    """
    Validate a release tag and classify prereleases.

    Parameters
    ----------
    tag : str
        Version tag including the leading ``v``.

    Returns
    -------
    bool
        Whether the tag has an alpha, beta, or release-candidate suffix.

    Raises
    ------
    ValueError
        If the tag does not follow the supported release convention.
    """
    match = _TAG.fullmatch(tag)
    if match is None:
        raise ValueError(
            f"unsupported release tag {tag!r}; expected vX.Y.Z with optional aN/bN/rcN"
        )
    return match["suffix"] is not None


def _heading(lines: list[str], name: str, path: Path) -> tuple[int, str]:
    """
    Locate a unique RST heading with a valid underline.

    Parameters
    ----------
    lines : list of str
        Release-note lines.
    name : str
        Exact heading text, excluding surrounding whitespace.
    path : Path
        Source path for diagnostics.

    Returns
    -------
    index : int
        Heading line index.
    adornment : str
        Underline character.

    Raises
    ------
    ValueError
        If the heading is missing, duplicated, or malformed.
    """
    indices = [i for i, line in enumerate(lines) if line.strip() == name]
    if len(indices) != 1:
        raise ValueError(f"{path}: expected exactly one {name!r} heading")
    index = indices[0]
    underline = lines[index + 1].strip() if index + 1 < len(lines) else ""
    if len(underline) < len(name) or not re.fullmatch(r"([^\w\s])\1*", underline):
        raise ValueError(f"{path}: malformed underline for {name!r}")
    if index + 2 < len(lines) and lines[index + 2].strip():
        raise ValueError(f"{path}: expected a blank line after {name!r} underline")
    return index, underline[0]


def _resolve_doc_links(overview: str, *, path: Path, root: Path, tag: str) -> str:
    """
    Resolve explicit Sphinx document links to tagged source documents.

    Parameters
    ----------
    overview : str
        RST overview to normalize.
    path : Path
        Release-note source path.
    root : Path
        Repository root.
    tag : str
        Validated release tag.

    Returns
    -------
    str
        RST containing external links instead of document roles.

    Raises
    ------
    ValueError
        If a document link is invalid or a Sphinx role remains unresolved.
    """
    source_root = (root / "docs/source").resolve()
    for role in list(re.finditer(r":doc:`([^`]*)`", overview))[::-1]:
        link = re.fullmatch(r"(.+?)\s*<([^<>]+)>", role[1])
        if link is None or not link[1].strip():
            raise ValueError(
                f"{path}: :doc: links require an explicit label and target"
            )
        label, target = link[1].strip(), link[2].strip()
        if not target or any(char in target for char in "#?\\:"):
            raise ValueError(
                f"{path}: invalid :doc: target {target!r}; no anchors/queries"
            )
        document = (
            source_root / target.lstrip("/")
            if target.startswith("/")
            else path.parent / target
        )
        if not target.endswith(".rst"):
            document = Path(f"{document}.rst")
        document = document.resolve()
        if (
            document.suffix != ".rst"
            or not document.is_relative_to(source_root)
            or not document.is_file()
        ):
            raise ValueError(
                f"{path}: :doc: target {target!r} must exist inside docs/source"
            )
        relative = quote(document.relative_to(source_root).as_posix(), safe="/")
        replacement = f"`{label} <{_SOURCE_URL}/{tag}/docs/source/{relative}>`_"
        overview = overview[: role.start()] + replacement + overview[role.end() :]
    if re.search(r":[\w:+.-]+:`|`[^`\n]+`:[\w:+.-]+:", overview):
        raise ValueError(
            f"{path}: unresolved Sphinx role in Quick Overview; "
            "use explicit :doc: links, ordinary RST links, or inline literals"
        )
    return overview


def prepare_release_notes(
    tag: str,
    *,
    root: Path,
    output_file: Path,
    preview: bool = False,
    previous_tag: str | None = None,
) -> tuple[str, bool]:
    """
    Convert prepared notes to a release introduction or offline preview.

    Parameters
    ----------
    tag : str
        Stable or alpha, beta, or release-candidate tag including ``v``.
    root : Path
        Repository checkout containing the prepared release notes.
    output_file : Path
        Destination for UTF-8 Markdown with LF endings.
    preview : bool, optional
        Include the prepared PR list, contributors, and tag-comparison link.
    previous_tag : str, optional
        Override the comparison baseline recorded in the release notes.

    Returns
    -------
    previous_tag : str
        Selected, validated comparison baseline.
    prerelease : bool
        Whether this release has a prerelease suffix.

    Raises
    ------
    ValueError
        If tags are invalid, notes are missing or malformed, or Pandoc fails.
    OSError
        If the Markdown output cannot be written.

    Notes
    -----
    Preview content comes from the checkout's prepared notes. GitHub supplies
    final PR author mentions and new-contributor annotations during publication.
    Overriding the baseline does not regenerate the prepared PR list.
    """
    prerelease = _validate_tag(tag)
    path = root / "docs/source/release_notes" / f"release{tag}.rst"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"{path}: cannot read release notes: {error}") from error

    titles = [
        line.strip() for line in lines if line.strip().startswith("Release notes ")
    ]
    if titles != [f"Release notes {tag}"]:
        raise ValueError(f"{path}: expected title 'Release notes {tag}'")
    title, _ = _heading(lines, titles[0], path)
    overview_start, overview_style = _heading(lines, "Quick Overview", path)
    details_start, details_style = _heading(lines, "Details", path)
    if not title < overview_start < details_start or overview_style != details_style:
        raise ValueError(f"{path}: malformed Quick Overview/Details section structure")
    overview = "\n".join(lines[overview_start + 2 : details_start]).strip()
    if not overview:
        raise ValueError(f"{path}: Quick Overview must not be empty")

    stats = [
        (i, line)
        for i, line in enumerate(lines)
        if line.startswith("GitHub stats for ")
    ]
    if len(stats) != 1 or stats[0][0] <= details_start + 1:
        raise ValueError(
            f"{path}: expected exactly one GitHub stats baseline in Details"
        )
    baseline = re.fullmatch(r"GitHub stats for [^()]+ \(tag: ([^()]+)\)", stats[0][1])
    if baseline is None:
        raise ValueError(f"{path}: malformed GitHub stats previous-tag metadata")
    recorded_previous_tag = baseline[1]
    for comparison_tag in (recorded_previous_tag, previous_tag):
        if comparison_tag is None:
            continue
        try:
            _validate_tag(comparison_tag)
        except ValueError as error:
            raise ValueError(f"{path}: invalid previous tag: {error}") from error
        if comparison_tag == tag:
            raise ValueError(f"{path}: previous tag must differ from {tag}")
    if previous_tag is None:
        previous_tag = recorded_previous_tag

    overview = _resolve_doc_links(overview, path=path, root=root, tag=tag)
    full_notes_url = f"{_SOURCE_URL}/{tag}/docs/source/release_notes/release{tag}.rst"
    if preview:
        pull_headers = [
            i
            for i, line in enumerate(lines)
            if re.fullmatch(r"Pull Requests \([0-9]+\):", line)
        ]
        issue_headers = [
            i
            for i, line in enumerate(lines)
            if re.fullmatch(r"Issues \([0-9]+\):", line)
        ]
        author_headers = [
            i
            for i, line in enumerate(lines)
            if re.fullmatch(
                r"The following [0-9]+ authors? contributed [0-9]+ commits?\.", line
            )
        ]
        if (
            len(pull_headers) != 1
            or len(issue_headers) != 1
            or len(author_headers) != 1
            or not details_start
            < author_headers[0]
            < pull_headers[0]
            < issue_headers[0]
        ):
            raise ValueError(f"{path}: malformed preview PR/contributor sections")
        pull_start, issue_start = pull_headers[0], issue_headers[0]
        expected_pulls = int(re.search(r"[0-9]+", lines[pull_start])[0])
        pull_requests, count = re.subn(
            r"(?m)^\* :gh(?:issue|pull):`(?P<number>[0-9]+)`:"
            r"[ \t]*(?P<title>[^\n]+)",
            r"* \g<title> (`#\g<number> "
            r"<https://github.com/fury-gl/fury/pull/\g<number>>`_)",
            "\n".join(lines[pull_start + 1 : issue_start]).strip(),
        )
        if count != expected_pulls:
            raise ValueError(
                f"{path}: expected {expected_pulls} prepared PR entries, found {count}"
            )
        contributors = "\n".join(
            line
            for line in lines[author_headers[0] + 1 : pull_start]
            if line.startswith("* ")
        )
        overview = (
            f"Quick Overview\n==============\n\n{overview}\n\n"
            f"`Full release notes <{full_notes_url}>`_\n\n"
            f"What's Changed\n==============\n\n{pull_requests}\n\n"
        )
        if contributors:
            overview += f"Contributors\n============\n\n{contributors}\n\n"
        overview += (
            f"**Full Changelog**: `{previous_tag}...{tag} "
            f"<https://github.com/fury-gl/fury/compare/{previous_tag}...{tag}>`_\n"
        )
    pandoc = which("pandoc")
    if pandoc is None:
        try:
            package = distribution("pypandoc_binary")
        except PackageNotFoundError as error:
            raise ValueError(
                f"{path}: Pandoc not found; install development dependencies "
                'with python -m pip install -e ".[dev]" or install system Pandoc'
            ) from error
        pandoc = next(
            (
                str(package.locate_file(file))
                for file in package.files or ()
                if file.name in ("pandoc", "pandoc.exe")
            ),
            None,
        )
        if pandoc is None:
            raise ValueError(f"{path}: pypandoc_binary does not contain Pandoc")
    command = [pandoc, "--from=rst", "--to=gfm", "--wrap=none", "--fail-if-warnings"]
    if preview:
        command.append("--shift-heading-level-by=1")
    try:
        converted = subprocess.run(
            command,
            input=overview,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
    except OSError as error:
        raise ValueError(f"{path}: cannot run Pandoc: {error}") from error
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"{path}: Pandoc conversion failed: {error.stderr.strip()}"
        ) from error
    if converted.stderr.strip() or not converted.stdout.strip():
        raise ValueError(
            f"{path}: Pandoc produced diagnostics or empty Markdown: "
            f"{converted.stderr.strip()}"
        )
    introduction = converted.stdout.strip() + "\n"
    if not preview:
        introduction = (
            f"## Quick Overview\n\n{introduction}\n"
            f"[Full release notes]({full_notes_url})\n"
        )
    output_file.write_text(introduction, encoding="utf-8", newline="\n")
    return previous_tag, prerelease


def main() -> int:
    """
    Prepare a local introduction or full preview and print release metadata.

    Returns
    -------
    int
        Zero on success, one on preparation failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument(
        "--previous-tag",
        help="Override the recorded comparison tag; prepared PRs are unchanged.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Include prepared PRs and contributors in an offline Markdown preview.",
    )
    args = parser.parse_args()
    try:
        previous_tag, prerelease = prepare_release_notes(
            args.tag,
            root=Path(__file__).resolve().parents[1],
            output_file=args.output_file,
            preview=args.preview,
            previous_tag=args.previous_tag,
        )
    except (ValueError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"previous_tag={previous_tag}")
    print(f"prerelease={str(prerelease).lower()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
