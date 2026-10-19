"""
Script to automate the release preparation and execution steps for FURY.

This script performs the following tasks:
1. Checks the latest tag in the specified series.
2. Updates the AUTHORS file with unique authors from git log.
3. Checks and updates the LICENSE year.
4. Updates the release history file with new release notes.
5. Creates a release announcement blog post.
6. Checks for deprecations.
7. Runs tests and builds documentation.
8. Prompts the user to confirm the completion of the release preparation.
"""

import datetime
import os
import platform
import re
import subprocess
import sys

from packaging.version import Version


def run(cmd, *, check=True):
    """
    Run a shell command and print it to the console.

    Parameters
    ----------
    cmd : str
        The command to run.
    check : bool, optional
        If True, raises an exception if the command fails. Default is True.
    """
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=check)


def get_latest_tag_from_series(*, series=None):
    """
    Get the latest tag from the specified series.

    Parameters
    ----------
    series : str, optional
        The series to check for tags (e.g., "0.x", "1.x", "2.x").

    Returns
    -------
    str
        The latest tag in the specified series, or None if no tags are found.
    """
    series = series.lower().strip() if series else None
    if not series or series not in ["0.x", "1.x", "2.x"]:
        raise ValueError("Series must be one of: '0.x', '1.x', '2.x'.")

    # Extract the initial series number
    current_series = int(series.split(".")[0])

    # Loop through each series from the given series down to 0
    for i in range(current_series, -1, -1):
        # Define the pattern to match the tags in the current series
        pattern = rf"^v{i}\.\d+\.\d+([a-zA-Z0-9]+)?(-\S+)?$"

        try:
            # Get the list of tags from the Git repository
            subprocess.run(
                ["git", "fetch", "https://github.com/fury-gl/fury.git", "--tags"],
                check=True,
            )
            tags = subprocess.check_output(["git", "tag"]).decode("utf-8").splitlines()

            # Filter tags that match the pattern
            series_tags = [tag for tag in tags if re.match(pattern, tag)]

            if series_tags:
                # Sort the tags in version order and get the latest one
                series_tags.sort(key=lambda v: Version(v.lstrip("v")))
                return series_tags[-1]

        except subprocess.CalledProcessError:
            continue

    return None


def update_authors(*, file_path="AUTHORS.rst"):
    """
    Update the AUTHORS.rst file with unique authors from git log.

    Parameters
    ----------
    file_path : str, optional
        The path to the AUTHORS.rst file. Default is "AUTHORS.rst".
    """
    print(f"--- Updating {file_path} file... ---")
    # Read the content of the authors.rst file
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Find the index of the "Contributors" line
    contributors_index = None
    for i, line in enumerate(lines):
        if line.strip() == "Contributors":
            contributors_index = (
                i + 2
            )  # +2 to skip the "Contributors" line and the next empty line
            break

    if contributors_index is None:
        raise ValueError("The 'Contributors' line was not found in the file.")

    # Remove any existing lines after the "Contributors" line
    lines = lines[: contributors_index + 1]

    # Get the list of unique authors from git log
    # git_log_cmd = "git log --format='%aN' | sort -u"
    try:
        git = subprocess.Popen(["git", "log", "--format=%aN"], stdout=subprocess.PIPE)
        authors = git.stdout.read().decode("utf-8").split("\n")
        authors = sorted(set(authors))
        authors.remove("")
        authors.remove("dependabot[bot]")
        authors.remove("Your Name")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute git command: {e}") from e

    # Filter out empty strings and format authors with indentation
    authors = [f"* {author}\n" for author in authors if author]

    # Insert the authors after the "Contributors" line
    lines[contributors_index + 1 : contributors_index + 1] = authors

    # Write the updated content back to the file
    with open(file_path, "w", encoding="utf-8", newline="\n") as file:
        file.writelines(lines)

    run("git diff AUTHORS.rst")


def update_release_history(
    *,
    file_path="docs/source/release-history.rst",
    notes_dir="docs/source/release_notes",
):
    """
    Update the release-history.rst file with new release notes.

    Parameters
    ----------
    file_path : str, optional
        The path to the release-history.rst file.
    notes_dir : str, optional
        The directory containing the release notes files.
    """
    print("--- Updating release-history.rst file... ---")
    # Read the content of the release-history.rst file
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Find the index to insert the new version
    insert_index = None
    for i, line in enumerate(lines):
        if ":maxdepth: 1" in line:
            insert_index = (
                i + 2
            )  # +2 to skip the ":maxdepth: 1" line and the next empty line
            break

    if insert_index is None:
        raise ValueError("The 'release' section was not found in the file.")

    # Remove any existing lines after the "Contributors" line
    lines = lines[:insert_index]
    release_notes = [
        f"   release_notes/{note.rstrip('.rst')}\n"
        for note in os.listdir(notes_dir)
        if note.startswith("releasev") and note.endswith(".rst")
    ]

    release_notes.sort(
        key=lambda v: Version(v.strip().split("releasev", 1)[1]),
        reverse=True,
    )
    print(release_notes)
    # Insert the authors after the "Contributors" line
    lines[insert_index:insert_index] = release_notes
    # Write the updated content back to the file
    with open(file_path, "w", encoding="utf-8", newline="\n") as file:
        file.writelines(lines)

    run("git diff docs/source/release-history.rst")


def check_license_year():
    """Check and update the LICENSE file with the current year."""
    print("--- Checking LICENSE year range... ---")
    current_year = datetime.datetime.now().year
    with open("LICENSE", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Copyright (c)" in line:
            if str(current_year) not in line:
                print(f"→ Updating LICENSE year to {current_year}")
                lines[i] = (
                    f"Copyright (c) 2014–{current_year}, FURY - Free Unified Rendering "
                    "in Python. All rights reserved.\n"
                )
                with open("LICENSE", "w", encoding="utf-8", newline="\n") as f_out:
                    f_out.writelines(lines)
                run("git diff LICENSE")
            else:
                print("→ LICENSE year is up to date.")
            break


def create_release_announcement(
    *, release_type, author="skoudoro", name="Serge Koudoro", version="0.11.0"
):
    """
    Create a release announcement blog post.

    Parameters
    ----------
    release_type : {"features", "maintenance"}
        The confirmed release type used in the announcement.
    author : str, optional
        The author of the blog post.
    name : str, optional
        The name of the author.
    version : str, optional
        The version of the release.

    Raises
    ------
    ValueError
        If release_type is not "features" or "maintenance".
    """
    if release_type not in ("features", "maintenance"):
        raise ValueError("release_type must be 'features' or 'maintenance'.")
    release_description = (
        "This release includes new features and improvements."
        if release_type == "features"
        else "This release focuses on maintenance."
    )
    print("--- Creating release announcement blog post... ---")
    author_id = author
    author = input(f"Enter blog post author (e.g., {author}): ").strip()
    if author:
        author_id = author

    author_name = name
    name = input(f"Enter blog post author name (e.g., {name}): ").strip()
    if name:
        author_name = name
    today = datetime.date.today()
    year = today.year
    posts_dir = os.path.join("docs", "source", "posts")
    year_dir = os.path.join(posts_dir, str(year))

    if not os.path.exists(year_dir):
        os.makedirs(year_dir)

    file_name = f"{today.strftime('%Y-%m-%d')}-release-announcement.rst"
    file_path = os.path.join(year_dir, file_name)

    release_title = f"FURY {version} Released"
    underline = "=" * len(release_title)
    release_title = f"{release_title}\n{underline}\n"

    content = f"""
{release_title}

.. post:: {today.strftime("%B %d, %Y")}
   :author: {author_id}
   :tags: fury
   :category: release


The FURY project is happy to announce the release of FURY {version}!
FURY is a free and open source software library for scientific visualization and 3D animations.

You can show your support by `adding a star <https://github.com/fury-gl/fury/stargazers>`_ on FURY github project.

{release_description} The **major highlights** of this release are:

.. include:: ../../release_notes/releasev{version}.rst
    :start-after: --------------
    :end-before: Details

.. note:: The complete release notes are available :ref:`here <releasev{version}>`

**To upgrade or install FURY**

Run the following command in your terminal::

    pip install --upgrade fury

or::

    conda install -c conda-forge fury


**Questions or suggestions?**

For any questions go to http://fury.gl, or send an e-mail to fury@python.org
You can also join our `discord community <https://discord.gg/6btFPPj>`_

We would like to thanks to :ref:`all contributors <community>` for this release:

.. include:: ../../release_notes/releasev{version}.rst
    :start-after: commits.
    :end-before: We closed


On behalf of the :ref:`FURY developers <community>`,

- {author_name}
"""  # noqa

    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

    print(f"→ Created blog post at {file_path}")


def get_new_version(default_version):
    """
    Prompt the user for a new version number.

    Parameters
    ----------
    default_version : str
        The default version to suggest if the user does not provide input.

    Returns
    -------
    str
        The new version number entered by the user, or the default version if no input
        is provided.
    """
    while True:
        new_version = input(f"Enter new version (e.g., {default_version}): ").strip()
        if not new_version:
            return default_version
        if not new_version.startswith("v"):
            print("Version must start with 'v'.")
            continue
        if not re.match(r"^v\d+\.\d+\.\d+[a-zA-Z0-9]*$", new_version):
            print(
                "Version must follow semantic versioning (e.g., v0.13.0 or v.0.13.0a1)."
            )
            continue
        return new_version


def get_release_type(notes_path):
    """
    Suggest a release type from pull requests and request confirmation.

    Parameters
    ----------
    notes_path : str or path-like
        Path to the UTF-8 release notes to review.

    Returns
    -------
    {"features", "maintenance"}
        The release type confirmed by the maintainer.

    Raises
    ------
    OSError
        If the release notes cannot be read.
    UnicodeDecodeError
        If the release notes are not valid UTF-8.
    """
    feature_prs = []
    in_pull_requests = False
    with open(notes_path, encoding="utf-8") as notes:
        for line in notes:
            if re.fullmatch(r"Pull Requests \(\d+\):", line.strip()):
                in_pull_requests = True
                continue
            if not in_pull_requests:
                continue
            if re.fullmatch(r"Issues \(\d+\):", line.strip()):
                break
            match = re.match(r"^\s*\*\s+:(?:ghissue|ghpull):`(\d+)`:\s*(.*)$", line)
            if match and re.match(r"^(?:NF|ENH)\s*:", match[2], re.IGNORECASE):
                feature_prs.append((match[1], match[2]))
        else:
            feature_prs.clear()

    if feature_prs:
        for number, title in feature_prs:
            print(f"* #{number}: {title}")
        print("Suggested release type: features.")
        prompt = "Release type [features/maintenance] (Enter accepts features): "
    else:
        print(
            "No feature-prefixed pull requests found; "
            "review the release notes and choose a release type."
        )
        prompt = "Release type [features/maintenance]: "

    while True:
        answer = input(prompt).strip().lower()
        if not answer and feature_prs:
            return "features"
        if answer in ("features", "maintenance"):
            return answer
        print("Please enter 'features' or 'maintenance'.")


def is_powershell():
    """
    Detect if the current shell is PowerShell or cmd.

    Returns
    -------
    bool
        True if the current shell is PowerShell or cmd, False otherwise.
    """
    return "PSModulePath" in os.environ


def main():
    """Main function to prepare the release."""
    series = input("Enter the FURY series (e.g. 0.x or 2.x): ").strip()
    if os.getcwd().endswith("fury"):
        _ = os.getcwd()
    else:
        print("Error: run this from the 'fury' directory", file=sys.stderr)
        sys.exit(1)

    # 1. Find last tag and show changes
    last = get_latest_tag_from_series(series=series)
    print(f"Last tag: {last}")
    print("--- Update .mailmap file ---")
    input(
        "please, check duplicated and unknown names, then update the .mailmap file."
        " okay? Press Enter to continue..."
    )
    run(f"git shortlog -nse {last}..HEAD")

    # 2. Update AUTHORS file
    update_authors()

    # 3. Check LICENSE year
    check_license_year()

    # 4. Generate release notes
    last_version = Version(last.lstrip("v"))
    major, minor, patch = last_version.release
    proposed_version = f"v{major}.{minor + 1}.0"

    new_version_v = get_new_version(proposed_version)
    new_version = new_version_v.lstrip("v")
    notes_path = os.path.join(
        "docs", "source", "release_notes", f"releasev{new_version}.rst"
    )
    print(f"--- Generating release notes for {new_version_v} ---")
    run(
        f"python docs/source/ext/github_tools.py --tag={last} "
        f"--save --version={new_version}"
    )
    print("→ Release notes created")
    print(f"Please, go to {notes_path} and update 'Quick overview' section.")
    input("Press Enter when done...")

    # 5. Remind to update release-history.rst
    update_release_history()

    # 6. Prepare blog post
    release_type = get_release_type(notes_path)
    create_release_announcement(version=new_version, release_type=release_type)

    # 7.  Check deprecations
    print("--- Deprecation cycles ---")
    while True:
        answer = input(
            "Have you checked deprecated functions/modules [yes/no]: "
        ).strip()
        if answer.lower() in ["yes", "y"]:
            break
        elif answer.lower() in ["no", "n"]:
            print("Please, check deprecated functions/modules before proceeding.")
            continue
        else:
            print("Invalid input. Please answer with 'yes' or 'no'.")

    # 7. Run tests and build docs
    while True:
        os.environ["FURY_OFFSCREEN"] = "1"
        run("pytest -c pyproject.toml -svv --doctest-modules fury")

        # Build docs using directory handling instead of shell chaining
        original_dir = os.getcwd()
        try:
            os.chdir("docs")
            if platform.system().lower() == "windows":
                run(".\\make.bat clean")
                run(".\\make.bat html")
            else:
                run("make clean")
                run("make -C . html")
            print(
                "Review the documentation at http://localhost:8000. "
                "Press Ctrl+C to stop the preview and continue."
            )
            try:
                if platform.system().lower() == "windows":
                    run(".\\make.bat serve")
                else:
                    run("make serve")
            except KeyboardInterrupt:
                print("Documentation preview stopped.")
        finally:
            os.chdir(original_dir)
            os.environ["FURY_OFFSCREEN"] = "0"

        answer = input("Have you checked the generated docs [yes/no]: ").strip()
        if answer.lower() in ["yes", "y"]:
            break
        elif answer.lower() in ["no", "n"]:
            print("Please, check tutorials, docs and figures before proceeding.")
            continue
        else:
            print("Invalid input. Please answer with 'yes' or 'no'.")

    print(
        "✅ Release preparation complete. \n"
        "You can create a PR now. Then, merge it and create a tag."
    )


if __name__ == "__main__":
    main()
