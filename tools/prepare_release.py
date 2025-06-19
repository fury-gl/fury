"""Script to automate the release preparation and execution steps for FURY.

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


def run(cmd, *, check=True):
    """Run a shell command and print it to the console.

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
    """Get the latest tag from the specified series.

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
        pattern = rf"^v{i}\.\d+.\d+$"

        try:
            # Get the list of tags from the Git repository
            tags = subprocess.check_output(["git", "tag"]).decode("utf-8").splitlines()

            # Filter tags that match the pattern
            series_tags = [tag for tag in tags if re.match(pattern, tag)]

            if series_tags:
                # Sort the tags in version order and get the latest one
                series_tags.sort(key=lambda v: list(map(int, v.lstrip("v").split("."))))
                return series_tags[-1]

        except subprocess.CalledProcessError:
            continue

    return None


def update_authors(*, file_path="AUTHORS.rst"):
    """Update the AUTHORS.rst file with unique authors from git log.

    Parameters
    ----------
    file_path : str, optional
        The path to the AUTHORS.rst file. Default is "AUTHORS.rst".
    """
    print(f"--- Updating {file_path} file... ---")
    # Read the content of the authors.rst file
    with open(file_path, "r") as file:
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
    git_log_cmd = "git log --format='%aN' | sort -u"
    try:
        authors = (
            subprocess.check_output(git_log_cmd, shell=True).decode("utf-8").split("\n")
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute git command: {e}") from e

    # Filter out empty strings and format authors with indentation
    authors = [f"* {author}\n" for author in authors if author]

    # Insert the authors after the "Contributors" line
    lines[contributors_index + 1 : contributors_index + 1] = authors

    # Write the updated content back to the file
    with open(file_path, "w") as file:
        file.writelines(lines)

    run("git diff AUTHORS.rst")


def update_release_history(
    *,
    file_path="docs/source/release-history.rst",
    notes_dir="docs/source/release_notes",
):
    """Update the release-history.rst file with new release notes.

    Parameters
    ----------
    file_path : str, optional
        The path to the release-history.rst file.
    notes_dir : str, optional
        The directory containing the release notes files.
    """
    print("--- Updating release-history.rst file... ---")
    # Read the content of the release-history.rst file
    with open(file_path, "r") as file:
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
        key=lambda v: list(
            map(int, re.search(r"v(\d+\.\d+\.\d+)", v).group(1).split("."))
        ),
        reverse=True,
    )
    print(release_notes)
    # Insert the authors after the "Contributors" line
    lines[insert_index:insert_index] = release_notes
    # Write the updated content back to the file
    with open(file_path, "w") as file:
        file.writelines(lines)

    run("git diff docs/source/release-history.rst")


def check_license_year():
    """Check and update the LICENSE file with the current year."""
    print("--- Checking LICENSE year range... ---")
    current_year = datetime.datetime.now().year
    with open("LICENSE", "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Copyright (c)" in line:
            if str(current_year) not in line:
                print(f"→ Updating LICENSE year to {current_year}")
                lines[i] = (
                    f"Copyright (c) 2014–{current_year}, FURY - Free Unified Rendering "
                    "in Python. All rights reserved.\n"
                )
                with open("LICENSE", "w") as f_out:
                    f_out.writelines(lines)
                run("git diff LICENSE")
            else:
                print("→ LICENSE year is up to date.")
            break


def create_release_announcement(*, author="skoudoro", version="0.11.0"):
    """Create a release announcement blog post.

    Parameters
    ----------
    author : str, optional
        The author of the blog post.
    version : str, optional
        The version of the release.
    """
    print("--- Creating release announcement blog post... ---")
    username = input(f"Enter blog post author (e.g., {author}): ").strip()
    if username:
        author = username
    today = datetime.date.today()
    year = today.year
    posts_dir = os.path.join("docs", "source", "posts")
    year_dir = os.path.join(posts_dir, str(year))

    if not os.path.exists(year_dir):
        os.makedirs(year_dir)

    file_name = f"{today.strftime('%Y-%m-%d')}-release-announcement.rst"
    file_path = os.path.join(year_dir, file_name)

    content = f"""
FURY {version} Released
====================

.. post:: {today.strftime("%B %d, %Y")}
   :author: {author}
   :tags: fury
   :category: release


The FURY project is happy to announce the release of FURY {version}!
FURY is a free and open source software library for scientific visualization and 3D animations.

You can show your support by `adding a star <https://github.com/fury-gl/fury/stargazers>`_ on FURY github project.

This Release is mainly a maintenance release. The **major highlights** of this release are:

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
We can also join our `discord community <https://discord.gg/6btFPPj>`_

We would like to thanks to :ref:`all contributors <community>` for this release:

.. include:: ../../release_notes/releasev{version}.rst
    :start-after: commits.
    :end-before: We closed


On behalf of the :ref:`FURY developers <community>`,

Serge K.
"""  # noqa

    with open(file_path, "w") as f:
        f.write(content)

    print(f"→ Created blog post at {file_path}")


def get_new_version(default_version):
    """Prompt the user for a new version number.

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
    # print("--- Update .mailmap file ---")
    # input("please, check duplicated and unknown names, then update the .mailmap file."
    #       " okay? Press Enter to continue...")
    # run(f"git shortlog -nse {last}..HEAD")

    # # 2. Update AUTHORS file
    # update_authors()

    # # 3. Check LICENSE year
    # check_license_year()

    # # 4. Generate release notes
    last_version_parts = last.lstrip("v").split(".")
    major, minor, patch = map(int, last_version_parts)
    proposed_version = f"v{major}.{minor + 1}.0"

    new_version_v = get_new_version(proposed_version)
    new_version = new_version_v.lstrip("v")
    # print(f"--- Generating release notes for {new_version_v} ---")
    # run(f"python docs/source/ext/github_tools.py --tag={last} "
    #     f"--save --version={new_version}")
    # print("→ Release notes created")
    # print(f"Please, go to docs/release_notes/release{new_version_v}.rst and update"
    #       " 'Quick overview' section.")
    # input("Press Enter when done...")

    # 5. Remind to update release-history.rst
    update_release_history()

    # 6. Prepare blog post
    create_release_announcement(version=new_version)

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
        if platform.system().lower() == "windows":
            run("set FURY_OFFSCREEN=1 && pytest -svv --doctest-modules fury")
            run(
                "cd docs && make.bat clean && set FURY_OFFSCREEN=1 && make.bat -C . html && cd .."  # noqa
            )
        else:
            run("FURY_OFFSCREEN=1 pytest -svv --doctest-modules fury")
            run("cd docs && make clean && FURY_OFFSCREEN=1 make -C . html && cd ..")

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
