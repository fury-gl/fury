#!/usr/bin/env python

"""Script to commit the doc build outputs into the github-pages repo.

Use:
  upload_to_gh-pages.py
"""

###############################################################################
# Imports
###############################################################################
import os
from os import chdir as cd
from os.path import join as pjoin
import re
import shutil
from subprocess import PIPE, CalledProcessError, Popen, check_call
import sys

if sys.version_info < (3, 4):
    raise RuntimeError('Python 3.4 and above is required' ' for running this script')
else:
    from pathlib import Path

###############################################################################
# Globals
###############################################################################

docs_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
pkg_name = 'fury'
pkg_path = os.path.realpath(pjoin(docs_dir, '..'))
pages_dir = os.path.realpath(pjoin(docs_dir, 'gh-pages'))
html_dir = os.path.realpath(pjoin(docs_dir, 'build', 'html'))
pdf_dir = os.path.realpath(pjoin(docs_dir, 'build', 'latex'))
pages_repo = 'https://github.com/fury-gl/fury-website.git'


###############################################################################
# Functions
###############################################################################
def sh(cmd):
    """Execute command in a subshell, return status code."""
    return check_call(cmd, shell=True)


def sh2(cmd):
    """Execute command in a subshell, return stdout.

    Stderr is unbuffered from the subshell.x
    """
    p = Popen(cmd, stdout=PIPE, shell=True)
    out = p.communicate()[0]
    retcode = p.returncode
    if retcode:
        print(out.rstrip())
        raise CalledProcessError(retcode, cmd)
    else:
        return out.rstrip()


def sh3(cmd):
    """Execute command in a subshell, return stdout, stderr.

    If anything appears in stderr, print it out to sys.stderr
    """
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = p.communicate()
    retcode = p.returncode
    if retcode:
        raise CalledProcessError(retcode, cmd)
    else:
        return out.rstrip(), err.rstrip()


###############################################################################
# Script starts
###############################################################################
if __name__ == '__main__':
    # Get starting folder
    current_dir = os.getcwd()

    # Load source package
    cd(pkg_path)
    mod = __import__(pkg_name)

    if pjoin(pkg_path, pkg_name).lower() != os.path.dirname(mod.__file__).lower():

        print(pjoin(pkg_path, pkg_name))
        print(mod.__file__)
        raise RuntimeError(
            'You should work with the source and not the ' 'installed package'
        )

    # find the version number
    tag = 'dev'
    if any(t in mod.__version__.lower() for t in ['dev', 'post']):
        tag = mod.__version__

    if len(sys.argv) == 2:
        tag = sys.argv[1]

    intro_msg = """
##############################################
#  Documentation version {}
#
#  using tag '{}'
##############################################""".format(
        mod.__version__, tag
    )

    print(intro_msg)

    if not os.path.exists(html_dir):
        raise RuntimeError(
            'Documentation build folder not found! You should '
            'generate the documentation first via this '
            "command: 'make -C docs html')"
        )

    if not os.path.exists(pages_dir):
        # clone the gh-pages repo if we haven't already.
        sh('git clone {0} {1}'.format(pages_repo, pages_dir))

    # ensure up-to-date before operating
    cd(pages_dir)
    try:
        sh('git checkout gh-pages')
        sh('git pull')
    except BaseException:
        print('\nLooks like gh-pages branch does not exist!')
        print('Do you want to create a new one? (y/n)')
        while 1:
            choice = str(input()).lower()
            if choice == 'y':
                sh('git checkout -b gh-pages')
                sh('rm -rf *')
                sh('git add .')
                sh("git commit -m 'cleaning gh-pages branch'")
                sh('git push origin gh-pages')
                break
            elif choice == 'n':
                print('Please manually create a new gh-pages branch' ' and try again.')
                sys.exit(0)
            else:
                print('Please enter valid choice ..')

    # delete tag version and copy the doc
    dest = os.path.join(pages_dir, tag)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(html_dir, dest)

    try:
        cd(pages_dir)
        status = sh2('LANG=en_US git status | head -1')
        branch = re.match(b'On branch (.*)$', status).group(1)
        if branch != b'gh-pages':
            e = 'On %r, git branch is %r, MUST be "gh-pages"' % (pages_dir, branch)
            raise RuntimeError(e)

        # Add no jekyll file
        if os.path.exists(pjoin(pages_dir, '.nojekyll')):
            Path('.nojekyll').touch()
            sh('git add .nojekyll')

        sh('git add --all {}'.format(tag))

        status = sh2('LANG=en_US git status | tail -1')
        if not re.match(b'nothing to commit', status):
            sh2('git commit -m "Updated doc release: {}"'.format(tag))
        else:
            print('\n! Note: no changes to commit\n')

        print('Most recent commit:')
        sys.stdout.flush()
        sh('git --no-pager log --oneline HEAD~1..')
        # update stable symlink
        # latest_tag = sh2('git describe --exact-match --abbrev=0')
        # os.symlink()
        #  ln -nsf 1.2 latest
    finally:
        cd(current_dir)

    print('')
    print('Now verify the build in: %r' % dest)
    print("If everything looks good, run 'git push' inside doc/gh-pages.")
