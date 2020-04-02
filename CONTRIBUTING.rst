============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/fury-gl/fury/issues.

If you are reporting a bug, please include:

* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "feature"
is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

FURY could always use more documentation, whether
as part of the official FURY docs, in docstrings,
or even on the web in blog posts, articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/fury-gl/fury/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `FURY` for local development.

1. Fork the `FURY` repo on GitHub.
2. Clone your fork locally::

    $ git clone https://github.com/your_name_here/fury.git

3. Add a tracking branch which can always have the last version of `FURY`::

    $ git remote add fury-gl https://github.com/fury-gl/fury.git
    $ git fetch fury-gl
    $ git branch fury-gl-master --track fury-gl/master
    $ git checkout fury-gl-master
    $ git pull

4. Create a branch from the last dev version of your tracking branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

5. Install it locally::

    $ pip install --user -e .

6. Now you can make your changes locally::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Install the required packages for running the unittests::

    $ pip install -r requirements/optional.txt
    $ pip install -r requirements/test.txt

8. When you're done making changes, check that your changes pass flake8 and pytest::

    $ flake8 fury
    $ pytest -svv fury

   To get flake8 and pytest, just pip install them into your virtualenv.

9. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 2.7, 3.4, 3.5, 3.6, 3.7 and for PyPy. Check
   https://travis-ci.org/fury-gl/fury/pull_requests
   and make sure that the tests pass for all supported Python versions.

Publishing Releases
--------------------

Checklist before Releasing
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Review the open list of `FURY issues <https://github.com/fury-gl/fury/issues>`_.  Check whether there are
  outstanding issues that can be closed, and whether there are any issues that
  should delay the release.  Label them !

* Check whether there are no build failing on `Travis`.

* Review and update the release notes.  Review and update the :file:`Changelog`
  file.  Get a partial list of contributors with something like::

      git shortlog -nse v0.1.0..

  where ``v0.1.0`` was the last release tag name.

  Then manually go over ``git shortlog v0.1.0..`` to make sure the release notes
  are as complete as possible and that every contributor was recognized.

* Use the opportunity to update the ``.mailmap`` file if there are any duplicate
  authors listed from ``git shortlog -ns``.

* Add any new authors to the ``AUTHORS`` file.

* Check the copyright years in ``docs/source/conf.py`` and ``LICENSE``

* Generate release notes. Go to ``docs/source/ext`` and run ``github_tools.py`` script the following way::

    $ python github_tools.py --tag=v0.1.0 --save --version=0.2.0

  This command will generate a new file named ``release0.2.0.rst`` in ``release_notes`` folder.

* Check the examples and tutorial - we really need an automated check here.

* Make sure all tests pass on your local machine (from the ``<fury root>`` directory)::

    cd ..
    pytest -s --verbose --doctest-modules fury
    cd fury # back to the root directory

* Check the documentation doctests::

    cd docs
    make -C . html
    cd ..

* The release should now be ready.

Doing the release
~~~~~~~~~~~~~~~~~

* Update release-history.rst in the documentation if you have not done so already.
  You may also highlight any additions, improvements, and bug fixes.

* Type git status and check that you are on the master branch with no uncommitted code.

* Now it's time for the source release. Mark the release with an empty commit, just to leave a marker.
  It makes it easier to find the release when skimming through the git history::

    git commit --allow-empty -m "REL: vX.Y.Z"

* Tag the commit::

    git tag -am 'Second public release' vX.Y.Z  # Don't forget the leading v

  This will create a tag named vX.Y.Z. The -a flag (strongly recommended) opens up a text editor where
  you should enter a brief description of the release.

* Verify that the __version__ attribute is correctly updated::

    import fury
    fury.__version__  # should be 'X.Y.Z'

  Incidentally, once you resume development and add the first commit after this tag, __version__ will take
  on a value like X.Y.Z+1.g58ad5f7, where +1 means “1 commit past version X.Y.Z” and 58ad5f7 is the
  first 7 characters of the hash of the current commit. The letter g stands for “git”. This is all managed
  automatically by versioneer and in accordance with the specification in PEP 440.

* Push the new commit and the tag to master::

    git push origin master
    git push origin vX.Y.Z

* Register for a PyPI account and Install twine, a tool for uploading packages to PyPI::

    python3 -m pip install --upgrade twine

* Remove any extraneous files::

    git clean -dfx

  If you happen to have any important files in your project directory that are not committed to git,
  move them first; this will delete them!

* Publish a release on PyPI::

    python setup.py sdist
    python setup.py bdist_wheel
    twine upload dist/*


* Check how everything looks on pypi - the description, the packages.  If
  necessary delete the release and try again if it doesn't look right.

* Set up maintenance / development branches

  If this is this is a full release you need to set up two branches, one for
  further substantial development (often called 'trunk') and another for
  maintenance releases.

  * Branch to maintenance::

      git co -b maint/X.Y.Z


    Push with something like ``git push upstream-rw maint/0.6.x --set-upstream``

  * Start next development series::

      git co main-master


    Next merge the maintenace branch with the "ours" strategy.  This just labels
    the maintenance branch `info.py` edits as seen but discarded, so we can
    merge from maintenance in future without getting spurious merge conflicts::

       git merge -s ours maint/0.6.x

    Push with something like ``git push upstream-rw main-master:master``

  If this is just a maintenance release from ``maint/0.6.x`` or similar, just
  tag and set the version number to - say - ``0.6.2.dev``.

* Push the tag with ``git push upstream-rw 0.6.0``

Other stuff that needs doing for the release
============================================

* Checkout the tagged release, build the html docs and upload them to
  the github pages website::

    make upload

* Announce to the mailing lists.  With fear and trembling.