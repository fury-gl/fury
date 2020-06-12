.. image:: https://travis-ci.com/fury-gl/fury.svg?branch=master
        :target: https://travis-ci.com/fury-gl/fury

.. image:: https://dev.azure.com/fury-gl/fury/_apis/build/status/fury-gl.fury?branchName=master
        :target: https://dev.azure.com/fury-gl/fury/_build/latest?definitionId=1&branchName=master

.. image:: https://img.shields.io/pypi/v/fury.svg
        :target: https://pypi.python.org/pypi/fury

.. image:: https://anaconda.org/conda-forge/fury/badges/version.svg
        :target: https://anaconda.org/conda-forge/fury

.. image:: https://codecov.io/gh/fury-gl/fury/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/fury-gl/fury

.. image:: https://api.codacy.com/project/badge/Grade/922600af9f94445ead5a12423b813576
        :alt: Codacy Badge
        :target: https://app.codacy.com/app/fury-gl/fury?utm_source=github.com&utm_medium=referral&utm_content=fury-gl/fury&utm_campaign=Badge_Grade_Dashboard

FURY - Free Unified Rendering in Python
=======================================


FURY is a software library for scientific visualization in Python

- **Website and Documentation:** https://fury.gl
- **Mailing list:** https://mail.python.org/mailman3/lists/fury.python.org
- **Official source code repo:** https://github.com/fury-gl/fury.git
- **Download releases:** https://pypi.org/project/fury/
- **Issue tracker:** https://github.com/fury-gl/fury/issues
- **Free software:** 3-clause BSD license

Dependencies
============

FURY requires:

- Numpy (>=1.7.1)
- Vtk (>=8.1.0)
- Scipy (>=0.9)

Installation
============

.. code-block::

    pip install fury

Development
===========

1. Installation from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1.** Get the latest source by cloning this repo::

    git clone https://github.com/fury-gl/fury.git

**Step 2.** Install requirements::

    pip install -r requirements/default.txt

**Step 3.** Install fury 

As a `local project installation <https://pip.pypa.io/en/stable/reference/pip_install/#id44>`__ using::

    pip install .

Or as an `"editable" installation <https://pip.pypa.io/en/stable/reference/pip_install/#id44>`__ using::

    pip install -e .

**If you are developing fury you should go with editable installation.**

**Step 4:** Enjoy!

For more information, see also `installation page on fury.gl <https://fury.gl/stable/installation.html>`_

2. Testing
~~~~~~~~~~

After installation, you can install test suite requirements::

    pip install -r requirements/test.txt

And to launch test suite::

    pytest -svv fury

Contribute
==========


We love contributions!

You've discovered a bug or something else you want to change - excellent! Create an `issue <https://github.com/fury-gl/fury/issues/new>`_!

You've worked out a way to fix it – even better! Submit a Pull Request!

Start with the `contributing guide <CONTRIBUTING.rst>`_!
