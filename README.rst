.. image:: https://img.shields.io/travis/fury-gl/fury.svg
        :target: https://travis-ci.org/fury-gl/fury

.. image:: https://ci.appveyor.com/api/projects/status/9asvp22cf5pkl45l?svg=true
        :target: https://ci.appveyor.com/project/skoudoro/fury-o608g

.. image:: https://img.shields.io/pypi/v/fury.svg
        :target: https://pypi.python.org/pypi/fury

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

**Step 3.** Install fury via::

    pip install .

or::

    pip install -e .

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
