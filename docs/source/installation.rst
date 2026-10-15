============
Installation
============

.. meta::
   :description: Install FURY, the open-source Python library for scientific visualization and 3D animations, via pip or conda.

FURY requires Python 3.12 or newer. Python 3.12, 3.13, and 3.14 are
supported stable versions, tested in blocking Linux, macOS, and Windows CI.
Python 3.12 support will be removed in an upcoming FURY release.

Dependencies
------------

The mandatory dependencies are:

- NumPy >= 2.0
- SciPy >= 1.13.0
- Pillow >= 10.4.0
- packaging >= 23.0
- lazy_loader >= 0.4
- pygfx >= 0.16.0
- GLFW >= 2.7.0
- aiohttp
- polyxios >= 0.2.0

Optional dependencies are available as extras:

- ``window``: PySide6, jupyter-rfb, and imgui_bundle (on Python 3.12 or newer).
- ``plot``: Matplotlib >= 3.9.
- ``medical``: DIPY and nibabel.
- ``compute``: Numba.
- ``optional``: all of the above, plus OpenCV (opencv-python).

For example, to install the window backends:

.. code-block:: shell

    pip install "fury[window]"


Installation with PyPi
----------------------

In a terminal, issue the following command

.. code-block:: shell

    pip install fury

Installation with Conda
-----------------------

Our conda package is on the Conda-Forge channel. You will need to run the following command

.. code-block:: shell

    conda install -c conda-forge fury

Installation via Source
-----------------------

**Step 1.** Get the latest source by cloning this repo

.. code-block:: shell

    git clone https://github.com/fury-gl/fury.git
    cd fury

**Step 2.** Install requirements

.. code-block:: shell

    pip install -r requirements/default.txt

**Step 3.** Install fury via

.. code-block:: shell

    pip install .

or

.. code-block:: shell

    pip install -e .

**Step 4:** Enjoy!

Test the Installation
---------------------

You can check your installation via this command

.. code-block:: shell

    python -c "from fury import get_info; print(get_info())"

This command will give you important information about FURY's installation. The next step will be to run a :doc:`tutorial <auto_examples/index>`.

Running the Tests
-----------------

From the repository root (the directory containing ``pyproject.toml``), install
the requirements for running the tests:

.. code-block:: shell

    pip install -r requirements/default.txt
    pip install -r requirements/test.txt

Examples, all run from the repository root:

- Run the full test suite:

.. code-block:: shell

    pytest -svv fury

- Run a specific test file:

.. code-block:: shell

    pytest -svv fury/tests/test_window.py

- Run a specific test directory:

.. code-block:: shell

    pytest -svv fury/actor/tests

- Run a specific test function:

.. code-block:: shell

    pytest -svv fury/tests/test_window.py::test_scene_initialization_default

Running the Tests Offscreen
---------------------------

FURY renders with pygfx and WebGPU through WGPU. Offscreen rendering still
requires a working graphics backend; it does not remove the need for compatible
graphics drivers.

For an existing scene, explicitly request an offscreen image with
``window.snapshot(scene=scene, fname="output.png")``, or create an offscreen
manager with ``window.ShowManager(scene=scene, window_type="offscreen")``.
In contrast, setting ``FURY_OFFSCREEN=true`` changes the capture behavior of
``ShowManager.start()`` but does not replace a windowed canvas with an offscreen
canvas.

For full headless tests on Ubuntu, install Mesa Vulkan drivers, Xvfb, and the
GLFW/Qt libraries. See `ci/setup_headless.sh
<https://github.com/fury-gl/fury/blob/master/ci/setup_headless.sh>`_ for the
complete CI package list; that GitHub Actions-specific provisioning script is
not a local installation command. From the repository root, run:

.. code-block:: shell

    QT_QPA_PLATFORM=offscreen FURY_OFFSCREEN=true xvfb-run -a --server-args="-screen 0 1024x768x24" pytest -svv fury

Coding style
------------

FURY uses the standard Python `PEP8 <https://peps.python.org/pep-0008/>`_
style to ensure the readability and consistency across the toolkit. FURY has
adopted an automated style checking framework that uses `pre-commit <https://pre-commit.com/>`_
and `ruff <https://docs.astral.sh/ruff/>`_. Style conformance rules are
specified in the ``ruff.toml`` configuration file. When requesting to push to
FURY, the compliance to the set of rules is automatically checked using a
GitHub Actions workflow.

In order to check the compliance locally, developers should install the
``[style]`` dependencies:

.. code-block:: shell

  pip install -e .[style]

The git hook scripts need to be installed then running:

.. code-block:: shell

  pre-commit install

``pre-commit`` will then run automatically on ``git commit``.

Most text editors can be configured to check the compliance of your code with
the set of rules specified in the ``ruff.toml`` file.

Populating our Documentation
----------------------------

Folder Structure
~~~~~~~~~~~~~~~~

The relevant repository folders are:

.. code-block:: text

    fury/
    ├── docs/
    │   ├── build/                 # Generated documentation output
    │   ├── examples/              # Maintained gallery inputs
    │   ├── examples_revamped/     # Generated/prepared gallery examples
    │   ├── source/                # Documentation prose and configuration
    │   ├── make.bat
    │   ├── Makefile
    │   └── README.md
    ├── requirements/              # Generated dependency requirements
    └── fury/
        ├── actor/                 # Runtime actor package
        └── ...

``docs/examples/`` contains the maintained Python gallery scripts.
``docs/examples/_valid_examples.toml`` selects which inputs are prepared for the
gallery in ``docs/examples_revamped/``; do not edit those generated/prepared
files. Maintain prose and configuration in ``docs/source/``. Documentation
builds write their output to ``docs/build/``.

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1.** From the repository root, install all required packages for
documentation generation:

.. code-block:: shell

    pip install -U -r requirements/default.txt
    pip install -U -r requirements/optional.txt
    pip install -U -r requirements/doc.txt

**Step 2.** Change to the ``docs`` folder:

.. code-block:: shell

    cd docs

Generate the documentation on Linux and macOS:

.. code-block:: shell

    make -C . clean && make -C . html

To generate the documentation without running the examples

.. code-block:: shell

    make -C . clean && make -C . html-no-examples

or under Windows

.. code-block:: shell

    .\make.bat clean
    .\make.bat html

To generate the documentation without running the examples under Windows

.. code-block:: shell

    .\make.bat clean
    .\make.bat html-no-examples


**Step 3.** Congratulations! the ``build`` folder has been generated! Go to ``build/html`` and open with browser ``index.html`` to see your generated documentation.
