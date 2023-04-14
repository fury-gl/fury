============
Installation
============

FURY supports Python 3.5+. You can currently still use Python 2.7 but it will soon stop being supported as the Python 2.7 end of life is on December 31st 2019.

Dependencies
------------

The mandatory dependencies are:

- numpy >= 1.7.1
- vtk >= 8.1.0
- scipy >= 0.9
- aiohttp
- pygltflib

The optional dependencies are:

- matplotlib >= 2.0.0
- dipy >= 0.16.0


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

This command will give you important information about FURY's installation. The next step will be to run a :doc:`tutorial <auto_tutorials/index>`.

Running the Tests
-----------------

Let's install all required packages for the running the test

.. code-block:: shell

    pip install -r requirements/default.txt
    pip install -r requirements/test.txt

There are two ways to run FURY tests:

- From the command line. You need to be on the FURY package folder

.. code-block:: shell

    pytest -svv fury

- To run a specific test file

.. code-block:: shell

    pytest -svv fury/tests/test_actor.py

- To run a specific test directory

.. code-block:: shell

    pytest -svv fury/tests

- To run a specific test function

.. code-block:: shell

    pytest -svv -k "test_my_function_name"

Running the Tests Offscreen
---------------------------

FURY is based on VTK which uses OpenGL for all its rendering. For a headless rendering, we recommend to install and use Xvfb software on linux or OSX.
Since Xvfb will require an X server (we also recommend to install XQuartz package on OSX). After Xvfb is installed you have 2 options to run FURY tests:

- First option

.. code-block:: shell

    export DISPLAY=:0
    Xvfb :0 -screen 1920x1080x24 > /dev/null 2>1 &
    pytest -svv fury

- Second option

.. code-block:: shell

    export DISPLAY=:0
    xvfb-run --server-args="-screen 0 1920x1080x24" pytest -svv fury


Populating our Documentation
----------------------------

Folder Structure
~~~~~~~~~~~~~~~~

Let’s start by showcasing the ``docs`` folder structure:

| fury
| ├── docs
| │   ├── build
| │   ├── make.bat
| │   ├── Makefile
| │   ├── Readme.md
| │   ├── upload_to_gh-pages.py
| │   ├── examples
| │   ├── experimental
| │   └── source
| ├── requirements.txt
| ├── fury
| │   ├── actor.py
| │   ├── ...
| │
| │── ...
|
|

In our ``docs`` folder structure above:

- ``source`` is the folder that contains all ``*.rst`` files.
- ``examples`` is the directory where we have all python scripts that describe how to use the library.
- ``experimental`` directory contains experimental Python scripts. The goal is to keep a trace of experimental work.

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1.** Install all required packages for the documentation generation

.. code-block:: shell

    pip install -U -r requirements/default.txt
    pip install -U -r requirements/optional.txt
    pip install -U -r requirements/docs.txt

**Step 2.** Go to the ``docs`` folder and run the following command to generate it (Linux and macOS)

.. code-block:: shell

    make -C . clean && make -C . html

To generate the documentation without running the examples

.. code-block:: shell

    make -C . clean && make -C . html-no-examples

or under Windows

.. code-block:: shell

    make clean
    make html

To generate the documentation without running the examples under Windows

.. code-block:: shell

    make clean
    make html-no-examples


**Step 3.** Congratulations! the ``build`` folder has been generated! Go to ``build/html`` and open with browser ``index.html`` to see your generated documentation.
