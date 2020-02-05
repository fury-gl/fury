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

The optional dependencies are:

- matplotlib >= 2.0.0
- dipy >= 0.16.0


Installation with PyPi
----------------------

In a terminal, issue the following command::

    $ pip install fury

Installation with Conda
-----------------------

Our conda package is on the Conda-Forge channel. You will need to run the following command::

    $ conda install -c conda-forge fury

Installation via Source
-----------------------

**Step 1.** Get the latest source by cloning this repo::

    $ git clone https://github.com/fury-gl/fury.git

**Step 2.** Install requirements::

    $ pip install -r requirements/default.txt

**Step 3.** Install fury via::

    $ pip install .

or::

    $ pip install -e .

**Step 4:** Enjoy!

Test the Installation
---------------------

You can check your installation via this command::

    $ python -c "from fury import get_info; print(get_info())"

This command will give you important information about FURY's installation. The next step will be to run a :doc:`tutorial <auto_tutorials/index>`.

Running the Tests
-----------------

There are two ways to run FURY tests:

- From python interpreter::

    $ from fury.tests import test
    $ test()

- From the command line. You need to be on the FURY package folder::

    pytest -svv fury

Running the Tests Offscreen
---------------------------

FURY is based on VTK which uses OpenGL for all its rendering. For a headless rendering, we recommend to install and use Xvfb software on linux or OSX.
Since Xvfb will require an X server (we also recommend to install XQuartz package on OSX). After Xvfb is installed you have 2 options to run FURY tests:

- First option::

    $ export DISPLAY=:0
    $ Xvfb :0 -screen 1920x1080x24 > /dev/null 2>1 &
    $ pytest -svv fury

- Second option::

    $ export DISPLAY=:0
    $ xvfb-run --server-args="-screen 0 1920x1080x24" pytest -svv fury

