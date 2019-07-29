============
Installation
============

FURY supports Python 3.5+. You can currently still use Python 2.7 but it will be removed as the Python 2.7 end of life
is on December 2019, 31rst

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

Just use the `pip` command line::

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

It should return all the informations about FURY. After this success, we recommend you to play with all tutorial!

Running the Tests
-----------------

There is two ways to run FURY tests:

- From python interpreter::

    $ from fury.tests import test
    $ test()

- From the command line. You need to be on FURY package folder::

    pytest -svv fury