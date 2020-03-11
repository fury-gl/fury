===============
Getting Started
===============

Start by importing FURY.

.. code-block:: python

    import numpy as np
    from fury import window, actor, ui, io, utils

To import a model, use :py:func:`.io.load_polydata`. Currently supported formats include OBJ, VTK, FIB, PLY, STL and XML.
Let us include the ``suzanne`` model used by Blender

.. code-block:: python

    suzanne = io.load_polydata('suzanne.obj')
    suzanne = utils.get_polymapper_from_polydata(suzanne)
    suzanne = utils.get_actor_from_polymapper(suzanne)

Set the opacity of the model::

    modelsuzanne.GetProperty().SetOpacity(0.5)

Let's create some random variables for the cylinder parameters

.. code-block:: python

    centers = np.random.rand(2, 3)
    directions = np.random.rand(2, 3)
    heights = np.random.rand(2)
    colors = np.random.rand(2, 3)

Now, we create a cylinder::

    cylinders = actor.cylinder(centers, directions, colors, heights=heights)

Anything that has to be rendered needs to be added to the scene so let's create a :py:class:`.Scene()`::

    scene = window.Scene()

We set the window scene variables e.g. (width, height)::

    showm = window.ShowManager(scene, size=(1024,720), reset_camera=False)
    showm.initialize()

We add a text block to add some information::

    tb = ui.TextBlock2D()
    tb.message = "Hello Fury"

The function :py:meth:`.Scene.add()` is used to add the created objects to the scene to be rendered::

    scene.add(suzanne)
    scene.add(cylinders)
    scene.add(tb)

Start the rendering of the scene::

    showm.start()