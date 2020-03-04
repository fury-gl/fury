===============
Getting Started
===============

Start by importing FURY.

.. code-block:: python

    import numpy as np
    from fury import window, actor, ui, io, utils

To import a model, use ``io.load_polydata()``. Currently supported formats include OBJ, VTK, FIB, PLY, STL and XML.
Let us include the Suzannemodel used by blender

.. code-block:: python

    suzanne = io.load_polydata('models/suzanne.obj')
    suzanne = utils.get_polymapper_from_polydata(suzanne)
    suzanne = utils.get_actor_from_polymapper(suzanne)

Set the Opacity of the model::

    modelsuzanne.GetProperty().SetOpacity(0.5)

Let's create random variables for cylinder parameters

.. code-block:: python

    centres = np.random.rand(2,3)
    directions = np.random.rand(2,3)
    heights = np.random.rand(2)
    colors = np.random.rand(2,3)

Now, we Create a cylinder::

    cylinders = actor.cylinder(centres,directions,colors,heights=heights)

We create a scene. Everything to be rendered is to be added to the scene::

    scene = window.Scene()

We set window scene variable Eg: (width, height)::

    showm = window.ShowManager(scene,size=(1024,720),reset_camera=False)
    showm.initialize()

We add a text block to add ome information::

    tb = ui.TextBlock2D()
    tb.message = "Hello Fury"

The funcition scene.add() is used to add the created objects to the scene to be rendered::

    scene.add(suzanne)
    scene.add(cylinders)
    scene.add(tb)

Start the rendering of the scene::

    showm.start()