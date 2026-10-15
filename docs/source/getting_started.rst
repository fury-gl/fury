===============
Getting Started
===============

.. meta::
   :description: A quick-start guide to building your first scientific visualization scene with FURY in Python.

This example builds a scene containing the Suzanne model, two cylinders, and
a text label. Install FURY first using the :doc:`installation` guide.

Import the current scene, actor, UI, and array-based I/O APIs:

.. code-block:: python

    from pathlib import Path

    import numpy as np

    from fury import actor, io, ui, window
    from fury.data import fetch_viz_models

Download the example models once. This requires network access on the first
run; subsequent runs reuse the cached files. The fetcher returns file
information and the model directory:

.. code-block:: python

    _, model_dir = fetch_viz_models()
    path_suzanne = Path(model_dir) / "suzanne.obj"

Read the mesh as NumPy arrays and create a partially transparent surface.
The Suzanne model has triangle faces; a point-only file cannot be used for
this surface example:

.. code-block:: python

    vertices, faces, colors = io.read_mesh(str(path_suzanne))
    if faces is None:
        raise ValueError("The Suzanne example requires a triangle mesh.")
    suzanne = actor.surface(vertices, faces, colors=colors, opacity=0.5)

Create two cylinders with explicit positions, directions, heights, and colors:

.. code-block:: python

    centers = np.array([[2., 0., 0.], [-2., 0., 0.]])
    directions = np.array([[0., 1., 0.], [0., 1., 0.]])
    heights = np.array([1., 1.5])
    colors = np.array([[1., 0., 0.], [0., 0., 1.]])
    cylinders = actor.cylinder(
        centers, directions=directions, colors=colors, height=heights, radii=0.2
    )

Add a dynamically sized text label and all actors to the scene before creating
the rendering window:

.. code-block:: python

    tb = ui.TextBlock2D(
        text="Hello FURY", position=(20, 20), font_size=23, dynamic_bbox=True
    )
    scene = window.Scene()
    scene.add(suzanne)
    scene.add(cylinders)
    scene.add(tb)

Interactive rendering
---------------------

Create the manager using keyword arguments and start the event loop. Close the
window to return to Python:

.. code-block:: python

    showm = window.ShowManager(scene=scene, size=(1024, 720))
    showm.start()

Offscreen alternative
---------------------

Instead of the interactive manager and ``start()`` block above, save the scene
directly to an image:

.. code-block:: python

    window.snapshot(scene=scene, fname="getting_started.png")

Offscreen rendering still requires a working graphics backend. See
:doc:`installation` for graphics requirements and headless testing guidance.
