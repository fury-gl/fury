Week 2 - Improving Fetcher and Exporting glTF
=============================================

.. post:: June 29 2022
   :author: Shivam Anand
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
This week I worked primarily on the glTF fetcher and exporting scenes to a glTF file. I added tests and docstrings for all functions. I modified the ``fetch_gltf`` function to return the downloaded files. As we planned, the PR for the glTF fetcher was merged this week.

I fixed the color issue of the glTF exporter by manually appending the color data from the actor to the ``polydata``. But there's another issue raised while using actors from ``vtkSource``. The ``utils.get_polydata_triangles(polydata)`` method only works with primitives and it raises an error when used with ``vtkSource`` actors.

Textures and Cameras can now be added to the glTF file. However, it supports baseTexture only. I'll be working on materials support later on.


What is coming up next week?
----------------------------

* Saving all models download link (from the Khronos glTF-Samples-Models repository) to a JSON file, create a separate branch and add the download script.
* Add tests and docstring for PR `#600 <https://github.com/fury-gl/fury/pull/600>`_.
* Create a PR for glTF exporter.

Did you get stuck anywhere?
---------------------------

* I managed to fix colors in polydata by adding them manually, but it raised another issue with indices (triangles) of the actor weren't of the correct shape. We decided to take a look at it later.
* Due to Github's API limit, it raised an error due to limits exceeding. We decided to save a JSON file with all model names and their download links. Then use that to download the model.