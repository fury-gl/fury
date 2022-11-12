Week 5 - Creating PR for glTF exporter and fixing the loader
============================================================
.. post:: July 19 2022
   :author: Shivam Anand
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- Finalised the glTF export PR `#630`_., adding tutorial, docs, and tests for all functions.
- Added support for exporting lines and dot actors.
- ``primitive.modes`` is now set if the rendering mode is a line, dot, or triangle.
- Working on importing different primitive modes, the current glTF importer can render only triangles.


What is coming up next week?
----------------------------

This week I'll be working on the following:

- Get the PR `#630`_. merged by this week. 
- Loading the animations (simple, morph, and skeletal) data as dictionaries from the glTF model so that it can be sent to the timeline.
- Try different examples on Mohamed's PR (`#626`_.) and try running glTF animations if time permits.


Did you get stuck anywhere?
---------------------------

- There wasn't any model in the Khronos glTF samples repository that uses the ``LINE`` or ``POINT`` modes. So I had to rely on the models that I exported using the glTF exporter.

.. _`#630`: https://github.com/fury-gl/fury/pull/630
.. _`#626`: https://github.com/fury-gl/fury/pull/626