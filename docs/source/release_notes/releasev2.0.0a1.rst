.. _releasev2.0.0a1:

==============================
 Release notes v2.0.0a1
==============================

Quick Overview
--------------

* Pre-release of the v2.0.0 version of the library.
* Switched to `pygfx` for rendering instead of `vtk`.
* Added support for various new actors including `VectorField`, `Slicer`, `Triangle`, `Disk`, `Ellipsoid`, `Line`, `Axes`, and more.
* Introduced new primitives and utilities for actor creation.
* Improved shader and material handling.
* Enhanced documentation and test coverage.
* Fixed various bugs and improved compatibility with existing code.

Details
--------



GitHub stats for 2024/12/11 - 2025/06/19 (tag: v0.12.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 4 authors contributed 146 commits.

* Maharshi Gor
* Manish Reddy Rakasi
* Mohamed Agour
* Serge Koudoro


We closed a total of 103 issues, 45 pull requests and 58 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (45):

* :ghpull:`1006`: test: fix doctest issues
* :ghpull:`1000`: ENH: Test cases and Tutorial for VectorField
* :ghpull:`997`: NF: Surface actor introduced.
* :ghpull:`1004`: BF: Long key press event None fixed
* :ghpull:`1003`: doc: fix doc generation and docstring
* :ghpull:`996`: NF: Volume slicer with affine
* :ghpull:`1002`: build(deps): bump codecov/codecov-action from 5.0.2 to 5.4.3 in the actions group
* :ghpull:`957`: Extra parameters for compatibility - StatefulSurface in Dipy
* :ghpull:`995`: NF: VectorField with arrows and thick lines
* :ghpull:`992`: NF: Vector field actor
* :ghpull:`993`: BF: remove legacy VTK warning handling
* :ghpull:`990`: RF: Added position argument for text actor and name fix.
* :ghpull:`980`: NF: Slicer Actor
* :ghpull:`991`: NF: Adding Triangle actor.
* :ghpull:`988`: NF: Adding disk actor
* :ghpull:`984`: NF: added `ellipsoid` actor
* :ghpull:`989`: CI: integration of numpydoc in precommit
* :ghpull:`983`: Fix `line` actor colors
* :ghpull:`979`: Adding `line` actor
* :ghpull:`981`: NF: Adding the axes actor
* :ghpull:`970`: NF: Adding text actor and its requirements
* :ghpull:`969`: NF: Qt support and window test cases.
* :ghpull:`978`: Relying on `pygfx` shaders for non-shader geometries
* :ghpull:`977`: BF: Fixed package name for jupyter_rfb
* :ghpull:`968`: NF: Adding point actor and all its requirements.
* :ghpull:`976`: fix: shader compatibility issue temp fix.
* :ghpull:`967`: NF: All shape actors based on primitives
* :ghpull:`963`: NF: Adding actor for frustum, TEST: Adding uni tests for frustum actor
* :ghpull:`965`: Changing color range from 0-255 o 0-1
* :ghpull:`962`: NF: actor_from_primitive
* :ghpull:`964`: RF: install wgpu and mesa driver for ubuntu CI's
* :ghpull:`959`: added actor for cylinder
* :ghpull:`960`: NF: shaders and materials modifiable & added smooth/flat shading
* :ghpull:`958`: Adding box actor
* :ghpull:`956`: RF: Examples working properly
* :ghpull:`954`: RF: Enable Test for v2 branch
* :ghpull:`955`: RF: Flatten the window folder to the file
* :ghpull:`946`: NF: Sphere, Geometry & Material
* :ghpull:`953`: RF: Lib introduced to consolidate the pygfx imports
* :ghpull:`950`: Adding utils.py and transform.py needed by prim_actors
* :ghpull:`951`: CI: Remove old script
* :ghpull:`952`: RF: Code Spell Fixed
* :ghpull:`949`: Release 0.12.0 preparation
* :ghpull:`947`: RF: update some settings files
* :ghpull:`948`: CI:  pin vtk<9.4.0

Issues (58):

* :ghissue:`1005`: BF: Limit increased.
* :ghissue:`1006`: test: fix doctest issues
* :ghissue:`1000`: ENH: Test cases and Tutorial for VectorField
* :ghissue:`997`: NF: Surface actor introduced.
* :ghissue:`1004`: BF: Long key press event None fixed
* :ghissue:`1003`: doc: fix doc generation and docstring
* :ghissue:`985`: NF: Adding Image actor.
* :ghissue:`996`: NF: Volume slicer with affine
* :ghissue:`1002`: build(deps): bump codecov/codecov-action from 5.0.2 to 5.4.3 in the actions group
* :ghissue:`957`: Extra parameters for compatibility - StatefulSurface in Dipy
* :ghissue:`995`: NF: VectorField with arrows and thick lines
* :ghissue:`992`: NF: Vector field actor
* :ghissue:`993`: BF: remove legacy VTK warning handling
* :ghissue:`986`: [WIP] NF: Peaks actor initial setup
* :ghissue:`990`: RF: Added position argument for text actor and name fix.
* :ghissue:`980`: NF: Slicer Actor
* :ghissue:`991`: NF: Adding Triangle actor.
* :ghissue:`988`: NF: Adding disk actor
* :ghissue:`984`: NF: added `ellipsoid` actor
* :ghissue:`989`: CI: integration of numpydoc in precommit
* :ghissue:`983`: Fix `line` actor colors
* :ghissue:`982`: Line Actor creates issue if not pass the colors
* :ghissue:`979`: Adding `line` actor
* :ghissue:`981`: NF: Adding the axes actor
* :ghissue:`970`: NF: Adding text actor and its requirements
* :ghissue:`969`: NF: Qt support and window test cases.
* :ghissue:`978`: Relying on `pygfx` shaders for non-shader geometries
* :ghissue:`942`: dipy has a problem calling fury
* :ghissue:`977`: BF: Fixed package name for jupyter_rfb
* :ghissue:`968`: NF: Adding point actor and all its requirements.
* :ghissue:`976`: fix: shader compatibility issue temp fix.
* :ghissue:`973`: Visualization not updating
* :ghissue:`967`: NF: All shape actors based on primitives
* :ghissue:`963`: NF: Adding actor for frustum, TEST: Adding uni tests for frustum actor
* :ghissue:`966`: NF: Adding actor for tetrahedron, TEST: Adding uni tests for tetrahedron actor
* :ghissue:`965`: Changing color range from 0-255 o 0-1
* :ghissue:`962`: NF: actor_from_primitive
* :ghissue:`964`: RF: install wgpu and mesa driver for ubuntu CI's
* :ghissue:`961`: Restored actors visual tests
* :ghissue:`959`: added actor for cylinder
* :ghissue:`960`: NF: shaders and materials modifiable & added smooth/flat shading
* :ghissue:`958`: Adding box actor
* :ghissue:`956`: RF: Examples working properly
* :ghissue:`954`: RF: Enable Test for v2 branch
* :ghissue:`955`: RF: Flatten the window folder to the file
* :ghissue:`946`: NF: Sphere, Geometry & Material
* :ghissue:`953`: RF: Lib introduced to consolidate the pygfx imports
* :ghissue:`950`: Adding utils.py and transform.py needed by prim_actors
* :ghissue:`870`: Issue with Navbar in the documentation site
* :ghissue:`872`: The dev version is not appropriate.
* :ghissue:`951`: CI: Remove old script
* :ghissue:`952`: RF: Code Spell Fixed
* :ghissue:`949`: Release 0.12.0 preparation
* :ghissue:`947`: RF: update some settings files
* :ghissue:`948`: CI:  pin vtk<9.4.0
* :ghissue:`875`: Make PyGLTFLib an optional dependency
* :ghissue:`873`: Make PyGLTFLib an optional dependency
* :ghissue:`941`: Fix: Resolve Documentation Generation HTTP Error
