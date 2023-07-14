.. _releasev0.8.0:

==================================
 Release notes v0.8.0 (2022/01/31)
==================================

Quick Overview
--------------

* New Physically Based Rendering (PBR) added. It includes anisotropic rotation and index of refraction among other material properties.
* New Principled BRDF shader unique to FURY added. BRDF stands for bidirectional reflectance distribution function.
* VTK 9.1.0 defined as minimum version.
* Continuous Integration (CI) platform updated.
* New actors added (Rhombicuboctahedron, Pentagonal Prism).
* New UI layouts added (Vertical and Horizontal).
* New module fury.molecular added.
* New module fury.lib added. Module improved loading speed.
* Demos added and updated.
* Documentation updated.


Details
-------

GitHub stats for 2021/08/03 - 2022/01/28 (tag: v0.7.1)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 12 authors contributed 500 commits.

* Anand Shivam
* Antriksh Misri
* Bruno Messias
* Eleftherios Garyfallidis
* Javier Guaje
* Marc-Alexandre Côté
* Meha Bhalodiya
* Praneeth Shetty
* PrayasJ
* Sajag Swami
* Serge Koudoro
* Shivam Anand


We closed a total of 81 issues, 34 pull requests and 47 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (34):

* :ghpull:`523`: Adding Anisotropy and Clear coat to PBR material
* :ghpull:`536`: Remove VTK_9_PLUS flag
* :ghpull:`535`: [ENH] Add missing shaders block
* :ghpull:`532`: Remove and replace vtkactor from docstring
* :ghpull:`503`: devmessias gsoc posts  part 2: weeks 09, 10 and 11
* :ghpull:`534`: [FIX] remove update_user_matrix from text3d
* :ghpull:`527`: [FIX] Allow sphere actor to use faces/vertices without casting issues. In addition, update versioning system (versioneer).
* :ghpull:`509`: adding `numpy_to_vtk_image_data` method to utility
* :ghpull:`507`: Deprecate and rename label to vector_text
* :ghpull:`524`: [WIP] Add debugging CI Tools
* :ghpull:`521`: Snapshot flipping bug fix
* :ghpull:`520`: Added rotation along the axis in Solar System Animations example
* :ghpull:`518`: Pytest patch
* :ghpull:`519`: Principled material
* :ghpull:`515`: Changing how we do things with our test suite.
* :ghpull:`516`: Adding Rhombicuboctahedron actor
* :ghpull:`514`: [FIX] Radio button and checkbox tests
* :ghpull:`513`: [FIX] Mesa installation
* :ghpull:`506`: update tutorial import
* :ghpull:`504`: Update molecular module import
* :ghpull:`470`: Update the way we import external libraries by using only the necessary modules
* :ghpull:`452`: Molecular module
* :ghpull:`491`: Method to process and load sprite sheets
* :ghpull:`496`: Added GSoC blog posts for remaining weeks
* :ghpull:`498`: Fix disk position outside the slider line
* :ghpull:`488`: Fix material docstrings, improved standard parameters and improved materials application support
* :ghpull:`449`: Add python3.9 for our CI's
* :ghpull:`493`: GSoC blogs 2021
* :ghpull:`474`: Add primitive and actor for pentagonal prism with test
* :ghpull:`362`: Animated Surfaces
* :ghpull:`433`: Peak representation improvements
* :ghpull:`432`: Fine-tuning of the OpenGL state
* :ghpull:`479`: Added Vertical Layout to `layout` module
* :ghpull:`480`: Added Horizontal Layout to `layout` module

Issues (47):

* :ghissue:`523`: Adding Anisotropy and Clear coat to PBR material
* :ghissue:`536`: Remove VTK_9_PLUS flag
* :ghissue:`535`: [ENH] Add missing shaders block
* :ghissue:`532`: Remove and replace vtkactor from docstring
* :ghissue:`503`: devmessias gsoc posts  part 2: weeks 09, 10 and 11
* :ghissue:`534`: [FIX] remove update_user_matrix from text3d
* :ghissue:`526`: Text justification in vtkTextActor3D
* :ghissue:`500`: Adding a utility function to convert a numpy array to vtkImageData
* :ghissue:`527`: [FIX] Allow sphere actor to use faces/vertices without casting issues. In addition, update versioning system (versioneer).
* :ghissue:`400`: Sphere actor does not appear when vertices and faces are used
* :ghissue:`509`: adding `numpy_to_vtk_image_data` method to utility
* :ghissue:`431`: Deprecation warning raised in from `utils.numpy_to_vtk_cells`
* :ghissue:`457`: Improve loading speed using partial imports
* :ghissue:`468`: Remove all vtk calls from tutorials and demos
* :ghissue:`507`: Deprecate and rename label to vector_text
* :ghissue:`524`: [WIP] Add debugging CI Tools
* :ghissue:`521`: Snapshot flipping bug fix
* :ghissue:`467`: Window snapshot inverts the displayed scene
* :ghissue:`520`: Added rotation along the axis in Solar System Animations example
* :ghissue:`505`: want a highlight feature
* :ghissue:`518`: Pytest patch
* :ghissue:`519`: Principled material
* :ghissue:`515`: Changing how we do things with our test suite.
* :ghissue:`512`: Flocking-simulation using boid rules
* :ghissue:`516`: Adding Rhombicuboctahedron actor
* :ghissue:`514`: [FIX] Radio button and checkbox tests
* :ghissue:`513`: [FIX] Mesa installation
* :ghissue:`511`: Flocking-simulation using boid rules
* :ghissue:`506`: update tutorial import
* :ghissue:`504`: Update molecular module import
* :ghissue:`404`: Parametric functions- actor, primitives
* :ghissue:`470`: Update the way we import external libraries by using only the necessary modules
* :ghissue:`452`: Molecular module
* :ghissue:`469`: Mismatch in parameter and docstring in manifest_standard() in material module
* :ghissue:`491`: Method to process and load sprite sheets
* :ghissue:`496`: Added GSoC blog posts for remaining weeks
* :ghissue:`498`: Fix disk position outside the slider line
* :ghissue:`488`: Fix material docstrings, improved standard parameters and improved materials application support
* :ghissue:`449`: Add python3.9 for our CI's
* :ghissue:`493`: GSoC blogs 2021
* :ghissue:`474`: Add primitive and actor for pentagonal prism with test
* :ghissue:`362`: Animated Surfaces
* :ghissue:`324`: Animate a wave function
* :ghissue:`433`: Peak representation improvements
* :ghissue:`432`: Fine-tuning of the OpenGL state
* :ghissue:`479`: Added Vertical Layout to `layout` module
* :ghissue:`480`: Added Horizontal Layout to `layout` module
