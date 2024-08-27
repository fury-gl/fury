
.. role:: raw-html(raw)
   :format: html

.. raw:: html

   <center><a href="https://summerofcode.withgoogle.com/programs/2022/projects/ZZQ6IrHq"><img src="https://developers.google.com/open-source/gsoc/resources/downloads/GSoC-logo-horizontal.svg" alt="gsoc" height="50"/></a></center>

.. raw:: html

   <center>
   <a href="https://summerofcode.withgoogle.com/projects/#6653942668197888"><img src="https://www.python.org/static/community_logos/python-logo.png" height="45"/></a>
   <a href="https://fury.gl/latest/community.html"><img src="https://python-gsoc.org/logos/FURY.png" alt="fury" height="45"/></a>
   </center>


Google Summer of Code Final Work Product
========================================

.. post:: January 29 2023
   :author: Shivam Anand
   :tags: google
   :category: gsoc

-  **Name:** Shivam Anand
-  **Organisation:** Python Software Foundation
-  **Sub-Organisation:** FURY
-  **Project:** `FURY - glTF Integration <https://github.com/fury-gl/fury/wiki/Google-Summer-of-Code-2022>`__


Proposed Objectives
-------------------

-  Ability to load glTF models

   -  Should be able to load static glTF models
   -  Should be able to add model to the scene.

-  Exporting scene data as glTF file
-  Materials & Textures

   -  Textures
   -  PBRs

-  Animations

   -  Simple Actor animations
   -  Skinning
   -  Morphing

-  Stretch Goals

   -  Ability to load ``.glb`` files

Objectives Completed
--------------------

Loading Static glTF models
**************************

A glTF file is a JSON like file format containing required data for 3D scenes. VTK has two built-in glTF loaders. However, they lack ability to animate and apply materials. Added methods to load binary
data and create actors from them. These actors can be added directly
to the scene. The glTF class reads texture data from either
``base64`` encoded string or from the image file, and maps the
texture to the actor using the given UV data. It is capable of doing
the following:

-  Loading both ``gltf`` and ``glb`` files. Get actors from the
   model.
-  Applying textures and colors from materials.
-  Setting cameras if the model contains multiple cameras.
-  Apply normals (for a smoother surface).

.. figure:: https://user-images.githubusercontent.com/74976752/174492510-b9f10816-3058-4a7b-a260-0627406354ba.png
   :alt: image


*Pull Requests:*

-  **Importing glTF files: (Merged)**
   https://github.com/fury-gl/fury/pull/600
-  **Loading glTF Demo (with textures) (Merged):**
   https://github.com/fury-gl/fury/pull/600


Exporting Scene as a glTF
*************************


The FURY scene can contain multiple objects such as actors, cameras,
textures, etc. We need to get the primitive information (such as
Vertices, Triangles, UVs, Normals, etc.) from these objects and store
them into a ``.bin`` file. Added methods that export these
information to a ``.gltf`` or ``.glb`` file format.

*Pull Requests:*

-  **Exporting scene as glTF: (Merged)**
   https://github.com/fury-gl/fury/pull/630
-  **Exporting scene as glTF Tutorial: (Merged)**
   https://github.com/fury-gl/fury/pull/630


Simple Actor Animations
***********************

Added simple actor animations (translation, rotation & scale of
actors) support. The animation data (transformation and timestamp) is
stored in buffers. It converts the binary data to ndarrays and
creates a timleline for each animation in glTF animations. This
timeline contains actors an can be added to the scene. We can animate
the scene by updating timeline inside a timer callback.

.. image:: https://user-images.githubusercontent.com/74976752/217645594-6054ea83-12e5-4868-b6a1-eee5a154bd26.gif
   :width: 480
   :align: center

*Pull Requests:*

-  **Simple Animations in glTF: (Merged)**
   https://github.com/fury-gl/fury/pull/643
-  **Simple Animations in glTF Tutorial: (Merged)**
   https://github.com/fury-gl/fury/pull/643


Morphing in glTF
****************

glTF allows us to animate meshes using morph targets. A morph target
stores displacements or differences for certain mesh attributes. At
runtime, these differences may be added to the original mesh, with
different weights, to animate parts of the mesh. Added methods to
extract this information, update the timeline and apply morphing to
each actor in the scene.

.. image:: https://user-images.githubusercontent.com/74976752/217645485-153ec403-6c87-4282-8907-30d921106b34.gif
   :width: 480
   :align: center

*Pull Requests:*

-  **Morphing support in glTF: (Under Review)**
   https://github.com/fury-gl/fury/pull/700
-  **Morphing in glTF demo: (Under Review)**
   https://github.com/fury-gl/fury/pull/700


Skeletal Animations (Skining)
*****************************

Another way of animating a glTF is by skinning. It allows the
geometry (vertices) of a mesh to be deformed based on the pose of a
skeleton. This is essential in order to give animated geometry. It
combines every parameter of a glTF file. While working with skinning,
we need to keep track of the parent-child hierarchy of
transformations. Vertex Skinning takes full advantage of newly
implemented ``Timeline`` & ``Animation`` modules to track
hierarchical transformation order. Though the current version of the
skinning implementation works with most of the glTF sample modes, It
struggles with models that have multiple actors (e.g. BrainStem). It
can be fixed by using the vertex shader to update the vertices. The
current implementation of skinning supports the following:

-  Multiple animation support
-  Multiple node and multiple actor animation with textures
-  Show or hide bones/skeleton of the model.


.. image:: https://user-images.githubusercontent.com/74976752/217645367-f901c6ed-ca20-40d6-92dd-f1cd8899ac7a.gif
   :width: 480
   :align: center

*Pull Requests:*

-  **Skinning support in glTF: (Under Review)**
   https://github.com/fury-gl/fury/pull/685
-  **Skinning in glTF demo: (Under Review)**
   https://github.com/fury-gl/fury/pull/685

Objectives in Progress
----------------------


PBR and emission materials in glTF
**********************************

The glTF format supports Physically based rendering also. PBR allow
renderers to display objects with a realistic appearance under
different lighting conditions, the shading model has to take the
physical properties of the object surface into account. There are
different representations of these physical material properties. One
that is frequently used is the metallic-roughness-model. We have
various material properties already in FURY, we need to apply it to
glTF models as well.


Skinning for models with no indices
***********************************

The glTF format supports non-indexed geometry (e.g., the ``Fox``
model). We currently do not know how to render the model without
indices. I tried estimating it in this
`branch <https://github.com/xtanion/fury/blob/gltf-indices-fix/fury/gltf.py>`__.
However, It fails to render in skinning.

*Branch URL:*

-  **Rendering glTF with no indices: (in-progress)**
   https://github.com/xtanion/fury/blob/gltf-indices-fix/fury/gltf.py

Other Objectives
----------------


Fetcher for importing glTF files from Khronos-glTF-Samples
**********************************************************

The
`KhronosGroup/gltf-samples <https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/>`__
contain multiple glTF sample models to test a glTF viewer for free.
Implemented new methods in fetcher that can load all of these models
by (using type) asynchronously. The glTF fetcher is capable
of the following:

-  Downloading multiple models asynchronously.
-  Get the path to the downloaded model using it   -  Download any model using the URL of the model.

*Pull Requests:*

-  **Fetching glTF sample models from github: (Merged)**
   https://github.com/fury-gl/fury/pull/602
-  **Fixing github API limit: (Merged)**
   https://github.com/fury-gl/fury/pull/616


Other Pull Requests
*******************

-  **Sphere actor uses repeat_primitive by default**:
   `fury-gl/fury/#533 <https://github.com/fury-gl/fury/pull/533>`__
-  **Cone actor uses repeat primitive by default**:
   `fury-gl/fury/#547 <https://github.com/fury-gl/fury/pull/547>`__
-  **Updated code of viz_network_animated to use fury.utils**:
   `fury-gl/fury/#556 <https://github.com/fury-gl/fury/pull/556>`__
-  **Added simulation for Tesseract**:
   `fury-gl/fury/#559 <https://github.com/fury-gl/fury/pull/559>`__
-  **GLTF actor colors from material**
   `fury-gl/fury/#689 <https://github.com/fury-gl/fury/pull/689>`__


GSoC weekly blogs
*****************

-  My blog posts can be found on the `FURY
   website <https://fury.gl/latest/blog/author/shivam-anand.html>`__
   and the `Python GSoC
   blog <https://blogs.python-gsoc.org/en/xtanions-blog/>`__.

Timeline
--------

+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Date                 | Description                                               |  Blog Post Link                                                                                                                                                                                           |
+======================+===========================================================+===========================================================================================================================================================================================================+
| Week 0 (24-05-2022)  | My journey to GSoC 2022                                   | `FURY <https://fury.gl/latest/posts/2022/2022-05-24-my-journey-to-gsoc-2022-shivam.html>`__ - `Python <https://blogs.python-gsoc.org/en/xtanions-blog/my-journey-to-gsoc-2022-1/>`__                      |
+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 1 (20-06-2022)  | A baic glTF Importer                                      | `FURY <https://fury.gl/latest/posts/2022/2022-06-20-week1-shivam.html>`__ - `Python <https://blogs.python-gsoc.org/en/xtanions-blog/week-1-a-basic-gltf-importer/>`__                                     |
+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 2 (29-06-2022)  | Improving Fetcher and Exporting glTF                      | `FURY <https://fury.gl/latest/posts/2022/2022-06-29-week2-shivam.html>`__ - `Python Blogs <https://blogs.python-gsoc.org/en/xtanions-blog/week-2-improving-fetcher-and-exporting-gltf/>`__                |
+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 3 (04-07-2022)  | Fixing fetcher adding tests and docs                      | `FURY <https://fury.gl/latest/posts/2022/2022-07-04-week3-shivam.html>`__ - `Python <https://blogs.python-gsoc.org/en/xtanions-blog/week-3-fixing-fetcher-adding-tests-and-docs/>`__                      |
+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 4 (12-07-2022)  | Finalizing glTF loader                                    | `FURY <https://fury.gl/latest/posts/2022/2022-07-12-week4-shivam.html>`__ -`Python<https://blogs.python-gsoc.org/en/xtanions-blog/week-4-finalizing-gltf-loader/>`__                                      |
+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 5 (19-07-2022)  | Creating PR for glTF exporter and fixing the loader       | `FURY <https://fury.gl/latest/posts/2022/2022-07-19-week5-shivam.html>`__ - `Python <https://blogs.python-gsoc.org/en/xtanions-blog/week-5-creating-pr-for-gltf-exporter-and-fixing-the-loader/>`__       |
+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 6 (25-07-2022)  | Extracting the animation data                             | `FURY <https://fury.gl/latest/posts/2022/2022-07-25-week-6-shivam.html>`__ - `Python <https://blogs.python-gsoc.org/en/xtanions-blog/week-6-extracting-the-animation-data/>`__                            |
+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 7 (01-08-2022)  | Fixing bugs in animations                                 | `FURY <https://fury.gl/latest/posts/2022/2022-08-01-week-7-shivam.html>`__ - `Python <https://blogs.python-gsoc.org/en/xtanions-blog/week-7-fixing-bugs-in-animations/>`__                                |
+----------------------+-----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 8 (09-08-2022)  | Fixing animation bugs                                     |  `FURY <https://fury.gl/latest/posts/2022/2022-08-09-week-08-shivam.html>`__ - `Python <https://blogs.python-gsoc.org/en/xtanions-blog/week-8-fixing-animation-bugs/>`__                                  |
+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 9 (17-08-2022)  | First working skeletal animation prototype                | `FURY <https://fury.gl/latest/posts/2022/2022-08-17-week-09-shivam.html>`__ - `Python <https://blogs.python-gsoc.org/en/xtanions-blog/week-9-first-working-skeletal-animation-prototype/>`__              |
+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 10 (25-08-2022) | Multi-node skinning support                               | `FURY <https://fury.gl/latest/posts/2022/2022-08-25-week-10-shivam.html>`__ - `Python <https://blogs.python-gsoc.org/en/xtanions-blog/week-10-multi-node-skinning-support/>`__                            |
+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 11 (31-08-2022) | Multiple transformations support and adding tests         | `FURY <https://fury.gl/latest/posts/2022/2022-08-31-week-11-shivam.html>`__ - `Python <https://blogs.python-gsoc.org/en/xtanions-blog/week-11-multiple-transformations-support-and-adding-tests/>`__      |
+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 12 (08-09-2022) | Adding skeleton as actors and fix global transformation   | `FURY <https://fury.gl/latest/posts/2022/2022-09-08-week-12-shivam.html>`__ - `Python <https://blogs.python-gsoc.org/en/xtanions-blog/week-12-adding-skeleton-as-actors-and-fix-global-transformation/>`__|
+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 13 (15-09-2022) | Multi bone skeletal animations                            | `FURY <https://fury.gl/latest/posts/2022/2022-09-15-week-13-shivam.html>`__ - `Python <https://blogs.python-gsoc.org/en/xtanions-blog/week-13-multi-bone-skeletal-animation-support/>`__                  |
+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 14 (28-09-2022) | Morphing is here !                                        | `FURY <https://fury.gl/latest/posts/2022/2022-09-28-week-14-shivam.html>`__ - `Python <https://blogs.python-gsoc.org/en/xtanions-blog/week-14-morphing-is-here/>`__                                       |
+----------------------+-----------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
