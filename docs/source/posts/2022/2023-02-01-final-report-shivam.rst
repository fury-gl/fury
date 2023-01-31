
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

.. post:: February 01 2023
   :author: Shivam Anand
   :tags: google
   :category: gsoc

-  **Name:** Shivam Anand
-  **Organisation:** Python Software Foundation
-  **Sub-Organisation:** FURY
-  **Project:** `FURY - glTF
   Intergration <https://github.com/fury-gl/fury/wiki/Google-Summer-of-Code-2022>`__


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


   A glTF file is a JSON like file format containing required data for
   3D scenes. VTK has two built-in glTF loaders. However, they lack
   ability to animate and apply materials. Added methods to load binary
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


   The fury scene can contain multiple objects such as actors, cameras,
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
   timeline containsactors an can be added to the scene. We can animate
   the scene by updating timline inside a timer callback.

   https://user-images.githubusercontent.com/74976752/194863125-58f3717d-d89e-48e7-8e2c-a8501e4f230b.mp4

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

   https://user-images.githubusercontent.com/74976752/194862679-ce239e11-5373-4fc5-95a7-4be12feb99cb.mp4

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

   https://user-images.githubusercontent.com/74976752/194862048-7ce65b42-2717-436b-b311-85368f3c3714.mp4

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
   by (usind type) asynchronously. The glTF fetcher is capable
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
      website <https://fury.gl/latest/blog/author/Shivam-Anand.html>`__
      and the `Python GSoC
      blog <https://blogs.python-gsoc.org/en/xtanions-blog/>`__.

Timeline
--------


+---------------------------+-----------------+------------------------+
| Date                      | Description     | Blog Link              |
+===========================+=================+========================+
| Week 0(24-05-2022)        | My Journey to   | `F                     |
|                           | GSoC 2022       | URY <https://fury.gl/l |
|                           |                 | atest/posts/2022/2022- |
|                           |                 | 05-24-my-journey-to-gs |
|                           |                 | oc-2022-shivam.html>`_ |
|                           |                 | _-`Python <https://blo |
|                           |                 | gs.python-gsoc.org/en/ |
|                           |                 | xtanions-blog/my-journ |
|                           |                 | ey-to-gsoc-2022-1/>`__ |
+---------------------------+-----------------+------------------------+
| Week 1(20-06-2022)        | Week 1 - A      | `FURY <ht              |
|                           | Basic glTF      | tps://fury.gl/latest/p |
|                           | Importer        | osts/2022/2022-06-20-w |
|                           |                 | eek1-shivam.html>`__-` |
|                           |                 | Python <https://blogs. |
|                           |                 | python-gsoc.org/en/xta |
|                           |                 | nions-blog/week-1-a-ba |
|                           |                 | sic-gltf-importer/>`__ |
+---------------------------+-----------------+------------------------+
| Week 2(29-06-2022)        | Week 2 -        | `F                     |
|                           | Improving       | URY <https://fury.gl/l |
|                           | Fetcher and     | atest/posts/2022/2022- |
|                           | Exporting glTF  | 06-29-week2-shivam.htm |
|                           |                 | l>`__-`Python <https:/ |
|                           |                 | /blogs.python-gsoc.org |
|                           |                 | /en/xtanions-blog/week |
|                           |                 | -2-improving-fetcher-a |
|                           |                 | nd-exporting-gltf/>`__ |
+---------------------------+-----------------+------------------------+
| Week 3(04-07-2022)        | Week 3 - Fixing | `F                     |
|                           | fetcher, adding | URY <https://fury.gl/l |
|                           | tests and docs  | atest/posts/2022/2022- |
|                           |                 | 07-04-week3-shivam.htm |
|                           |                 | l>`__-`Python <https:/ |
|                           |                 | /blogs.python-gsoc.org |
|                           |                 | /en/xtanions-blog/week |
|                           |                 | -3-fixing-fetcher-addi |
|                           |                 | ng-tests-and-docs/>`__ |
+---------------------------+-----------------+------------------------+
| Week 4(12-07-2022)        | Week 4 -        | `FURY <htt             |
|                           | Finalizing glTF | ps://fury.gl/latest/po |
|                           | loader          | sts/2022/2022-07-12-we |
|                           |                 | ek4-shivam.html>`__-`P |
|                           |                 | ython <https://blogs.p |
|                           |                 | ython-gsoc.org/en/xtan |
|                           |                 | ions-blog/week-4-final |
|                           |                 | izing-gltf-loader/>`__ |
+---------------------------+-----------------+------------------------+
| Week 5(19-07-2022)        | Week 5 -        | `FURY <https://fu      |
|                           | Creating PR for | ry.gl/latest/posts/202 |
|                           | glTF exporter   | 2/2022-07-19-week5-shi |
|                           | and fixing the  | vam.html>`__-`Python < |
|                           | loader          | https://blogs.python-g |
|                           |                 | soc.org/en/xtanions-bl |
|                           |                 | og/week-5-creating-pr- |
|                           |                 | for-gltf-exporter-and- |
|                           |                 | fixing-the-loader/>`__ |
+---------------------------+-----------------+------------------------+
| Week 6(25-07-2022)        | Week 6 -        | `FURY <https://fur     |
|                           | Extracting the  | y.gl/latest/posts/2022 |
|                           | animation data  | /2022-07-25-week-6-shi |
|                           |                 | vam.html>`__-`Python < |
|                           |                 | https://blogs.python-g |
|                           |                 | soc.org/en/xtanions-bl |
|                           |                 | og/week-6-extracting-t |
|                           |                 | he-animation-data/>`__ |
+---------------------------+-----------------+------------------------+
| Week 7(01-08-2022)        | Week 7 - Fixing | `FURY <https:/         |
|                           | bugs in         | /fury.gl/latest/posts/ |
|                           | animations      | 2022/2022-08-01-week-7 |
|                           |                 | -shivam.html>`__-`Pyth |
|                           |                 | on <https://blogs.pyth |
|                           |                 | on-gsoc.org/en/xtanion |
|                           |                 | s-blog/week-7-fixing-b |
|                           |                 | ugs-in-animations/>`__ |
+---------------------------+-----------------+------------------------+
| Week 8(09-08-2022)        | Week 8 - Fixing | `FURY <http            |
|                           | animation bugs  | s://fury.gl/latest/pos |
|                           |                 | ts/2022/2022-08-09-wee |
|                           |                 | k-08-shivam.html>`__-` |
|                           |                 | Python <https://blogs. |
|                           |                 | python-gsoc.org/en/xta |
|                           |                 | nions-blog/week-8-fixi |
|                           |                 | ng-animation-bugs/>`__ |
+---------------------------+-----------------+------------------------+
| Week 9(17-08-2022)        | Week 9 - First  | `FURY <htt             |
|                           | working         | ps://fury.gl/latest/po |
|                           | skeletal        | sts/2022/2022-08-17-we |
|                           | animation       | ek-09-shivam.html>`__- |
|                           | prototype       | `Python <https://blogs |
|                           |                 | .python-gsoc.org/en/xt |
|                           |                 | anions-blog/week-9-fir |
|                           |                 | st-working-skeletal-an |
|                           |                 | imation-prototype/>`__ |
+---------------------------+-----------------+------------------------+
| Week 10(25-08-2022)       | Week 10 -       | `FURY <https://fur     |
|                           | Multi-node      | y.gl/latest/posts/2022 |
|                           | skinning        | /2022-08-25-week-10-sh |
|                           | support         | ivam.html>`__-`Python  |
|                           |                 | <https://blogs.python- |
|                           |                 | gsoc.org/en/xtanions-b |
|                           |                 | log/week-10-multi-node |
|                           |                 | -skinning-support/>`__ |
+---------------------------+-----------------+------------------------+
| Week 11(31-08-2022)       | Week 11 -       | `FURY <https://fur     |
|                           | Multiple        | y.gl/latest/posts/2022 |
|                           | transformations | /2022-08-31-week-11-sh |
|                           | support and     | ivam.html>`__-`Python  |
|                           | adding tests    | <https://blogs.python- |
|                           |                 | gsoc.org/en/xtanions-b |
|                           |                 | log/week-11-multiple-t |
|                           |                 | ransformations-support |
|                           |                 | -and-adding-tests/>`__ |
+---------------------------+-----------------+------------------------+
| Week 12(08-09-2022)       | Week 12 -       | `F                     |
|                           | Adding skeleton | URY <https://fury.gl/l |
|                           | as actors and   | atest/posts/2022/2022- |
|                           | fix global      | 09-08-week-12-shivam.h |
|                           | transformation  | tml>`__-`Python <https |
|                           |                 | ://blogs.python-gsoc.o |
|                           |                 | rg/en/xtanions-blog/we |
|                           |                 | ek-12-adding-skeleton- |
|                           |                 | as-actors-and-fix-glob |
|                           |                 | al-transformation/>`__ |
+---------------------------+-----------------+------------------------+
| Week 13(15-09-2022)       | Week 13 - Multi | `FURY                  |
|                           | bone skeletal   | <https://fury.gl/lates |
|                           | animations      | t/posts/2022/2022-09-1 |
|                           |                 | 5-week-13-shivam.html> |
|                           |                 | `__-`Python <https://b |
|                           |                 | logs.python-gsoc.org/e |
|                           |                 | n/xtanions-blog/week-1 |
|                           |                 | 3-multi-bone-skeletal- |
|                           |                 | animation-support/>`__ |
+---------------------------+-----------------+------------------------+
| Week 14(28-09-2022)       | Week 14 -       | `FURY <                |
|                           | Morphing is     | https://fury.gl/latest |
|                           | here !          | /posts/2022/2022-09-28 |
|                           |                 | -week-14-shivam.html>` |
|                           |                 | __-`Python <https://bl |
|                           |                 | ogs.python-gsoc.org/en |
|                           |                 | /xtanions-blog/week-14 |
|                           |                 | -morphing-is-here/>`__ |
+---------------------------+-----------------+------------------------+
