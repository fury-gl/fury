.. image:: https://developers.google.com/open-source/gsoc/resources/downloads/GSoC-logo-horizontal.svg
   :height: 50
   :align: center
   :target: https://summerofcode.withgoogle.com/programs/2023

.. image:: https://www.python.org/static/community_logos/python-logo.png
   :width: 40%
   :target: https://summerofcode.withgoogle.com/programs/2023/organizations/python-software-foundation

.. image:: https://python-gsoc.org/logos/FURY.png
   :width: 25%
   :target: https://fury.gl/latest/index.html

Google Summer of Code Final Work Product
========================================

.. post:: August 24 2023
   :author: Tania Castillo
   :tags: google
   :category: gsoc

-  **Name:** Tania Castillo
-  **Organisation:** Python Software Foundation
-  **Sub-Organisation:** FURY
-  **Project:** `SDF-based uncertainty representation for dMRI glyphs <https://github.com/fury-gl/fury/wiki/Google-Summer-of-Code-2023-(GSOC2023)#project-3-sdf-based-uncertainty-representation-for-dmri-glyphs>`_


Proposed Objectives
-------------------

- Implement a parallelized version of computer-generated billboards using geometry shaders for amplification.
- Model the mathematical functions that express the geometry of ellipsoids glyphs and implement them using Ray Marching techniques.
- Model the mathematical functions that express the geometry of ODF glyphs and implement them using Ray Marching techniques.
- Use SDF properties and techniques to represent the uncertainty of dMRI reconstruction models.

Abstract
--------------------
Diffusion Magnetic Resonance Imaging (dMRI) is a non-invasive imaging technique used by neuroscientists to measure the diffusion of water molecules in biological tissue. The directional information is reconstructed using either a Diffusion Tensor Imaging (DTI) or High Angular Resolution Diffusion Imaging (HARDI) based model, which is graphically represented as tensors and Orientation Distribution Functions (ODF). Traditional rendering engines discretize Tensor and ODF surfaces using triangles or quadrilateral polygons, making their visual quality depending on the number of polygons used to build the 3D mesh which might compromise real-time display performance. This project proposes a methodological approach to further improve the visualization of DTI tensors and HARDI ODFs glyphs by using well-established techniques in the field of computer graphics such as geometry amplification, billboarding, signed distance functions (SDFs), and ray marching.

Objectives Completed
--------------------

Ellipsoid actor implemented with SDF
************************************

A first approach for tensor glyph generation has been made, using raymarching and SDF applied to a box. The current implementation with tensor slicer requires a sphere with a specific number of vertices to be deformed based on this model, to get a higher resolution a sphere with more vertices is needed. Because the raymarching technique does not use polygonal meshes it is possible to define perfectly smooth surfaces and still obtain a fast rendering.

Details of the implementation:
- The are some minor calculations done in the vertex shader, corresponding to the tensor matrix calculation, and data normalization.
- The implementation of the raymarching algorithm and the definition of the SDF is done in the fragment shader. We define the SDF in a simpler way by transforming a sphere into an ellipsoid, considering that the SDF of a sphere is easily computed and the definition of a tensor gives us a linear transformation of a given geometry. Also, as scaling is not a rigid body transformation, we multiply the final result by a factor to compensate for the difference.
- The central differences method was used to compute the normals necessary for the scene’s illumination. In addition, we used the Blinn-Phong lighting technique which is high-quality and computationally cheap.

.. image:: https://user-images.githubusercontent.com/31288525/244503195-a626718f-4a13-4275-a2b7-6773823e553c.png
    :width: 376
    :align: center

This implementation does show a better quality in the displayed glyphs, and support the display of a large amount of data, as seen in the image below. For this reason a tutorial was made to justify in more detail the value of this new implementation.

**Future work:** In line with one of the initial objectives, it is expected to implement billboards later on, to improve the performance, i.e., higher frame rate and less memory usage for the tensor ellipsoid creation. In addition to looking for ways to optimize the naive raymarching algorithm and the definition of SDFs.

*Pull Requests:*

-  **Ellipsoid actor implemented with SDF (Merged)** https://github.com/fury-gl/fury/pull/791
-  **Tutorial on using ellipsoid actor to visualize tensor ellipsoids for DTI (Merged)** https://github.com/fury-gl/fury/pull/818

Objectives in Progress
----------------------

DTI uncertainty visualization
*****************************

The DTI visualization pipeline is fairly complex, a level of uncertainty arises, which, if visualized, helps to assess the accuracy of the model. This measure is not currently implemented, and even though the are several methods to calculate a visualize the uncertainty in the DTI model, because of its simplicity and visual representation, we considered Matrix Perturbation Analysis (MPA) proposed by Basser [Bas97]. This measurement is visualized as double cones representing the variance of the main direction of diffusion, for which raymarching tecnique was also used in the creation of these objects.

Below is a demo of how this new feature is intended to be used, an image of diffusion tensor ellipsoids and their associated uncertainty cones.

.. image:: https://user-images.githubusercontent.com/31288525/254747296-09a8674e-bfc0-4b3f-820f-8a1b1ad8c5c9.png
    :width: 530
    :align: center

The implementation is almost complete, but as it is a new addition that includes mathematical calculations and for which there is no direct reference for comparison, it requires a more detail review before it can be incorporated. For this reason, a tutorial explaining in more detail how to use this feature will be added later.

*Pull Request:*

-  **DTI uncertainty visualization (Under Review)** https://github.com/fury-gl/fury/pull/810

**Future work:** When ODF implementation is complete, uncertainty for this other reconstruction model is expected to be added, using semitransparent glyphs representing the mean directional information proposed by Tournier [TCGC04].

ODF actor implemented with SDF
******************************

.. image:: https://user-images.githubusercontent.com/31288525/260909561-fd90033c-018a-465b-bd16-3586bb31ca36.png
    :width: 600
    :align: center

*Working branch:*

-  **ODF implementation (Under Development)**
   https://github.com/tvcastillod/fury/tree/SH-for-ODF-impl


GSoC Weekly Blogs
-----------------

-  My blog posts can be found on the `FURY website <https://fury.gl/latest/blog/author/tania-castillo.html>`__ and the `Python GSoC blog <https://blogs.python-gsoc.org/en/tvcastillods-blog/>`__.


Timeline
--------

+---------------------+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Date                | Description                                                            | Blog Post Link                                                                                                                                                           |
+=====================+========================================================================+==========================================================================================================================================================================+
| Week 0(02-06-2022)  | Community Bounding Period                                              | `FURY <https://fury.gl/latest/posts/2023/2023-06-02-week-0-tvcastillod.html>`__ - `Python <https://blogs.python-gsoc.org/en/tvcastillods-blog/weekly-check-in-0-2>`__    |
+---------------------+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 1(05-06-2022)  | Ellipsoid actor implemented with SDF                                   | `FURY <https://fury.gl/latest/posts/2023/2023-06-15-week-1-tvcastillod.html>`__ - `Python <https://blogs.python-gsoc.org/en/tvcastillods-blog/weekly-check-in-1-23>`__   |
+---------------------+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 2(12-06-2022)  | Making adjustments to the Ellipsoid Actor                              | `FURY <https://fury.gl/latest/posts/2023/2023-06-12-week-2-tvcastillod.html>`__ - `Python <https://blogs.python-gsoc.org/en/tvcastillods-blog/weekly-check-in-2-18>`__   |
+---------------------+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 3(19-06-2022)  | Working on uncertainty and details of the first PR                     | `FURY <https://fury.gl/latest/posts/2023/2023-06-19-week-3-tvcastillod.html>`__ - `Python <https://blogs.python-gsoc.org/en/tvcastillods-blog/weekly-check-in-3-27>`__   |
+---------------------+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 4(27-06-2022)  | First draft of the DTI uncertainty visualization                       | `FURY <https://fury.gl/latest/posts/2023/2023-06-27-week-4-tvcastillod.html>`__ - `Python <https://blogs.python-gsoc.org/en/tvcastillods-blog/weekly-check-in-4-24>`__   |
+---------------------+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 5(03-07-2022)  | Preparing the data for the Ellipsoid tutorial                          | `FURY <https://fury.gl/latest/posts/2023/2023-07-03-week-5-tvcastillod.html>`__ - `Python <https://blogs.python-gsoc.org/en/tvcastillods-blog/weekly-check-in-5-27>`__   |
+---------------------+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 6(10-07-2022)  | First draft of the Ellipsoid tutorial                                  | `FURY <https://fury.gl/latest/posts/2023/2023-07-10-week-6-tvcastillod.html>`__ - `Python <https://blogs.python-gsoc.org/en/tvcastillods-blog/weekly-check-in-6-26>`__   |
+---------------------+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 7(17-07-2022)  | Adjustments on the Uncertainty Cones visualization                     | `FURY <https://fury.gl/latest/posts/2023/2023-07-17-week-7-tvcastillod.html>`__ - `Python <https://blogs.python-gsoc.org/en/tvcastillods-blog/weekly-check-in-7-26>`__   |
+---------------------+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 8(25-07-2022)  | Working on Ellipsoid Tutorial and exploring SH                         | `FURY <https://fury.gl/latest/posts/2023/2023-07-25-week-8-tvcastillod.html>`__ - `Python <https://blogs.python-gsoc.org/en/tvcastillods-blog/weekly-check-in-8-17>`__   |
+---------------------+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 9(31-07-2022)  | Tutorial done and polishing DTI uncertainty                            | `FURY <https://fury.gl/latest/posts/2023/2023-07-31-week-9-tvcastillod.html>`__ - `Python <https://blogs.python-gsoc.org/en/tvcastillods-blog/weekly-check-in-9-22>`__   |
+---------------------+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 10(08-08-2022) | Start of SH implementation experiments                                 | `FURY <https://fury.gl/latest/posts/2023/2023-08-08-week-10-tvcastillod.html>`__ - `Python <https://blogs.python-gsoc.org/en/tvcastillods-blog/weekly-check-in-10-16>`__ |
+---------------------+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 11(16-08-2022) | Adjusting ODF implementation and looking for solutions on issues found | `FURY <https://fury.gl/latest/posts/2023/2023-08-16-week-11-tvcastillod.html>`__ - `Python <https://blogs.python-gsoc.org/en/tvcastillods-blog/weekly-check-in-11-17>`__ |
+---------------------+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 12(24-08-2022) | Experimenting with ODFs implementation                                 | `FURY <https://fury.gl/latest/posts/2023/2023-08-24-week-12-tvcastillod.html>`__ - `Python <https://blogs.python-gsoc.org/en/tvcastillods-blog/weekly-check-in-12-9>`__  |
+---------------------+------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


References
~~~~~~~~~~