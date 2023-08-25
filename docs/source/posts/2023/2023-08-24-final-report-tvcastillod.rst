.. image:: https://developers.google.com/open-source/gsoc/resources/downloads/GSoC-logo-horizontal.svg
   :height: 50
   :align: center
   :target: https://summerofcode.withgoogle.com/programs/2022/projects/a47CQL2Z

.. image:: https://www.python.org/static/community_logos/python-logo.png
   :width: 40%
   :target: https://summerofcode.withgoogle.com/programs/2022/organizations/python-software-foundation

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
-  **Project:** `FURY - Improve UI elements for drawing geometrical
   shapes <https://github.com/fury-gl/fury/wiki/Google-Summer-of-Code-2023-(GSOC2023)#project-3-sdf-based-uncertainty-representation-for-dmri-glyphs>`_


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

Future work: In line with one of the initial objectives, it is expected to implement billboards later on, to improve the performance, i.e., higher frame rate and less memory usage for the tensor ellipsoid creation. In addition to looking for ways to optimize the naive raymarching algorithm and the definition of SDFs.

*Pull Requests:*

-  **Ellipsoid actor implemented with SDF (Merged)**
    https://github.com/fury-gl/fury/pull/791
-  **Tutorial on using ellipsoid actor to visualize tensor ellipsoids for DTI (Merged)**
    https://github.com/fury-gl/fury/pull/818

Objectives in Progress
----------------------

Uncertainty representation
**************************

The DTI visualization pipeline is fairly complex, a level of uncertainty arises, which, if visualized, helps to assess the accuracy of the model. This measure is not currently implemented, and even though the are several methods to calculate a visualize the uncertainty in the DTI model, because of its simplicity and visual representation, we considered Matrix Perturbation Analysis (MPA) proposed by Basser [Bas97]. This measurement is visualized as double cones representing the variance of the main direction of diffusion, for which raymarching tecnique was also used in the creation of these objects.

Below is a demo of how this new feature is intended to be used, an image of diffusion tensor ellipsoids and their associated uncertainty cones.

.. image:: https://user-images.githubusercontent.com/31288525/254747296-09a8674e-bfc0-4b3f-820f-8a1b1ad8c5c9.png
    :width: 530
    :align: center

The implementation is almost complete, but as it is a new addition that includes mathematical calculations and for which there is no direct reference for comparison, it requires a more detail review before it can be incorporated. For this reason, a tutorial explaining in more detail how to use this feature will be added later.

*Pull Request:*

-  **DTI uncertainty visualization (Under Review)**
   https://github.com/fury-gl/fury/pull/810
