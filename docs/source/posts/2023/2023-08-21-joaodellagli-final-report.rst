.. image:: https://developers.google.com/open-source/gsoc/resources/downloads/GSoC-logo-horizontal.svg
   :height: 50
   :align: center
   :target: https://summerofcode.withgoogle.com/programs/2023/projects/ED0203De

.. image:: https://www.python.org/static/community_logos/python-logo.png
   :width: 40%
   :target: https://summerofcode.withgoogle.com/programs/2023/organizations/python-software-foundation

.. image:: https://python-gsoc.org/logos/FURY.png
   :width: 25%
   :target: https://fury.gl/latest/index.html

Google Summer of Code Final Work Product
========================================

.. post:: August 21 2023
   :author: João Victor Dell Agli Floriano
   :tags: google
   :category: gsoc

-  **Name:** João Victor Dell Agli Floriano
-  **Organisation:** Python Software Foundation
-  **Sub-Organisation:** FURY
-  **Project:** `FURY - Project 2. Fast 3D kernel-based density rendering using billboards. <https://github.com/fury-gl/fury/wiki/Google-Summer-of-Code-2023-(GSOC2023)#project-2-fast-3d-kernel-based-density-rendering-using-billboards>`_


Proposed Objectives
-------------------

- **First Phase** : Implement framebuffer usage in FURY
    * Investigate the usage of float framebuffers inside FURY's environment.
    * Implement a float framebuffer API.

- **Second Phase** : Shader-framebuffer integration
    * Implement a shader that uses a colormap to render framebuffers.
    * Escalate this rendering for composing multiple framebuffers.

- **Third Phase** : KDE Calculations
    * Investigate KDE calculation for point-cloud datasets.
    * Implement KDE calculation inside the framebuffer rendering shaders.
    * Test KDE for multiple datasets.

Objectives Completed
--------------------

- **Implement framebuffer usage in FURY** (partially)

    The first phase, adressed from *May/29* to *July/07*, took longer than expected and could not be totally completed. The project started with the investigation of
    `VTK's Frambuffer Object <https://vtk.org/doc/nightly/html/classvtkOpenGLFramebufferObject.html#details>`_, a vital part of this project, to understand 
    how to use it properly. 

    Framebuffer Objects, abbreviated as FBOs, are the key to post-processing effects in OpenGL, as they are used to render things offscreen and save it to a texture
    that will be later used to apply the desired post-processing effects within the object's `fragment shader <https://www.khronos.org/opengl/wiki/Fragment_Shader>`_ 
    rendered to screen, in this case, a `billboard <http://www.opengl-tutorial.org/intermediate-tutorials/billboards-particles/billboards/>`_. In the case of the 
    `Kernel Density Estimation <https://en.wikipedia.org/wiki/Kernel_density_estimation>`_ post-processing effect, we need a special kind of FBO, one that stores textures' 
    values as floats, different from the standard 8-bit unsigned int storage. This is necessary because the KDE rendering involves rendering every KDE point calculation 
    to separate billboards, rendered to the same scene, which will have their intensities, divided by the number of points rendered, blended with 
    `OpenGL Additive Blending <https://www.khronos.org/opengl/wiki/Blending>`_, and if a relative big number of points are rendered at the 
    same time, 32-bit float precision is needed to guarantee that small-intensity values will not be capped to zero, and disappear.

    After a month going through VTK's FBO documentation and weeks spent trying different approaches to it, I could not make it work properly, possible due to it 
    being broken and not very well documented. Reporting that to my mentors, which unsuccessfully tried themselves to make it work, they decided it was better if I took 
    another path, using `VTK's WindowToImageFilter <https://vtk.org/doc/nightly/html/classvtkWindowToImageFilter.html>`_ method as a workaround, described in this 
    :doc:`blogpost <2023-07-03-week-5-joaodellagli.rst>`. This is not the ideal way to make it work, as this method does not seem to support float textures, however, 
    a workaround to that is currently being worked on, as I will describe later on.

    *Pull Requests:*

    - **KDE Rendering Experimental Program:** 
        The result of this whole FBO and WindowToImageFilter experimentation is well documented in PR 
        `#804 <https://github.com/fury-gl/fury/pull/804>`_ that implements an experimental version of a KDE rendering program. 

- **Shader-framebuffer integration**


Other Objectives
----------------

- **Stretch Goals** : SDE Implementation, Network/Graph visualization using SDE/KDE, Tutorials
    * Investigate SDE calculation for surface datasets.
    * Implement SDE calculation inside the framebuffer rendering
    shaders.
    * Test SDE for multiple datasets.
    * Develop comprehensive tutorials that explain SDE concepts and
    FURY API usage.
    * Create practical, scenario-based tutorials using real datasets and/or
    simulations.

Objectives in Progress
----------------------

- **KDE Calculations** (ongoing)


GSoC Weekly Blogs
-----------------

-  My blog posts can be found at `FURY website <https://fury.gl/latest/blog/author/praneeth-shetty.html>`__
   and `Python GSoC blog <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/>`__.

Timeline
--------

+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Date                | Description                                        | Blog Post Link                                                                                                                                                                                            |
+=====================+====================================================+===========================================================================================================================================================================================================+
| Week 0(29-05-2023)  | The Beginning of Everything                        | `FURY <https://fury.gl/latest/posts/2023/2023-05-29-week-0-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/the-beggining-of-everything-week-0/>`__                  |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 1(05-06-2022)  | The FBO Saga                                       | `FURY <https://fury.gl/latest/posts/2023/2023-06-05-week-1-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-1-laying-the-foundation-of-drawpanel-ui>`__         |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 2(12-06-2022)  | The Importance of (good) Documentation             | `FURY <https://fury.gl/latest/posts/2023/2023-06-12-week-2-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/the-importance-of-good-documentation-week-2/>`__         |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 3(19-06-2022)  | Watch Your Expectations                            | `FURY <https://fury.gl/latest/posts/2023/2023-06-19-week-3-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-3-watch-your-expectations/>`__                      |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 4(26-06-2022)  | Nothing is Ever Lost                               | `FURY <https://fury.gl/latest/posts/2023/2023-06-26-week-4-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-4-nothing-is-ever-lost/>`__                         |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 5(03-07-2022)  | All Roads Lead to Rome                             | `FURY <https://fury.gl/latest/posts/2023/2023-07-03-week-5-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-5-all-roads-lead-to-rome/>`__                       |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 6(10-07-2022)  | Things are Starting to Build Up                    | `FURY <https://fury.gl/latest/posts/2023/2023-07-10-week-6-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-6-things-are-starting-to-build-up/>`__              |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 7(17-07-2022)  | Experimentation Done                               | `FURY <hhttps://fury.gl/latest/posts/2023/2023-07-17-week-7-joaodellagli.html>`__ - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-7-experimentation-done/>`__                         |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 8(24-07-2022)  | The Birth of a Versatile API                       | `FURY <https://fury.gl/latest/posts/2023/2023-07-24-week-8-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-8-the-birth-of-a-versatile-api/>`__                 |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 9(31-07-2022)  | It is Polishing Time!                              | `FURY <https://fury.gl/latest/posts/2023/2023-07-31-week-9-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-9-it-is-polishing-time/>`__                         |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 10(07-08-2022) | Ready for Review!                                  | `FURY <https://fury.gl/latest/posts/2023/2023-08-07-week-10-joaodellagli.html>`__ - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/ready-for-review/>`__                                    |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 11(14-08-2022) | A Refactor is Sometimes Needed                     | `FURY <https://fury.gl/latest/posts/2023/2023-08-14-week-11-joaodellagli.html>`__ - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/a-refactor-is-sometimes-needed/>`__                      |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 12(24-08-2022) |                                                    | `FURY <>`__ - `Python <>`__                                                                                                                                                                               |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
