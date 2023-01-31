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

.. post:: January 31 2023
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc

-  **Name:** Mohamed Abouagour
-  **Organisation:** Python Software Foundation
-  **Sub-Organisation:** FURY
-  **Project:** `FURY - Keyframe animations API <https://github.com/fury-gl/fury/wiki/Google-Summer-of-Code-2022-(GSOC2022)#project-2-keyframe-animations-in-fury>`_



Proposed Objectives
-------------------


* Keyframe animations API

  * Basic playback functions, such as playing, pausing, and rewinding the timeline.
  * Adding keyframes at a specific time for transformations of FURY actors and cameras such as translation, scale, rotation, color, and opacity.
  * Implement quaternion-based interpolation (SLERP)
  * Allow the camera to be interpolated by the keyframe system.
  * Allow the creation and removal of actors from the scene according to the keyframes.
  * Visualize the motion path of positional animation.
  * Speed the animation using GLSL vertex and fragment shaders.

Modified Objectives
-------------------


* Adding a playback panel for controlling the timeline.
* Billboard actor using the geometry shader.
* Hierarchical animation support.
* Animating primitives of the same Fury actor separately.
* Color interpolators.

Objectives Completed
--------------------


* 
  Keyframes Animation API


  * ``Animation`` Class
  * 
    The ``Animation`` class is the main part of the FURY animation module. It is responsible for keyframe animations for a single or a group of FURY actors.  The ``Animation`` is able to handle multiple attributes and properties of Fury actors such as position, color, scale, rotation, and opacity.  It is also capable of doing the following:


    * Set animation keyframes and events.
    * Animate custom properties.
    * Support add-to-scene/remove-from-scene events.
    * Can animate other animations (Hierarchical animation)
    * Set or change the keyframes interpolation method.
    * Visualize the motion path of the positional animation.


    * ``Timeline`` Class
      The ``Timeline`` is the player of FURY ``Animations``\ ; it controls the playback of one or more FURY animations. It also has the option to include a very useful playback panel to help control the playback of the animation. The ``Timeline`` can have a fixed length or get its duration from the animations added to it dynamically. It can loop, play once, change speed, play, pause, and stop.

  * 
    Keyframes Interpolators
       Interpolation is also a core part of the keyframes animation. It is responsible for filling in the blanks between the keyframes so that we have transitional states between the set keyframes. Another factor is that interpolators must be super-optimized to interpolate data in a minimum amount of time as possible or else it would be the bottleneck that lags the animation. 
       The following interpolators have been implemented as a part of the keyframes animation API:


    * Step Interpolator
    * Linear Interpolator
    * Spherical Linear Interpolator (Slerp)
    * Spline Interpolator 
    * Cubic Spline Interpolator 
    * Cubic Bézier Interpolator
    * Color Interpolators:

      * XYZ Color Interpolator
      * Lab Color Interpolator
      * HSV Color Interpolator

    .. raw:: html

        <iframe id="player" type="text/html"   width="440" height="390" src="https://user-images.githubusercontent.com/63170874/194437472-e9ad04b5-0d80-40bf-bd06-64952aeaa65f.mp4" frameborder="0"></iframe>

    .. raw:: html

        <iframe id="player" type="text/html"   width="440" height="390" src="https://user-images.githubusercontent.com/63170874/194441472-a33f66cb-4860-4659-83f7-720a69696c12.mp4" frameborder="0"></iframe>



    * Tutorials
       Also included 11 tutorials  demonstrating gow the FURY keyframe animation API works and how to use it to make some interesting animations. These tutorial will be added soon to the FURY website.
       Subjects explained in the tutorials are:

      * **Introduction**
      * **Timeline**
      * **Interpolators**
      * **Camera Animation**
      * **Hierarchical Animation**
      * **Using Color Interpolators**
      * **Using Bezier Interpolator**
      * **Using Spline Interpolator**
      * **Using time-based functions**
      * **Creating Custom Interpolators**
      * **Arm Robot Animation**  

  *Pull Requests:*


  * **Keyframe animations and interpolators (Merged):** https://github.com/fury-gl/fury/pull/647
  * **Seperating the  ``Timeline`` (Ready to be Merged):** https://github.com/fury-gl/fury/pull/694
  * **Timeline hierarchical transformation (Merged):** https://github.com/fury-gl/fury/pull/665
  * **Add Timelines to ShowManager directly (Ready to be Merged):** https://github.com/fury-gl/fury/pull/690
  * **Updating animation tutorials (Ready to be Merged):** https://github.com/fury-gl/fury/pull/680
  * **Record keyframe animation as GIF and MP4 (Under Development):** https://github.com/fury-gl/fury/pull/687

* 
  PlaybackPanel UI component

  At first, while in the early development stage of the FURY keyframe animation API, basic playback buttons were used to play, pause, and stop the animation. As the API kept growing, more controllers needed to be implemented, such as the time progress slider, the speed changer, and the loop toggle. And composing all of these controllers into a single UI element was inevitable.
  While the PlaybackPanel is a main part of the ``Timeline``\ ,  the goal was to make it completely independent from the keyframes animation API so that it can be used for anything else, i.e. a video player actor or a continuous time simulation or any other time-dependent applications.

    
  .. image:: https://user-images.githubusercontent.com/63170874/194377387-bfeeea2c-b4ee-4d26-82c0-b76c27fa0f90.png
     :target: https://user-images.githubusercontent.com/63170874/194377387-bfeeea2c-b4ee-4d26-82c0-b76c27fa0f90.png
     :alt: image


  *Pull Requests:*


  * **\ ``PlaybackPanel`` initial implementation (Merged):** https://github.com/fury-gl/fury/pull/647

    * **Set position and width of the  ``PlaybackPanel`` (Merged):** https://github.com/fury-gl/fury/pull/692


* 
  Billboard actor using the geometry shader
    Fury already has a billboard actor implemented using two triangles to construct the billboard. But the new approach uses only one vertex and the canvas of the billboard is generated by the geometry shader. This approach is faster in initialization since only the center is needed and no additional computations to generate the primitive on the CPU side. Also, animating these new billboards using the method mentioned above in the previous objective is way much faster, and faster is one of the reasons why we use billboards.

  *Pull Requests:*


  * **billboards using geometry shader (Ready to be Merged):** https://github.com/fury-gl/fury/pull/631

Objectives in Progress
----------------------


* 
  Animating primitives of the same FURY Actor separately
    Animating FURY actors is not a problem and can be done easily using the FURY animation module. The problem appears when trying to animate a massive amount of actors, thousands or even hundreds of thousands of actors, it's impossible to do that using the animation module. Instead, primitives of the same actor can be animated by changing their vertices and then sending the new vertices buffer to the GPU. This also needs some discussion to find the cleanest way to implement it.

  *Pull Requests:*


  * **Animating primitives of the same actor (Draft):** https://github.com/fury-gl/fury/pull/660
  * **Added primitives count to the polydata (Merged):** https://github.com/fury-gl/fury/pull/617

* 
  Speeding up the animation using GLSL shaders 
    Using the power of the GPU to help speed up the animations since some interpolators are relatively slow, such as the spline interpolator. Besides, morphing and skeletal animation would be tremendously optimized if they were computed on the GPU side! 

  *Pull Requests:*


  * **Adding shader support for doing the animations (Open):** https://github.com/fury-gl/fury/pull/702

Other Objectives
----------------


* 
  Added more enhancements to the ``vector_text`` actor
    Added the ability to change the direction of the ``vector_text`` actor, as well as giving it the option to follow the camera. Also added the option to extrude the text which makes it more like 3D text.

    *Pull Requests:*


  * **Improving  ``vector_text`` (Merged)** : https://github.com/fury-gl/fury/pull/661

* 
  Other PRs


  * **Fixed multi_samples not being used (Merged)**\ :  https://github.com/fury-gl/fury/pull/594
  * **Added an accurate way to calculate FPS (Merged)**\ :   https://github.com/fury-gl/fury/pull/597
  * **Implemented two new hooks for UI sliders (Merged)**\ :   https://github.com/fury-gl/fury/pull/634
  * **Fixed some old tutorials (Merged)**\ :   https://github.com/fury-gl/fury/pull/591
  * **Returning the Timer id while initialization (Merged)**\ :   https://github.com/fury-gl/fury/pull/598

* 
  GSoC Weekly Blogs


  * My blog posts can be found on `the FURY website <https://fury.gl/latest/blog/author/mohamed-abouagour.html>`_ and `the Python GSoC blog <https://blogs.python-gsoc.org/en/m-agours-blog/>`_.

Timeline
--------

.. list-table::
   :header-rows: 1

   * - Date
     - Description
     - Blog Post Link
   * - Week 0\  :raw-html:`<br>`\ (23-05-2022)
     - My journey till getting accepted into GSoC22
     - `FURY <https://fury.gl/latest/posts/2022/2022-05-23-first-post-mohamed.html>`_ - `Python <https://blogs.python-gsoc.org/en/m-agours-blog/my-journey-till-getting-accepted-into-gsoc22/>`_
   * - Week 1\ :raw-html:`<br>`\ (08-06-2022)
     - Implementing a basic Keyframe animation API
     - `FURY <https://fury.gl/latest/posts/2022/2022-06-08-week-1-mohamed.html>`_ - `Python <https://blogs.python-gsoc.org/en/m-agours-blog/week-1-implementing-a-basic-keyframe-animation-api/>`_
   * - Week 2\ :raw-html:`<br>`\ (28-06-2022)
     - Implementing non-linear and color interpolators
     - `FURY <https://fury.gl/latest/posts/2022/2022-06-28-week-2-mohamed.html>`_ - `Python <https://blogs.python-gsoc.org/en/m-agours-blog/week-2-implementing-non-linear-and-color-interpolators/>`_
   * - Week 3\ :raw-html:`<br>`\ (04-07-2022)
     - Redesigning the API,\ :raw-html:`<br>` Implementing cubic Bezier Interpolator,\ :raw-html:`<br>` and making progress on the GPU side!
     - `FURY <https://fury.gl/latest/posts/2022/2022-07-04-week-3-mohamed.html>`_ - `Python <https://blogs.python-gsoc.org/en/m-agours-blog/week-3-redesigning-the-api-implementing-cubic-bezier-interpolator-and-making-progress-on-the-gpu-side/>`_
   * - Week 4\ :raw-html:`<br>`\ (11-07-2022)
     - Camera animation, :raw-html:`<br>`\ interpolation in GLSL, and a single Timeline!
     - `FURY <https://fury.gl/latest/posts/2022/2022-07-11-week-4-mohamed.html>`_ - `Python <https://blogs.python-gsoc.org/en/m-agours-blog/week-4-camera-animation-interpolation-in-glsl-and-a-single-timeline/>`_
   * - Week 5\ :raw-html:`<br>`\ (19-07-2022)
     - Slerp implementation, :raw-html:`<br>`\ documenting the Timeline, and adding unit tests
     - `FURY <https://fury.gl/latest/posts/2022/2022-07-19-week-5-mohamed.html>`_ - `Python <https://blogs.python-gsoc.org/en/m-agours-blog/week-5-slerp-implementation-documenting-the-timeline-and-adding-unit-tests/>`_
   * - Week 6\ :raw-html:`<br>`\ (25-07-2022)
     - Fixing the Timeline issues and equipping it with\ :raw-html:`<br>` more features
     - `FURY <https://fury.gl/latest/posts/2022/2022-07-25-week-6-mohamed.html>`_ - `Python <https://blogs.python-gsoc.org/en/m-agours-blog/week-6-fixing-the-timeline-issues-and-equipping-it-with-more-features/>`_
   * - Week 7\ :raw-html:`<br>`\ (01-08-2022)
     - Billboard spheres and implementing interpolators\ :raw-html:`<br>` using closures
     - `FURY <https://fury.gl/latest/posts/2022/2022-08-01-week-7-mohamed.html>`_ - `Python <https://blogs.python-gsoc.org/en/m-agours-blog/week-7-billboard-spheres-and-implementing-interpolators-using-closures/>`_
   * - Week 8\ :raw-html:`<br>`\ (09-08-2022)
     - Back to the shader-based version of the Timeline
     - `FURY <https://fury.gl/latest/posts/2022/2022-08-09-week-8-mohamed.html>`_ - `Python <https://blogs.python-gsoc.org/en/m-agours-blog/week-8-back-to-the-shader-based-version-of-the-timeline/>`_
   * - Week 9\ :raw-html:`<br>`\ (16-08-2022)
     - Animating primitives of the same actor
     - `FURY <https://fury.gl/latest/posts/2022/2022-08-16-week-9-mohamed.html>`_ - `Python <https://blogs.python-gsoc.org/en/m-agours-blog/week-9-animating-primitives-of-the-same-actor/>`_
   * - Week 10\ :raw-html:`<br>`\ (23-08-2022)
     - Supporting hierarchical animating
     - `FURY <https://fury.gl/latest/posts/2022/2022-08-23-week-10-mohamed.html>`_ - `Python <https://blogs.python-gsoc.org/en/m-agours-blog/week-10-supporting-hierarchical-animations/>`_
   * - Week 11\ :raw-html:`<br>`\ (30-08-2022)
     - Improving tutorials a little
     - `FURY <https://fury.gl/latest/posts/2022/2022-08-30-week-11-mohamed.html>`_ - `Python <https://blogs.python-gsoc.org/en/m-agours-blog/week-11-improving-tutorials-a-little/>`_
   * - Week 12\ :raw-html:`<br>`\ (7-09-2022)
     - Adding new tutorials
     - `FURY <https://fury.gl/latest/posts/2022/2022-09-7-week-12-mohamed.html>`_ - `Python <https://blogs.python-gsoc.org/en/m-agours-blog/week-12-adding-new-tutorials/>`_
   * - Week 13\ :raw-html:`<br>`\ (20-09-2022)
     - Keyframes animation is now a bit easier in FURY
     - `FURY <https://fury.gl/latest/posts/2022/2022-09-20-week-13-mohamed.html>`_ - `Python <https://blogs.python-gsoc.org/en/m-agours-blog/week-13-keyframes-animation-is-now-a-bit-easier-in-fury/>`_

