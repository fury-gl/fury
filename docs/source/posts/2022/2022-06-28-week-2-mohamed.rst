Week 2: Implementing non-linear and color interpolators
=======================================================

.. post:: June 28 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- Implemented some other general interpolators such as n-th degree spline and cubic spline interpolators. Also implemented HSV and LAB color interpolators.

    PRs `#612`_ and `#613`_


  .. raw:: html

        <iframe id="player" type="text/html" width="440" height="390" src="https://user-images.githubusercontent.com/63170874/174503916-7ce0554b-9943-43e3-9d5c-c97c9ce48eaf.mp4" frameborder="0"></iframe>


  .. raw:: html

        <iframe id="player" type="text/html" width="440" height="390" src="https://user-images.githubusercontent.com/63170874/176550105-81f23462-43a5-44b1-84ce-3bbd4196f5be.mp4" frameborder="0"></iframe>



- Added animation slider to seek a particular time and visualize the timeline in real-time.

  .. raw:: html

        <iframe id="player" type="text/html" width="440" height="390" src="https://user-images.githubusercontent.com/63170874/176545652-19160248-f1d3-4fff-952c-4512ab889055.mp4" frameborder="0"></iframe>


- Managed to do the transformation on the GPU side using GLSL using matrices. And did some digging about how and when we interpolate the camera and also how to do this on the GPU side.


What is coming up next week?
----------------------------

This week I will do the following

- Allow FURY actors to maintain the number of primitives as an object property so that it can be used to manipulate only a subset of primitives in a single actor.
- Change the structure of the Animation API to a newer one designed by Filipi to solve performance issues when creating a large number of timelines.
- Implement the Bézier curve interpolation.



Did you get stuck anywhere?
---------------------------

I got stuck trying to fix the clipping plans not being appropriately set. Which caused actors to disappear at some point.

.. _`#612`: https://github.com/fury-gl/fury/pull/612
.. _`#613`: https://github.com/fury-gl/fury/pull/613
