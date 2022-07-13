Week 3: Redesigning the API, Implementing cubic Bezier Interpolator, and making progress on the GPU side!
=========================================================================================================

.. post:: July 04 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- Redesigned the keyframe animations API to optimize having a lot of timelines by composing them into a parent ``Timeline`` called ``CompositeTimeline`` while maintaining playing individual timelines separately.

- Implemented the cubic Bezier Interpolator using two control points for each keyframe. Also made a tutorial for it below:

  .. raw:: html

        <iframe id="player" type="text/html"   width="440" height="390" src="https://user-images.githubusercontent.com/63170874/177091785-d46817f1-f81e-4ee8-889b-0a7f799261ce.mp4" frameborder="0"></iframe>



- Also Implemented linear and cubic Bezier in GLSL interpolations to be computed by the GPU by sending two keyframes as uniforms and the current animation time.

- Composed the playback panel as a single component ``PlaybackPanel`` (slider does not function yet).

- As we tried to come up with a way to do keyframe animations on a partial subset of the actor's primitives not all of them, we found that there is no currently implemented way to get the primitives count of a single actor. So I made this PR `#617`_ so that the primitives' count is saved inside the polydata of the actor as suggested by Filipi and Javier so that the primitives' count can be used to distinguish the vertices of different primitives.


What is coming up next week?
----------------------------

This week I will do the following

- Finish up the ``PlaybackPanel``.
- Implement all other interpolators in GLSL (color interpolators included).
- Start Implementing slerp interpolation using quaternions in both Python and GLSL.
- Add tests for the previous work.
- Now that I made some progress with the keyframe animation API, I should be able to make a mergeable PR!


Did you get stuck anywhere?
---------------------------

- Couldn't get the ``LineSlider2D`` to work inside my new ``PlaybackPanel``.


.. _`#617`: https://github.com/fury-gl/fury/pull/617