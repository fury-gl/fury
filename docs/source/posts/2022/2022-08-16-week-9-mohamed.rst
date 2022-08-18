Week 9: Animating primitives of the same actor
==============================================

.. post:: August 16 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- Made two tutorials in this `PR`_ that show two approaches on how to animate primitives of the same FURY actor using the ``Timeline``.

    .. raw:: html

        <iframe id="player" type="text/html"   width="440" height="390" src="https://user-images.githubusercontent.com/63170874/184627836-6b022832-043b-4c28-85b3-d5911808e1a4.mp4" frameborder="0"></iframe>

- Tried sending all keyframes at once as uniforms, but I faced a performance issue doing this.

- Set uniforms that are not being sent by VTK for the billboard actor.


What is coming up next week?
----------------------------

- Alter the ``Timeline`` to use matrices instead of setting values directly to allow hierarchical transformation.

- Improve the API of the ``PartialActor`` to act almost like a normal actor.


Did you get stuck anywhere?
---------------------------

I had some issues trying to get shader’s uniforms to hold their data, and solving this issue led to another issue, which was a performance issue.



.. _`PR`: https://github.com/fury-gl/fury/pull/660