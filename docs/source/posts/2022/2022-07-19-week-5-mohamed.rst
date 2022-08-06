Week 5: Slerp implementation, documenting the Timeline, and adding unit tests
=============================================================================

.. post:: July 19 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- Implemented Slerp (spherical linear interpolation) for rotation keyframes.
- Controlling the speed of the animation is now an option.
- Added the tests and documented the ``Timeline``.
- Used the geometry shader to generate billboard actors using a minimal set of calculations `#631`_.

    .. raw:: html

        <iframe id="player" type="text/html"   width="440" height="390" src="https://user-images.githubusercontent.com/63170874/179493243-55b28d24-2c94-485d-af7e-ccb296733f34.mp4" frameborder="0"></iframe>


What is coming up next week?
----------------------------

This week I will do the following:

- Focus on finalizing PR `#626`_ to be merged.
- Make some upgrades to the playback panel to make it more functional and responsive.
- Add more tutorials to explain how to use all the functionalities of the ``Timeline``.

Did you get stuck anywhere?
---------------------------
I didn't get stuck this week.

.. _`#626`: https://github.com/fury-gl/fury/pull/626
.. _`#631`: https://github.com/fury-gl/fury/pull/631