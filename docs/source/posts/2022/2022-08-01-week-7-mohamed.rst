Week 7: Billboard spheres and implementing interpolators using closures
=======================================================================

.. post:: August 1 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- Restructured the keyframe animation interpolators using closures to be functions instead of classes `#647`_. Now it is easier to implement new interpolators with the help of the functions existing in `fury/animation/helpers.py`_. Also, now unit tests can be added for the latter functions.

- Added two examples explaining how to implement a custom interpolator that can be used by the ``Timeline``, one using `classes`_ and the other using `closures`_.

- Fixed rotation issue that Shivam discovered while using the ``Timeline`` to animate glTF models. So, rotation in VTK is done by rotating first around Z-axis, then X-axis, then finally around Y-axis, which was not the order I was using to convert from quaternions to Euler degrees.

- Made changes requested by Javier and Filipi on the billboards using geometry shader `PR`_, and made an example of how to use this billboard to show an impostor sphere which looks almost like a real high poly sphere. Also benchmarked using this version of billboard vs using a quad-based billboard, and how they both affect the FPS of the animation.

    .. raw:: html

        <iframe id="player" type="text/html"   width="440" height="390" src="https://user-images.githubusercontent.com/63170874/182064895-27fdd00a-6372-4caa-aff6-3a4bad64e407.mp4" frameborder="0"></iframe>

What is coming up next week?
----------------------------
- Document the new closure-based interpolators. And make some slight renaming changes that we discussed in today's meeting.
- Add tests to the functions inside `fury/animation/helpers.py`_.
- Make changes to the geometry-shader-based billboard to make it more like the quad-based billboard actor while maintaining the new features.
- Integrate the already-implemented shader functionality to the new ``Timeline`` in a separate draft or PR.

Did you get stuck anywhere?
---------------------------
I got stuck trying to get and modify the vertices (centers) of the billboard actor.

.. _`PR`: https://github.com/fury-gl/fury/pull/631
.. _`#647`: https://github.com/fury-gl/fury/pull/647
.. _`fury/animation/helpers.py`: https://github.com/fury-gl/fury/blob/670d3a41645eb7bcd445a7d8ae9ddd7bebc376b7/fury/animation/helpers.py
.. _`closures`: https://github.com/fury-gl/fury/blob/670d3a41645eb7bcd445a7d8ae9ddd7bebc376b7/docs/tutorials/05_animation/viz_keyframe_custom_interpolator.py
.. _`classes`: https://github.com/fury-gl/fury/blob/e0539269adc2a51e35282f83b8b0672bbe047a39/docs/tutorials/05_animation/viz_keyframe_custom_interpolator.py