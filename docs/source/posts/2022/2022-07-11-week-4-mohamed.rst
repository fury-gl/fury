Week 4: Camera animation, interpolation in GLSL, and a single ``Timeline``!
===========================================================================

.. post:: July 11 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- Managed to implement a single ``Timeline`` using the ``Container`` class. So, instead of having two classes: ``Timeline`` and ``CompositeTimeline``, now the ``Timeline`` can have multiple ``Timeline`` objects and controls them as in the code below.

    .. code-block:: python

        main_timeline = Timeline()
        sub_timeline = Timeline(actors_list)
        main_timeline.add_timeline(sub_timeline)

|

- Implemented the HSV, Lab, and XYZ color interpolators in GLSL.

- Added support for animating two camera properties: position and focal position, which can be interpolated using any general ``Interpolator``.

    .. raw:: html

        <iframe id="player" type="text/html"   width="440" height="390" src="https://user-images.githubusercontent.com/63170874/178276182-531b7d0d-414d-41f8-8db8-c3e9f4885e59.mp4" frameborder="0"></iframe>


- Allowed all ``actors`` inside all timelines to be added to the ``Scene`` automatically when the main (parent) ``Timeline`` is added to the ``Scene``.

- Fixed the ``PlaybackPanel``, added a counter to display the animation time as in the video above, and added an option to attach a ``PlaybackPanel`` to the ``Timeline``.

    .. code-block:: python

        main_timeline = Timeline(playback_panel=True)


|

What is coming up next week?
----------------------------

This week I will do the following:

- Start Implementing slerp interpolation using quaternions in both Python and GLSL. And use slerp to apply camera rotation.
- Add tests for the previous work.
- Make a PR to merge the non-shader-based version of the ``Timeline``.
- Test how shader attributes can be used to hold keyframes for each vertex and benchmark it.
- Study 'colormaps' and implement some 'colormaps' in GLSL.


Did you get stuck anywhere?
---------------------------

- Uniforms don't maintain their data after shaders are unbounded and another uniform with the same name in a different shader is set.
