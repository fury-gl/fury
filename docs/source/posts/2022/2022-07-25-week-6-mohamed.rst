Week 6: Fixing the ``Timeline`` issues and equipping it with more features
==========================================================================

.. post:: July 25 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- Improved the ``PlaybackPanel`` by adding speed control and the ability to loop the animation. Also, fixed the lagging issue of play and pause buttons and composed them into a single play/pause button.

- Updated the old tutorials' syntax to match the other tutorials and added a new tutorial on position animation using spline interpolation. Added unit tests for the ``PlaybackPanel`` and the newly added color converters in ``colormap.py``.

- Added more hooks to the 2D sliders to cover two more states:

  1. ``on_value_changed``, which gets called whenever the value of the slider is changed without interacting with the slider.

  2. ``on_moving_slider``, which gets called when the position of the slider is changed by user interaction. `#634`_.

  - The reason for adding these two hooks is that there was only the ``on_change`` hook, which always gets called when the value of the slider is changed without considering how the value is changed, hence, the functionality of the slider was limited.

- Provided the ability to add static actors to the ``Timeline``, which might be needed in the animation part of Shivam's glTF project `#643`_.

  - If an ``actor`` is added to the ``Timeline`` as a static actor, it won't be animated by the ``Timeline``, but it will get added to the scene along with the ``Timeline``.

- Implemented a custom evaluator for the ``Timeline``'s properties.

  - A custom evaluator uses a user-provided function that takes time as input and evaluates the property at that time. This feature is yet to be discussed more in today's meeting.

- Fixed camera rotation and the view-up issue when interacting with the scene.


What is coming up next week?
----------------------------
- Make a tutorial on how to implement a new custom Interpolator to work with the ``Timeline``.
- Add motion path visualization feature for both position and color properties.
- Add custom evaluation functions directly, without using the ``CustomInterpolator``.
- Implement an already existing complex interpolator using closures instead of classes.

Did you get stuck anywhere?
---------------------------
I didn't get stuck this week.

.. _`#634`: https://github.com/fury-gl/fury/pull/634
.. _`#643`: https://github.com/fury-gl/fury/pull/643