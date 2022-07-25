Week 6: Fixing the ``Timeline`` issues and equipping it with more features
==========================================================================

.. post:: July 25 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- Improved the ``PlaybackPanel`` by adding speed control and the ability to loop the animation. Also, fixed the lagging issue of play and pause buttons and composed them into a single play/pause button.

- Updated the old tutorials' syntax to match the other tutorials and added a new tutorial on position animation using spline interpolation. And added unit tests for the ``PlaybackPanel`` and the newly added color converters in ``colormap.py``.

- Added more hooks to the 2D sliders to cover two more states: ``on_value_changed`` and ``on_moving_slider`` `#634`_.

- Provided the ability to add static actors to the ``Timeline``, which might be needed in the animation part of shivam's glTF project.

  - If an ``actor`` is added to the ``Timeline`` as a static actor, it won't be animated by the ``Timeline``, but it will get added to the scene along with the ``Timeline`` when the ``Timeline`` is added to the scene.

- Implemented a custom evaluator for the ``Timeline``'s properties.

  - A custom evaluator uses a user-provided function that takes time as input and evaluates the property at that time. This feature is yet to be discussed more in today's meeting.

- Fixed camera rotation, and view-up issue when interacting with the scene.


What is coming up next week?
----------------------------
Next week's work is yet to be determined.


Did you get stuck anywhere?
---------------------------
I didn't get stuck this week.

.. _`#634`: https://github.com/fury-gl/fury/pull/634