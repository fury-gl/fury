Week 14: Keyframes animation is now a bit easier in FURY
========================================================

.. post:: September 28 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- Separated the ``Timeline`` into a ``Timeline`` and an ``Animation``. So, instead of having the ``Timeline`` do everything. It's now more like doing animations in Blender and other 3D editing software where the timeline handles the playback of several animations `#694`_.

- Added unit tests for the billboards based on geometry shader.

- Tried to solve the issue with actors not being rendered when their positions are changed in the vertex shader. For now, I just found a way to force invoke the shader callbacks, but force rendering the actor itself still needs more investigation.


What is coming up next week?
----------------------------

- Add unit testing for the ``Animation``, document it well, and implement more properties suggested by Shivam (@xtanion).

- Modify `Add Timelines to ShowManager directly`_ PR to allow adding ``Animation`` to the ``ShowManager`` as well.

- Update tutorials to adapt to all the new changes in the ``animation`` module.


Did you get stuck anywhere?
---------------------------

- I got stuck trying to solve the issue mentioned above with actors not being rendered.



.. _`#694`: https://github.com/fury-gl/fury/pull/694
.. _`Add Timelines to ShowManager directly`: https://github.com/fury-gl/fury/pull/690
