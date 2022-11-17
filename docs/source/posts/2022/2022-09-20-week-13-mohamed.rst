Week 13: Keyframes animation is now a bit easier in FURY
========================================================

.. post:: September  20 2022
   :author: Mohamed Abouagour
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

- Added the ability to have the ShowManager stored inside the Timeline. That way the user does not have to update and render the animations because it will be done internally.

- Added a record method to the Timeline that records the animation and saves it as either GIF or MP4 (requires OpenCV). This record functionality has the option to show/hide the PlaybackPanel which makes it better than recording the animation using a third-party software.

    .. image:: https://user-images.githubusercontent.com/63170874/190892795-f47ceaf1-8dd0-4235-99be-2cf0aec323bb.gif
        :width: 600
        :align: center

- Fixed some issues that Serge mentioned while reviewing PR `#665`_.


What is coming up next week?
----------------------------

- Instead of adding the ShowManager to the Timeline, doing it the other way around is a better choice and makes the code more readable.

- Add tests for the Timeline's record method.

- Add tests for the billboard actor to test consistency among different approaches..


Did you get stuck anywhere?
---------------------------

I didn't get stuck this week.



.. _`#665`: https://github.com/fury-gl/fury/pull/665