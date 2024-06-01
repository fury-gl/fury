Week 10: Its time for a Spin-Box!
=================================

.. post:: August 05, 2023
   :author: Praneeth Shetty
   :tags: google
   :category: gsoc

What did you do this week?
--------------------------
This week, my focus shifted to the ``SpinBoxUI`` after wrapping up work on ``TextBlock2D``. ``SpinBoxUI`` is a component that allows users to select a value by spinning through a range. To ensure a smooth transition, I made adjustments in ``SpinBoxUI`` to align it with the recent updates in ``TextBlock2D``. To make things even clearer and more user-friendly, I initiated a continuous code improvement process. I introduced setters and getters that enable easier customization of ``TextBlock2D``'s new features, such as **auto_font_scale** and **dynamic_bbox**. These tools simplify the process of adjusting these settings, and you can see the ongoing changes in pull request `#830 <https://github.com/fury-gl/fury/pull/830>`_.

Simultaneously, I worked on improving the ``FileDialog`` component. Since the ``FileDialog`` PR was based on an older version, it required updates to match the recent developments in ``TextBlock2D``. This involved restructuring the code and making sure that everything worked smoothly together. You can checkout the progress here at PR `#832 <https://github.com/fury-gl/fury/pull/832>`_.

Did you get stuck anywhere?
---------------------------
Thankfully, this week was quite smooth sailing without any major roadblocks.

What is coming up next?
-----------------------
Looking ahead, my plan is to finalize the integration of the updated ``TextBlock`` and ``SpinBoxUI`` components. This entails making sure that everything works seamlessly together and is ready for the next stages of development.
