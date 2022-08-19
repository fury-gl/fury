==========================================================
Week 6 - Supporting Rotation of the Shapes from the Center
==========================================================

.. post:: July 13 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
This week I started implementing a new feature to rotate the shapes from the center using ``RingSlider2D``. I already had a `rotate` function that rotates the shape around its pivot vertex, so I updated it to support rotation from the center.

`Rotation from center <https://github.com/fury-gl/fury/pull/623>`_

.. image:: https://user-images.githubusercontent.com/64432063/180257893-196baafe-3c42-4152-b5f4-643b794176d2.gif
    :align: center
    :width: 300

Then I tried to group the shapes to transform and modify them all at once. For this, I had to investigate more about the key press and release events. Then I managed to select and deselect shapes by holding the ``Ctrl`` key.

`Grouping Shapes <https://github.com/ganimtron-10/fury/tree/grouping-shapes>`_

.. image:: https://user-images.githubusercontent.com/64432063/180261113-39760cba-0343-41e7-924a-c741eb838f0b.gif
    :align: center
    :width: 300

I improved the `polyline` feature by adding a separate class to manage the creation and manipulation of the lines but still; I was facing the same issue with the dragging event, which initialized a new object every time a new line was created.

Did you get stuck anywhere?
---------------------------
It was difficult to rotate the shape from the center because the pivot(or the reference point) of the shape wasn't consistent. As you can see below it changed depending on how the shape was created.

.. image:: https://user-images.githubusercontent.com/64432063/176093855-6129cc25-d03d-45ba-872e-c8d2c6329a1e.gif
    :width: 400
    :align: center

To handle this, I created an interface between the actual pivot point and the center of the bounding box, which made it easy to calculate and set the positions.

Also, I wasn't able to find a way by which I could manually call the dragging event without a preceding click event which was required for the `polyline`` feature.

What is coming up next?
-----------------------
Complete `Rotation from center <https://github.com/fury-gl/fury/pull/623>`_ PR.