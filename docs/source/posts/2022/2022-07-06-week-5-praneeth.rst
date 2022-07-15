================================
Week 5 - Working on new features
================================

.. post:: July 06 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
This week I tried to create a base for some upcoming new features.
The first thing I updated was the Properties panel which I prototyped in `Week 3 <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-3-dealing-with-problems/>`_.
So previously, it was just displaying the properties but now after the update, it is able to modify the properties(such as `color`, `position`, and `rotation`) too. This was a quick change to test the callbacks.

`Properties Panel: <https://github.com/ganimtron-10/fury/tree/properties-panel>`_

.. image:: https://user-images.githubusercontent.com/64432063/178412630-a0013a1a-3bfd-46fa-8445-fb5cff728e9c.gif
    :align: center
    :width: 300

Then I worked with the bounding box to make it visible whenever a shape is selected.
For this, I used the existing functionality of the ``Panel2D`` actor to create borders around the bounding box.

`Bounding Box <https://github.com/ganimtron-10/fury/tree/bb-border>`_

.. image:: https://user-images.githubusercontent.com/64432063/178413769-5e4626d6-a207-489a-9789-59777c3e0522.gif
    :align: center
    :width: 300

Also along with this, I managed to add the `polyline` feature on user interactions. This creation isn't that smooth but works as intended.

`Poly Line <https://github.com/ganimtron-10/fury/tree/polyline>`_

.. image:: https://user-images.githubusercontent.com/64432063/178414652-f47f3b25-a2c5-484a-bdbe-94f4ba1eff1f.gif
    :align: center
    :width: 300


Did you get stuck anywhere?
---------------------------
Handling interactions for the `polyline` was complicated. I wasn't able to invoke the `left_mouse_click` event, then as  I was trying to invoke the events internally, it started creating multiple copies of the same line.


What is coming up next?
-----------------------
I will be enhancing the `polyline` feature.
