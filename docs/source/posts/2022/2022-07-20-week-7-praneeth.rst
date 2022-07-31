===========================================================
Week 7 - Working on Rotation PR and Trying Freehand Drawing
===========================================================

.. post:: July 20 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
I continued PR `#623`_ and fixed the displacement of the shape from its original position when applying rotation. This was happening because most of the calculations resulted in `float` values, but as the pixel position were integers we had to explicitly convert these values into `int`. This conversion rounded off the values and as we call this function continuously, each time the round-off would happen, the shape would get displaced.

.. image:: https://user-images.githubusercontent.com/64432063/181723334-ef9ec75d-d3bf-4b79-83bf-272545c4dd12.gif
    :width: 400
    :align: center

To fix this, I converted all the calculations into integer calculations using floor division ``//`` instead of normal division ``/``.

.. image:: https://user-images.githubusercontent.com/64432063/181723783-352092aa-b7a7-4d13-8d26-553315c7e1aa.gif
    :width: 400
    :align: center

The tests were failing in `Ubuntu` and the `macOS` because the mouse click event wasn't propagated to the line due to some unknown reasons. When investigating more about the issue, Mohamed suggested and helped me implement another approach to select the shapes. In this approach, we calculate the distance between the mouse click event position and each of the shapes, and if any of the distances is less than a specific value (which I call as limit value), then we send this current event to that element.

`New shape selection technique: <https://github.com/ganimtron-10/fury/tree/new-selection>`_


.. image:: https://user-images.githubusercontent.com/64432063/181730428-debd0617-dc32-4232-93ab-18ab903e92de.gif
    :width: 400
    :align: center

I also tried to implement a freehand drawing mode by adding ``Disk2D`` as points. But as you can see below, there are many flaws in this.

- First of all, we add many ``Disk2D`` objects which make it memory consuming process.
- If the mouse moves very fast, then we can see the gaps between points.
- Storing, managing, and transforming are difficult.

.. image:: https://user-images.githubusercontent.com/64432063/181731181-1f242c65-ccb8-4589-a2ed-40bfb3718cfd.gif
    :width: 400
    :align: center


Did you get stuck anywhere?
---------------------------
It was hard to debug why the tests were failing in `Ubuntu` and `macOS`. I tried investigating it by installing Ubuntu and got nothing, but then while implementing the new selection approach, it automatically got fixed.

What is coming up next?
-----------------------
Getting PR `#623`_ merged and working on the polyline feature.

.. _`#623`: https://github.com/fury-gl/fury/pull/623