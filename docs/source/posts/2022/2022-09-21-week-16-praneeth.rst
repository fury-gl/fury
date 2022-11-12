=================================
Week 16 - Working with Rotations!
=================================

.. post:: September 21 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
Last week my mentors noticed that each `DrawShape` has its individual `rotation_slider` which increases redundancy and complexity in setting its visibility on and off. Instead, they suggested moving the `rotation_slider` to `DrawPanel` and keeping a common slider for all the shapes.

PR `#688 <https://github.com/fury-gl/fury/pull/688>`_ does the above mentioned thing.
There isn't any visual difference as everything is as it was earlier, just the code was modified a bit to make it work properly.

After this, I started working with the rotation for the `Polyline` feature. For rotating the `Polyline`, I implemented something similar to what I had did while rotating the individual shapes. Firstly I calculate the bounding box and the center of the shape, then apply the rotation to the points through which the polyline was generated.

`Polyline: <https://github.com/ganimtron-10/fury/tree/polyline-with-grouping>`_

.. image:: https://user-images.githubusercontent.com/64432063/193308748-6bc14acb-b687-4d88-9c41-12991186a104.gif
    :width: 400
    :align: center

As we can see above the rotation seems correct but as earlier the shape is translating from its original center. This should be easy to fix.

Did you get stuck anywhere?
---------------------------
Instead of implementing the approaches for creating and managing the `Polyline`, I kept on thinking of various ideas on how I could do so, which wasted my time. I should have thought about some approaches and tried to implement them so that I would get an idea of whether things would work or not.

What is coming up next?
-----------------------
Working on `Polyline` to make sure everything works fine.