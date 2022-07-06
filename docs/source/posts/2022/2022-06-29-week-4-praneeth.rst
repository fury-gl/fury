==================================
Week 4 - Fixing the Clamping Issue
==================================

.. post:: June 29 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
Phew!! This week was a tedious week for me as parallelly my End-Sem exams also started. So as usual I started from where I left off last week, *The Clamping Issue*. As per the discussion with the mentors, we decided to use the *AABB bounding box* method to calculate the bounding box around the shape and then reposition or transform respectively, as now we had a fixed reference point. So after doing some calculations with the mouse positions and the bounding box points at last, I was able to clamp all the shapes sucessfully.

`DrawPanel UI <https://github.com/fury-gl/fury/pull/599>`_


.. image:: https://user-images.githubusercontent.com/64432063/175497817-21974f43-d82b-4245-b93d-2db1081e6b04.gif
    :width: 450
    :align: center

While testing these changes, I found an issue that whenever we just do a single *left mouse click* on the shape, it automatically translated a bit. As you can see below.

.. image:: https://user-images.githubusercontent.com/32108826/175790660-e4b05269-e8d3-44e9-92e1-336c0eeb34ca.gif
    :width: 400
    :align: center 

This was due to the difference between the global mouse click position and the canvas position, which was then fixed by converting the mouse click position to the relative canvas position.

Along with this, I tried to add some control points using `Disk2D` for the shape so that we can use them later on to transform the shapes.

`Control points <https://github.com/ganimtron-10/fury/tree/control-points>`_

.. image:: https://user-images.githubusercontent.com/64432063/177264804-15e67715-b714-4e33-ac68-423375d81740.gif
    :width: 300
    :align: center

Also, to enhance the visualization of the bounding box, I added a box border covering the shapes.

`Bounding Box Borders <https://github.com/ganimtron-10/fury/tree/bb-border>`_

.. image:: https://user-images.githubusercontent.com/64432063/177264077-a8859a96-e3f7-444c-9760-8b9b17542f0f.gif
    :width: 300
    :align: center


Did you get stuck anywhere?
---------------------------
No, Everything worked out fine.

What is coming up next?
-----------------------
Enhancing the control points to work perfectly.
