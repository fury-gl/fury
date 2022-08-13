=========================================
Week 9 - Grouping and Transforming Shapes
=========================================

.. post:: August 3 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
I started this week by creating a quick PR `#645 <https://github.com/fury-gl/fury/pull/645>`_ for the UI sliders. The sliders raised ``ZeroDivsionError`` when the min and max values were the same. To solve this, I handled the case where the value_range becomes zero and then manually set the handle position to zero.

Then I updated the implementation of the Grouping Shapes feature to support the translation and rotation of the shapes grouped together, as you can see below.

.. image:: https://user-images.githubusercontent.com/64432063/183248609-4281087c-c930-4141-907a-5a906732524a.gif
    :width: 400
    :align: center

After this, I worked on the `PolyLine` and removed the extra point being added to the start of the `PolyLine` whenever a new instance was created.

.. image:: https://user-images.githubusercontent.com/64432063/183280803-5d7ae350-f080-478d-8a2f-a71460037ea4.gif
    :width: 400
    :align: center

Did you get stuck anywhere?
---------------------------
No, everything went well.

What is coming up next?
-----------------------
Completing the PR `#623 <https://github.com/fury-gl/fury/pull/623>`_ and `#653 <https://github.com/fury-gl/fury/pull/653>`_.
Searching different approaches for implementing the freehand drawing.