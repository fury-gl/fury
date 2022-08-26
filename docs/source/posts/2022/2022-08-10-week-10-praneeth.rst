========================================================
Week 10 - Understanding Codes and Playing with Animation
========================================================

.. post:: August 10 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
I started working on the PR `#645 <https://github.com/fury-gl/fury/pull/645>`_ created last week and tested a few more corner cases such as What happens if the maximum value is exceeded?? What if we give a value less than the minimum value range??
Most of these worked as intended.

Then I moved on to `vtkImageTracer`, I tried implementing this locally. It required an ImageSource on which the tracing was done. As you can see below, I managed to implement the pure VTK version of the widget. After this, I tried integrating this with FURY and besides the ImageSource all things were successfully handled by FURY.

.. image:: https://user-images.githubusercontent.com/64432063/185802405-b289dbc9-08a3-496a-8cc4-ef8c4d40bf60.gif
    :width: 400
    :align: center

As per the suggestions from the mentors, I started looking at the implementation of the `vtkImageTracer <https://github.com/Kitware/VTK/blob/master/Interaction/Widgets/vtkImageTracerWidget.cxx>`_ to see how they manage the widget. My prior experience with C++ helped me a lot with this because the original implementation of vtk is in C++.

Here, I found an approach similar to the polyline. They first grab the current point, check whether it's inside the area, and then use the last point to draw a line by calculating some kind of motion vector.

Using the Animation Architecture created by Mohamed, I created a Bouncing Text Animation using UI elements to check the compatibility of the UI System with the Animation.

`Bouncing text animation: <https://github.com/ganimtron-10/fury/blob/354e56338d197fe2a29b628e86a16ad7c7a845b5/docs/tutorials/02_ui/viz_ui_text_animation.py>`_

.. image:: https://user-images.githubusercontent.com/64432063/185803066-70e320de-0777-478d-87bf-30767b02efe2.gif
    :width: 400
    :align: center

Did you get stuck anywhere?
---------------------------
Proper tutorials weren't there to implement `vtkImageTracer`, which took time to make it work locally.

What is coming up next?
-----------------------
Working on the Freehand Drawing.