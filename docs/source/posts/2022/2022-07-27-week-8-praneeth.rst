========================================
Week 8 - Working on the polyline feature
========================================

.. post:: July 27 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
This week I started working on the `polyline` feature. After a lot of investigating and trying out different things, I found a way to call the dragging event manually without any prior click event. VTK actually captures the mouse movement using the ``MouseMoveEvent``. This event is then modified by FURY to only be called after the click event. So I added a new callback to track the mouse movement and set the current canvas as an active prop because it is required to capture the drag event happening on it.

.. image:: https://user-images.githubusercontent.com/64432063/182432684-abd015e5-b63d-4aab-b6a5-c8ba5dab3252.gif
    :width: 400
    :align: center

I had to look for some ways by which we could make the icons look smoother. For this, I created an icon test file, which consisted of a set of icons of varying scales. Then on these icons, I used some vtkTexture methods discussed in the meeting, such as ``MipmapOn`` and ``InterpolateOn``. You can see some differences in the icons below:

.. image:: https://user-images.githubusercontent.com/64432063/182910990-fe4934ee-4201-4c3c-8ab4-1a4f7bfa9276.png
    :width: 600
    :align: center

Did you get stuck anywhere?
---------------------------
It took some time to get settled with all the things as my college reopened.
I was trying to use shaders with the UI elements to implement the freehand drawing, but then my mentors suggested that I should look  into the tracer widget in the VTK.

What is coming up next?
-----------------------
Updating PR `#623`_ to keep `rotation_slider` on the top of the screen.
Looking into various vtk tracer widgets to see if we can use that to create freehand drawings.

.. _`#623`: https://github.com/fury-gl/fury/pull/623