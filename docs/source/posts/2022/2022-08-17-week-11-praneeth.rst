==============================================
Week 11 - Creating a base for Freehand Drawing
==============================================

.. post:: August 17 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
This week I tried to imitate the working of `vtkImageTracer`. Previously, I had created a small prototype for freehand drawing by adding points at the mouse position (which you can check out `here <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-7-working-on-rotation-pr-and-trying-freehand-drawing/>`_). As mentioned, there were some drawback of this method.
So to overcome these issues, I tried combining both methods. Considering points using the previous method and instead of adding points I tried creating lines between them which looks promising. Below you can see a demo.

`Freehand Drawing: <https://github.com/ganimtron-10/fury/tree/freehand-drawing>`_

.. image:: https://user-images.githubusercontent.com/64432063/186952329-636a0d81-6631-4e8d-9486-9a8c5e88a9a7.gif
    :width: 400
    :align: center

While doing this, I started working on how I could efficiently draw the lines and smoothen them out. My mentors referred me `this <https://github.com/rougier/python-opengl/blob/master/09-lines.rst>`_  to learn more about constructing and rendering lines.

Along with this, I updated a few tests and some small bugs in PR `#623 <https://github.com/fury-gl/fury/pull/623>`_ and `#653 <https://github.com/fury-gl/fury/pull/653>`_.

Did you get stuck anywhere?
---------------------------
No, I didn't get stuck anywhere.

What is coming up next?
-----------------------
Enhancing the freehand drawing feature.