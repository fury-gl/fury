==============================
Week 3 - Dealing with Problems
==============================

.. post:: June 22 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
This week was full of researching, playing around with things, prototyping ideas, etc.
I started with last week's clamping issue and managed to solve this issue while drawing shapes by clipping the mouse position according to canvas size, but it didn't solve the problem with translating these shapes. I tried solving this using various approaches, but if one thing would get fixed, other things would raise a new error.

Instead of spending too much time on this, I thought it would be great to switch to other work and discuss this problem with the mentors. So I started to create a basic prototype for the properties panel, which would be used to manipulate different properties of the selected shape.

.. image:: https://user-images.githubusercontent.com/64432063/176094716-6be012b8-36c5-43e7-a981-612dbd37ab09.gif
    :width: 300
    :align: center

But then the question arises `How to efficiently take data from the user?`, `Which data format would be easy to compute and best for user experience?` and so on.

Alongside I was trying to create polylines but whenever I wanted to start the creation of the second line the dragging event wasn't triggered as each dragging event required a left mouse click event associated with it. 
I tried to do that manually but it didn't work.

Did you get stuck anywhere?
---------------------------
As I mentioned when I translated the shapes it would go out of the canvas bounds. Here the problem was with the reference point by which I was calculating all the transformations it changed depending on various cases as shown below.

.. image:: https://user-images.githubusercontent.com/64432063/176093855-6129cc25-d03d-45ba-872e-c8d2c6329a1e.gif
    :width: 400
    :align: center

This became more complex when working with a line because then the cases again differ depending on the quadrant in which the line lies.
I worked around and tried to compute the bounds but it wasn't getting updated as the shape transformed.

What is coming up next?
-----------------------
Solving the clamping issue to restrict the shape's position to be in the canvas boundaries.
