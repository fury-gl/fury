==============================================
Week 1 - Laying the Foundation of DrawPanel UI
==============================================

.. post:: June 8 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc

This week we started with our first technical meeting in which the weekly tasks were assigned. So I had to start with some background or canvas and draw a line using mouse clicks.


What did you do this week?
--------------------------

I started with a simple ``Panel2D`` (which is basically a movable rectangle on which we can add other UI elements) as a background and then assigned its mouse click callback to print “**Clicked!**” to verify the event triggering.

Then I modified the event callback to create a ``Rectangle2D`` at that current mouse position(Now you would ask, Why ``Rectangle2D``?? We wanted to change the width of the line, which wasn't possible with the regular line). This would create a rectangle at that point but it had size.
So I had to then calculate the distance between the first point where the mouse was clicked and the current position to resize it accordingly. 

.. image:: https://user-images.githubusercontent.com/64432063/174661567-76251ce9-380f-4a41-a572-65865d028a9c.gif
   :width: 400

This thing creates a Rectangle, not a line. So I had to think of some other approach.
The first thing that came to my mind was to keep the width of the rectangle constant and apply some rotation to the rectangle according to the mouse position and this worked!

.. image:: https://user-images.githubusercontent.com/64432063/174661632-98e1c4ec-31a2-4c4d-8e52-7bf2a47592c7.gif
   :width: 400

As previously we created an interactive rectangle(unintentionally), I thought it would be great if I could add different modes for creating different shapes(ie. line, rectangle, circle as these shapes already existed in the UI System).

Considering this I implemented a class to create and manage these shapes and a panel to select which shape is to be drawn along with a ``TextBlock2D`` to show the current mode.

``DrawPanel UI``:

https://github.com/fury-gl/fury/pull/599

.. image:: https://user-images.githubusercontent.com/64432063/174661680-8a5120ff-ec88-4739-945b-b87074f9742b.gif
   :width: 500
   :align: center


Did you get stuck anywhere?
---------------------------
I was facing issues with implementing the rotation. I scanned the utils to find some method to do the same but there wasn't any for ``Actor2D``. Then I read some docs and tried out various functions.
At last, I decided to implement it by the most general method which is to calculate the new rotated points by multiplying them with the rotation matrix and that seemed fine for now!!


What is coming up next?
-----------------------
Deletion of the shapes is to be implemented along with tests and tutorials. 
