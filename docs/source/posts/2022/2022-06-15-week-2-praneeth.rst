===============================
Week 2 - Improving DrawPanel UI
===============================

.. post:: June 15 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
This week I had to refactor and make my code cleaner along with some bug fixing, I started by adding tests and tutorials so that my changes could be tested by everyone. Then I separated the mode for *selection* so that it would be easy to select an individual element and work along with it. Once the selection mode was complete I started with the *deletion* of the elements.

Now, as we have various modes the question arises, How do we know which mode is currently active and how would users know it?? For this, I took references from some other similar applications and came up with an idea to toggle the icon of the selected mode whenever the mode is selected/deselected as we had individual buttons with an icon for each mode. 

https://github.com/fury-gl/fury/pull/599

.. image:: https://user-images.githubusercontent.com/64432063/174710174-87604e78-e563-458d-8db7-28941301b02c.gif
   :width: 300
   :height: 300
   :align: center

Along with this I also started to learn how to access key events so that in near future some keyboard shortcuts can be added.


Did you get stuck anywhere?
---------------------------
No, Everything went well.


What is coming up next?
-----------------------
In the current version of the DrawPanel, the shapes can be created and translated beyond the canvas boundaries, so we have to clamp the positions according to the canvas.
