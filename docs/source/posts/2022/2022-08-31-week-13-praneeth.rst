==========================================
Week 13 - Separating tests and fixing bugs
==========================================

.. post:: August 31 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
This week I managed to fix the translation issue in PR `#653 <https://github.com/fury-gl/fury/pull/653>`_. This was happening because of the calculation error while repositioning the shapes. Now it works as intended.

.. image:: https://user-images.githubusercontent.com/64432063/187058183-840df649-163e-44dd-8104-27d6c2db87a9.gif
    :width: 400
    :align: center

Also, the PR `#623 <https://github.com/fury-gl/fury/pull/623>`_ got merged as now the tests were passing after the update.

As we are now adding new features to the DrawPanel, the current tests are becoming bigger and bigger.
Due to this creating, debugging, and managing tests are becoming harder.
So to keep things simple, separating the tests to validate individual parts and features of the DrawPanel. This was done in PR `#674 <https://github.com/fury-gl/fury/pull/674>`_.

Along this, there was a redundant parameter called `in_progress`, which was used to keep a check whether the shapes are added to the panel or not, but it was confusing, and discarding it didn't affect the codebase. So PR `#673 <https://github.com/fury-gl/fury/pull/673>`_ removed that parameter and separated the two functions.


Did you get stuck anywhere?
---------------------------
Debugging the issue in PR `#653 <https://github.com/fury-gl/fury/pull/653>`_ took most of the time this week. I had to manually calculate and check whether the calculations were correct or not. 

What is coming up next?
-----------------------
Currently, the borders around the shapes are not actually the borders, they are the bounding box of that shape. So I have to find out some ways to highlight the shapes when selected by the user.