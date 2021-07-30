Week #1: Welcome to my weekly Blogs!
====================================

.. post:: June 08 2021
   :author: Antriksh Misri
   :tags: google
   :category: gsoc

Hi everyone! I am **Antriksh Misri**. I am a Pre-Final year student at MIT Pune. This summer, I will be working on **Layout Management** under FURY's `UI <https://fury.gl/latest/reference/fury.ui.html>`_ module as my primary goal. This includes addition of different classes under Layout Management to provide different layouts/arrangements in which the UI elements can be arranged. As my stretch goals, I will be working on a couple of UI components for FURY’s UI module. These 2D and 3D components will be sci-fi like as seen in the movie “**Guardians of The Galaxy**”. My objective for the stretch goals would be to develop these UI components with their respective test and tutorials such that it adds on to the UI module of FURY and doesn’t hinder existing functionalities/performance.

What did I do this week?
------------------------
During the community bonding period I got to know the mentors as well as other participants. We had an introductory meeting, in which the rules and code of conduct was explained. Also, my proposal was reviewed and modified slightly. Initially, I had to develop UI elements as my primary goal and I had to work on layout management as my stretch goals but the tasks were switched. Now I have to work on Layout Management as my primary task and develop UI in the stretch goals period. I also started coding before hand to actually make use of this free period. I worked on different PR's which are described below:-

* `Added tests for Layout module <https://github.com/fury-gl/fury/pull/434>`_ : The layout module of FURY didn't had any tests implemented, so I made this PR to add tests for **Layout** & **GridLayout** class.
* `Complied available classes for Layout Management in different libraries <https://docs.google.com/document/d/1zo981_cyXZUgMDA9QdkVQKAHTuMmKaixDRudkQi4zlc/edit>`_ : In order to decide the behavior and functionality of Layout Management in FURY, I made a document that has all classes available in different libraries to manage layout of UI elements. This document also contains code snippets for these classes.
* `Resize Panel2D UI on WindowResizeEvent <https://github.com/antrikshmisri/fury/tree/panel-resize>`_ : Currently, the **Panel2D** UI is not responsive to window resizing which means its size is static. In this branch I implemented this feature.

Did I get stuck anywhere?
-------------------------
I got stuck at Panel resizing feature. I couldn't figure out how to propagate the window invoked events to a specific actor. Fortunately, the mentors helped me to solve this problem by using **partial** from **functools**.

What is coming up next?
-----------------------
The next tasks will be decided in this week's open meeting with the mentors.

**See you guys next week!**