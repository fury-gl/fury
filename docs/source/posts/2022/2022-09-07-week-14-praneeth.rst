=========================================
Week 14 - Updating DrawPanel architecture
=========================================

.. post:: September 7 2022
   :author: Praneeth Shetty 
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
This week I continued updating the DrawShape and DrawPanel. 

So as we can see below, whenever we create, translate, or rotate the shapes on the panel, it sometimes overlaps the `mode_panel` or `mode_text` which are used to select and show the current mode respectively.

.. image:: https://user-images.githubusercontent.com/64432063/188268649-65ea24f0-3f46-4545-8e52-e07513a94b9f.gif
    :width: 400
    :align: center

To resolve this, I created a PR `#678 <https://github.com/fury-gl/fury/pull/678>`_ which moves the `mode_panel` and the `mode_text` to be on the borders of the panel.

.. image:: https://user-images.githubusercontent.com/64432063/188268804-949ec656-7da3-4310-ba8b-7e4f0281faa1.gif
    :width: 400
    :align: center

Along this, there were some similar functionalities in the `Grouping Shapes PR <https://github.com/fury-gl/fury/pull/653>`_ and the `DrawShape` due to which some lines of code were repeating. To remove this duplicacy I created a PR `#679 <https://github.com/fury-gl/fury/pull/679>`_ to move these functions to the `helper.py` file.

Then I tried different ways of highlighting the shapes,

1. To create a copy of the shape and scale it in the background so that it looks like a border or highlighted edges.

2. Add yellow color to the shape so that it looks brighter.

Did you get stuck anywhere?
---------------------------
No, I didn't get stuck this week.

What is coming up next?
-----------------------
Working on these new PRs to get them merged. Implement a highlighting feature.
