.. image:: https://developers.google.com/open-source/gsoc/resources/downloads/GSoC-logo-horizontal.svg
   :height: 50
   :align: center
   :target: https://summerofcode.withgoogle.com/programs/2022/projects/a47CQL2Z

.. image:: https://www.python.org/static/community_logos/python-logo.png
   :width: 40%
   :target: https://summerofcode.withgoogle.com/programs/2022/organizations/python-software-foundation

.. image:: https://python-gsoc.org/logos/FURY.png
   :width: 25%
   :target: https://fury.gl/latest/index.html

Google Summer of Code Final Work Product
========================================

.. post:: January 24 2023
   :author: Praneeth Shetty
   :tags: google
   :category: gsoc

-  **Name:** Praneeth Shetty
-  **Organisation:** Python Software Foundation
-  **Sub-Organisation:** FURY
-  **Project:** `FURY - Improve UI elements for drawing geometrical
   shapes <https://github.com/fury-gl/fury/wiki/Google-Summer-of-Code-2022-(GSOC2022)#project-5-improve-ui-elements-for-drawing-geometrical-shapes>`_


Proposed Objectives
-------------------

-  Visualization UI:

   -  Drawing Geometrical Objects
   -  Moving Components
   -  Rotating Components
   -  Erasing Components
   -  Resizing Components

-  Stretch Goals:

   -  Converting 2D shapes to 3D

Objectives Completed
--------------------


-  **DrawPanel (previously known as Visualization UI)**

   ``DrawPanel`` is the parent component that contains and manages all the other sub-components to efficiently visualize the shapes. The main functions of the ``DrawPanel`` are capturing interactions from user, managing modes, drawing shapes and transforming them. The ``DrawPanel`` is mainly divided into three parts :

   i. **Main Panel**
        It is the main background panel(``Panel2D``) on which the main interaction and visualization happen. Here user can interactively draw shapes, reposition and rotate them. This panel also defines the boundaries for the shapes. It can also be called as a container element as it contains all the shapes and other DrawPanel components.
   ii. **Mode Panel**
        It is a composite UI element consisting of the main panel(``Panel2D``) on which buttons(``Button2D``) are arranged which can toggle the current working mode. Each button has an icon associated with it which tries to depict the information about the mode. Here mode is nothing but the different channels which on selection can perform different tasks. Some of the modes present in the Mode Panel are discussed below:
            -  Selection:  This mode is used to select an individual or group of shapes.
            -  Deletion:  This mode is used to delete an individual or group of shapes.
            -  The modes mentioned below create an element on the Panel which is described below.
                -  Line
                -  Quad
                -  Circle
                -  Polyline
                -  Freehand drawing
            -  To activate any of these above mode the user has to click on the button with the respective icon present in the mode panel and then interact with the main panel.
   iii. **Mode Text** It is a ``TextBlock2D`` which displays the current mode of the ``DrawPanel``. It automatically updates whenever the mode is changed. This helps the user to quickly identify which mode is he currently in.

   *Pull Requests:*

   -  **Creating DrawPanel UI (Merged) :**
      `https://github.com/fury-gl/fury/pull/599 <https://github.com/fury-gl/fury/pull/599>`_

    .. image:: https://user-images.githubusercontent.com/64432063/194766188-c6f83b75-82d1-455c-9be1-d2a1cada945a.png
        :width: 400
        :align: center

-  **Drawing Shapes:**

   A new class called ``DrawShape`` was create to manage all the transformation and to handle the user interaction which are passed by the ``DrawPanel``. To create a shape the required mode can be selected from the mode panel and then on the left_mouse_click event the shape creation starts. Then to resize user needs to drag the mouse depending on how he wants the shape to be. These interactions follow WYSIWYG (What You See Is What You Get) principle. Currently, the following shapes are supported:

   1. Line:  Creates a line joining two points using ``Rectangle2D``.
   2. Quad:  Creates a rectangle using ``Rectangle2D``.
   3. Circle:  Create a Circle using ``Disk2D``.
   4. Polyline:  Creates a chain of lines that can either end at the starting point and create a loop or remain an independent collection of lines. Individual line is created using ``Rectangle2D``.

      -  **DrawPanel Feature: Polyline (Under Review) :**
         `https://github.com/fury-gl/fury/pull/695 <https://github.com/fury-gl/fury/pull/695>`__

   5. Freehand drawing:  Here you can draw any freehand object something similar to doodling. Internally we use ``Polyline`` for doing this.

      -  **DrawPanel Feature: Freehand Drawing (Under Review) :**
         `https://github.com/fury-gl/fury/pull/696 <https://github.com/fury-gl/fury/pull/696>`__

      .. image:: https://user-images.githubusercontent.com/64432063/194773058-b074fde0-e2e1-4719-93e3-38a34032cd88.jpg
        :width: 400
        :align: center

-  **Transforming Shapes:**

   Following transformation are supported by every ``DrawShape``

   -  **Translation**

      The translation is nothing but repositioning the shapes on the main panel. It is made sure that the shapes don't exceed the panel boundaries by clamping the new position between the panel bounds. All the UI elements have a center property which can be used to do the above-mentioned thing but the reference point of the Shape may change depending on how it was created. So to resolve this I created an interface that would calculate and return the bounding box data around the shape and which could be then used to reposition the shape on the panel.

      .. image:: https://user-images.githubusercontent.com/64432063/194772993-289e10bd-199d-4692-bcb0-5cccdb1b32fe.gif
        :width: 400
        :align: center

   -  **Rotation**

      Each ``DrawShape`` can be rotated from the center of that shape. Whenever you select a shape using the selection mode a rotation slider(RingSlider2D) appears at the lower right corner of the ``DrawPanel``. This rotation slider can be used to rotate the shapes by some specific angle which is displayed at the center of the slider.

      .. image:: https://user-images.githubusercontent.com/64432063/194773295-4303ec78-3f2b-44e5-8c85-ff01140a8c95.gif
        :width: 400
        :align: center

   *Pull Requests:*

   -  **DrawPanel Feature: Adding Rotation of shape from Center (Merged) :**

      `https://github.com/fury-gl/fury/pull/623 <https://github.com/fury-gl/fury/pull/623>`__

-  **Deleting Shapes:**

   Whenever we create anything it's never perfect we change, modify, and at last delete. Here too every DrawShape is never perfect so to delete the shapes we also have a delete option that can be chosen from the mode panel and by clicking the shape they are removed from the panel.

      .. image:: https://user-images.githubusercontent.com/64432063/194862464-387edc59-a942-4675-ab44-53c899e70e29.gif
        :width: 400
        :align: center

Other Objectives
----------------

-  **Grouping Shapes**

   Many times we need to perform some actions on a group of shapes so here we are with the grouping feature using which you can group shapes together, reposition them, rotate them and delete them together. To activate grouping of shapes you have to be on selection mode then by holding **Ctrl** key select the required shapes and they will get highlighted. To remove shape from the group just hold the **Ctrl** and click the shape again it will get deselected. Then once everything is grouped you can use the normal transformation as normal i.e. for translation just drag the shapes around and for rotation the rotation slider appears at usual lower left corner which can be used.

   *Pull Requests:*

   -  **DrawPanel Feature: Grouping Shapes (Under Review)** - `https://github.com/fury-gl/fury/pull/653 <https://github.com/fury-gl/fury/pull/653>`__

      .. image:: https://user-images.githubusercontent.com/64432063/194926770-e1031181-04c6-491b-89ca-275213060a13.gif
        :width: 400
        :align: center

-  **Creating icons**

   As most of the things in the DrawPanel are visually seen, each mode also require some icons so that users easily understand the use of that mode, so to achieve this I have created some icons by using the pre-existing icons in the FURY. These icons are stored `here <https://github.com/fury-gl/fury-data>`__. Whenever FURY requires these icons they are fetched using the fetchers present in FURY. To fetch these new icons I created some new fetchers.

   *Pull Requests:*

   -  **Adding new icons required for DrawPanel UI (Merged)** - `https://github.com/fury-gl/fury-data/pull/9 <https://github.com/fury-gl/fury-data/pull/9>`__
   -  **Creating a fetcher to fetch new icons (Merged)** - `https://github.com/fury-gl/fury/pull/609 <https://github.com/fury-gl/fury/pull/609>`__
   -  **Adding polyline icons (Merged)** - `https://github.com/fury-gl/fury-data/pull/10 <https://github.com/fury-gl/fury-data/pull/10>`__
   -  **Adding resize and freehand drawing icon (Merged)** - `https://github.com/fury-gl/fury-data/pull/11 <https://github.com/fury-gl/fury-data/pull/11>`__
   -  **Updating fetch_viz_new_icons to fetch new icons (Under Review)** - `https://github.com/fury-gl/fury/pull/701 <https://github.com/fury-gl/fury/pull/701>`__


-  **Other PRs**

   -  **Fixing ZeroDivisionError thrown by UI sliders when the value_range is zero (0) (Merged)**: `https://github.com/fury-gl/fury/pull/645 <https://github.com/fury-gl/fury/pull/645>`__
   -  **DrawPanel Update: Removing in_progress parameter while drawing shapes (Merged)**: `https://github.com/fury-gl/fury/pull/673 <https://github.com/fury-gl/fury/pull/673>`__
   -  **DrawPanel Update: Separating tests to test individual features (Merged)**: `https://github.com/fury-gl/fury/pull/674 <https://github.com/fury-gl/fury/pull/674>`__
   -  **DrawPanel Update: Repositioning the mode_panel and mode_text (Merged)**: `https://github.com/fury-gl/fury/pull/678 <https://github.com/fury-gl/fury/pull/678>`__
   -  **DrawPanel Update: Moving repetitive functions to helpers (Merged)**: `https://github.com/fury-gl/fury/pull/679 <https://github.com/fury-gl/fury/pull/679>`__
   -  **DrawPanel Update: Moving rotation_slider from DrawShape to DrawPanel (Under Review)**: `https://github.com/fury-gl/fury/pull/688 <https://github.com/fury-gl/fury/pull/688>`__

Objectives in Progress
----------------------

-  **Resizing Shapes:**

   Currently after the shape is created we can only transform it but we might need to resize it. To be able to resize I am currently using the borders of the shape itself. You can switch to resize mode and then select the shape. It would display the bounding box around the shape which act as interactive slider and resizes the shape as shown below.

      .. image:: https://user-images.githubusercontent.com/64432063/194775648-04c2fa7a-b22f-4dda-a73b-2f8161bb4f3a.gif
        :width: 400
        :align: center

   -  **DrawPanel Feature: Resizing Shapes (Under Development)**: `https://github.com/ganimtron-10/fury/blob/resize_shapes/fury/ui/elements.py <https://github.com/ganimtron-10/fury/blob/resize_shapes/fury/ui/elements.py>`__

GSoC Weekly Blogs
-----------------

-  My blog posts can be found at `FURY website <https://fury.gl/latest/blog/author/praneeth-shetty.html>`__
   and `Python GSoC blog <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/>`__.

Timeline
--------

+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Date                | Description                                        | Blog Post Link                                                                                                                                                                                            |
+=====================+====================================================+===========================================================================================================================================================================================================+
| Week 0(25-05-2022)  | Pre-GSoC Journey                                   | `FURY <https://fury.gl/latest/posts/2022/2022-05-25-pre-gsoc-journey-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/my-pre-gsoc-22-journey>`__                          |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 1(08-06-2022)  | Laying the Foundation of DrawPanel UI              | `FURY <https://fury.gl/latest/posts/2022/2022-06-08-week-1-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-1-laying-the-foundation-of-drawpanel-ui>`__              |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 2(15-06-2022)  | Improving DrawPanel UI                             | `FURY <https://fury.gl/latest/posts/2022/2022-06-15-week-2-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-2-improving-drawpanel-ui>`__                             |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 3(22-06-2022)  | Dealing with Problems                              | `FURY <https://fury.gl/latest/posts/2022/2022-06-22-week-3-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-3-dealing-with-problems>`__                              |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 4(29-06-2022)  | Fixing the Clamping Issue                          | `FURY <https://fury.gl/latest/posts/2022/2022-06-29-week-4-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-4-fixing-the-clamping-issue>`__                          |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 5(06-07-2022)  | Working on new features                            | `FURY <https://fury.gl/latest/posts/2022/2022-07-06-week-5-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-5-working-on-new-features>`__                            |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 6(13-07-2022)  | Supporting Rotation of the Shapes from the Center  | `FURY <https://fury.gl/latest/posts/2022/2022-07-13-week-6-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-6-supporting-rotation-of-the-shapes-from-the-center>`__  |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 7(20-07-2022)  | Working on Rotation PR and Trying Freehand Drawing | `FURY <https://fury.gl/latest/posts/2022/2022-07-20-week-7-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-7-working-on-rotation-pr-and-trying-freehand-drawing>`__ |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 8(27-07-2022)  | Working on the polyline feature                    | `FURY <https://fury.gl/latest/posts/2022/2022-07-27-week-8-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-8-working-on-the-polyline-feature>`__                    |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 9(03-08-2022)  | Grouping and Transforming Shapes                   | `FURY <https://fury.gl/latest/posts/2022/2022-08-03-week-9-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-9-grouping-and-transforming-shapes>`__                   |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 10(10-08-2022) | Understanding Codes and Playing with Animation     | `FURY <https://fury.gl/latest/posts/2022/2022-08-10-week-10-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-10-understanding-codes-and-playing-with-animation>`__   |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 11(17-08-2022) | Creating a base for Freehand Drawing               | `FURY <https://fury.gl/latest/posts/2022/2022-08-17-week-11-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-11-creating-a-base-for-freehand-drawing>`__             |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 12(24-08-2022) | Fixing translating issues and updating tests       | `FURY <https://fury.gl/latest/posts/2022/2022-08-24-week-12-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-12-fixing-translating-issues-and-updating-tests>`__     |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 13(31-08-2022) | Separating tests and fixing bugs                   | `FURY <https://fury.gl/latest/posts/2022/2022-08-31-week-13-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-13-separating-tests-and-fixing-bugs>`__                 |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 14(07-09-2022) | Updating DrawPanel architecture                    | `FURY <https://fury.gl/latest/posts/2022/2022-09-07-week-14-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-14-updating-drawpanel-architecture>`__                  |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 15(14-09-2022) | Highlighting DrawShapes                            | `FURY <https://fury.gl/latest/posts/2022/2022-09-14-week-15-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-15-highlighting-drawshapes>`__                          |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 16(21-09-2022) | Working with Rotations!                            | `FURY <https://fury.gl/latest/posts/2022/2022-09-21-week-16-praneeth.html>`__ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog/week-16-working-with-rotations/>`__                          |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
