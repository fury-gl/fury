.. image:: https://developers.google.com/open-source/gsoc/resources/downloads/GSoC-logo-horizontal.svg
   :height: 50
   :align: center
   :target: https://summerofcode.withgoogle.com/projects/#6653942668197888

.. image:: https://www.python.org/static/community_logos/python-logo.png
   :width: 40%
   :target: https://blogs.python-gsoc.org/en/nibba2018s-blog/

.. image:: https://python-gsoc.org/logos/FURY.png
   :width: 25%
   :target: https://fury.gl/latest/community.html

Google Summer of Code Final Work Product
========================================

.. post:: August 24 2020
   :author: Soham Biswas
   :tags: google
   :category: gsoc

-  **Name:** Soham Biswas
-  **Organisation:** Python Software Foundation
-  **Sub-Organisation:** FURY
-  **Project:** `FURY - Create new UI Widgets & Physics Engine
   Integration <https://github.com/fury-gl/fury/wiki/Google-Summer-of-Code-2020>`_

Proposed Objectives
-------------------

-  ComboBox
-  Tab UI
-  File Dialog Improvements

Modified Objectives
-------------------

-  Combobox
-  Tab UI
-  File Dialog Improvements
-  Double Click Callback
-  TextBlock2D Improvements
-  Scrollbars as a Standalone Component
-  Physics Engine Integration

Objectives Completed
--------------------

-  **ComboBox2D UI Component**

   A combobox is a commonly used graphical user interface widget.
   Traditionally, it is a combination of a drop-down list or list box and a
   single-line textbox, allowing the user to select a value from the list.
   The term "combo box" is sometimes used to mean "drop-down list".
   Respective components, tests and tutorials were created.

   *Pull Requests:*

   -  **Combobox UI component:** https://github.com/fury-gl/fury/pull/240
   -  **Combobox UI Tutorial:** https://github.com/fury-gl/fury/pull/246

-  **Tab UI Component**

   In interface design, a tabbed document interface or Tab is a graphical
   control element that allows multiple documents or panels to be contained
   within a single window, using tabs as a navigational widget for
   switching between sets of documents. Respective components, tests and
   tutorials were created.

   *Pull Requests:*

   -  **Tab UI component:** https://github.com/fury-gl/fury/pull/252
   -  **Tab UI tutorial:** https://github.com/fury-gl/fury/pull/275

-  **Double Click Callback**

   Double click callbacks aren't implemented in VTK by default so they need
   to be implemented manually. With my mentor's help I was able to
   implement double click callbacks for all the three mouse buttons
   successfully.

   *Pull Requests:*

   -  **Adding Double Click Callback:**
      https://github.com/fury-gl/fury/pull/231

-  **TextBlock2D Improvements**

   The previous implementation of ``TextBlock2D`` was lacking a few
   features such as size arguments and text overflow. There was no specific
   way to create Texts occupying a said height or width area. Apart from
   that UI components like ``ListBoxItem2D``, ``FileMenu2D`` etc had an
   issue where text would overflow from their specified width. In order to
   tackle these problems, a modification was done to ``TextBlock2D`` to
   accept size as an argument and a new method was added to clip
   overflowing text based on a specified width and to replace the
   overflowing characters with ``...``.

   *Pull Requests:*

   -  **Setting Bounding Box for TextBlock2D:**
      https://github.com/fury-gl/fury/pull/248
   -  **Clip Text Overflow:** https://github.com/fury-gl/fury/pull/268

-  **Physics Engine Integration**

   Optional support for Physics engine integration of Pybullet was added to
   Fury. Pybullet's engine was used for the simulations and FURY was used
   for rendering the said simulations. Exhaustive examples were added to
   demonstrate various types of physics simulations possible using pybullet
   and fury. The said examples are as follows:

   -  Brick Wall Simulation

      -  Explains how to render and simulate external forces, objects and
         gravity.

   -  Ball Collision Simulation

      -  Explains how collisions work and how to detect said collisions.

   -  Chain Simulation

      -  Explains how to render and simulate joints.

   -  Wrecking Ball Simulation

      -  A more complicated simulation that combines concepts explained by
         the other examples.

   Apart from that, a document was created to explain the integration
   process between pybullet and fury in detail.

   *Pull Requests:*

   -  **Physics Simulation Examples:**
      https://github.com/fury-gl/fury/pull/287
   -  **Fury-Pybullet Integration Docs:**
      https://docs.google.com/document/d/1XJcG1TL5ZRJZDyi8V76leYZt_maxGp0kOB7OZIxKsTA/edit?usp=sharing

Objectives in Progress
----------------------

-  **Scrollbars as a standalone component**

   The previous implementation of scrollbars were hard coded into
   ``ListBox2D``. Therefore, it was not possible to use scrollbars with any
   other UI component. Apart from that, scrollbars in terms of design were
   limited. Creating a horizontal scrollbar was not possible. The objective
   of this PR is to make scrollbars separate so that other UI elements can
   also make use of it.

   Currently, the skeletal and design aspects of the scrollbars are
   implemented but the combination of scrollbars with other UI components
   are still in progress.

   *Pull Requests:*

   -  **Scrollbars as a Standalone API:**
      https://github.com/fury-gl/fury/pull/285

-  **File Dialog Improvements**

   Currently, we have access to ``FileMenu2D`` which allows us to navigate
   through the filesystem but it does not provide a user friendly Dialog to
   read and write files in Fury. Hence the idea is to create a file dialog
   which can easily open or save file at runtime. As of now, ``Open`` and
   ``Save`` operations are implemented. Corresponding tests and tutorials
   are in progress.

   *Pull Requests:*

   -  **File Dialog UI component:**
      https://github.com/fury-gl/fury/pull/294

Other Objectives
----------------

-  **Radio Checkbox Tutorial using FURY API**

   The objects for Radio button and Checkbox tutorial were rendered using
   VTK's method by a fellow contributor so I decided to replace them with
   native FURY API. The methods were rewritten keeping the previous commits
   intact.

   *Pull Requests:*

   -  **Radio Checkbox tutorial using FURY API:**
      https://github.com/fury-gl/fury/pull/281

-  **GSoC weekly Blogs**

   Weekly blogs were added for FURY's Website.

   *Pull Requests:*

   -  **First & Second Evaluation:**
      https://github.com/fury-gl/fury/pull/272
   -  **Third Evaluation:** https://github.com/fury-gl/fury/pull/286

Timeline
--------

+-----------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
| Date                  | Description                                                      | Blog Link                                                                                          |
+=======================+==================================================================+====================================================================================================+
| Week 1(30-05-2020)    | Welcome to my GSoC Blog!!                                        | `Weekly Check-in #1 <https://blogs.python-gsoc.org/en/nibba2018s-blog/weekly-check-in-1-5/>`__     |
+-----------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
| Week 2(07-06-2020)    | First Week of Coding!!                                           | `Weekly Check-in #2 <https://blogs.python-gsoc.org/en/nibba2018s-blog/weekly-check-in-2-3/>`__     |
+-----------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
| Week 3(14-06-2020)    | ComboBox2D Progress!!                                            | `Weekly Check-in #3 <https://blogs.python-gsoc.org/en/nibba2018s-blog/weekly-check-in-3-4/>`__     |
+-----------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
| Week 4(21-06-2020)    | TextBlock2D Progress!!                                           | `Weekly Check-in #4 <https://blogs.python-gsoc.org/en/nibba2018s-blog/weekly-check-in-4-4/>`__     |
+-----------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
| Week 5(28-06-2020)    | May the Force be with you!!                                      | `Weekly Check-in #5 <https://blogs.python-gsoc.org/en/nibba2018s-blog/weekly-check-in-5-4/>`__     |
+-----------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
| Week 6(05-07-2020)    | Translation, Reposition, Rotation.                               | `Weekly Check-in #6 <https://blogs.python-gsoc.org/en/nibba2018s-blog/weekly-check-in-6-7/>`__     |
+-----------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
| Week 7(12-07-2020)    | Orientation, Sizing, Tab UI.                                     | `Weekly Check-in #7 <https://blogs.python-gsoc.org/en/nibba2018s-blog/weekly-check-in-7-4/>`__     |
+-----------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
| Week 8(19-07-2020)    | ComboBox2D, TextBlock2D, ClippingOverflow.                       | `Weekly Check-in #8 <https://blogs.python-gsoc.org/en/nibba2018s-blog/weekly-check-in-8-2/>`__     |
+-----------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
| Week 9(26-07-2020)    | Tab UI, TabPanel2D, Tab UI Tutorial.                             | `Weekly Check-in #9 <https://blogs.python-gsoc.org/en/nibba2018s-blog/weekly-check-in-9-4/>`__     |
+-----------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
| Week 10(02-08-2020)   | Single Actor, Physics, Scrollbars.                               | `Weekly Check-in #10 <https://blogs.python-gsoc.org/en/nibba2018s-blog/weekly-check-in-10-2/>`__   |
+-----------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
| Week 11(09-08-2020)   | Chain Simulation, Scrollbar Refactor,Tutorial Update.            | `Weekly Check-in #11 <https://blogs.python-gsoc.org/en/nibba2018s-blog/weekly-check-in-11-1/>`__   |
+-----------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
| Week 12(16-08-2020)   | Wrecking Ball Simulation, ScrollbarsUpdate, Physics Tutorials.   | `Weekly Check-in #12 <https://blogs.python-gsoc.org/en/nibba2018s-blog/weekly-check-in-12/>`__     |
+-----------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
| Week 13(23-08-2020)   | Part of the Journey is the end unless itsOpen Source!            | `Weekly Check-in #13 <https://blogs.python-gsoc.org/en/nibba2018s-blog/weekly-check-in-13/>`__     |
+-----------------------+------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+

Detailed weekly tasks and work done can be found
`here <https://blogs.python-gsoc.org/en/nibba2018s-blog/>`__.
