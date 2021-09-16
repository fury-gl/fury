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

.. post:: August 23 2021
   :author: Antriksh Misri
   :tags: google
   :category: gsoc

-  **Name:** Antriksh Misri
-  **Organisation:** Python Software Foundation
-  **Sub-Organisation:** FURY
-  **Project:** `FURY: Create new user interface widget <https://github.com/fury-gl/fury/wiki/Google-Summer-of-Code-2021#project-3-create-new-user-interface-widget>`_

Proposed Objectives
-------------------

* Add support for Layouts in UI elements
* Add support for Horizontal Layout
* Add support for Vertical Layout
* Add support for Layout along X, Y, Z axes.
* Stretch Goals:

  * Add Tree2D UI element to the UI sub-module
  * Add Accordion2D UI element to the UI sub-module
  * Add SpinBox2D UI element to the UI sub-module

Objectives Completed
--------------------


-  **Add support for Horizontal Layout**

   Added support for Horizontal Layout in the layout module. This layout allows the user to stack actors in a horizontal fashion. Primarily, should be used for laying out UI elements as there is no meaning of horizontal/vertical in 3D space.

   *Pull Requests:*

   -  **Horizontal Layout:** https://github.com/fury-gl/fury/pull/480
   -  **Ribbon Representation demo:** https://github.com/fury-gl/fury/pull/480

- **Add support for Vertical Layout**

  Added support for Vertical Layout in the layout module. This layout allows the user to stack actors in a vertical fashion. Primarily, should be used for laying out UI elements as there is no meaning of horizontal/vertical in 3D space.

  *Pull Requests:*

  - **Vertical Layout:** https://github.com/fury-gl/fury/pull/479
  - **Vertical Layout demo:** https://github.com/fury-gl/fury/pull/479

- **Add support for Layout along X, Y, Z axes**

  Added support for Layout along x, y, z axes. Allows user to layout different actors along any given axes. Also it allows users to switch the stacking order by passing a axis+ or axis- to the constructor.

  *Pull Requests:*

  - **X, Y, Z axes Layout:** https://github.com/fury-gl/fury/pull/486
  - **X, Y, Z axes Layout demo:** https://github.com/fury-gl/fury/pull/486

- **Add Tree2D UI element to the UI sub-module**

  Added Tree2D UI element to the UI sub-module. This allows user to visualize some data in a hierarchical fashion. Each node inside the tree can have N child nodes and the depth can be infinite. Each node can be clicked to trigger a user defined callback to perform some action. Tests and two demos were added for this UI element. Below is a screenshot for reference:

  .. image:: https://camo.githubusercontent.com/dd23b7c8503e4d01c80f2d9e84ee173e06c61eeb7c348c35aeadc75f722647ca/68747470733a2f2f692e696d6775722e636f6d2f4e49334873746c2e706e67
        :width: 200
        :height: 200

  *Pull Requests:*

  - **Tree2D UI element:** https://github.com/fury-gl/fury/pull/460
  - **Tree2D UI element demo:** https://github.com/fury-gl/fury/pull/460

- **Add Accordion2D UI element to the UI sub-module**

  Added Accordion2D to the UI sub-module. This Ui element allows users to visulize data in a tree with depth of one. Each node has a title and a content panel. The children for each node can be N if and only if the children are not nodes themselves. The child UIs can be placed inside the content panel by passing some coordinates, which can be absolute or normalized w.r.t the node content panel size. Tests and two demos were added for this UI element. Below is a screenshot for reference

  .. image:: https://camo.githubusercontent.com/9395d0ea572d7f253a051823f02496450c9f79d19ff0baf32841ec648b6f2860/68747470733a2f2f692e696d6775722e636f6d2f7854754f645a742e706e67
        :width: 200
        :height: 200

  *Pull Requests:*

  - **Accordion2D UI element:** https://github.com/fury-gl/fury/pull/487
  - **Accordion2D UI element demo:** https://github.com/fury-gl/fury/pull/487

Objectives in Progress
----------------------

-  **Add support for Layout in UI elements**

   Currently all the available layouts are only available for actors i.e. of type vtkActor2D. In order to add support for the layouts in UI elements there needs to be some tweaking in the base Layout class. Currently, the PR that adds these functionalities in stalling because of some circular imports. These will hopefully be fixed soon and as soon as the circular imports are fixed, the PR will be merged.

   *Pull Requests:*

   - **Add support for Layout in UI elements:** https://github.com/fury-gl/fury/pull/443

-  **Method to process and load sprite sheets**

   This method adds support for loading and processing a sprite sheet. This will be very useful in playing animations from a n*m sprite sheet. This also has a flag to convert the processed chunks into vtkimageData which can be directly used to update the texture in some UI elements. The primary use of this method will in a tutorial for Card2D, wherein, the image panel of the card will play the animation directly from the sprite sheet.

   *Pull Requests:*

   - **Method to process and load sprite sheets:** https://github.com/fury-gl/fury/pull/491

Other Objectives
----------------

-  **Add Card2D UI element to UI sub-module**

   Added Card2D UI element to the UI sub-module. A Card2D is generally divided into two parts i.e. the image content and the text content. This version of card has an image which can be fetched from a URL and the text content which is yet again divided into two parts i.e. the title and the body. The space distribution between the image and the text content is decided by a float between 0 and 1. A value of 0 means the image takes up no space and a value of 1 means the image consumes the whole space. Below is a demonstration:

   .. image:: https://camo.githubusercontent.com/a2e461352799b6490088de15ac041162d7bf8adf9c07485ea921b525fecd0a8e/68747470733a2f2f692e696d6775722e636f6d2f446c69537066302e676966
        :width: 200
        :height: 200
 
   *Pull Requests:*

   - **Add Card2D UI element to UI sub-module:**  https://github.com/fury-gl/fury/pull/398

-  **Resize Panel2D with WindowResizeEvent or from corner placeholder**

   Currently, the size of the Panel2D is static and cannot be changed dynamically. The size is passed in during the initialization and cannot be changed easily at runtime. This PR adds support for resizing the Panel2D dynamically by adding a placeholder icon at the bottom right corner of the panel. This icon can be click and dragged on to change the size accordingly. Other than this, the panel also retains a specific size ratio when the window is resized. This means if the window is resized in any direction the panel adapts itself w.r.t the updated size. This is done by adding relevant observers for the WindowResizeEvent and binding the relevant callback to it. Below is a quick demonstration:

    .. image:: https://camo.githubusercontent.com/3b1bf6a1b6522a6079055ff196551362fcf89a41b35ac4b32315ce02333e496d/68747470733a2f2f692e696d6775722e636f6d2f3837504e3754512e676966
        :width: 200
        :height: 200

   *Pull Requests:*

   - **Resize Panel2D with WindowResizeEvent or from corner placeholder:**  https://github.com/fury-gl/fury/pull/446

-  **Added the watcher class to UI**

   This PR adds support for a watcher class in the UI elements. The purpose of this class is to monitor a particular attribute from the UI element after it has been added to the scene. If the attribute changes in the real time, a user defined callback is triggered and the scene is force rendered.

   *Pull Requests:*

   - **Added wathcer class to the UI sub-module:**  https://github.com/fury-gl/fury/pull/448

-  **Added support for borders in Panel2D**

   The Panel2D previously, didn't support any sort of effect, the main reason behind this is that, all UI elements are individual entities that are comprised of different actors. These are not the widgets provided by vtk and in order to have some effects provided by vtk shaders must be involved. This obviously makes the whole system very complicated. The border on the other hand uses 4 Rectangle2Ds to draw the 4 borders. This makes the whole process easier but makes the Panel2D very performance heavy as we are adding 5 actors to the scene. Future iterations will replace these rectangles by textures, that way we don't compromise performance and we can have different patterns in the border. Below is a demonstration:

   .. image:: https://user-images.githubusercontent.com/54466356/121709989-bd340280-caf6-11eb-9b8a-81c65260d277.png
        :width: 200
        :height: 200
 
   *Pull Requests:*

   - **Added support for borders in Panel2D:**  https://github.com/fury-gl/fury/pull/441

-  **GSoC weekly Blogs**

    Weekly blogs were added for FURY's Website.

    *Pull Requests:*

    - **First Evaluation:** https://github.com/fury-gl/fury/pull/477

    - **Second Evaluation:** https://github.com/fury-gl/fury/pull/494

Timeline
--------

+-----------------------+------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| Date                  | Description                                                      | Blog Link                                                                                                                                             |
+=======================+==================================================================+=======================================================================================================================================================+
| Week 1(08-06-2021)    | Welcome to my weekly Blogs!                                      | `Weekly Check-in #1 <https://blogs.python-gsoc.org/en/antrikshmisris-blog/week-1-welcome-to-my-weekly-blogs/>`__                                      |
+-----------------------+------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 2(14-06-2021)    | Feature additions in UI and IO modules                           | `Weekly Check-in #2 <https://blogs.python-gsoc.org/en/antrikshmisris-blog/week-2-feature-additions-in-ui-and-io-modules/>`__                          |
+-----------------------+------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 3(21-06-2021)    | Adapting GridLayout to work with UI                              | `Weekly Check-in #3 <https://blogs.python-gsoc.org/en/antrikshmisris-blog/week-3-adapting-gridlayout-to-work-with-ui/>`__                             |
+-----------------------+------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 4(28-06-2021)    | Adding Tree UI to the UI module                                  | `Weekly Check-in #4 <https://blogs.python-gsoc.org/en/antrikshmisris-blog/week-4-adding-tree-ui-to-the-ui-module/>`__                                 |
+-----------------------+------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 5(05-07-2021)    | Rebasing all PR's w.r.t the UI restructuring, Tree2D, Bug Fixes  | `Weekly Check-in #5 <https://blogs.python-gsoc.org/en/antrikshmisris-blog/week-5-rebasing-all-pr-s-w-r-t-the-ui-restructuring-tree2d-bug-fixes/>`__   |
+-----------------------+------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 6(12-07-2021)    | Bug fixes, Working on Tree2D UI                                  | `Weekly Check-in #6 <https://blogs.python-gsoc.org/en/antrikshmisris-blog/week-6-bug-fixes-working-on-tree2d-ui/>`__                                  |
+-----------------------+------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 7(19-07-2021)    | Finalizing the stalling PR's, finishing up Tree2D UI.            | `Weekly Check-in #7 <https://blogs.python-gsoc.org/en/antrikshmisris-blog/week-7-finalizing-the-stalling-pr-s-finishing-up-tree2d-ui/>`__             |
+-----------------------+------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 8(26-07-2020)    | Code Cleanup, Finishing up open PR's, Continuing work on Tree2D. | `Weekly Check-in #8 <https://blogs.python-gsoc.org/en/antrikshmisris-blog/week-8-code-cleanup-finishing-up-open-pr-s-continuing-work-on-tree2d/>`__   |
+-----------------------+------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 9(02-08-2021)    | More Layouts!                                                    | `Weekly Check-in #9 <https://blogs.python-gsoc.org/en/antrikshmisris-blog/week-9-more-layouts/>`__                                                    |
+-----------------------+------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 10(09-08-2021)   | Accordion UI, Support for sprite sheet animations.               | `Weekly Check-in #10 <https://blogs.python-gsoc.org/en/antrikshmisris-blog/week-10-accordion-ui-support-for-sprite-sheet-animations/>`__              |
+-----------------------+------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 11(16-08-2021)   | More tutorials for Accordion2D, Finalizing remaining PRs.        | `Weekly Check-in #11 <https://blogs.python-gsoc.org/en/antrikshmisris-blog/week-11-2/>`__                                                             |
+-----------------------+------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------+



Detailed weekly tasks and work done can be found
`here <https://blogs.python-gsoc.org/en/antrikshmisris-blog/>`_.