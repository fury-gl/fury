.. image:: https://developers.google.com/open-source/gsoc/resources/downloads/GSoC-logo-horizontal.svg
   :height: 50
   :align: center
   :target: https://summerofcode.withgoogle.com/programs/2023/projects/BqfBWfwS

.. image:: https://www.python.org/static/community_logos/python-logo.png
   :width: 40%
   :target: https://summerofcode.withgoogle.com/programs/2023/organizations/python-software-foundation

.. image:: https://python-gsoc.org/logos/FURY.png
   :width: 25%
   :target: https://fury.gl/latest/index.html

Google Summer of Code Final Work Product
========================================

.. post:: August 25 2023
   :author: Praneeth Shetty
   :tags: google
   :category: gsoc

-  **Name:** Praneeth Shetty
-  **Organisation:** Python Software Foundation
-  **Sub-Organisation:** FURY
-  **Project:** `FURY - Update user interface widget + Explore new UI Framework <https://github.com/fury-gl/fury/wiki/Google-Summer-of-Code-2023-(GSOC2023)#project-5-update-user-interface-widget--explore-new-ui-framework>`_


Proposed Objectives
-------------------

- SpinBoxUI
- Scrollbar as Independent Element
- FileDialog
- TreeUI
- AccordionUI
- ColorPickerUI

- Stretch Goals:
    - Exploring new UI Framework
    - Implementing Borders for UI elements

Objectives Completed
--------------------


- **SpinBoxUI:**
	The ``SpinBoxUI`` element is essential for user interfaces as it allows users to pick a numeric value from a set range. While we had an active pull request (PR) to add this element, updates in the main code caused conflicts and required further changes for added features. At one point, we noticed that text alignment wasn't centered properly within the box due to a flaw. To fix this, we began a PR to adjust the alignment, but it turned into a larger refactoring of the ``TextBlock2D``, a core component connected to various parts. This was a complex task that needed careful handling. After sorting out the ``TextBlock2D``, we returned to the ``SpinBoxUI`` and made a few tweaks. Once we were confident with the changes, the PR was successfully merged after thorough review and testing.
	
	**Pull Requests:**
  - **SpinBoxUI (Merged)** - https://github.com/fury-gl/fury/pull/499

      .. image:: https://user-images.githubusercontent.com/64432063/263165327-c0b19cdc-9ebd-433a-8ff1-99e706a76508.gif
        :height: 500
        :align: center
        :alt: SpinBoxUI
	


- **`TextBlock2D` Refactoring:**
	This was a significant aspect of the GSoC period and occupied a substantial portion of the timeline. The process began when we observed misaligned text in the ``SpinBoxUI``, as previously discussed. The root cause of the alignment issue was the mispositioning of the text actor concerning the background actor. The text actor's independent repositioning based on justification conflicted with the static position of the background actor, leading to the alignment problem.
	
	To address this, the initial focus was on resolving the justification issue. However, as the work progressed, we recognized that solely adjusting justification would not suffice. The alignment was inherently linked to the UI's size, which was currently retrieved only when a valid scene was present. This approach lacked scalability and efficiency, as it constrained size retrieval to scene availability.
	
	To overcome these challenges, we devised a solution involving the creation of a bounding box around the ``TextBlock2D``. This bounding box would encapsulate the size information, enabling proper text alignment. This endeavor spanned several weeks of development, culminating in a finalized solution that underwent rigorous testing before being merged.
	
	As a result of this refactoring effort, the ``TextBlock2D`` now offers three distinct modes:
  
	1. **Fully Static Background:** This mode requires background setup during initialization.
	2. **Dynamic Background:** The background dynamically scales based on the text content.
	3. **Auto Font Scale Mode:** The font within the background box automatically scales to fill the available space.
	
	An issue has been identified with ``TextBlock2D`` where its text actor aligns with the top boundary of the background actor, especially noticeable with letters like "g," "y," and "j". These letters extend beyond the baseline of standard alphabets, causing the text box to shift upwards.
	
	However, resolving this matter is complex. Adjusting the text's position might lead to it touching the bottom boundary, especially in font scale mode, resulting in unexpected positioning and transformations. To address this, the plan is to defer discussions about this matter until after GSoC, allowing for thorough consideration and solutions.
	
	For more detailed insights into the individual steps and nuances of this process, you can refer to the comprehensive weekly blog post provided below. It delves into the entire journey of this ``TextBlock2D`` refactoring effort.
	
	**Pull Requests:**
  - **Fixing Justification Issue - 1st Draft (Closed)** - https://github.com/fury-gl/fury/pull/790
  - **Adding BoundingBox and fixing Justificaiton (Merged)** - https://github.com/fury-gl/fury/pull/803
  - **Adding getters and setter for properties (Merged)** - https://github.com/fury-gl/fury/pull/830
  - **Text Offset PR (Closed)** - https://github.com/fury-gl/fury/pull/837


    .. image:: https://user-images.githubusercontent.com/64432063/258603191-d540105a-0612-450e-8ae3-ca8aa87916e6.gif
        :height: 500
        :align: center
        :alt: TextBlock2D Feature Demo
    
    .. image:: https://github-production-user-asset-6210df.s3.amazonaws.com/64432063/254652569-94212105-7259-48da-8fdc-41ee987bda84.png
        :height: 500
        :align: center
        :alt: TextBlock2D All Justification

- **ScrollbarUI as Independent Element:**
	We initially planned to make the scrollbar independent based on PR `#16 <https://github.com/fury-gl/fury/pull/16>`_. The main goal was to avoid redundancy by not rewriting the scrollbar code for each element that requires it, such as the ``FileMenu2D``. However, upon further analysis, we realized that elements like the ``FileMenu2D`` and others utilize the ``Listbox2D``, which already includes an integrated scrollbar. We also examined other UI libraries and found that they also have independent scrollbars but lack a proper use case. Typically, display containers like ``Listbox2D`` are directly used instead of utilizing an independent scrollbar.

	Based on these findings, we have decided to close all related issues and pull requests for now. If the need arises in the future, we can revisit this topic.
	
	**Topic:** - https://github.com/fury-gl/fury/discussions/816


Other Objectives
----------------

- **Reviewing & Merging:**
	In this phase, my focus was not on specific coding but rather on facilitating the completion of ongoing PRs. Here are two instances where I played a role:

	1. **CardUI PR:**
	   I assisted with the ``CardUI`` PR by aiding in the rebase process and reviewing the changes. The CardUI is a simple UI element consisting of an image and a description, designed to function like a flash card. I worked closely with my mentor to ensure a smooth rebase and review process.

	2. **ComboBox Issue:**
	   There was an issue with the ``ComboBox2D`` functionality, where adding it to a ``TabUI`` caused all elements to open simultaneously, which shouldn't be the case. I tested various PRs addressing this problem and identified a suitable solution. I then helped the lead in reviewing the PR that fixed the issue, which was successfully merged.

	**Pull Requests:**
  - **CardUI (Merged)** - https://github.com/fury-gl/fury/pull/398
  - **ComboBox Flaw (Merged)** - https://github.com/fury-gl/fury/pull/768
	

    .. image:: https://user-images.githubusercontent.com/54466356/112532305-b090ef80-8dce-11eb-90a0-8d06eed55993.png
        :height: 500
        :align: center
        :alt: CardUI
	

- **Updating Broken Website Links:**
	I addressed an issue with malfunctioning links in the Scientific Section of the website. The problem emerged from alterations introduced in PR `#769 <https://github.com/fury-gl/fury/pull/769>`_. These changes consolidated demos and examples into a unified "auto_examples" folder, and a toml file was utilized to retrieve this data and construct examples. However, this led to challenges with the paths employed in website generation. My responsibility was to rectify these links, ensuring they accurately direct users to the intended content.

	**Pull Requests:**
  - **Updating Broken Links (Merged)** - https://github.com/fury-gl/fury/pull/820


Objectives in Progress
----------------------

- **FileDialogUI:**
	An existing ``FileDialog`` PR by Soham (`#294 <https://github.com/fury-gl/fury/pull/294>`_) was worked upon. The primary task was to rebase the PR to match the current UI structure, resolving compatibility concerns with the older base. In PR `#832 <https://github.com/fury-gl/fury/pull/832>`_, we detailed issues encompassing resizing ``FileDialog`` and components, addressing text overflow, fixing ``ZeroDivisionError``, and correcting ``ListBox2D`` item positioning. The PR is complete with comprehensive testing and documentation. Presently, it's undergoing review, and upon approval, it will be prepared for integration.
	
	**Pull Requests:**
  - **Soham's FileDialog (Closed)** - https://github.com/fury-gl/fury/pull/294
  - **FileDialogUI (Under Review)** - https://github.com/fury-gl/fury/pull/832
	

    .. image:: https://user-images.githubusercontent.com/64432063/263189092-6b0891d5-f0ef-4185-8b17-c7104f1a7d60.gif
        :height: 500
        :align: center
        :alt: FileDialogUI


- **TreeUI:**
	Continuing Antriksh's initial PR for ``TreeUI`` posed some challenges. Antriksh had set the foundation, and I picked up from there. The main issue was with the visibility of TreeUI due to updates in the ``set_visibility`` method of ``Panel2D``. These updates affected how ``TreeUI`` was displayed, and after investigating the actors involved, it was clear that the visibility features had changed. This took some time to figure out, and I had a helpful pair programming session with my mentor, Serge, to narrow down the problem. Now, I've updated the code to address this issue. However, I'm still a bit cautious about potential future problems. The PR is now ready for review.
	
	**Pull Requests:**
  - **TreeUI (In Progress)** - https://github.com/fury-gl/fury/pull/821
	

    .. image:: https://user-images.githubusercontent.com/64432063/263237308-70e77ba0-1ce8-449e-a79c-d5e0fbb58b45.gif
        :height: 500
        :align: center
        :alt: TreeUI

GSoC Weekly Blogs
-----------------

-  My blog posts can be found at `FURY website <https://fury.gl/latest/blog/author/praneeth-shetty.html>`__
   and `Python GSoC blog <https://blogs.python-gsoc.org/en/ganimtron_10s-blog-copy-2/>`__.

Timeline
--------

.. list-table::
   :widths: 40 40 20
   :header-rows: 1

   * - Date
     - Description
     - Blog Post Link
   * - Week 0 (27-05-2023)
     - Community Bounding Period
     - `FURY <https://fury.gl/latest/posts/2023/2023-06-02-week-0-praneeth.html>`_ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog-copy-2/gsoc-2023-community-bonding-period/>`_
   * - Week 1 (03-06-2023)
     - Working with SpinBox and TextBox Enhancements
     - `FURY <https://fury.gl/latest/posts/2023/2023-06-03-week-1-praneeth.html>`_ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog-copy-2/week-1-working-with-spinbox-and-textbox-enhancements/>`_
   * - Week 2 (10-06-2023)
     - Tackling Text Justification and Icon Flaw Issues
     - `FURY <https://fury.gl/latest/posts/2023/2023-06-11-week-2-praneeth.html>`_ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog-copy-2/week-2-tackling-text-justification-and-icon-flaw-issues/>`_
   * - Week 3 (17-06-2023)
     - Resolving Combobox Icon Flaw and TextBox Justification
     - `FURY <https://fury.gl/latest/posts/2023/2023-06-17-week-3-praneeth.html>`_ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog-copy-2/week-3-resolving-combobox-icon-flaw-and-textbox-justification/>`_
   * - Week 4 (24-06-2023)
     - Exam Preparations and Reviewing
     - `FURY <https://fury.gl/latest/posts/2023/2023-06-24-week-4-praneeth.html>`_ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog-copy-2/week-4-exam-preparations-and-reviewing/>`_
   * - Week 5 (01-07-2023)
     - Trying out PRs and Planning Ahead
     - `FURY <https://fury.gl/latest/posts/2023/2023-07-01-week-5-praneeth.html>`_ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog-copy-2/week-5-testing-out-prs-and-planning-ahead/>`_
   * - Week 6 (08-07-2023)
     - BoundingBox for TextBlock2D!
     - `FURY <https://fury.gl/latest/posts/2023/2023-07-08-week-6-praneeth.html>`_ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog-copy-2/week-6-boundingbox-for-textblock2d/>`_
   * - Week 7 (15-07-2023)
     - Sowing the seeds for TreeUI
     - `FURY <https://fury.gl/latest/posts/2023/2023-07-15-week-7-praneeth.html>`_ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog-copy-2/week-7-sowing-the-seeds-for-treeui/>`_
   * - Week 8 (22-07-2023)
     - Another week with TextBlockUI
     - `FURY <https://fury.gl/latest/posts/2023/2023-07-22-week-8-praneeth.html>`_ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog-copy-2/week-8-another-week-with-textblockui/>`_
   * - Week 9 (29-07-2023)
     - TextBlock2D is Finally Merged!
     - `FURY <https://fury.gl/latest/posts/2023/2023-07-29-week-9-praneeth.html>`_ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog-copy-2/week-9-textblock2d-is-finally-merged/>`_
   * - Week 10 (05-08-2023)
     - Its time for a Spin-Box!
     - `FURY <https://fury.gl/latest/posts/2023/2023-08-05-week-10-praneeth.html>`_ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog-copy-2/week-10-its-time-for-a-spin-box/>`_
   * - Week 11 (12-08-2023)
     - Bye Bye SpinBox
     - `FURY <https://fury.gl/latest/posts/2023/2023-08-12-week-11-praneeth.html>`_ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog-copy-2/week-11-bye-bye-spinbox/>`_
   * - Week 12 (19-08-2023)
     - FileDialog Quest Begins!
     - `FURY <https://fury.gl/latest/posts/2023/2023-08-19-week-12-praneeth.html>`_ - `Python <https://blogs.python-gsoc.org/en/ganimtron_10s-blog-copy-2/week-12-filedialog-quest-begins/>`_
