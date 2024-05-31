Week 12: FileDialog Quest Begins!
=================================

.. post:: August 19, 2023
   :author: Praneeth Shetty
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
During this week, I initiated my work on the ``FileDialog`` PR, which had been started by Soham. The initial version of the ``FileDialog`` can be found at `#294 <https://github.com/fury-gl/fury/pull/294>`_. To start, I focused on rebasing the PR. Since this PR was based on an older version, there were some updates to the overall UI structure that needed to be addressed for compatibility. While handling this, I identified a set of issues that I documented in the current PR `#832 <https://github.com/fury-gl/fury/pull/832>`_. These mainly revolved around:

1. Resizing ``FileDialog`` and related components.
2. Rectifying the text overflow problem.
3. Dealing with a ``ZeroDivisionError``.
4. Fixing the positioning of items in the ``ListBox2D``.

I systematically approached each of these challenges:

**Resizing FileMenu and Related Components:** This was a fairly complex task since it involved intricate dependencies, such as the ``FileDialog`` relying on the ``FileMenu``, which, in turn, was dependent on ``ListBox2D`` and ``Panel2D`` resizing. To make the process manageable, I decided to progress incrementally in a separate PR a bit later.

**Text Overflow Issue:** The problem with text overflow was rooted in our previous approach, which involved executing these actions only when the ``TextBlock2D`` had a scene property. Although this approach suited the previous version of ``TextBlock2D``, the recent refactoring led to the removal of this property. The scene was previously utilized to determine the text actor's size. However, we had new methodologies to calculate these sizes, which are detailed in `#803 <https://github.com/fury-gl/fury/pull/803>`_.

.. image:: https://github.com/fury-gl/fury/assets/64432063/b001f9d3-a5e8-45ad-8605-85df595b5654
   :align: center
   :alt: Text Overflow Before

.. image:: https://github.com/fury-gl/fury/assets/64432063/d3c9c3a3-e601-45ab-8975-2b1e98acf1d3
   :align: center
   :alt: Text Overflow After

**Addressing ZeroDivisionError:** The ``ZeroDivisionError`` emerged when the total number of values was the same as the number of slots. The issue lay in the separation of these values for calculating the scrollbar's height parameter. Unfortunately, this calculation error occurred when this would return us zero while updating the scrollbar. To counter this, I implemented a conditional check to ascertain whether the value is zero or not.

**Correcting ``ListBox2D`` Item Positioning:** Another challenge I encountered related to the improper positioning of ``ListBox2D`` item's background. When a slot was not visible, its background was resized to zero, and visibility was set to off. Consequently, during the calculation of updated positions, the height was considered zero, leading to mispositioning. I resolved this by refraining from resizing and solely toggling visibility, achieving the desired result.

.. image:: https://github.com/fury-gl/fury/assets/64432063/e2805934-b037-47fd-872c-0b284b298d3c
   :align: center
   :alt: ListBox2D mispositioning Before

.. image:: https://github.com/fury-gl/fury/assets/64432063/3bc1aabb-bb79-4e26-817d-a2a2ddd20ea3
   :align: center
   :alt: Fixed ListBox2D mispositioning

Did you get stuck anywhere?
---------------------------
Among the challenges I faced, one notable instance involved addressing the visibility issue in ``TreeUI``. Despite my attempts at various solutions, none yielded the desired outcome. The ``TreeUI`` exhibited either full visibility or no visibility at all. In this situation, I sought guidance from my mentor to find a viable solution.


What is coming up next?
-----------------------
The ``FileDialog`` implementation is nearly finalized, and my plan is to work on any review, feedback or suggestions that might arise. Following this, I will shift my attention towards addressing the ``TreeUI``.
