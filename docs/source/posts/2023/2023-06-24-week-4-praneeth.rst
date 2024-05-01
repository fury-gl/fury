Week 4: Exam Preparations and Reviewing
=======================================

.. post:: June 24, 2023
   :author: Praneeth Shetty
   :tags: google
   :category: gsoc

What did I do this week?
------------------------

This week, amidst end-semester exams, I managed to accomplish a few notable tasks. Let's dive into the highlights:

1. Merging **CardUI**: The PR `#398 <https://github.com/fury-gl/fury/pull/398>`_ introduced the **CardUI** to the UI system of FURY. After a successful review and test check, it was merged into the codebase.

2. Revisiting PR `#540 <https://github.com/fury-gl/fury/pull/540>`_: I restarted working on PR `#540 <https://github.com/fury-gl/fury/pull/540>`_ as I wasn't satisfied with the previous approach when I checked it for rebasing. I took the opportunity to update the code and ensure that the unit tests passed successfully. Although there are a few issues remaining in the tests, I am determined to resolve them and move forward with the implementation. This PR aims to improve the usage of the **numpy_to_vtk_image_data** utility function.

3. Independent Scrollbar Consideration: We are currently evaluating the necessity of making the Scrollbar an independent element. Currently it is only used by the **ListBox2D**, we are exploring various use cases to determine if there are other scenarios where the Scrollbar can be employed independently. This evaluation will help us make an informed decision about its future implementation.

4. PR Reviews: In the brief intervals between exams, I utilized the time to review two PRs: `#446 <https://github.com/fury-gl/fury/pull/446>`_ - Resize panel and `#460 <https://github.com/fury-gl/fury/pull/460>`_ - Tree UI.

Did I get stuck anywhere?
-------------------------

No, fortunately, I didn't encounter any major obstacles or challenges during my tasks this week.

What is coming up next?
-----------------------

Once the exams are over, I am eagerly looking forward to making a full comeback to development. My immediate plans include addressing the remaining issues in PR `#540 <https://github.com/fury-gl/fury/pull/540>`_ and completing the pending tasks. I will also catch up on any missed discussions and sync up with the team to align our goals for the upcoming weeks.
