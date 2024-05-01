Week 7: Sowing the seeds for TreeUI
===================================

.. post:: July 15, 2023
   :author: Praneeth Shetty
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------
This week, I focused on completing the **TextBlock2D** Bounding Box feature. However, the tests were failing due to automatic background resizing based on content and improper text actor alignment during setup. I encountered difficulties while positioning the text, which caused the text to appear offset and led to test failures.

Text background greater than the actual maximum size:

.. image:: https://github.com/fury-gl/fury/assets/64432063/aaf4a764-4480-4f96-9adf-29d9e28135a6
   :align: center
   :alt: Text background greater than the actual maximum size in ComboBox2D

Text offset from center:

.. image:: https://github.com/fury-gl/fury/assets/64432063/0a3bc1e6-a566-4c08-9ca4-a191525b9c97
   :align: center
   :alt: Text offset from center in RingSlider2D

Additionally, I reviewed PR `#814 <https://github.com/fury-gl/fury/pull/814>`_ and noticed that after PR `#769 <https://github.com/fury-gl/fury/pull/769>`_, all demos and examples were merged into a single folder, which affected the paths used in the Scientific Domain Section. To address this, I created PR `#820 <https://github.com/fury-gl/fury/pull/820>`_ to redirect the links to the correct path.

As I faced issues with the **TextBlock2D** PR, I took the opportunity to rebase and continue working on the **TreeUI** PR since there were no updates from the author.

Did you get stuck anywhere?
---------------------------
While fixing the issues with the tests for the **TextBlock2D** bounding box, I encountered a weird behavior in text positioning when using the center alignment. The output varied depending on the sequence of repositioning which we are still investigating.

What is coming up next?
-----------------------
I will continue working on the **TreeUI** and resolve the **TextBlock2D** error to ensure both PRs progress smoothly.
