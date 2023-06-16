Week 2: Tackling Text Justification and Icon Flaw Issues
========================================================

.. post:: June 11, 2023
   :author: Praneeth Shetty
   :tags: google
   :category: gsoc


What did you do this week?
--------------------------

This week, I continued tweaking the text justification PR `#790 <https://github.com/fury-gl/fury/pull/790>`_ and encountered a new issue when combining both **justification** and **vertical_justification**. The problem arose because the **vertical_justification** did not take into account the applied **justification**, resulting in unexpected behavior. I focused on resolving this issue by ensuring that both justifications work together correctly. Additionally, during the weekly meeting, we discussed the problem and decided to introduce new properties such as boundaries and padding to enhance the functionality of the text justification feature.

Furthermore, I started working on PR `#576 <https://github.com/fury-gl/fury/pull/576>`_, which aimed to address the flaw in the icon of the **combobox**. While investigating this issue, I discovered related problems and PRs, including `#562 <https://github.com/fury-gl/fury/pull/562>`_, `#731 <https://github.com/fury-gl/fury/pull/731>`_, and `#768 <https://github.com/fury-gl/fury/pull/768>`_. The main challenge was related to the propagation of the **set_visibility** feature of the UI, causing the **combobox** to automatically open its options. To overcome this issue, I requested the author of PR `#768 <https://github.com/fury-gl/fury/pull/768>`_ to rebase their pull request as it can be a solution for the issue.

Did you get stuck anywhere?
---------------------------

A significant portion of my time was dedicated to resolving the text justification issue when both justification types were combined. It required careful analysis and adjustments to ensure the desired behavior.

What is coming up next?
-----------------------

For the upcoming week, I have the following plans:

1. Work on modifying the text justification implementation to address any remaining issues and ensure compatibility with other features.
2. Begin the implementation of the scrollbar class from scratch to provide a robust and customizable scrollbar element.
3. Focus on completing the resolution of the icon flaw issue by collaborating with the relevant stakeholders and ensuring the necessary modifications are made.
