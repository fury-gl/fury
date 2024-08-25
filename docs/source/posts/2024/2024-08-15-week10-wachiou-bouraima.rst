WEEK 10: Investigating Footer Deformation and Limited Progress on Warnings
==========================================================================

.. post:: August 15, 2024
   :author: Wachiou BOURAIMA
   :tags: google
   :category: gsoc

Welcome to the week 10 of my Google Summer of Code (GSoC) 2024 journey. This week, I focused on resolving an issue with the ``FURY``  website’s footer and made limited progress on fixing documentation warnings. Here’s a detailed overview of my activities and challenges.


Investigating Footer Deformation Issue: `#874 <https://github.com/fury-gl/fury/issues/874>`_
--------------------------------------------------------------------------------------------

During this week, I identified the root cause of the footer deformation on the FURY website. The problem arose when hovering over an element, which caused the element’s size to increase. This, in turn, increased the padding of its container and subsequently affected the layout of following elements.
To address this, I initially considered avoiding changes in font size on hover and opted to make elements bold instead. While this approach resolved the deformation issue, it did not align with the design requirements for the homepage footer.
Due to health constraints, I was unable to continue this work. I plan to explore alternative solutions to align with the design specifications in the upcoming week.


Limited Progress on Documentation Warnings
------------------------------------------

I also aimed to make progress on fixing Sphinx warnings related to documentation typos. Unfortunately, I could not advance significantly in this area due to the time constraints imposed by the footer issue and my health.


What's Next ?
-------------

For week 11, I plan to:

- Explore alternative solutions to address the footer deformation issue while aligning with the design requirements.
- Continue fixing Sphinx warnings in the documentation and address any remaining issues.
