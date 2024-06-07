WEEK 1: Progress and challenges at Google Summer of Code (GSoC) 2024
====================================================================

.. post:: June 06, 2024
   :author: Wachiou BOURAIMA
   :tags: google
   :category: gsoc

Hello👋🏾,

Welcome back to my Google Summer of Code (GSoC) 2024 journey!
This week has been filled with progress and challenges as I continue to work on modernizing the FURY code base.


Applying the keyword_only decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

My main task this week was to apply the keyword_only decorator to several functions.
The decorator ensures that all arguments except the first are keyword-only,
which helps to make the code clearer and parameter passing more explicit.
Some warnings appeared after applying this decorator, and to resolve them,
I updated all the code where these functions were called with the necessary format. This was a very important step in maintaining the integrity and functionality of the code base.


Managing the challenges of Git rebasing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rebasing the branch I was working on was the other major activity of my week.
It was a real challenge because of the conflicts that arose and had to be resolved.
It involved a lot of research and problem-solving on how to resolve these conflicts,
which greatly enhanced my understanding of Git. It was a challenging but satisfying experience of version control management and complex mergers.


Peer code review
~~~~~~~~~~~~~~~~

In addition to my duties, I was also tasked with reviewing the code of my peers.
This exercise was very rewarding, as it enabled me to understand different coding styles and approaches.
The constructive comments and suggestions were beneficial not only for teammates,
but also for improving my own coding and reviewing skills.


Acknowledgements
~~~~~~~~~~~~~~~~~

I would like to thank all my classmates and my guide for their constructive suggestions on my work.
Their ideas and suggestions were of great help to me and I am grateful for their support and advice.


What happens next?
~~~~~~~~~~~~~~~~~~

Here's a summary of what I plan to do in week three:

- Apply the keyword_only decorator to all other necessary functions.
- Update the calling of these functions in the code to ensure consistency and avoid raising warnings.
- Rename the decorator with a more descriptive name.
- Add two parameters to the decorator, specifying from which version of FURY it will work.


🥰Thanks for reading! Your comments are most welcome, and I look forward to giving you a sneak preview of my work next week.
