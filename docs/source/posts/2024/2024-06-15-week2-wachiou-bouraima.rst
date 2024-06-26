WEEK 2: Refinements and Further Enhancements
============================================

.. post:: June 15, 2024
   :author: Wachiou BOURAIMA
   :tags: google
   :category: gsoc

Hello again,
~~~~~~~~~~~~~

Welcome back to my Google Summer of Code (GSoC) 2024 journey! This week has been dedicated to refining and improving the work done so far, with a particular focus on the keyword_only decorator.


Renaming and Updating the Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This week, I've updated `this Pull Request <https://github.com/fury-gl/fury/pull/888>`_ by renaming the ``keyword_only`` decorator to ``warn_on_args_to_kwargs`` for greater clarity. The updated decorator now includes version parameters from_version and until_version. This enhancement ensures that the decorator will raise a RuntimeError if the current version of FURY is greater than until_version.


Peer Code Review
~~~~~~~~~~~~~~~~~

I also spent time reviewing `Kaustav Deka's <https://github.com/deka27>`_ code. This exercise remains rewarding, as it helps me understand different coding styles and approaches. Constructive feedback and suggestions from my classmates were invaluable, not only in helping my teammates but also in improving my own coding and reviewing skills.


Research into lazy loading
~~~~~~~~~~~~~~~~~~~~~~~~~~

In parallel, I started researching the lazy loading feature and thinking about how to implement it. This feature will optimize performance by loading resources only when they're needed, which is crucial to improving the efficiency of FURY's code base.


Acknowledgements
~~~~~~~~~~~~~~~~

I am deeply grateful to my classmates `Iñigo Tellaetxe Elorriaga <https://github.com/itellaetxe>`_, `Robin Roy <https://github.com/robinroy03>`_, and `Kaustav Deka <https://github.com/deka27>`_ for their insightful suggestions and comments on my work.
Special thanks to my mentor, `Serge Koudoro <https://github.com//skoudoro>`_, whose guidance and support enabled me to meet the challenges of this project.
Their combined efforts have greatly contributed to my progress, and I appreciate their continued help.


What happens next?
~~~~~~~~~~~~~~~~~~

For week 3, I plan to :

- Ensure that the ``warn_on_args_to_kwargs`` decorator is applied consistently in all necessary functions.
- Continue to update the calling of these functions in the code to maintain consistency and avoid warnings.
- Refine decorator as necessary based on feedback and testing.
- Start implementing lazy loading functionality based on my research to optimize performance.


🥰 Thank you for taking the time to follow my progress. Your feedback is always welcome and I look forward to sharing more updates with you next week.
