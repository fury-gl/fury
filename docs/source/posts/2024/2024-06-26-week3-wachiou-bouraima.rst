WEEK 3: Refinements and Further Enhancements
============================================

.. post:: June 26, 2024
   :author: Wachiou BOURAIMA
   :tags: google
   :category: gsoc

Hello everyone,
---------------

Welcome to the fourth week of my Google Summer of Code (GSoC) 2024 journey!
This week I've been delving into the technical aspects of my project,
focusing on the consistent application of the ``warn_on_args_to_kwargs`` decorator and the initial implementation of lazy loading.


Consistent application of ``warn_on_args_to_kwargs``
----------------------------------------------------

This week I continued to apply the decorator to functions.
To ensure consistency across the code base, I audited all functions that could benefit from the ``warn_on_args_to_kwargs`` decorator.
To do this, I had to:

1. Identify target functions:

   * Identify functions that could benefit from the decorator.
   * continue reviewing the code base to identify functions that accept both positional and keyword arguments.

2. Applying the Decorator:

   * For each identified function, I added the ``warn_on_args_to_kwargs`` decorator.
   * Example:

.. code-block::  python

   @warn_on_args_to_kwargs()
   def get_actor_from_primitive(
      vertices,
      triangles,
      *,
      colors=None,
      normals=None,
      backface_culling=True,
      prim_count=1,
      ):

3. Updating Unit Tests:

* updated all the unit tests for the functions where the ``warn_on_args_to_kwargs`` decorator is applied to ensure they respect the new format.
* Example:

.. code-block:: python

   actr = get_actor_from_primitive(big_verts, big_faces, colors=big_colors)

- You can find more details and the implementation in my pull request: `https://github.com/fury-gl/fury/pull/888 <https://github.com/fury-gl/fury/pull/888>`_.


What Happens Next?
------------------

For week 4, I plan to:

* Continue refining the ``warn_on_args_to_kwargs`` decorator based on feedback from my Peers `Iñigo Tellaetxe Elorriaga <https://github.com/itellaetxe>`_, `Robin Roy <https://github.com/robinroy03>`_, `Kaustav Deka <https://github.com/deka27>`_, my guide: `Serge Koudoro <https://github.com/skoudoro>`_ and the other community members.
* Apply the ``warn_on_args_to_kwargs`` decorator to all the remaining modules and update all the unit tests of these modules too, to respect the desired format.
* Dive deep into the lazy loading functionality based on my research to optimize performance.
* Further engage in code reviews to support my peers and improve our project.

Did I get stuck?
----------------

I didn't get stuck.

Thank you for following my progress. Your feedback is always welcome.
