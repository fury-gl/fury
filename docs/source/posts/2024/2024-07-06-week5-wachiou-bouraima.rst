WEEK 5: Implementing Lazy Loading in FURY with ``lazy_loader``
==============================================================

.. post:: July 6, 2024
   :author: Wachiou BOURAIMA
   :tags: google
   :category: gsoc

Hello everyone,
---------------

Welcome back to my Google Summer of Code (GSoC) 2024 journey! This week has been particularly exciting as I introduced a significant performance optimization feature: lazy loading. Here's an overview of my progress and contributions.


**Introduction of lazy loading**
--------------------------------

This week, I focused on implementing the ``lazy_loader`` feature of `Scientific Python <https://scientific-python.org/>`_ to optimize module loading in FURY. Lazy loading improves performance by deferring the loading of modules until they are actually needed, thus reducing start-up times and memory footprint.

The implementation involved:

1. Implementation of Lazy Loading:

   - Application of lazy loading in several FURY modules using the ``lazy_loader`` module to improve performance

2. Update ``__init__.py`` files:

   - Modified ``__init__.py`` files to support lazy loading where necessary. This ensures that modules are only loaded when they are accessed for the first time

3. Added Type Stubs (``__init__.pyi``):

   - Adding type stubs (``__init__.pyi``) provides type hints for lazy-loading modules, improving code readability and maintainability

4. **Improved module organization:**

   - Improved module organization in ``__init__.py`` and ``__init__.pyi`` files, to effectively support lazy loading.


**Example Implementation**
---------------------------

To give you an idea, here's the actual implementation of how lazy loading was done using the ``lazy_loader`` module in FURY:

``__init__.py`` File:

.. code-block:: python

    import lazy_loader as lazy
    from fury.pkg_info import __version__, pkg_commit_hash

    __getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

    _all__ += [
    "__version__",
    "disable_warnings",
    "enable_warnings",
    "get_info",
    ]

    # ... (functions)

``__init__.pyi`` File:

.. code-block:: python

    # This file is a stub type for the fury package. It provides information about types
    # to help type-checking tools like mypy and improve the development experience
    # with better autocompletion and documentation in code editors.

    __all__ = [
        "actor",
        "actors",
        "animation",
        "colormap",
        # ... (other modules)
        ,
        ]

        from . import (
            actor,
            actors,
            animation,
            colormap,
            # ... (other modules)
            ,
            )
            # ... (other functions)

You can review the implementation in `this pull request <https://github.com/fury-gl/fury/pull/907/>`_.


Reading ``SPEC1``
-----------------

To align myself with best practice, I read the `SPEC1 <https://scientific-python.org/specs/spec-0001/>`_ document available at Scientific Python SPEC1. This document provided valuable hints and guidelines that I took into account when implementing the lazy loading feature.


Did I get stuck anywhere?
--------------------------
No, I didn't encounter any major blockers this week. The implementation of lazy loading went smoothly, and I was able to complete the task.


**What's Next?**
-----------------

For the next week, I plan to:

1. Review all my Pull Requests with my mentor `Serge Koudoro <https://github.com/skoudoro/>`_, to ensure everything is up to FURY's standards.
2. Start working on the redesign of the FURY website, making it more user-friendly and visually appealing.


Thank you for reading. Stay tuned for more updates on my progress!
