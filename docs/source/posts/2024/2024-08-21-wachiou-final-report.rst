.. image:: https://developers.google.com/open-source/gsoc/resources/downloads/GSoC-logo-horizontal.svg
   :height: 40
   :target: https://summerofcode.withgoogle.com/programs/2023/projects/ED0203De

.. image:: https://www.python.org/static/img/python-logo@2x.png
   :height: 40
   :target: https://summerofcode.withgoogle.com/programs/2023/organizations/python-software-foundation

.. image:: https://python-gsoc.org/logos/fury_logo.png
   :width: 40
   :target: https://fury.gl/latest/index.html



Google Summer of Code Final Work Product
========================================

.. post:: August 21 2024
   :author: Wachiou BOURAIMA
   :tags: google
   :category: gsoc

-  **Name:** `Wachiou BOURAIMA <https://github.com/WassCodeur>`_
-  **Organisation:** `Python Software Foundation <https://www.python.org/psf-landing/>`_
-  **Sub-Organisation:** `FURY <https://fury.gl/latest/index.html>`_
-  **Mentor:** `Serge Koudoro <https://github.com/skoudoro>`_
-  **Project:** `Modernization of Codebase FURY. <https://github.com/fury-gl/fury/wiki/Google-Summer-of-Code-2024-(GSOC2024)#project-1-modernize-fury-codebase>`_


Introduction
------------

The Google Summer of Code (GSoC) 2024 was an enriching journey where I had the opportunity to work on modernizing the codebase of the ``FURY`` project, an open-source Python library for scientific visualization and 3D rendering. My project focused on optimizing the performance, improving documentation, and enhancing the overall user experience.


Abstract
--------

This project focused on modernizing the FURY codebase through the implementation of key features aimed at improving performance, maintainability, and usability. The initial phase involved the introduction of the ``warn_on_args_to_kwargs`` decorator, which enforces keyword-only arguments to ensure robust and future-proof API usage. Following this, the project moved to the implementation of lazy loading across various FURY modules, significantly optimizing the import process by deferring module loading until it is actually needed.
Additionally, considerable effort was directed towards addressing and resolving Sphinx documentation warnings caused by inconsistencies in the docstrings. The solution involved aligning the documentation with the "numpydoc" standard, ensuring smooth compilation and enhancing the clarity and comprehensiveness of the project’s documentation. These enhancements culminated in a more efficient, well-organized FURY library, offering an improved developer experience and a solid foundation for future developments.


Proposed Objectives
-------------------

The primary objectives of my GSoC project were:

- Implement ``Keyword-Only`` Arguments Decorator: Ensure backward compatibility by warning users when positional arguments are passed instead of keyword arguments.
- Implement ``Lazy Loading``: Introduce lazy loading to optimize module imports and reduce the memory footprint.
- Enhance Documentation: Address documentation warnings and improve the structure and clarity of the documentation.
- Modernize Codebase: Refactor existing code to ensure compatibility with the latest Python standards and improve maintainability.


Objectives Completed
--------------------

1. Implement ``warn_on_args_to_kwargs`` Decorator in FURY:

The first phase of this project was to develop the ``warn_on_args_to_kwargs`` decorator. This decorator is designed to check that all arguments after the first are passed as named arguments, and to ensure that users respect this convention.
The decorator works by inspecting the signature of the decorated function using the inspect module's signature function. It identifies function arguments that must be passed as named arguments (`KEYWORD_ONLY_ARGS`) and positional arguments (`POSITIONAL_ARGS`).
When a decorated function is called, the decorator compares the arguments supplied with those expected by the signature. If mandatory named arguments are missing, or if positional arguments are used where named arguments are expected, the decorator attempts to correct these errors by transforming the remaining positional arguments into named arguments.
In addition, the decorator issues a warning (`UserWarning`) if the current version of FURY is within a certain range (`from_version` to `until_version`). This warning informs the user that the current function call method will no longer be supported in future versions of FURY. If the FURY version exceeds until_version, an error is raised to indicate that the calling method is obsolete.

This decorator was applied to several key FURY modules, including ``actors``, ``ui``, ``animation``, ``shares``, ``data``, and ``stream``. This enhanced the robustness and maintainability of the code, minimizing potential errors due to incorrect use of arguments.

code source: https://github.com/fury-gl/fury/blob/master/fury/decorators.py

*Pull Requests*:

- Applied the ``warn_on_args_to_kwargs`` decorator to multiple modules:

  - actors: https://github.com/fury-gl/fury/pull/898 (merged)
  - animation: https://github.com/fury-gl/fury/pull/899 (merged)
  - data: https://github.com/fury-gl/fury/pull/900 (merged)
  - shares: https://github.com/fury-gl/fury/pull/901 (merged)
  - ui: https://github.com/fury-gl/fury/pull/902 (merged)
  - stream: https://github.com/fury-gl/fury/pull/909 (merged)

- Initial implementation of the decorator: https://github.com/fury-gl/fury/pull/888 (merged)

2. **Implement Lazy loading in FURY**

Overview:
The implementation of lazy loading in FURY was a key enhancement aimed at optimizing performance by delaying the loading of modules until they are actually needed. This approach reduces startup time and memory usage by only loading modules when they are accessed.

- Implementation of lazy loading: Lazy loading was applied to various FURY modules using the lazy_loader module. This technique ensures that modules are only loaded into memory when their functionalities are first accessed, rather than during the initial import of the package.
- Reading `SPEC1 <https://scientific-python.org/specs/spec-0001/>`__: To ensure best practices, I reviewed the `SPEC1 <https://scientific-python.org/specs/spec-0001/>`__ document from Scientific Python. This document offered valuable insights and guidelines on implementing lazy loading effectively.
- Update ``__init__.py`` Files: The ``__init__.py`` files were modified to support lazy loading. By leveraging ``lazy_loader``, these files were configured to load modules only when accessed. This change helps in managing dependencies efficiently and improves the overall performance of the package.

*Examples Implementation*:


.. code-block:: python
   :caption: fury/__init__.py

   import lazy_loader as lazy
   from fury.pkg_info import __version__, pkg_commit_hash

   # Configure lazy loading
   __getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

   # Append additional attributes
   __all__ += [
      "__version__",
      "disable_warnings",
      "enable_warnings",
      "get_info",
   ]

   # other functions and classes


.. code-block:: python
   :caption: fury/__init__.pyi

   # Type stub file for the fury package to support type-checking tools

   __all__ = [
      "actor",
      "actors",
      "animation",
      "colormap",
      # ... (other modules)
      ]

   from . import (
      actor,
      actors,
      animation,
      colormap,
      # ... (other modules)
      )

   # ... code block

- Added Type Stubs (``__init__.pyi``):

Type stubs (``__init__.pyi`` files) were added to provide type `hints` for the lazy loaded modules. This helps with type checking tools like `mypy` and enhances the development experience by offering better autocompletion and documentation in code editors.

- Improved Module Organization:

The organization of the ``__init__.py`` and ``__init__.pyi`` files was refined to better support lazy loading. This included restructuring imports and ensuring that module dependencies were managed efficiently.
  - Import Simplification: One significant change was simplifying how FURY is imported in example modules. Previously, the import statements were more complex, like from fury ``import ....`` To align with the lazy loading principles and reduce unnecessary overhead, I updated these statements to a more straightforward import fury This change ensures that only the necessary components are loaded when they are actually needed, improving performance.

*Pull Requests*:

- https://github.com/fury-gl/fury/pull/907 (merged)
- https://github.com/fury-gl/fury/pull/919 (merged)
- Simpply imports in FURY's examples: https://github.com/fury-gl/fury/pull/918 (merged)

3. **Handling Sphinx Warnings and Footer Deformation Issues**

   1. Addressing Sphinx Warnings:
      During the third phase, significant focus was placed on resolving Sphinx warnings related to documentation inconsistencies. The core issue stemmed from a mismatch between the documentation conventions used in the docstrings and the configuration settings in ``conf.py``.

      - Problem Identification:
      - Mismatch between ``numpydoc`` and ``napoleon``: The docstrings in the modules followed the numpydoc convention, while the conf.py file was configured to use sphinx.ext.napoleon. This discrepancy caused Sphinx to struggle with parsing and generating documentation correctly.
      - Solutions Explored:

        - Documentation Style Review: Examined the differences between ``numpydoc`` and ``napoleon`` documentation styles to understand the root of the issues.
        - Configuration Update vs. Docstring Conversion: Evaluated whether to update docstrings to match napoleon style or configure Sphinx to support numpydoc.

      - Decision and Implementation:
      - Configuration Update: Chose to update the Sphinx configuration to support the ``numpydoc`` style, aligning it with the existing docstrings. This adjustment resolved the conflicts and allowed Sphinx to compile the documentation without warnings.

   Updated Configuration in ``conf.py``:

   .. code-block:: python
      :caption: conf.py

      extensions = [
         ...
         "sphinx.ext.autodoc",
         "numpydoc",
         "sphinx.ext.autosummary",
         "sphinx.ext.githubpages",
         "sphinx.ext.intersphinx",
         ...,
         ]


   2. Investigating and Fixing the Footer Deformation Issue (https://github.com/fury-gl/fury/issues/874 (closed)):

      - Issue Identification: The footer on the FURY website deformed when hovering over elements due to size increases, which affected the padding of its container and the layout of subsequent elements.
      - Initial Approach: Bold Styling: Attempted to resolve the issue by making the elements bold on hover instead of changing font size. While this approach fixed the deformation, it did not meet the design requirements for the homepage footer.
      - Final Fix: CSS Adjustments: Added properties to the ``.class-columns`` in ``styles.css`` to better manage the footer style and prevent layout issues, ensuring that the design integrity was maintained.

      Video Demonstrations:
      Before Fixing the Footer Issue:

      Video demonstrating the footer deformation before the fix.

      .. raw:: html

         <iframe src="https://github.com/user-attachments/assets/2f5d4021-b661-4be9-944f-7a2638376f2c" width="640" height="390" frameborder="0" allowfullscreen></iframe>



      After Fixing the Footer Issue:

      Video showing the footer after applying the fix.

      .. raw:: html

         <iframe src="https://github.com/user-attachments/assets/b0ec74df-827b-4280-b3e4-bf968b97a654" width="640" height="390" frameborder="0" allowfullscreen></iframe>



*Pull Requests*:

- Addressed Sphinx warnings and updated configuration: https://github.com/fury-gl/fury/pull/922 (merged)
- https://github.com/fury-gl/fury/pull/915 (merged)
- fix typos in blog posts: https://github.com/fury-gl/fury/pull/911 (merged)
- Fix footer deformation issue: https://github.com/fury-gl/fury/pull/925 (merged)

Other Objectives
----------------

1. Peer Review Contributions:

   - Reviewed and provided feedback on PRs from other contributors to the FURY and DIPY projects.
   - Assisted in resolving issues and improving code quality in the PRs.

*Pull Requests*:

- https://github.com/fury-gl/fury/pull/913 (merged)


Objectives in Progress
----------------------

1. Enhancing Documentation:

   - Ongoing work to improve the documentation structure and content.
   - Addressing additional Sphinx warnings.

2. Modernizing Codebase:

   - Refactoring existing code to align with the latest Python standards.
   - Implementing best practices for code maintainability and readability.
   - Separating FURY's codebase and website.

3. Community Engagement:

   - Engaging with the FURY community through discussions, feedback, and contributions.
   - Participating in community events and meetings to share progress and gather insights.


GSoC Weekly Blogs
-----------------

- My blog posts can be found at `FURY website <https://fury.gl/latest/blog/author/wachiou-bouraima.html>`_ and `Python GSoC Mastodon server <https://social.python-gsoc.org/@wasscodeur>`_.

Timeline
--------

.. list-table::
   :widths: 20 60 20
   :header-rows: 1

   * - Date
     - Description
     - Blog Post Link
   * - Week 0 (28-05-2024)
     - The beginning of my journey in Google Summer of Code
     - `Blog post <https://fury.gl/latest/posts/2024/2024-05-28-week0-wachiou-bouraima.html>`__
   * - Week 1 (06-06-2024)
     - Progress and challenges at Google Summer of Code
     - `Blog Post <https://fury.gl/latest/posts/2024/2024-06-06-week1-wachiou-bouraima.html>`__
   * - Week 2 (15-06-2024)
     - Refinements and Further Enhancements
     - `Blog post <https://fury.gl/latest/posts/2024/2024-06-15-week2-wachiou-bouraima.html>`__
   * - Week 3 (19-06-2024)
     - Refinements and Further Enhancements
     - `Blog post <https://fury.gl/latest/posts/2024/2024-06-26-week3-wachiou-bouraima.html>`__
   * - Week 4 (26-06-2024)
     - Updating Decorator, Exploring Lazy Loading, and Code Reviews
     - `Blog post <https://fury.gl/latest/posts/2024/2024-06-26-week4-wachiou-bouraima.html>`__
   * - Week 5 (06-07-2024)
     - Implementing Lazy Loading in FURY with lazy_loader
     - `Blog post <https://fury.gl/latest/posts/2024/2024-07-06-week5-wachiou-bouraima.html>`__
   * - Week 6 (06-08-2024)
     - Code reviews, relining and crush challenges
     - `Blog post <https://fury.gl/latest/posts/2024/2024-08-06-week6-wachiou-bouraima.html>`__
   * - Week 7 (06-08-2024)
     - Fixing Sphinx Warnings in Blog Posts
     - `Blog post <https://fury.gl/latest/posts/2024/2024-08-06-week7-wachiou-bouraima.html>`__
   * - Week 8 (12-08-2024)
     - Refining Lazy Loading Implementation and Simplifying Imports in FURY
     - `Blog post <https://fury.gl/latest/posts/2024/2024-08-12-week8-wachiou-bouraima.html>`__
   * - Week 9 (13-08-2024)
     - Fixing Sphinx Warnings and Investigating Web Footer Issues
     - `Blog post <https://fury.gl/latest/posts/2024/2024-08-13-week9-wachiou-bouraima.html>`__
   * - Week 10 (15-08-2024)
     - Investigating Footer Deformation and Limited Progress on Warnings
     - `Blog post <https://fury.gl/latest/posts/2024/2024-08-15-week10-wachiou-bouraima.html>`__
   * - Week 11 (14-08-2024)
     - Resolving the Footer Issue and Addressing Sphinx Warnings
     - `Blog post <https://fury.gl/latest/posts/2024/2024-08-15-week11-wachiou-bouraima.html>`__
   * - Week 12 (17-08-2024)
     - The final straight
     - `Blog post <https://fury.gl/latest/posts/2024/2024-08-17-week12-wachiou-bouraima.html>`__
