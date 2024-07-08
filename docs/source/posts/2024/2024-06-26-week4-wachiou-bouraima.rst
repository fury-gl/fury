WEEK 4: Updating Decorator, Exploring Lazy Loading, and Code Reviews
====================================================================

.. post:: June 26, 2024
   :author: Wachiou BOURAIMA
   :tags: google
   :category: gsoc

Hello everyone,
---------------

Welcome again to my Google summer of code 2024 (GSoC' 2024) journey 2024!.
This week, I focused on updating the ``warn_on_args_to_kwargs`` decorator, applying it across multiple modules, exploring lazy loading, and continuing with code reviews.


Updating the ``warn_on_args_to_kwargs`` decorator
-------------------------------------------------

Based on feedback from my mentor `Serge Koudoro <https://github.com/skoudoro>`_  and peers  `Iñigo Tellaetxe Elorriaga <https://github.com/itellaetxe>`_, `Robin Roy <https://github.com/robinroy03>`_, `Kaustav Deka <https://github.com/deka27>`_, I refined the ``warn_on_args_to_kwargs`` decorator and its associated unit tests:

1. Improvements:

   - Added conditions to verify if the values of ``from_version``, ``until_version``, and the current version of FURY are respected. This includes handling cases where ``from_version`` is greater than the current version of FURY, ``until_version`` is less than the current version of FURY, and ``until_version`` is greater than or equal to the current version of FURY.
   - Ensured the decorator and tests cover a broader range of edge cases.
   - Enhanced the warning messages for better clarity and guidance.

2. Doctest Updates:

   - Updated the doctest considering the values of `from_version` and `until_version`.
   - Moved the doctest from the `def decorator()` function to the root function.

3. Unit Tests:

.. code-block:: python

    def test_warn_on_args_to_kwargs():
        @warn_on_args_to_kwargs()
        def func(a, b, *, c, d=4, e=5):
            return a + b + c + d + e

    # if FURY_CURRENT_VERSION is less than from_version
    fury.__version__ = "0.0.0"
    npt.assert_equal(func(1, 2, 3, 4, 5), 15)
    npt.assert_equal(func(1, 2, c=3, d=4, e=5), 15)
    npt.assert_raises(TypeError, func, 1, 3)

- This ensures robust validation and helps catch potential issues early.


Applying the ``warn_on_args_to_kwargs`` Decorator
-----------------------------------------------

This week, I applied the ``warn_on_args_to_kwargs`` decorator to several modules, ensuring consistent usage and improved code quality. The modules updated include:

- `actors`
- `ui`
- `animation`
- `shares`
- `data`

For each module, I opened a pull request to track the changes and facilitate reviews:

- `actors`: https://github.com/fury-gl/fury/pull/898
- `animation`: https://github.com/fury-gl/fury/pull/899
- `data`: https://github.com/fury-gl/fury/pull/900
- `shares`: https://github.com/fury-gl/fury/pull/901
- `ui`: https://github.com/fury-gl/fury/pull/902


Exploring lazy loading
----------------------

In order to optimize performance, I've started exploring and implementing lazy loading. This week, the focus was on the following points:

- Getting to grips with how the lazy loader works
- Implementing some small script to understand how the lazy loader works
- I also read the SPEC1 document available at `SPEC1 <https://scientific-python.org/specs/spec-0001/>`_
- Understanding the benefits of lazy loading and how it can be applied to the FURY code base
- Planning the integration of lazy loading into the FURY code base

Code sample: `<https://gist.github.com/WassCodeur/98297d7a59b27979d27945760e3ffb10>`_


Peer Code Review
----------------

This week, I continued to dedicate time to reviewing the code of my peers. Specifically, I reviewed Kaustav Deka’s work, providing constructive feedback and suggestions for improvement. You can view the pull request here: `https://github.com/dipy/dipy/pull/3239 <https://github.com/dipy/dipy/pull/3239>`_.


Acknowledgements
----------------

I am deeply grateful to my classmates `Iñigo Tellaetxe Elorriaga <https://github.com/itellaetxe>`_, `Robin Roy <https://github.com/robinroy03>`_, `Kaustav Deka <https://github.com/deka27>`_  for their continuous support and insightful suggestions. Special thanks to my mentor, `Serge Koudoro <https://github.com/skoudoro>`_ , whose expertise and guidance have been invaluable in navigating these technical challenges.


Did I get stuck?
-----------------

Yes, I was a bit confused about understanding lazy loader, but thanks to the help of my mentor `Serge Koudoro <https://github.com/skoudoro>`_ , I was able to understand it better.


What's next?
------------

For the upcoming week, I plan to:

- Implement lazy loading in the FURY code base
- Continue refining the ``warn_on_args_to_kwargs`` decorator based on feedback
- Engage in more code reviews to support my peers
- Prepare to working on the FURY website to improve the documentation and user experience

Thank you for following my progress. Your feedback is always welcome.
