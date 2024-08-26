WEEK 9: Fixing Sphinx Warnings and Investigating Web Footer Issues
==================================================================

.. post:: August 13, 2024
   :author: Wachiou BOURAIMA
   :tags: google
   :category: gsoc

Hello everyone,
welcome to my Google Summer of Code (GSoC) 2024 journey! Week 9 was devoted to fixing Sphinx warnings caused by indentation errors in some docstrings of some examples in the ``auto_examples`` folder. I also started investigating why the footer of the ``FURY`` site distorts when you try to move the mouse over an element.


Continuing the Fight Against Sphinx Warnings
--------------------------------------------

My main task this week has been to continue dealing with the Sphinx warnings in our documentation. I focused on the 188 warnings related to the main documentation, as well as 2 warnings in the blog posts.
I fixed 19 warnings caused by the docstring of ``viz_**.py`` modules in the ``source/auto_examples`` folder
by fixing the indentation errors in the docstrings of the examples.

Here is an example of indentation error in the docstring of the example ``auto_examples/04_demos/viz_animated_surfaces.py``:

.. code-block:: python

    ... Code block ...

    ###############################################################################
    # Variables and their usage:
    # :time - float: initial value of the time variable i.e. value of the time variable at
    #               the beginning of the program; (default = 0)
    # dt: float
    #     amount by which ``time`` variable is incremented for every iteration
    #     of timer_callback function (default = 0.1)
    # lower_xbound: float

    ... Code block ...


Investigating Web Footer Issues
-------------------------------

In parallel, I started investigating an issue with the FURY website's footer, which deforms when hovering over an element with the mouse. This problem affects the user experience and visual consistency of the site. My work this week has focused on diagnosing the underlying cause of this issue and planning the necessary steps to fix it. This task has been both technically intriguing and a great opportunity to sharpen my web development skills.

issue number: `#874 <https://github.com/fury-gl/fury/issues/874>`_


Did I get stuck anywhere ?
--------------------------

I encountered some challenges while fixing the indentation errors in the docstrings of the examples. The errors were caused by inconsistent indentation in the docstrings, which made it difficult to identify the root cause of the warnings. However, I was able to resolve these issues and make progress in addressing the Sphinx warnings.

What's next ?
-------------

For week 10, I plan to:

- Continue fixing the Sphinx warnings in the documentation.
- Start fixing the issue with the footer of the FURY website.
