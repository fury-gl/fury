Week 12: Now That is (almost) a Wrap!
=====================================

.. post:: August 21, 2023
   :author: João Victor Dell Agli Floriano
   :tags: google
   :category: gsoc

Hello everyone, it's time for another GSoC blogpost! Today, I am going to talk about some minor details I worked on last week on my
project.

Last Week's Effort
------------------
After the API refactoring was done last week, I focused on addressing the reviews I would get from it. The first issues I addressed was related to
style, as there were some minor details my GSoC contributors pointed out that needed change. Also, I have addressed an issue I was having
with the `typed hint` of one of my functions. Filipi, my mentor, showed me there is a way to have more than one typed hint in the same parameter,
all I needed to do was to use the `Union` class from the `typing` module, as shown below:

.. code-block:: python

   from typing import Union as tUnion
   from numpy import ndarray

   def function(variable : tUnion(float, np.ndarray)):
      pass

Using that, I could set the typedhint of the `bandwidth` variable to `float` and `np.ndarray`.

So how did it go?
-----------------
All went fine with no difficult at all, thankfully.

The Next Steps
--------------
My next plans are, after having PR `#826 <https://github.com/fury-gl/fury/pull/826>`_ merged, to work on the float encoding issue described in
:doc:`this blogpost<2023-07-31-week-9-joaodellagli>`. Also, I plan to tackle the UI idea once again, to see if I can finally give the user
a way to control the intensities of the distributions.

Wish me luck!
