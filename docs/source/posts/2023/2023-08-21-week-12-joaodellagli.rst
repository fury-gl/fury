Week 12: Now That is (almost) a Wrap!
==========================

.. post:: August 28, 2023
   :author: João Victor Dell Agli Floriano
   :tags: google
   :category: gsoc

Hello everyone, it's time for my last GSoC blogpost! We made it!

Last Week's Effort
------------------
After the API refactoring done last week, I focused on adressing the reviews I would get from it. The first issues I adressed was related to 
style, as there were some minor details my GSoC contributors pointed out that needed change. Also, I have adressed an issue I was having 
with the `typed hint` of one of my functions. Filipi, my mentor, showed me there is a way to have more than one typed hint in the same parameter, 
all I needed to do was to use the `Union` class from the `typing` module, as shown below:

.. code-block::python

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
My next plans are, after having PR `#826 <https://github.com/fury-gl/fury/pull/826>`_ merged, to work on the float enconding issue described in 
:doc:`this blogpost<2023-07-31-week-9-joaodellagli>`. Also, I plan to tackle the UI idea once again, to see if I can finally give the user 
a way to control de intensities of the distributions.

Thank you all for riding with me through this amazing journey that was GSoC! I can't describe how enlightening and awesome all of that was,
I met incredibly skilled people with a set of ideas so broad my mind opened like never before in coding. I would like to, first, thank Google and Python Software Foundation 
for hosting this unique opportunity, it certainly opened doors that will be hard to close again. Second, I would like to thank my mentors,
Filipi and Bruno, that were always beside me whenever needed, and helped me with problems I almost gave up bcause of how difficult they
seemed to be, you were essential. Third, I would like to thank my fellow GSoC contributors, Praneeth and Tania, that divided with me this
experience of weekly blogposting and developed amazing projects as well, congratulations guys! And last, but definetely not least, I would 
like to thank Fury, specially Serge, Javier, and other Fury core maintainers, for keeping such beautiful project that for sure helps a 
lot of people around the world to render some stunnig graphics!

I will be keep working on my projects in Fury and contributing the way I can, but won't post more weekly blogposts, so goodbye everyone, 
see you on my final report!