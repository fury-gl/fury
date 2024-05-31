Week 10: Ready for Review!
==========================

.. post:: August 07, 2023
   :author: João Victor Dell Agli Floriano
   :tags: google
   :category: gsoc

Hello everyone, it's time for another weekly blogpost!

Last Week's Effort
------------------
After talking with my mentors, I was tasked with getting my API PR `#826 <https://github.com/fury-gl/fury/pull/826>`_ ready for review,
as it still needed some polishing, and the most important of all, it needed its tests working, as this was something I haven't invested time since its creation.
Having that in mind, I have spent the whole week cleaning whatever needed, writing the tests, and also writing a simple example of its
usage. I also tried implementing a little piece of UI so the user could control the intensity of the bandwidth of the KDE render, but
I had a little problem I will talk about below.


So how did it go?
-----------------
Fortunately, for the cleaning part, I didn't have any trouble, and my PR is finally ready for review! The most complicated part was to write the tests, as this is something that
requires attention to understand what needs to be tested, exactly. As for the UI part, I managed to have a slider working for the
intensity, however, it was crashing the whole program for a reason, so I decided to leave this idea behind for now.
Below, an example of how this should work:

.. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/slider.gif
   :align: center
   :alt: Buggy slider for the intensity control of the bandwidth of the KDE

This Week's Goals
-----------------
After a meeting with my mentors, we decided that this week's focus should be on finding a good usage example of the KDE rendering feature,
to have it as a showcase of the capability of this API. Also, they hinted me some changes that need to be done regarding the API, so I
will also invest some time on refactoring it.

Wish me luck!
