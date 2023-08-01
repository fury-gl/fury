Week 9: It is Polishing Time!
=============================

.. post:: July 31, 2023
   :author: João Victor Dell Agli Floriano
   :tags: google
   :category: gsoc


Hello everyone, it's time for another weekly blogpost! Today, I am going to update you on my project's latest changes.

Last Week's Effort
------------------
After having finished a first draft of the API that will be used for the KDE rendering, and showing how it could be used 
for other post-processing effects, my goal was to clean the code and try some details that would add to it so it could be better 
complete. Having that in mind, I invested in three work fronts:

1. Fixing some bugs related to the rendering more than one post-processing effect actor.
2. Experimenting with other rendering kernels (I was using the *gaussian* one only).
3. Completing the KDE render by renormalizing the values in relation to the number of points (one of the core KDE details). 

Both three turned out more complicated than it initially seemed, as I will show below.

So how did it go?
-----------------
The first one I did on monday-tuesday, and I had to deal with some issues regarding scaling and repositioning. Due to implementation 
choices, the final post-processed effects



This Week's Goals
-----------------
