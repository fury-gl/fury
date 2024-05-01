Week 11: A Refactor is Sometimes Needed
=======================================

.. post:: August 14, 2023
   :author: João Victor Dell Agli Floriano
   :tags: google
   :category: gsoc

Hello everyone, it's time for another weekly blogpost! Today I am going to share some updates on the API refactoring
I was working on with my mentors.

Last Week's Effort
------------------
As I shared with you :doc:`last week <2023-08-07-week-10-joaodellagli>`, the first draft of my API was finally ready for review, as
I finished tweaking some remaining details missing. I was tasked with finding a good example of the usage of the tools we proposed,
and I started to do that, however after testing it with some examples, I figured out some significant bugs were to be fixed. Also,
after some reviews and hints from some of my mentors and other GSoC contributors, we realised that some refactoring should be done,
mainly focused on avoiding bad API usage from the user.

So how did it go?
-----------------
Initially, I thought only one bug was the source of the issues the rendering presented, but it turned out to be two, which I will
explain further.

The first bug was related to scaling and misalignment of the KDE render. The render of the points being post-processed was not only
with sizes different from the original set size, but it was also misaligned, making it appear in positions different from the points'
original ones. After some time spent, I figured out the bug was related to the texture coordinates I was using. Before, this is how
my fragment shader looked:

.. code-block:: C

    vec2 res_factor = vec2(res.y/res.x, 1.0);
    vec2 tex_coords = res_factor*normalizedVertexMCVSOutput.xy*0.5 + 0.5;
    float intensity = texture(screenTexture, tex_coords).r;

It turns out using this texture coordinates for *this case* was not the best choice, as even though it matches the fragment positions,
the idea here was to render the offscreen window, which has the same size as the onscreen one, to the billboard actor. With that in mind,
I realised the best choice was using texture coordinates that matched the whole screen positions, coordinates that were derived from the
``gl_FragCoord.xy``, being the division of that by the resolution of the screen, for normalization. Below, the change made:

.. code-block:: C

    vec2 tex_coords = gl_FragCoord.xy/res;
    float intensity = texture(screenTexture, tex_coords).r;

This change worked initially, although with some problems, that later revealed the resolution of the offscreen window needed to be
updated inside the callback function as well. Fixing that, it was perfectly aligned and scaled!

The second bug was related with the handling of the bandwidth, former sigma parameter. I realised I wasn't dealing properly with the option of the user passing only
one single bandwidth value being passed, so when trying that, only the first point was being rendered. I also fixed that and it worked,
so cheers!

As I previously said, the bugs were not the only details I spent my time on last week. Being reviewed, the API design, even
though simple, showed itself vulnerable to bad usage from the user side, requiring some changes. The changes suggested by mentors were,
to, basically, take the ``kde`` method out of the ``EffectManager`` class, and create a new class from it inside an ``effects`` module,
like it was a special effects class. With this change, the KDE setup would go from:

.. code-block:: python

    em = EffectManager(show_manager)

    kde_actor = em.kde(...)

    show_manager.scene.add(kde_actor)

To:

.. code-block:: python

    em = EffectManager(show_manager)

    kde_effect = KDE(...)

    em.add(kde_effect)

Not a gain in line shortening, however, a gain in security, as preventing users from misusing the kde_actor. Something worth noting is
that I learned how to use the ``functools.partial`` function, that allowed me to partially call the callback function with only some
parameters passed.


This Week's Goals
-----------------
Having that refactoring made, now I am awaiting for a second review so we could finally wrap it up and merge the first stage of this API.
With that being done, I will write the final report and wrap this all up.

Let's get to work!
