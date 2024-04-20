Week 7: Experimentation Done
============================

.. post:: July 17, 2023
   :author: João Victor Dell Agli Floriano
   :tags: google
   :category: gsoc

Hello everyone, welcome to another weekly blogpost! Let's talk about the current status of my project (spoiler: it is beautiful).

Last Week's Effort
------------------
Having accomplished a KDE rendering to a billboard last week, I was then tasked with trying a different approach to how the
rendering was done. So, to recap, below was how I was doing it:

1. Render one point's KDE offscreen to a single billboard, passing its position and sigma to the fragment shader as uniforms.
2. Capture the last rendering's screen as a texture.
3. Render the next point's KDE, and sum it up with the last rendering's texture.
4. Do this until the end of the points.
5. Capture the final render screen as a texture.
6. Apply post processing effects (colormapping).
7. Render the result to the screen.

This approach was good, but it had some later limitations and issues that would probably take more processing time and attention to details (correct matrix
transformations, etc) than the ideal. The different idea is pretty similar, but with some differences:

1. Activate additive blending in OpenGL.
2. Render each point's KDE to its own billboard, with position defined by the point's position, all together in one pass.
3. Capture the rendered screen as a texture.
4. Pass this texture to a billboard.
5. Apply post processing effects (colormapping).
6. Render the result to the screen.

So I needed to basically do that.

Was it Hard?
------------
Fortunately, it wasn't so hard to do it in the end. Following those steps turned out pretty smooth, and after some days,
I had the below result:

.. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/final_2d_plot.png
   :align: center
   :alt: Final 2D KDE render

This is a 2D KDE render of random 1000 points. For this I used the *"viridis"* colormap from `matplotlib`. Some details worth noting:

* For this to work, I have implemented three texture helper functions: `window_to_texture()`, `texture_to_actor()` and `colormap_to_texture()`. The first one captures a window and pass it as a texture to an actor, the second one passes an imported texture to an actor, and the last one passes a colormap, prior passed as an array, as a texture to an actor.
* The colormap is directly get from `matplotlib`, available in its `colormaps` object.
* This was only a 2D flatten plot. At first, I could not figure out how to make the connection between the offscreen interactor and the onscreen one, so rotating and moving around the render was not happening. After some ponder and talk to my mentors, they told me to use *callback* functions inside the interactor, and after doing that, I managed to make the 3D render work, which had the following result:

.. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/3d_kde_gif.gif
   :align: center
   :alt: 3D KDE render

After those results, I refactored my PR `#804 <https://github.com/fury-gl/fury/pull/804>`_ to better fit its current status, and it is
now ready for review. Success!


This Week's Goals
-----------------
After finishing the first iteration of my experimental program, the next step is to work on an API for KDE rendering. I plan to meet
with my mentors and talk about the details of this API, so expect an update next week. Also, I plan to take a better look on my fellow GSoC FURY
contributors work so when their PRs are ready for review, I will have to be better prepared for it.

Let's get to work!
