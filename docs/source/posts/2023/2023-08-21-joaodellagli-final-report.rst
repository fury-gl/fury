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

.. post:: August 21 2023
   :author: João Victor Dell Agli Floriano
   :tags: google
   :category: gsoc

-  **Name:** João Victor Dell Agli Floriano
-  **Organisation:** Python Software Foundation
-  **Sub-Organisation:** FURY
-  **Project:** `FURY - Project 2. Fast 3D kernel-based density rendering using billboards. <https://github.com/fury-gl/fury/wiki/Google-Summer-of-Code-2023-(GSOC2023)#project-2-fast-3d-kernel-based-density-rendering-using-billboards>`_


Abstract
--------
This project had the goal to implement 3D Kernel Density Estimation rendering to FURY. Kernel Density Estimation, or KDE, is a
statistical method that uses kernel smoothing for modeling and estimating the density distribution of a set of points defined
inside a given region. For its graphical implementation, it was used post-processing techniques such as offscreen rendering to
framebuffers and colormap post-processing as tools to achieve the desired results. This was completed with a functional basic KDE
rendering result, that relies on a solid and easy-to-use API, as well as some additional features.

Proposed Objectives
-------------------

- **First Phase** : Implement framebuffer usage in FURY
    * Investigate the usage of float framebuffers inside FURY's environment.
    * Implement a float framebuffer API.

- **Second Phase** : Shader-framebuffer integration
    * Implement a shader that uses a colormap to render framebuffers.
    * Escalate this rendering for composing multiple framebuffers.

- **Third Phase** : KDE Calculations
    * Investigate KDE calculation for point-cloud datasets.
    * Implement KDE calculation inside the framebuffer rendering shaders.
    * Test KDE for multiple datasets.

Objectives Completed
--------------------

- **Implement framebuffer usage in FURY**
    The first phase, addressed from *May/29* to *July/07*, started with the investigation of
    `VTK's Framebuffer Object <https://vtk.org/doc/nightly/html/classvtkOpenGLFramebufferObject.html#details>`_, a vital part of this project, to understand
    how to use it properly.

    Framebuffer Objects, abbreviated as FBOs, are the key to post-processing effects in OpenGL, as they are used to render things offscreen and save the resulting image to a texture
    that will be later used to apply the desired post-processing effects within the object's `fragment shader <https://www.khronos.org/opengl/wiki/Fragment_Shader>`_
    rendered to screen, in this case, a `billboard <http://www.opengl-tutorial.org/intermediate-tutorials/billboards-particles/billboards/>`_. In the case of the
    `Kernel Density Estimation <https://en.wikipedia.org/wiki/Kernel_density_estimation>`_ post-processing effect, we need a special kind of FBO, one that stores textures'
    values as floats, different from the standard 8-bit unsigned int storage. This is necessary because the KDE rendering involves rendering every KDE point calculation
    to separate billboards, rendered to the same scene, which will have their intensities, divided by the number of points rendered, blended with
    `OpenGL Additive Blending <https://www.khronos.org/opengl/wiki/Blending>`_, and if a relative big number of points are rendered at the
    same time, 32-bit float precision is needed to guarantee that small-intensity values will not be capped to zero, and disappear.

    After a month going through VTK's FBO documentation and weeks spent trying different approaches to this method, it would not work
    properly, as some details seemed to be missing from the documentation, and asking the community haven't solved the problem as well.
    Reporting that to my mentors, which unsuccessfully tried themselves to make it work, they decided it was better if another path was taken, using
    `VTK's WindowToImageFilter <https://vtk.org/doc/nightly/html/classvtkWindowToImageFilter.html>`_ method as a workaround, described
    in this `blogpost <https://fury.gl/latest/posts/2023/2023-07-03-week-5-joaodellagli.html>`_. This method helped the development of
    three new functions to FURY, *window_to_texture()*, *texture_to_actor()* and *colormap_to_texture()*, that allow the passing of
    different kinds of textures to FURY's actor's shaders, the first one to capture a window and pass it as a texture to an actor,
    the second one to pass an external texture to an actor, and the third one to specifically pass a colormap as a texture to an
    actor. It is important to say that *WindowToImageFilter()* is not the ideal way to make it work, as this method does not seem to
    support float textures. However, a workaround to that is currently being worked on, as I will describe later on.

    *Pull Requests:*

    - **KDE Rendering Experimental Program (Needs major revision):** `https://github.com/fury-gl/fury/pull/804 <https://github.com/fury-gl/fury/pull/804>`_

    The result of this whole FBO and WindowToImageFilter experimentation is well documented in PR
    `#804 <https://github.com/fury-gl/fury/pull/804>`_ that implements an experimental version of a KDE rendering program.
    The future of this PR, as discussed with my mentors, is to be better documented to be used as an example for developers on
    how to develop features in FURY with the tools used, and it shall be done soon.

- **Shader-framebuffer integration**
    The second phase, which initially was thought of as "Implement a shader that uses a colormap to render framebuffers" and "Escalate this
    rendering for composing multiple framebuffers" was actually a pretty simple phase that could be addressed in one week, *July/10*
    to *July/17*, done at the same time as the third phase goal, documented in this
    `blogpost <https://fury.gl/latest/posts/2023/2023-07-17-week-7-joaodellagli.html>`_. As FURY already had a tool for generating and
    using colormaps, they were simply connected to the shader part of the program as textures, with the functions explained above.
    Below, is the result of the *matplotlib viridis* colormap passed to a simple gaussian KDE render:

    .. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/final_2d_plot.png
       :align: center
       :alt: Final 2D plot

    That is also included in PR `#804 <https://github.com/fury-gl/fury/pull/804>`_. Having the 2D plot ready, some time was taken to
    figure out how to enable a 3D render, that includes rotation and other movement around the set rendered, which was solved by
    learning about the callback properties that exist inside *VTK*. Callbacks are ways to enable code execution inside the VTK rendering
    loop, enclosed inside *vtkRenderWindowInteractor.start()*. If it is desired to add a piece of code that, for example, passes a time
    variable to the fragment shader over time, a callback function can be declared:

    .. code-block:: python

        from fury import window
        t = 0
        showm = window.ShowManager(...)

        def callback_function:
            t += 0.01
            pass_shader_uniforms_to_fs(t, "t")

        showm.add_iren_callback(callback_function, "RenderEvent")

    The piece of code above created a function that updates the time variable *t* in every *"RenderEvent"*, and passes it to the
    fragment shader. With that property, the camera and some other parameters could be updated, which enabled 3D visualization, that
    then, outputted the following result, using *matplotlib inferno* colormap:

    .. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/3d_kde_gif.gif
       :align: center
       :alt: 3D Render gif

- **KDE Calculations** (ongoing)
    As said before, the second and third phases were done simultaneously, so after having a way to capture the window and use it as a
    texture ready, the colormap ready, and an initial KDE render ready, all it was needed to do was to improve the KDE calculations.
    As this `Wikipedia page <https://en.wikipedia.org/wiki/Kernel_density_estimation>`_ explains, a KDE calculation is to estimate an
    abstract density around a set of points defined inside a given region with a kernel, that is a function that models the density
    around a point based on its associated distribution :math:`\sigma`.

    A well-known kernel is, for example, the **Gaussian Kernel**, that says that the density around a point :math:`p` with distribution
    :math:`\sigma` is defined as:

    .. math::

        GK_{\textbf{p}, \sigma} (\textbf{x}) = e^{-\frac{1}{2}\frac{||\textbf{x} - \textbf{p}||^2}{\sigma^2}}

    Using that kernel, we can calculate the KDE of a set of points :math:`P` with associated distributions :math:`S` calculating their individual
    Gaussian distributions, summing them up and dividing them by the total number of points :math:`n`:

    .. math::

        KDE(A, S)=\frac{1}{n}\sum_{i = 0}^{n}GK(x, p_{i}, \sigma_{i})

    So I dove into implementing all of that into the offscreen rendering part, and that is when the lack of a float framebuffer would
    charge its cost. As it can be seen above, just calculating each point's density isn't the whole part, as I also need to divide
    everyone by the total number of points :math:`n`, and then sum them all. The problem is that, if the number of points its big enough,
    the individual densities will be really low, and that would not be a problem for a 32-bit precision float framebuffer, but that is
    *definitely* a problem for a 8-bit integer framebuffer, as small enough values will simply underflow and disappear. That issue is
    currently under investigation, and some solutions have already being presented, as I will show in the **Objectives in Progress**
    section.

    Apart from that, after having the experimental program ready, I focused on modularizing it into a functional and simple API
    (without the :math:`n` division for now), and I could get a good set of results from that. The API I first developed implemented the
    *EffectManager* class, responsible for managing all of the behind-the-scenes steps necessary for the kde render to work,
    encapsulated inside the *ÈffectManager.kde()* method. It had the following look:

    .. code-block:: python
        from fury.effect_manager import EffectManager
        from fury import window

        showm = window.ShowManager(...)

        # KDE rendering setup
        em = EffectManager(showm)
        kde_actor = em.kde(...)
        # End of KDE rendering setup

        showmn.scene.add(kde_actor)

        showm.start()

    Those straightforward instructions, that hid several lines of code and setup, could manage to output the following result:

    .. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/fianl_3d_plot.png
       :align: center
       :alt: API 3D KDE plot

    And this was not the only feature I had implemented for this API, as the use of *WindowToImageFilter* method opened doors for a
    whole new world for FURY: The world of post-processing effects. With this features setup, I managed to implement a *gaussian blur*
    effect, a *grayscale* effect and a *Laplacian* effect for calculating "borders":

    .. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/gaussian_blur.png
       :align: center
       :alt: Gaussian Blur effect

    .. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/grayscale.png
       :align: center
       :alt: Grayscale effect

    .. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/laplacian1.gif
       :align: center
       :alt: Laplacian effect

    As this wasn't the initial goal of the project and I still had several issues to deal with, I have decided to leave these features as a
    future addition.

    Talking with my mentors, we realized that the first KDE API, even though simple, could lead to bad usage from users, as the
    *em.kde()* method, that outputted a *FURY actor*, had dependencies different from any other object of its kind, making it a new
    class of actors, which could lead to confusion and bad handling. After some pair programming sessions, they instructed me to take
    a similar, but different road from what I was doing, turning the kde actor into a new class, the *KDE* class. This class would
    have almost the same set of instructions present in the prior method, but it would break them in a way it would only be completely
    set up after being passed to the *EffectManager* via its add function. Below, how the refactoring handles it:

    .. code-block:: python

        from fury.effects import EffectManager, KDE
        from fury import window

        showm = window.ShowManager(...)

        # KDE rendering setup
        em = EffectManager(showm)
        kde_effect = KDE(...)
        em.add(kde_effect)
        # End of KDE rendering setup

        showm.start()

    Which outputted the same results as shown above. It may have cost some simplicity as we are now one line farther from having it
    working, but it is more explicit in telling the user this is not just a normal actor.

    Another detail I worked on was the kernel variety. The Gaussian Kernel isn't the only one available to model density distributions,
    there are several others that can do that job, as it can be seen in this `scikit-learn piece of documentation <https://scikit-learn.org/stable/modules/density.html>`_
    and this `Wikipedia page on kernels <https://en.wikipedia.org/wiki/Kernel_(statistics)>`_. Based on the scikit-learn KDE
    implementation, I worked on implementing the following kernels inside our API, that can be chosen as a parameter when calling the
    *KDE* class:

    * Cosine
    * Epanechnikov
    * Exponential
    * Gaussian
    * Linear
    * Tophat

    Below, the comparison between them using the same set of points and bandwidths:

    .. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/kernels.png
       :align: center
       :alt: Comparison between the six implemented kernels


    *Pull Requests*:

    - **First Stage of the KDE Rendering API (will be merged soon)**: `https://github.com/fury-gl/fury/pull/826 <https://github.com/fury-gl/fury/pull/826>`_

    All of this work culminated in PR `#826 <https://github.com/fury-gl/fury/pull/826/>`_, that proposes to add the first stage of
    this API (there are some details yet to be completed, like the :math:`n` division) to FURY. This PR added the described API, and also
    proposed some minor changes to some already existing FURY functions related to callbacks, changes necessary for this and other
    future applications that would use it to work. It also added the six kernels described, and a simple documented example on how
    to use this feature.

Other Objectives
----------------

- **Stretch Goals** : SDE Implementation, Network/Graph visualization using SDE/KDE, Tutorials
    * Investigate SDE calculation for surface datasets.
    * Implement SDE calculation inside the framebuffer rendering shaders.
    * Test SDE for multiple datasets.
    * Develop comprehensive tutorials that explain SDE concepts and FURY API usage.
    * Create practical, scenario-based tutorials using real datasets and/or simulations.

Objectives in Progress
----------------------

- **KDE Calculations** (ongoing)
    The KDE rendering, even though almost complete, have the $n$ division, an important step, missing, as this normalization allows colormaps
    to cover the whole range o values rendered. The lack of a float FBO made a big difference in the project, as the search for a functional implementation of it not only delayed the project, but it is vital for
    the correct calculations to work.

    For the last part, a workaround thought was to try an approach I later figured out is an old one, as it can be check in
    `GPU Gems 12.3.3 section <https://developer.nvidia.com/gpugems/gpugems/part-ii-lighting-and-shadows/chapter-12-omnidirectional-shadow-mapping>`_:
    If I need 32-bit float precision and I got 4 8-bit integer precision available, why not trying to pack this float into this RGBA
    texture? I have first tried to do one myself, but it didn't work for some reason, so I tried `Aras Pranckevičius <https://aras-p.info/blog/2009/07/30/encoding-floats-to-rgba-the-final/>`_
    implementation, that does the following:

    .. code-block:: GLSL

        vec4 float_to_rgba(float value) {
            vec4 bitEnc = vec4(1.,256.,65536.0,16777216.0);
            vec4 enc = bitEnc * value;
            enc = fract(enc);
            enc -= enc.yzww * vec2(1./255., 0.).xxxy;
            return enc;
        }

    That initially worked, but for some reason I am still trying to understand, it is resulting in a really noisy texture:

    .. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/noisy%20kde.png
       :align: center
       :alt: Noisy KDE render

    One way to try to mitigate that while is to pass this by a gaussian blur filter, to try to smooth out the result:

    .. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/blurred_kde.png
       :align: center
       :alt: Blurred result

    But it is not an ideal solution as well, as it may lead to distortions in the actual density values, depending on the application of
    the KDE. Now, my goal is to first find the root of the noise problem, and then, if that does not work, try to make the gaussian filter
    work.

    Another detail that would be a good addition to the API is UI controls. Filipi, one of my mentors, told me it would be a good feature
    if the user could control the intensities of the bandwidths for a better structural visualization of the render, and knowing FURY already
    have a good set of `UI elements <https://fury.gl/latest/auto_examples/index.html#user-interface-elements>`_, I just needed to integrate
    that into my program via callbacks. I tried implementing an intensity slider. However, for some reason, it is making the program crash
    randomly, for reasons I still don't know, so that is another issue under investigation. Below, we show a first version of that feature,
    which was working before the crashes:

    .. image:: https://raw.githubusercontent.com/JoaoDell/gsoc_assets/main/images/slider.gif
       :align: center
       :alt: Slider for bandwidths

    *Pull Requests*

    - **UI intensity slider for the KDE rendering API (draft)**: `https://github.com/fury-gl/fury/pull/849 <https://github.com/fury-gl/fury/pull/849>`_
    - **Post-processing effects for FURY Effects API (draft)**: `https://github.com/fury-gl/fury/pull/850 <https://github.com/fury-gl/fury/pull/850>`_


GSoC Weekly Blogs
-----------------

- My blog posts can be found at `FURY website <https://fury.gl/latest/blog/author/joao-victor-dell-agli-floriano.html>`_ and `Python GSoC blog <https://blogs.python-gsoc.org/en/joaodellaglis-blog/>`_.

Timeline
--------

+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Date                | Description                                        | Blog Post Link                                                                                                                                                                                            |
+=====================+====================================================+===========================================================================================================================================================================================================+
| Week 0 (29-05-2023) | The Beginning of Everything                        | `FURY <https://fury.gl/latest/posts/2023/2023-05-29-week-0-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/the-beggining-of-everything-week-0/>`__                  |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 1 (05-06-2022) | The FBO Saga                                       | `FURY <https://fury.gl/latest/posts/2023/2023-06-05-week-1-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/the-fbo-saga-week-1/>`__                                 |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 2 (12-06-2022) | The Importance of (good) Documentation             | `FURY <https://fury.gl/latest/posts/2023/2023-06-12-week-2-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/the-importance-of-good-documentation-week-2/>`__         |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 3 (19-06-2022) | Watch Your Expectations                            | `FURY <https://fury.gl/latest/posts/2023/2023-06-19-week-3-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-3-watch-your-expectations/>`__                      |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 4 (26-06-2022) | Nothing is Ever Lost                               | `FURY <https://fury.gl/latest/posts/2023/2023-06-26-week-4-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-4-nothing-is-ever-lost/>`__                         |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 5 (03-07-2022) | All Roads Lead to Rome                             | `FURY <https://fury.gl/latest/posts/2023/2023-07-03-week-5-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-5-all-roads-lead-to-rome/>`__                       |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 6 (10-07-2022) | Things are Starting to Build Up                    | `FURY <https://fury.gl/latest/posts/2023/2023-07-10-week-6-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-6-things-are-starting-to-build-up/>`__              |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 7 (17-07-2022) | Experimentation Done                               | `FURY <https://fury.gl/latest/posts/2023/2023-07-17-week-7-joaodellagli.html>`__ - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-7-experimentation-done/>`__                         |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 8 (24-07-2022) | The Birth of a Versatile API                       | `FURY <https://fury.gl/latest/posts/2023/2023-07-24-week-8-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-8-the-birth-of-a-versatile-api/>`__                 |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 9 (31-07-2022) | It is Polishing Time!                              | `FURY <https://fury.gl/latest/posts/2023/2023-07-31-week-9-joaodellagli.html>`__  - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-9-it-is-polishing-time/>`__                         |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 10 (07-08-2022)| Ready for Review!                                  | `FURY <https://fury.gl/latest/posts/2023/2023-08-07-week-10-joaodellagli.html>`__ - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/ready-for-review/>`__                                    |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 11 (14-08-2022)| A Refactor is Sometimes Needed                     | `FURY <https://fury.gl/latest/posts/2023/2023-08-14-week-11-joaodellagli.html>`__ - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/a-refactor-is-sometimes-needed/>`__                      |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Week 12 (21-08-2022)| Now That is (almost) a Wrap!                       | `FURY <https://fury.gl/latest/posts/2023/2023-08-21-week-12-joaodellagli.html>`__ - `Python <https://blogs.python-gsoc.org/en/joaodellaglis-blog/week-12-now-that-is-almost-a-wrap/>`__                   |
+---------------------+----------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
