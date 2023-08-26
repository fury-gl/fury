Week 12 : Experimenting with ODFs implementation
================================================

.. post:: August 24, 2023
   :author: Tania Castillo
   :tags: google
   :category: gsoc

What did I do this week?
------------------------

There were different issues I needed to address for the ODF implementation. Even though I could not solve any of them completely, I did check each of the issues and made some progress. All the work in progress is being recorded in the following branch `SH-for-ODF-impl <https://github.com/tvcastillod/fury/tree/SH-for-ODF-impl>`_, which when ready will be associated with a well-structured PR.

First, about the scaling, I was suggested to check Generalized Fractional Anisotropy **gfa** metric to adjust the scaling depending on the shape of the ODF glyph, i.e., the less the **gfa** the more sphere-shaped and smaller, so I had to associate a greater scaling for those. However, this did not work very well as I was unable to define an appropriate scale relation that would give an equitable result for each glyph. For this reason, I opted to use peak values which are extracted from the ODFs, setting the scales as 1/peak_value*0.4 and I got a more uniformly sized glyph without the need of setting it manually. That is a temporal solution as I would like to see better why this happens and if possible do the adjustment inside the shader instead of a precalculation.

Second, for the direction, I made a small adjustment to the spherical coordinates which affected the direction of the ODF glyph. As you can see below,

.. image:: https://user-images.githubusercontent.com/31288525/263122770-b9ee19d2-d82b-4d7f-a5bb-1cbbf5907049.png
    :width: 400
    :align: center

All the glyphs are aligned over the y-axis but not over the z-axis, to correct this I precalculated the main direction of each glyph using peaks and passed it to the shader as a *vec3*, then used *vec2vecrotmat* to align the main axis vector of the ODF to the required direction vector, the only problem with this is that not all the glyps are equally aligned to the axis, i.e., the first 3 glyphs are aligned with the x-axis but the last one is aligned with the y-axis, so the final rotation gives a different result for that one.

.. image:: https://user-images.githubusercontent.com/31288525/263122752-b2aa696f-62a5-4b09-b8dd-0cb1ec49431c.png
    :width: 400
    :align: center

As with the first small adjustment of the coordinates the direction was partially correct, I need to double check the theta, phi and r definitions to see if I can get the right direction without the need of the additional data of direction. Also, there might be a way to get the specific rotation angles associated to each individual glyph from the data associated with the ODFs.

Third, about passing the coefficients data through textures, I understand better now how to pass textures to the shaders but I still have problems understanding how to retrieve the data inside the shader. I used `this base implementation <https://github.com/fury-gl/fury/blob/master/docs/experimental/viz_shader_texture.py>`_, suggested by one of my mentors, to store the data as a `texture cubemap <http://www.khronos.org/opengl/wiki/Cubemap_Texture#:~:text=A%20Cubemap%20Texture%20is%20a,the%20value%20to%20be%20accessed.>`_, "a texture, where each mipmap level consists of six 2D images which must be square. The 6 images represent the faces of a cube". I had 4x15 coefficients and inside the function, a grid of RGB colors is made so then it can be mapped as a texture. To check if was passing the data correctly, I used the same value, .5, for all the textures, so then I could pick a random texel get a specific color (gray), and pass it as *fragOutput0* to see if the value was correct. However, it didn't appear to work correctly as I couldn't get the expected color. To get the specific color I used `texture(sampler, P) <https://registry.khronos.org/OpenGL-Refpages/gl4/html/texture.xhtml>`_ which samples texels from the texture bound to sampler at texture coordinate P. Now, what I still need to figure out is which should be the corresponding texture coordinate. I have tried with random coordinates, as they are supposed to correspond to a point on the cube and since the information I have encoded in the texture is all the same, I assumed that I would get the expected result for any set of values. It might be a problem with the data normalization, or maybe there is something failing on the texture definition, but I need to review it in more detail to see where is the problem.

Lastly, about the colormapping, I created the texture based on a generic colormap from `matplotlib <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_. I was able to give some color to the glyph but it does not match correctly its shape. Some adjustment must be done regarding the texels, as the colormap is mapped on a cube, but I need it to fit the shape of the glyph correctly.

.. image:: https://user-images.githubusercontent.com/31288525/263122760-7d1fff5e-7787-473c-8053-ea69f3009fb4.png
    :width: 250
    :align: center

What is coming up next?
-----------------------

I will continue to explore more on how to handle textures so I can solve the issues related to the coefficient data and colormapping. Also, take a deeper look at the SH implementation and check what is the information needed to adjust the main direction of the ODF correctly.

Did I get stuck anywhere?
-------------------------

As I mentioned I had some drawbacks in understanding the use of textures and how to retrieve the data inside the shaders. This is a topic that might take some time to manage properly but if I can master it and understand it better, it is a tool that can be useful later. Additionally, there are details of the SH implementation that I still need to understand and explore better in order to make sure I get exactly the same result as the current *odf_slicer* implementation.
