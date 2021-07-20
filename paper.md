---
title: 'FURY: advanced scientific visualization'
tags:
  - Python
  - scientific visualization
  - 3D rendering
  - Shaders
  - GLSL
  - Vulkan
authors:
  - name: Eleftherios Garyfallidis
    orcid: 0000-0002-3991-2774
    affiliation: 1
  - name: Serge Koudoro
    orcid: 0000-0002-9819-9884
    affiliation: 1
  - name: Javier Guaje
    orcid: 0000-0003-3534-3794
    affiliation: 1
  - name: Marc-Alex Côté
    orcid: 0000-0002-5147-7859
    affiliation: 3
  - name: Soham Biswas
    orcid: 0000-0002-8449-2107
    affiliation: 4
  - name: David Reagan
    orcid: 0000-0002-8359-7580
    affiliation: 5
  - name: Nasim Anousheh
    orcid: 0000-0002-5931-7753
    affiliation: 1
  - name: Filipi Silva
    orcid: 0000-0002-9151-6517
    affiliation: 2
  - name: Geoffrey Fox
    orcid: 0000-0003-1017-1391
    affiliation: 1
  - name: FURY Contributors
    affiliation: 6
affiliations:
 - name: Department of Intelligent Systems Engineering, Luddy School of Informatics, Computing and Engineering, Indiana University, Bloomington, IN, USA
   index: 1
 - name: Network Science Institute, Indiana University, Bloomington, IN, USA
   index: 2
 - name: Microsoft Research, Montreal, Canada
   index: 3
 - name: Department of Computer Science and Engineering, Institute of Engineering and Management, Kolkata, India
   index: 4
 - name: Advanced Visualization Lab, University Information Technology Services, Indiana University, Bloomington, IN, USA
   index: 5
 - name: Anywhere in the Universe
   index: 6

date: 8 April 2021
bibliography: paper.bib
---



# Summary

Free Unified Rendering in pYthon (FURY), is a community-driven, open-source, and high-performance scientific visualization library that harnesses the graphics processing unit (GPU) for improved speed, precise interactivity, and visual clarity. FURY provides an integrated API in Python that allows UI elements and 3D graphics to be programmed together. FURY is designed to be fully interoperable with most projects of the Pythonic ecosystem that use NumPy [@harris2020array] for processing numerical arrays. In addition, FURY uses core parts of VTK [@schroeder1996visualization] and enhances them using customized shaders. FURY provides access to the latest technologies such as raytracing, signed distance functionality, physically based rendering, and collision detection for direct use in research. More importantly, FURY enables students and researchers to script their own 3D animations in Python and simulate dynamic environments.


# Statement of need

The massive amount of data collected and analyzed by scientists in several disciplines requires powerful tools and techniques able to handle these whilst still managing efficiently the computational resources available. In some particular disciplines, these datasets not only are large but also encapsulate the dynamics of their environment, increasing the demand for resources. Although 3D visualization technologies are advancing quickly [@sellers2016vulkan], their sophistication and focus on non-scientific domains makes it hard for researchers to use them.  In other words, most of the existing 3D visualization and computing APIs are low-level (close to the hardware) and made for professional specialist developers.  Because of these issues, there is a significant barrier to many scientists and these powerful technologies are rarely deployed to everyday research practices.

Therefore, FURY is created to address this necessity of high-performance 3D scientific visualization in an easy-to-use API fully compatible with the Pythonic ecosystem.

# FURY Architecture

FURY is built to be modular, scalable, and to respect software engineering principles including a well-documented codebase and unit integration testing. The framework runs in all major operating systems including multiple Linux distributions, Windows, and macOS. Also, it can be used on the desktop and the web. The framework contains multiple interconnected engines, modules, API managers as illustrated in \autoref{fig:architecture}.

![The FURY framework contains multiple interconnected engines to bring forward advanced visualization capabilities. Additionally, it contains an integrated user interface module and an extendable I/O module. One of the most important classes is the Scene Manager that connects the actors to the shaders, animations, and interactors for picking 3D objects. The actors are directly connected to NumPy arrays with vertices, triangles, and connectivity information that is provided by the core engine. These are then connected to the physics and networks  engines.\label{fig:architecture}](https://raw.githubusercontent.com/fury-gl/fury-communication-assets/main/fury_paper/architecture.png)


**Rendering Engine**: This engine includes managers like scene, animation, shader, and picking manager. The scene manager allows the visual objects to appear on a canvas. The picking manager allows selecting specific objects in the scene. The animation manager allows users to script their own 3D animations and videos with timelines allowing objects to act in specific times. Lastly, the shader manager provides several interfaces to different elements in the OpenGL rendering pipeline. This manager allows developers to add customized shaders snippets to the existing shaders included in the API.

**Core Engine**: This engine contains utilities for building actors from primitives and transforming them. A primitive is an object that describes its shape and connectivity with elements such as vertices and triangles.

**Physics Engine**: This engine allows us to either build collision mechanisms as used in molecular dynamics or integrate well-established engines such as Bullet [@coumans2013bullet] and NVIDIA PhysX [@harris2009cuda].

**Networks Engine**:  This engine allows for the creation and use of graph systems and layouts.

**Integrated User Interfaces Module**: FURY contains its own user interfaces. This module provides a range of UI 2D / 3D elements such as buttons, combo boxes, and file dialogues. Nevertheless, users can easily connect to other known GUIs such as Qt or IMGUI if necessary.

**I/O module**: FURY supports a range of file formats from the classic OBJ format to the more advanced GLTF format that can be used to describe a complete scene with many actors and animations.

**Interoperability**: FURY can be used together with projects such as SciPy [@virtanen2020scipy], Matplotlib [@hunter2007matplotlib], pandas [@mckinney2010data], scikit-learn [@pedregosa2011scikit], NetworkX [@hagberg2008exploring], PyTorch [@paszke2019pytorch] and TensorFlow [@abadi2016tensorflow].

FURY’s visualization API can be compared with VisPy [@campagnola2015vispy], glumpy [@rougier2015glumpy], Mayavi [@ramachandran2011mayavi], and others. VisPy and glumpy directly connect to OpenGL. FURY uses OpenGL through Python VTK, which can be advantageous because it can use the large stack of visualization algorithms available in VTK. This is similar to Mayavi, however, FURY provides an easy and efficient way to ease interaction with 3D scientific data via integrated user interface elements and allows to reprogram the low-level shaders for the creation of stunning effects (see \autoref{fig:features}) not available in VTK. Historically, FURY had also a different path than these libraries as it was originally created for heavy-duty medical visualization purposes for DIPY [@garyfallidis2014dipy]. As the project grew it spinned off as an independent project with applications across the domains of science and engineering including visualization of nanomaterials and robotics simulations.




![**Top**. Dynamic changes are shown as diffused waves on the surface of the horse visualized with FURY. Showing here 4 frames at 4 different time points (t1−t4). A vertex and fragment shader are used to calculate in real-time the mirroring texture and blend its colors with the blue-yellow wave. **Bottom**. In FURY we create actors that contain multiple visual objects controlled by NumPy arrays.  Here an actor is generating 5 superquadrics with different properties (e.g. colors, directions, metallicity) by injecting the information as NumPy arrays in a single call.  This is one of the important design choices that make FURY easier to use but also faster to render. Actors in FURY can contain many objects. The user can select any of the objects in the actor. Here the user selected the first object (spherical superquadric).\label{fig:features}](https://raw.githubusercontent.com/fury-gl/fury-communication-assets/main/fury_paper/features.png)



# Acknowledgements
FURY is partly funded through NSF #1720625 Network for Computational Nanotechnology - Engineered nanoBIO Node [@klimeck2008nanohub].


# References