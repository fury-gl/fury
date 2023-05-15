.. _releasev0.7.0:

===================================
 Release notes v0.7.0 (2021/03/13)
===================================

Quick Overview
--------------

* New SDF actors added.
* Materials module added.
* ODF slicer actor performance improved.
* New primitive (Cylinder) added.
* Compatibility with VTK 9 added.
* Five new demos added.
* Large Documentation Update.
* Migration from Travis to Github Action.


Details
-------

GitHub stats for 2020/08/20 - 2021/03/13 (tag: v0.6.1)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 14 authors contributed 195 commits.

* Eleftherios Garyfallidis
* Serge Koudoro
* Charles Poirier
* Javier Guaje
* Soham Biswas
* Sajag Swami
* Lenix Lobo
* Pietro Astolfi
* Sanjay Marreddi
* Tushar
* ganimtron-10
* haran2001
* Aju100
* Aman Soni


We closed a total of 98 issues, 37 pull requests and 61 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (37):

* :ghpull:`388`: added simulation for brownian motion
* :ghpull:`389`: ENH: peaks_slicer option for asymmetric peaks visualization
* :ghpull:`370`: Materials module including Physically Based Rendering (PBR)
* :ghpull:`385`: fixed the example for superquadric function
* :ghpull:`387`: [fix]   Propagate update_actor
* :ghpull:`382`: Added an sdf for rendering a Capsule actor
* :ghpull:`383`: Minor documentation fix
* :ghpull:`376`: Added animations for some electromagnetic phenomena
* :ghpull:`374`: ENH: Refactor actor.odf_slicer for increased performances
* :ghpull:`373`: Updated actor.py
* :ghpull:`368`: Solving The Minor Documentation Error
* :ghpull:`343`: Adding physics engine integration docs
* :ghpull:`353`: fix: Minor docs changes
* :ghpull:`346`: Fix the sdf bug by checking the arguments passed
* :ghpull:`351`: Opacity bug fix for point and sphere actors
* :ghpull:`350`: modelsuzanne to suzanne
* :ghpull:`348`: Added center forwarding in billboard shaders.
* :ghpull:`341`: Add Option to generate the documentation without examples
* :ghpull:`342`: From Travis to Github Actions
* :ghpull:`339`: Update Readme information
* :ghpull:`340`: Pass OAuth token through header
* :ghpull:`337`: Add support for clipping side in clip_overflow_text
* :ghpull:`336`: Update UI tutorials.
* :ghpull:`334`: Added Domino-Simulation-file for Review
* :ghpull:`332`: Fixing UI warnings
* :ghpull:`328`: Added cylinder primitive
* :ghpull:`329`: [FIX] Force LUT to be RGB
* :ghpull:`286`: GSoC blogs for Third Evaluation.
* :ghpull:`319`: fixed discord icon bug in documentation
* :ghpull:`311`: Remove python35 from Travis
* :ghpull:`307`: Fixed translating and scaling issues on billboard and SDF actors
* :ghpull:`304`: Blogs for the final review
* :ghpull:`306`: merged basic UI and advanced UI tutorials into one
* :ghpull:`302`: moved physics tutorials to examples under the heading 'Integrate physics using pybullet'
* :ghpull:`303`: FIX vtp reader
* :ghpull:`300`: BF: Out should be varying and alpha is not passed to shader
* :ghpull:`295`: Update fetcher

Issues (61):

* :ghissue:`388`: added simulation for brownian motion
* :ghissue:`389`: ENH: peaks_slicer option for asymmetric peaks visualization
* :ghissue:`370`: Materials module including Physically Based Rendering (PBR)
* :ghissue:`385`: fixed the example for superquadric function
* :ghissue:`387`: [fix]   Propagate update_actor
* :ghissue:`382`: Added an sdf for rendering a Capsule actor
* :ghissue:`383`: Minor documentation fix
* :ghissue:`376`: Added animations for some electromagnetic phenomena
* :ghissue:`374`: ENH: Refactor actor.odf_slicer for increased performances
* :ghissue:`364`: New Animated Network Demo/Example
* :ghissue:`379`: Merge pull request #2 from fury-gl/master
* :ghissue:`361`: Closes #352
* :ghissue:`373`: Updated actor.py
* :ghissue:`372`: Ellipsoid primitive needs to be added in the comment section of sdf actor.
* :ghissue:`369`: Added Special Character Support
* :ghissue:`363`: Minor error in documentation of create_colormap function
* :ghissue:`368`: Solving The Minor Documentation Error
* :ghissue:`366`: added special character support for TextBox2D
* :ghissue:`357`: Patches: vulnerable code that can lead to RCE
* :ghissue:`359`: unwanted objects rendering randomly
* :ghissue:`343`: Adding physics engine integration docs
* :ghissue:`312`: Adding Physics Integration Docs to FURY's Website
* :ghissue:`353`: fix: Minor docs changes
* :ghissue:`346`: Fix the sdf bug by checking the arguments passed
* :ghissue:`310`: Rendering bug in SDF actor when not all primitives are defined
* :ghissue:`351`: Opacity bug fix for point and sphere actors
* :ghissue:`335`: _opacity argument for point doesn't seem to work
* :ghissue:`345`: Fixes the opacity bug for sphere and point actors (unit tests are included)
* :ghissue:`350`: modelsuzanne to suzanne
* :ghissue:`348`: Added center forwarding in billboard shaders.
* :ghissue:`341`: Add Option to generate the documentation without examples
* :ghissue:`342`: From Travis to Github Actions
* :ghissue:`338`: From travis (pricing model changed) to github Actions ?
* :ghissue:`339`: Update Readme information
* :ghissue:`340`: Pass OAuth token through header
* :ghissue:`315`: Deprecation notice for authentication via URL query parameters
* :ghissue:`337`: Add support for clipping side in clip_overflow_text
* :ghissue:`308`: Clipping overflowing text from the left.
* :ghissue:`336`: Update UI tutorials.
* :ghissue:`334`: Added Domino-Simulation-file for Review
* :ghissue:`309`: Domino Physics Simulation
* :ghissue:`333`: Unable to set up the project locally for python 32bit system
* :ghissue:`332`: Fixing UI warnings
* :ghissue:`239`: Superquadric Slicer
* :ghissue:`328`: Added cylinder primitive
* :ghissue:`318`: Cylinder primitive generation
* :ghissue:`329`: [FIX] Force LUT to be RGB
* :ghissue:`286`: GSoC blogs for Third Evaluation.
* :ghissue:`319`: fixed discord icon bug in documentation
* :ghissue:`313`: Discord icon should appear in doc too
* :ghissue:`311`: Remove python35 from Travis
* :ghissue:`307`: Fixed translating and scaling issues on billboard and SDF actors
* :ghissue:`274`: SDF rendering bug for low scale values
* :ghissue:`304`: Blogs for the final review
* :ghissue:`306`: merged basic UI and advanced UI tutorials into one
* :ghissue:`297`: Update Demos/tutorial
* :ghissue:`302`: moved physics tutorials to examples under the heading 'Integrate physics using pybullet'
* :ghissue:`298`: Wrecking Ball Simulation
* :ghissue:`303`: FIX vtp reader
* :ghissue:`300`: BF: Out should be varying and alpha is not passed to shader
* :ghissue:`295`: Update fetcher
