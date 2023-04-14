.. _releasev0.9.0:

==============================
 Release notes v0.9.0
==============================

Quick Overview
--------------

* New Streaming System added.
* Large improvement of Signed Distance Functions actors (SDF).
* Continuous Integration (CI) platform updated. Migrate Windows CI from Azure to Github Actions
* Migration from setuptools to hatching. versioning system updated also.
* New actors added (Rhombicuboctahedron, Pentagonal Prism).
* New module fury.animation added.
* New module fury.gltf added. Module to support glTF 2.0.
* Multiple tutorials added and updated.
* Documentation updated.
* Website updated.


Details
-------

GitHub stats for 2022/01/31 - 2023/04/14 (tag: v0.8.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 24 authors contributed 1835 commits.

* Anand Shivam
* Antriksh Misri
* Bruno Messias
* Dwij Raj Hari
* Eleftherios Garyfallidis
* Filipi Nascimento Silva
* Francois Rheault
* Frank Cerasoli
* Javier Guaje
* Johny Daras
* Mohamed Agour
* Nasim Anousheh
* Praneeth Shetty
* Rohit Kharsan
* Sara Hamza
* Serge Koudoro
* Siddharth Gautam
* Soham Biswas
* Sreekar Chigurupati
* Tania Castillo
* Zhiwen Shi
* maharshigor
* sailesh
* sparshg


We closed a total of 379 issues, 166 pull requests and 213 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (166):

* :ghpull:`687`: Record keyframe animation as GIF and MP4
* :ghpull:`782`: Add Codespell and update codecov
* :ghpull:`587`: Billboard tutorial
* :ghpull:`781`: Tab customization
* :ghpull:`779`: versions-corrected
* :ghpull:`741`: Remove unneeded multithreading call
* :ghpull:`778`: TabUI collapsing/expanding improvements
* :ghpull:`777`: Remove alias keyword on documentation
* :ghpull:`771`: add one condition in `repeat_primitive` to handle direction [-1, 0, 0], issue #770
* :ghpull:`766`: Cylinder repeat primitive
* :ghpull:`769`: Merge Demo and examples
* :ghpull:`767`: Update Peak actor shader
* :ghpull:`677`: Cylindrical billboard implementation
* :ghpull:`765`: add instruction about how to get Suzanne model
* :ghpull:`764`: ComboBox2D drop_down_button mouse callback was inside for loop
* :ghpull:`748`: some fixs and ex addition in docstrings in actor.py
* :ghpull:`754`: update viz_roi_contour.py
* :ghpull:`760`: update deprecated function get.data() to get.fdata()
* :ghpull:`761`: add instruction of how to download suzanne model for getting started page
* :ghpull:`762`: update the deprecated get_data() to get_fdata in viz_roi_contour.py in the demo section.
* :ghpull:`756`: Triangle strips 2 Triangles
* :ghpull:`747`: Connected the sliders to the right directions
* :ghpull:`744`: Update initialize management
* :ghpull:`710`: Principled update
* :ghpull:`688`: DrawPanel Update: Moving rotation_slider from `DrawShape` to `DrawPanel`
* :ghpull:`734`: Added GSoC'22 Final Report
* :ghpull:`736`: Adding GSoC'22 final report
* :ghpull:`727`: Feature/scientific domains
* :ghpull:`478`: Resolving GridUI caption error
* :ghpull:`502`: Multithreading support and examples
* :ghpull:`740`: Multithreading example simplified and refactored
* :ghpull:`739`: added a check for operating system before executing the tput command through popen in fury/data/fetcher.py update_progressbar() function
* :ghpull:`737`: remove object keyword from class
* :ghpull:`726`: Adding GSoC'22 Final Report
* :ghpull:`735`: Add precommit
* :ghpull:`728`: Fix flipped images in load, save, and snapshot
* :ghpull:`730`: Update CI and add pyproject.toml
* :ghpull:`729`: Fix links in CONTRIBUTING.rst
* :ghpull:`725`: Improve Doc management + quick fix
* :ghpull:`724`: Feature/community page
* :ghpull:`721`: Fix: Color changes on docs pages fixed
* :ghpull:`723`: Update CI's
* :ghpull:`722`: Fix failing tests due to last numpy release
* :ghpull:`719`: Logo changes
* :ghpull:`718`: Home page mobile friendly
* :ghpull:`717`: Scientific domains enhancement
* :ghpull:`680`: Updating animation tutorials
* :ghpull:`690`: Add Timelines to ShowManager directly
* :ghpull:`694`: Separating the Timeline into Timeline and Animation
* :ghpull:`712`: Fix: segfault created by record method
* :ghpull:`706`: fix: double render call with timeline obj causes a seg fault
* :ghpull:`700`: Adding morphing support in `gltf.py`
* :ghpull:`697`: Adding week 14 blog
* :ghpull:`693`: Adding Week 15 Blogpost
* :ghpull:`701`: Updating `fetch_viz_new_icons` to fetch new icons
* :ghpull:`685`: glTF skinning animation implementation
* :ghpull:`699`: Adding Week 16 Blogpost
* :ghpull:`698`: Added blog post for week 14
* :ghpull:`667`: [WIP] Remove initialize call from multiple places
* :ghpull:`689`: GLTF actor colors from material
* :ghpull:`643`: [WIP] Adding ability to load glTF animations
* :ghpull:`665`: Timeline hierarchical transformation and fixing some issues
* :ghpull:`686`: Adding week 13 blog post
* :ghpull:`684`: Adding Week 14 Blogpost
* :ghpull:`692`: Set position and width of the `PlaybackPanel`
* :ghpull:`691`: Added week 13 post
* :ghpull:`683`: Adding Week 13 Blogpost
* :ghpull:`682`: Adding week 12 blog post
* :ghpull:`681`: Added blog post for week 12
* :ghpull:`672`: Adding Week 12 Blogpost
* :ghpull:`678`: DrawPanel Update: Repositioning the `mode_panel` and `mode_text`
* :ghpull:`661`: Improving `vector_text`
* :ghpull:`679`: DrawPanel Update: Moving repetitive functions to helpers
* :ghpull:`674`: DrawPanel Update: Separating tests to test individual features
* :ghpull:`675`: Week 11 blog post
* :ghpull:`673`: DrawPanel Update: Removing `in_progress` parameter while drawing shapes
* :ghpull:`676`: Adding week 11 blog post
* :ghpull:`671`: Adding Week 11 Blogpost
* :ghpull:`623`: DrawPanel Feature: Adding Rotation of shape from Center
* :ghpull:`670`: Adding week 10 blog post
* :ghpull:`666`: Adding Week 10 Blogpost
* :ghpull:`669`: Added blog post for week 10
* :ghpull:`647`: Keyframe animations and interpolators
* :ghpull:`620`: Tutorial on making a primitive using polygons and SDF
* :ghpull:`630`: Adding function to export scenes as glTF
* :ghpull:`663`: Adding week 9 blog post
* :ghpull:`656`: Week 8 blog post
* :ghpull:`662`: Week 9 blog post
* :ghpull:`654`: Adding Week 9 Blogpost
* :ghpull:`659`: Adding week 8 blog post
* :ghpull:`650`: Adding Week 8 Blogpost
* :ghpull:`655`: Fix test skybox
* :ghpull:`645`: Fixing `ZeroDivisionError` thrown by UI sliders when the `value_range` is zero (0)
* :ghpull:`648`: Adding week 7 blog post
* :ghpull:`649`: Added week 7 blog post
* :ghpull:`646`: Adding Week 7 Blogpost
* :ghpull:`641`: Week 6 blog post
* :ghpull:`644`: Adding week 6 blog post
* :ghpull:`638`: Adding Week 6 Blogpost
* :ghpull:`639`: Migrate Windows from Azure to GHA
* :ghpull:`634`: Prevented calling `on_change` when slider value is set without user intervention
* :ghpull:`637`: Adding week 5 blog post
* :ghpull:`632`: Bugfix: Visibility issues with ListBox2D
* :ghpull:`610`: Add DPI support for window snapshots
* :ghpull:`633`: Added week 5 blog post
* :ghpull:`617`: Added primitives count to the the Actor's polydata
* :ghpull:`624`: Adding Week 5 BlogPost
* :ghpull:`627`: Adding week 4 blog post
* :ghpull:`625`: Added week 4 blog post
* :ghpull:`600`: Adding support for importing simple glTF files
* :ghpull:`622`: Adding week 3 blog post
* :ghpull:`619`: Week 3 blog post.
* :ghpull:`621`: Adding Week 4 Blogpost
* :ghpull:`616`: Fixing API limits reached issue in gltf fetcher
* :ghpull:`611`: Adding Week 3 BlogPost
* :ghpull:`614`: Added week 2 blog
* :ghpull:`615`: Added blog post for week 2
* :ghpull:`607`: Adding Week 2 Blog Post
* :ghpull:`599`: Creating `DrawPanel` UI
* :ghpull:`606`: Added week 1 post
* :ghpull:`608`: Adding week 1 blog post
* :ghpull:`597`: Added an accurate way to get the FPS for the showManager
* :ghpull:`605`: Adding Week1 Blog Post
* :ghpull:`501`: Creating an `off_focus` hook in `TextBox2D`
* :ghpull:`602`: Added support for fetching gltf samples
* :ghpull:`609`: Creating a fetcher to fetch new icons
* :ghpull:`601`: Updating author's name in README
* :ghpull:`593`: Support empty ArraySequence in saving (for empty vtk)
* :ghpull:`598`: Timer id is returned after creating the timer.
* :ghpull:`581`: Keep original dtype for offsets in vtk format
* :ghpull:`595`: changed `use_primitive` to false by default
* :ghpull:`589`: First blog: GSoC
* :ghpull:`586`: Added my first blog post
* :ghpull:`594`: Fixed multi_samples not being used.
* :ghpull:`591`: Fixed some old tutorials.
* :ghpull:`590`: Adding Pre-GSoC Journey Blog Post
* :ghpull:`584`: Changing dot actor
* :ghpull:`582`: Deprecation of the function shaders.load
* :ghpull:`580`: Update website
* :ghpull:`437`: FURY Streaming System Proposal
* :ghpull:`574`: symmetric parameter for peak
* :ghpull:`561`: Shader API improvements
* :ghpull:`533`: Sphere actor uses repeat_primitive by default
* :ghpull:`577`: Added play/pause buttons
* :ghpull:`443`: Adapt GridLayout to work with UI
* :ghpull:`570`: Function to save screenshots with magnification factor
* :ghpull:`486`: Added `x,y,z` layouts to the layout module.
* :ghpull:`547`: Cone actor uses `repeat_primitive` by default
* :ghpull:`552`: Modified Arrow actor to use repeat primitive by default
* :ghpull:`555`: Fixed the rotation matrix in repeat_primitive.
* :ghpull:`569`: Add new example/demo: three-dimensional fractals
* :ghpull:`572`: Fixed the static path in configuration file for docs
* :ghpull:`571`: Fix vertex order in prim_tetrahedron
* :ghpull:`567`: Replace theme in requirements/docs.txt
* :ghpull:`566`: Update Website Footer
* :ghpull:`551`: Fixed #550 : Added necessary alignment between glyph creation and ac…
* :ghpull:`559`: Added simulation for Tesseract
* :ghpull:`556`: Updated code of `viz_network_animated` to use `fury.utils`
* :ghpull:`565`: Minor documentation fixes
* :ghpull:`563`: New website changes
* :ghpull:`564`: Record should not make the window appear
* :ghpull:`557`: Check to see if file exists before opening
* :ghpull:`560`: Force mesa update
* :ghpull:`544`: Improve setuptools
* :ghpull:`542`: Re-enabling nearly all under investigation tests
* :ghpull:`537`: Add OpenGL flags for offscreen rendering

Issues (213):

* :ghissue:`713`: The docs generation fails with pyData theme v0.11.0
* :ghissue:`687`: Record keyframe animation as GIF and MP4
* :ghissue:`782`: Add Codespell and update codecov
* :ghissue:`587`: Billboard tutorial
* :ghissue:`781`: Tab customization
* :ghissue:`779`: versions-corrected
* :ghissue:`741`: Remove unneeded multithreading call
* :ghissue:`776`: TabUI collapsing/expanding improvements
* :ghissue:`778`: TabUI collapsing/expanding improvements
* :ghissue:`777`: Remove alias keyword on documentation
* :ghissue:`770`:  Directions of arrow actor do not change in `repeat_primitive = False` method (VTK)
* :ghissue:`732`: [WIP] integrating latex to fury
* :ghissue:`771`: add one condition in `repeat_primitive` to handle direction [-1, 0, 0], issue #770
* :ghissue:`766`: Cylinder repeat primitive
* :ghissue:`769`: Merge Demo and examples
* :ghissue:`772`: test for peak_slicer() cannot pass
* :ghissue:`767`: Update Peak actor shader
* :ghissue:`82`: GLTF 2.0
* :ghissue:`354`: Some Typos & Grammatical Errors to be fixed in WIKI GSOC 2021
* :ghissue:`677`: Cylindrical billboard implementation
* :ghissue:`765`: add instruction about how to get Suzanne model
* :ghissue:`764`: ComboBox2D drop_down_button mouse callback was inside for loop
* :ghissue:`748`: some fixs and ex addition in docstrings in actor.py
* :ghissue:`754`: update viz_roi_contour.py
* :ghissue:`760`: update deprecated function get.data() to get.fdata()
* :ghissue:`761`: add instruction of how to download suzanne model for getting started page
* :ghissue:`762`: update the deprecated get_data() to get_fdata in viz_roi_contour.py in the demo section.
* :ghissue:`756`: Triangle strips 2 Triangles
* :ghissue:`708`: Strips to triangles
* :ghissue:`747`: Connected the sliders to the right directions
* :ghissue:`745`: Getting error in installation
* :ghissue:`743`: Missing fury.animation
* :ghissue:`709`: Commented the self.initialize
* :ghissue:`744`: Update initialize management
* :ghissue:`710`: Principled update
* :ghissue:`688`: DrawPanel Update: Moving rotation_slider from `DrawShape` to `DrawPanel`
* :ghissue:`734`: Added GSoC'22 Final Report
* :ghissue:`736`: Adding GSoC'22 final report
* :ghissue:`727`: Feature/scientific domains
* :ghissue:`463`: `GridUI` throws error when captions are `None`
* :ghissue:`478`: Resolving GridUI caption error
* :ghissue:`502`: Multithreading support and examples
* :ghissue:`740`: Multithreading example simplified and refactored
* :ghissue:`738`: Download progress bar tries to use the tput command to determine the width of the terminal to adjust the width of the progress bar, however, when run on windows, this leaves an error message
* :ghissue:`739`: added a check for operating system before executing the tput command through popen in fury/data/fetcher.py update_progressbar() function
* :ghissue:`737`: remove object keyword from class
* :ghissue:`726`: Adding GSoC'22 Final Report
* :ghissue:`735`: Add precommit
* :ghissue:`664`: Improve animation module tutorial
* :ghissue:`720`: fix image load flip issue
* :ghissue:`642`: Textures are inverted in the tutorials
* :ghissue:`728`: Fix flipped images in load, save, and snapshot
* :ghissue:`730`: Update CI and add pyproject.toml
* :ghissue:`729`: Fix links in CONTRIBUTING.rst
* :ghissue:`725`: Improve Doc management + quick fix
* :ghissue:`724`: Feature/community page
* :ghissue:`721`: Fix: Color changes on docs pages fixed
* :ghissue:`316`: Build a sphinx theme
* :ghissue:`714`: Earth coordinates tutorial example upsidedown
* :ghissue:`723`: Update CI's
* :ghissue:`722`: Fix failing tests due to last numpy release
* :ghissue:`719`: Logo changes
* :ghissue:`718`: Home page mobile friendly
* :ghissue:`717`: Scientific domains enhancement
* :ghissue:`680`: Updating animation tutorials
* :ghissue:`716`: tensor_slicer function has an issue with sphere argument
* :ghissue:`690`: Add Timelines to ShowManager directly
* :ghissue:`694`: Separating the Timeline into Timeline and Animation
* :ghissue:`603`: UI tests are failing in Ubuntu OS due to a "segmentation error"
* :ghissue:`712`: Fix: segfault created by record method
* :ghissue:`705`: [BUG] Segmentation fault error  caused by Morph Stress Test
* :ghissue:`706`: fix: double render call with timeline obj causes a seg fault
* :ghissue:`435`: Fury/VTK Streaming: webrtc/rtmp
* :ghissue:`704`: seg fault investigation
* :ghissue:`700`: Adding morphing support in `gltf.py`
* :ghissue:`697`: Adding week 14 blog
* :ghissue:`693`: Adding Week 15 Blogpost
* :ghissue:`701`: Updating `fetch_viz_new_icons` to fetch new icons
* :ghissue:`685`: glTF skinning animation implementation
* :ghissue:`699`: Adding Week 16 Blogpost
* :ghissue:`698`: Added blog post for week 14
* :ghissue:`667`: [WIP] Remove initialize call from multiple places
* :ghissue:`689`: GLTF actor colors from material
* :ghissue:`643`: [WIP] Adding ability to load glTF animations
* :ghissue:`665`: Timeline hierarchical transformation and fixing some issues
* :ghissue:`686`: Adding week 13 blog post
* :ghissue:`684`: Adding Week 14 Blogpost
* :ghissue:`692`: Set position and width of the `PlaybackPanel`
* :ghissue:`691`: Added week 13 post
* :ghissue:`683`: Adding Week 13 Blogpost
* :ghissue:`682`: Adding week 12 blog post
* :ghissue:`681`: Added blog post for week 12
* :ghissue:`672`: Adding Week 12 Blogpost
* :ghissue:`678`: DrawPanel Update: Repositioning the `mode_panel` and `mode_text`
* :ghissue:`661`: Improving `vector_text`
* :ghissue:`679`: DrawPanel Update: Moving repetitive functions to helpers
* :ghissue:`674`: DrawPanel Update: Separating tests to test individual features
* :ghissue:`675`: Week 11 blog post
* :ghissue:`673`: DrawPanel Update: Removing `in_progress` parameter while drawing shapes
* :ghissue:`676`: Adding week 11 blog post
* :ghissue:`671`: Adding Week 11 Blogpost
* :ghissue:`623`: DrawPanel Feature: Adding Rotation of shape from Center
* :ghissue:`670`: Adding week 10 blog post
* :ghissue:`666`: Adding Week 10 Blogpost
* :ghissue:`669`: Added blog post for week 10
* :ghissue:`419`: Controlling Fury windows by HTC VIVE
* :ghissue:`647`: Keyframe animations and interpolators
* :ghissue:`620`: Tutorial on making a primitive using polygons and SDF
* :ghissue:`630`: Adding function to export scenes as glTF
* :ghissue:`663`: Adding week 9 blog post
* :ghissue:`656`: Week 8 blog post
* :ghissue:`662`: Week 9 blog post
* :ghissue:`654`: Adding Week 9 Blogpost
* :ghissue:`659`: Adding week 8 blog post
* :ghissue:`650`: Adding Week 8 Blogpost
* :ghissue:`655`: Fix test skybox
* :ghissue:`645`: Fixing `ZeroDivisionError` thrown by UI sliders when the `value_range` is zero (0)
* :ghissue:`657`: Put text next to a roi
* :ghissue:`626`: Keyframe animation with camera support
* :ghissue:`648`: Adding week 7 blog post
* :ghissue:`649`: Added week 7 blog post
* :ghissue:`646`: Adding Week 7 Blogpost
* :ghissue:`641`: Week 6 blog post
* :ghissue:`644`: Adding week 6 blog post
* :ghissue:`638`: Adding Week 6 Blogpost
* :ghissue:`639`: Migrate Windows from Azure to GHA
* :ghissue:`618`: Theme issues when docs compiled with latest sphinx-theme version
* :ghissue:`634`: Prevented calling `on_change` when slider value is set without user intervention
* :ghissue:`637`: Adding week 5 blog post
* :ghissue:`632`: Bugfix: Visibility issues with ListBox2D
* :ghissue:`418`: ListBox2D has resizing issues when added into TabUI
* :ghissue:`610`: Add DPI support for window snapshots
* :ghissue:`612`: [WIP] Implemented a functional prototype of the keyframes animation API
* :ghissue:`613`: [WIP] Added three tutorials to test the animation system and the interpolators
* :ghissue:`633`: Added week 5 blog post
* :ghissue:`617`: Added primitives count to the the Actor's polydata
* :ghissue:`624`: Adding Week 5 BlogPost
* :ghissue:`627`: Adding week 4 blog post
* :ghissue:`625`: Added week 4 blog post
* :ghissue:`600`: Adding support for importing simple glTF files
* :ghissue:`622`: Adding week 3 blog post
* :ghissue:`619`: Week 3 blog post.
* :ghissue:`621`: Adding Week 4 Blogpost
* :ghissue:`616`: Fixing API limits reached issue in gltf fetcher
* :ghissue:`611`: Adding Week 3 BlogPost
* :ghissue:`614`: Added week 2 blog
* :ghissue:`615`: Added blog post for week 2
* :ghissue:`607`: Adding Week 2 Blog Post
* :ghissue:`599`: Creating `DrawPanel` UI
* :ghissue:`606`: Added week 1 post
* :ghissue:`608`: Adding week 1 blog post
* :ghissue:`597`: Added an accurate way to get the FPS for the showManager
* :ghissue:`605`: Adding Week1 Blog Post
* :ghissue:`501`: Creating an `off_focus` hook in `TextBox2D`
* :ghissue:`602`: Added support for fetching gltf samples
* :ghissue:`609`: Creating a fetcher to fetch new icons
* :ghissue:`553`: Refresh code of all tutorials and demos
* :ghissue:`601`: Updating author's name in README
* :ghissue:`593`: Support empty ArraySequence in saving (for empty vtk)
* :ghissue:`598`: Timer id is returned after creating the timer.
* :ghissue:`581`: Keep original dtype for offsets in vtk format
* :ghissue:`588`: Fixed Sphere Creation Error on viz_pbr_interactive Tutorial
* :ghissue:`596`: Segmentation Faults when running Fury demos
* :ghissue:`585`: Double requirement given for Pillow in default.txt
* :ghissue:`595`: changed `use_primitive` to false by default
* :ghissue:`589`: First blog: GSoC
* :ghissue:`525`: Implemented vtkBillboardTextActor
* :ghissue:`586`: Added my first blog post
* :ghissue:`594`: Fixed multi_samples not being used.
* :ghissue:`591`: Fixed some old tutorials.
* :ghissue:`590`: Adding Pre-GSoC Journey Blog Post
* :ghissue:`584`: Changing dot actor
* :ghissue:`582`: Deprecation of the function shaders.load
* :ghissue:`580`: Update website
* :ghissue:`575`: Button and footer changes in docs
* :ghissue:`437`: FURY Streaming System Proposal
* :ghissue:`574`: symmetric parameter for peak
* :ghissue:`561`: Shader API improvements
* :ghissue:`546`: No replacement option for Geometry Shaders
* :ghissue:`533`: Sphere actor uses repeat_primitive by default
* :ghissue:`528`: Sphere actor needs to use repeat_primitives by default
* :ghissue:`577`: Added play/pause buttons
* :ghissue:`443`: Adapt GridLayout to work with UI
* :ghissue:`570`: Function to save screenshots with magnification factor
* :ghissue:`486`: Added `x,y,z` layouts to the layout module.
* :ghissue:`547`: Cone actor uses `repeat_primitive` by default
* :ghissue:`529`: Cone actor needs to use repeat_primitives by default
* :ghissue:`530`: Arrow actor needs to use repeat_primitives by default
* :ghissue:`552`: Modified Arrow actor to use repeat primitive by default
* :ghissue:`545`: Fix some tests in `test_material.py`
* :ghissue:`554`: The rotation done by repeat_primitive function is not working as it should.
* :ghissue:`555`: Fixed the rotation matrix in repeat_primitive.
* :ghissue:`573`: Segmentation Fault
* :ghissue:`569`: Add new example/demo: three-dimensional fractals
* :ghissue:`572`: Fixed the static path in configuration file for docs
* :ghissue:`571`: Fix vertex order in prim_tetrahedron
* :ghissue:`567`: Replace theme in requirements/docs.txt
* :ghissue:`566`: Update Website Footer
* :ghissue:`550`: Cylinder direction not unique.
* :ghissue:`551`: Fixed #550 : Added necessary alignment between glyph creation and ac…
* :ghissue:`541`: Allow offscreen rendering in window.record.
* :ghissue:`548`: Black window on screen on "window.record".
* :ghissue:`559`: Added simulation for Tesseract
* :ghissue:`556`: Updated code of `viz_network_animated` to use `fury.utils`
* :ghissue:`565`: Minor documentation fixes
* :ghissue:`563`: New website changes
* :ghissue:`564`: Record should not make the window appear
* :ghissue:`557`: Check to see if file exists before opening
* :ghissue:`560`: Force mesa update
* :ghissue:`549`: Add time step to brownian animation and velocity components to helica…
* :ghissue:`544`: Improve setuptools
* :ghissue:`542`: Re-enabling nearly all under investigation tests
* :ghissue:`537`: Add OpenGL flags for offscreen rendering
