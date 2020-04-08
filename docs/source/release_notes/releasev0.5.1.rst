.. _releasev0.5.1:

=========================================
 Release notes v0.5.1 (2020-04-01)
=========================================

Quick Overview
--------------

* Remove python 2 compatibility
* Added texture management
* Added multiples primitives.
* Added multiples actors (contour_from_label, billboard...)
* Huge improvement of multiple UI (RangeSlider, ...)
* Improved security (from md5 to sha256)
* Large documentation update, examples and tutorials
* Increased tests coverage and code quality

Details
-------
GitHub stats for 2019/10/29 - 2020/04/02 (tag: v0.4.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 20 authors contributed 407 commits.

* ChenCheng0630
* Devanshu Modi
* Eleftherios Garyfallidis
* Etienne St-Onge
* Filipi Nascimento Silva
* Gottipati Gautam
* Javier Guaje
* Jon Haitz Legarreta Gorroño
* Liam Donohue
* Marc-Alexandre Côté
* Marssis
* Naman Bansal
* Nasim
* Saransh Jain
* Serge Koudoro
* Shreyas Bhujbal
* Soham Biswas
* Vivek Choudhary
* ibrahimAnis
* lenixlobo


We closed a total of 153 issues, 49 pull requests and 104 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (49):

* :ghpull:`227`: [Fix] update streamlines default color
* :ghpull:`210`: Added contour_from_label method
* :ghpull:`225`: update tutorial folder structure
* :ghpull:`223`: [Fix] sphere winding issue
* :ghpull:`218`: Changed options attribute from list to dict and updated respective tests
* :ghpull:`220`: bumping scipy version to 1.2.0
* :ghpull:`213`: Utils vtk
* :ghpull:`215`: Remove more than one actors at once
* :ghpull:`207`: updated fetcher
* :ghpull:`206`: [FIX] avoid in-place replacements
* :ghpull:`203`: Namanb009 windowtitlefix
* :ghpull:`204`: Vertical Layout for RangeSlider
* :ghpull:`190`: Add initial state to checkbox
* :ghpull:`201`: [FIX] icons flipping
* :ghpull:`181`: Vertical Layout for LineDoubleSlider2D
* :ghpull:`198`: Utils test and winding order algorithm
* :ghpull:`192`: Tetrahedron, Icosahedron primitives
* :ghpull:`189`: Added dynamic text positioning
* :ghpull:`194`: [FIX] Update superquadrics test
* :ghpull:`182`: [Doc] Reshape the documentation
* :ghpull:`177`: [Fix] Flipping during save
* :ghpull:`191`: DOC: Fix `actor.line` parameter type and add `optional` keyword
* :ghpull:`173`: Fixing Text Overflow of ListBox2D
* :ghpull:`167`: Animated Network Visualization Example
* :ghpull:`165`: Vertical Layout for LineSlider2D
* :ghpull:`154`: Added Shader tutorial
* :ghpull:`153`: Sep viz ui
* :ghpull:`132`: Add Billboard actor
* :ghpull:`164`: Documentation
* :ghpull:`163`: Spelling error
* :ghpull:`157`: Corrected Disk2D comments
* :ghpull:`148`: Replace md5 by sha 256
* :ghpull:`145`: DOC: Fix `io:load_image` and `io:save_image` docstrings
* :ghpull:`144`: STYLE: Change examples `README` file extension to reStructuredText
* :ghpull:`143`: STYLE: Improve the requirements' files' style.
* :ghpull:`139`: [Fix] some docstring for doc generation
* :ghpull:`140`: [DOC] Add demo for showing an network
* :ghpull:`136`: Started new tutorial about using normals to make spiky spheres
* :ghpull:`134`: Add event parameter on add_window_callback method in ShowManager class.
* :ghpull:`129`: update loading and saving IO for polydata
* :ghpull:`131`: Add Superquadric primitives and actors
* :ghpull:`130`: Adding Sphere primitives
* :ghpull:`128`: Update Deprecated function
* :ghpull:`126`: Add basic primitives
* :ghpull:`125`: Add Deprecated decorator
* :ghpull:`124`: Texture utilities and actors
* :ghpull:`118`: Remove python2 compatibility
* :ghpull:`120`: Replace pickle with JSON for "events_counts" dict serialization
* :ghpull:`115`: Release 0.4.0 preparation

Issues (104):

* :ghissue:`150`: Re-compute Bounds in Slicer
* :ghissue:`227`: [Fix] update streamlines default color
* :ghissue:`135`: Backward compatibilities problem with streamtube
* :ghissue:`77`: contour_from_label
* :ghissue:`210`: Added contour_from_label method
* :ghissue:`225`: update tutorial folder structure
* :ghissue:`223`: [Fix] sphere winding issue
* :ghissue:`137`: Issues with provided spheres
* :ghissue:`152`: Improve checkbox options cases
* :ghissue:`218`: Changed options attribute from list to dict and updated respective tests
* :ghissue:`76`: Improve Checkbox options access
* :ghissue:`219`: Issue occur when I Start testing the project
* :ghissue:`220`: bumping scipy version to 1.2.0
* :ghissue:`217`: Transformed options attribute from list to dict and updated respective tests
* :ghissue:`213`: Utils vtk
* :ghissue:`179`: Utility functions are needed for getting numpy arrays from actors
* :ghissue:`212`: Namanb009 issue 133 fix
* :ghissue:`214`: Namanb009 Remove mulitple actors
* :ghissue:`215`: Remove more than one actors at once
* :ghissue:`211`: Namanb009 hexadecimal color support
* :ghissue:`187`: New utility functions are added in utils.py and tests are added in te…
* :ghissue:`209`: Namanb009 viz_ui.py does not show render window when run
* :ghissue:`207`: updated fetcher
* :ghissue:`206`: [FIX] avoid in-place replacements
* :ghissue:`203`: Namanb009 windowtitlefix
* :ghissue:`202`: Window Title name does not change
* :ghissue:`204`: Vertical Layout for RangeSlider
* :ghissue:`190`: Add initial state to checkbox
* :ghissue:`75`: Improve Checkbox initialisation
* :ghissue:`201`: [FIX] icons flipping
* :ghissue:`199`: Loading of Inverted icons using read_viz_icons
* :ghissue:`181`: Vertical Layout for LineDoubleSlider2D
* :ghissue:`175`: LineDoubleSlider2D vertical layout
* :ghissue:`198`: Utils test and winding order algorithm
* :ghissue:`192`: Tetrahedron, Icosahedron primitives
* :ghissue:`189`: Added dynamic text positioning
* :ghissue:`176`: Allowing to change text position on Sliders
* :ghissue:`185`: NF: winding order in utils
* :ghissue:`170`: NF: adding primitive stars, 3D stars, rhombi.
* :ghissue:`195`: Added dynamic text position on sliders
* :ghissue:`194`: [FIX] Update superquadrics test
* :ghissue:`171`: bug-in-image 0.1
* :ghissue:`182`: [Doc] Reshape the documentation
* :ghissue:`156`: Test Case File Updated
* :ghissue:`155`: There are libraries we have to install not mentioned in the requirement.txt file to run the test case.
* :ghissue:`122`: Documentation not being rendered correctly
* :ghissue:`177`: [Fix] Flipping during save
* :ghissue:`160`: Saved Images are vertically Inverted
* :ghissue:`193`: Merge pull request #2 from fury-gl/master
* :ghissue:`191`: DOC: Fix `actor.line` parameter type and add `optional` keyword
* :ghissue:`178`: changed text position
* :ghissue:`188`: Added dynamic text positioning
* :ghissue:`173`: Fixing Text Overflow of ListBox2D
* :ghissue:`15`: viz.ui.ListBoxItem2D text overflow
* :ghissue:`166`: Build Native File Dialogs
* :ghissue:`180`: Native File Dialog Text Overflow Issue
* :ghissue:`186`: add name
* :ghissue:`184`: Added winding order algorithm to utils
* :ghissue:`183`: Added star2D and 3D, rhombicuboctahedron to tests_primitive
* :ghissue:`54`: generating directed arrows
* :ghissue:`174`: List box text overflow
* :ghissue:`167`: Animated Network Visualization Example
* :ghissue:`165`: Vertical Layout for LineSlider2D
* :ghissue:`108`: Slider vertical layout
* :ghissue:`172`: window.show() is giving Attribute error.
* :ghissue:`154`: Added Shader tutorial
* :ghissue:`151`: Prim shapes
* :ghissue:`162`: Winding order 2
* :ghissue:`168`: Prim test
* :ghissue:`158`: nose is missing
* :ghissue:`71`: viz_ui.py example needs expansion
* :ghissue:`153`: Sep viz ui
* :ghissue:`132`: Add Billboard actor
* :ghissue:`164`: Documentation
* :ghissue:`163`: Spelling error
* :ghissue:`161`: Merge pull request #1 from fury-gl/master
* :ghissue:`157`: Corrected Disk2D comments
* :ghissue:`121`: Replace md5 by sha2 or sha3 for security issue
* :ghissue:`148`: Replace md5 by sha 256
* :ghissue:`147`: update md5 to sha256
* :ghissue:`146`: Shapes
* :ghissue:`145`: DOC: Fix `io:load_image` and `io:save_image` docstrings
* :ghissue:`144`: STYLE: Change examples `README` file extension to reStructuredText
* :ghissue:`142`: STYLE: Change examples `README` file extension to markdown
* :ghissue:`143`: STYLE: Improve the requirements' files' style.
* :ghissue:`139`: [Fix] some docstring for doc generation
* :ghissue:`140`: [DOC] Add demo for showing an network
* :ghissue:`136`: Started new tutorial about using normals to make spiky spheres
* :ghissue:`134`: Add event parameter on add_window_callback method in ShowManager class.
* :ghissue:`81`: Add superquadric function in actor.py
* :ghissue:`129`: update loading and saving IO for polydata
* :ghissue:`131`: Add Superquadric primitives and actors
* :ghissue:`130`: Adding Sphere primitives
* :ghissue:`128`: Update Deprecated function
* :ghissue:`126`: Add basic primitives
* :ghissue:`125`: Add Deprecated decorator
* :ghissue:`124`: Texture utilities and actors
* :ghissue:`99`: [WIP] Adding util to get Numpy 3D array of RGBA values
* :ghissue:`118`: Remove python2 compatibility
* :ghissue:`117`: Remove compatibility with python 2
* :ghissue:`123`: WIP: Texture support
* :ghissue:`119`: Improve data Serialization
* :ghissue:`120`: Replace pickle with JSON for "events_counts" dict serialization
* :ghissue:`115`: Release 0.4.0 preparation
