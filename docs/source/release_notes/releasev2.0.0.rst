.. _releasev2.0.0:

==============================
 Release notes v2.0.0
==============================

Quick Overview
--------------

* First stable release of FURY built on `pygfx`/`wgpu`, replacing the legacy VTK-based rendering pipeline.
* Added a :doc:`migration guide </migration_guide_v2.0>` to help users transition from FURY 0.12.x to v2.0.0.
* Completed the UI framework port to the v2 architecture: ``TextBox2D``, ``ComboBox2D``, ``ListBox2D``, ``Card2D``, ``TabUI``, ``ImageContainer2D``, ``RangeSlider``, ``LineDoubleSlider2D``, ``RingSlider2D``, ``PlaybackPanel``, and ``Radio Button/Checkbox`` components.
* Ported the Animation ``Timeline and Camera Animation`` systems from VTK to `pygfx`.
* Added new actors and features, including Network Visualization, per-instance geometry parameters for actors, and a chunked Vector Field actor for large datasets.
* Revamped the FURY website with a new homepage, light/dark theming, community page, and improved documentation footer.
* Added new demos and examples: Flight Simulator, Solar System Animation, and an interactive science domain showcase.
* Numerous bug fixes and stability improvements across actors, UI event handling, and the documentation build pipeline.

Details
--------

GitHub stats for 2026/04/03 - 2026/08/03 (tag: v2.0.0a7)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 7 authors contributed 254 commits.

* Aditya Gupta
* JigyasuRajput
* Maharshi Gor
* Medha Bhardwaj
* Praneeth Shetty
* Serge Koudoro
* williamcancodee


We closed a total of 123 issues, 99 pull requests and 24 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (99):

* :ghissue:`1302`: Add migration guide for FURY 0.12.x to v2.0.0
* :ghissue:`1316`: Update pre-commit hooks
* :ghissue:`1314`: Seo optimization cherrypick
* :ghissue:`1315`: BF: backreference warning fix
* :ghissue:`1312`: Revamp FURY website
* :ghissue:`1313`: RF: Update the code to set unset env variables without needing to set…
* :ghissue:`1310`: BF: macOS glfw warning fix
* :ghissue:`1305`: NF: Porting Radio button and check box.
* :ghissue:`1309`: Update pre-commit hooks
* :ghissue:`1308`: Bump actions/setup-python from 6 to 7 in the actions group
* :ghissue:`1306`: BF: Fix phantom sphere in the case of imposter (billboard) Sphere.
* :ghissue:`1299`: Week 7: Rounded Rectangle Research and Website Enhancements
* :ghissue:`1304`: Update pre-commit hooks
* :ghissue:`1297`: Demo: Adding Flight Simulator
* :ghissue:`1303`: UI: Combobox expands on `TextBlock` clicks
* :ghissue:`1296`: Week 6: Combobox UI Component and Global Theme Integration
* :ghissue:`1298`: Update pre-commit hooks
* :ghissue:`1295`: Aligns FURY text in line with the logo
* :ghissue:`1259`: Feat: Adding UI Event Recorder
* :ghissue:`1286`: BF: Dev doc builds were not generating the examples.
* :ghissue:`1293`: Community page (both light and dark mode)
* :ghissue:`1290`: MNT: Added analyze-snapshot test cases.
* :ghissue:`1289`: NF: Polyxios Integration.
* :ghissue:`1292`: Week 5: Advancing the FURY Homepage Revamp
* :ghissue:`1287`: UI/normalize uniform color
* :ghissue:`1291`: Update pre-commit hooks
* :ghissue:`1288`: Bump actions/cache from 5 to 6 in the actions group
* :ghissue:`1284`: Adds `FURY` next to logo in homepage in navbar
* :ghissue:`1282`: Override Default CSS with Custom Styles to Update Doc Page Colors
* :ghissue:`1271`: Introduces Combobox
* :ghissue:`1280`: RF: Updated the package name.
* :ghissue:`1285`: BF: Fixed chipping of text.
* :ghissue:`1281`: Implements global light mode theme for internal pages (same as homepage)
* :ghissue:`1279`: Week 4: UI Components and Website Homepage Revamp
* :ghissue:`1272`: NF: Record functionality properly added.
* :ghissue:`1278`: Implements global dark mode theme for internal pages (same as homepage)
* :ghissue:`1274`: Bump actions/checkout from 6 to 7 in the actions group
* :ghissue:`1277`: Update pre-commit hooks
* :ghissue:`1276`: Extracts shared navbar/footer styles into fury_theme.css and scope homepage.css to home.html
* :ghissue:`1275`: Migrate images to PNG and update logo card styling
* :ghissue:`1267`: refactor: enhance homepage responsiveness and refine mobile UI components
* :ghissue:`1266`: Implement and integrate custom documentation footer
* :ghissue:`1265`: Adds community and support section to homepage
* :ghissue:`1270`: RF: Title available in show method.
* :ghissue:`1264`: Adds examples gallery and installation section to the homepage
* :ghissue:`1263`: RF: Animation Timeline 3rd part.
* :ghissue:`1268`: docs: add new examples to valid list
* :ghissue:`1262`: Add interactive science domain showcase section to homepage (light and dark mode)
* :ghissue:`1260`: Adds interactive features to homepage (both dark and light mode)
* :ghissue:`1258`: Improve copy-to-clipboard functionality with improved UI feedback and minor refactor
* :ghissue:`1257`: Week 3: Expanding UI Components and Automated GIF Recording
* :ghissue:`1261`: CI: skip deploy and cron jobs on forks
* :ghissue:`1223`: Feat: Adding Network Visualization
* :ghissue:`1134`: Ports `Textbox2d` to v2 architecture
* :ghissue:`1255`: Update pre-commit hooks
* :ghissue:`1256`: Implement dark mode for the homepage on top of light mode
* :ghissue:`1254`: Implement custom homepage layout with hero section and navigation overrides (light mode)
* :ghissue:`1241`: Introduces `Rangeslider` and `LineDoubleSlider2D`
* :ghissue:`1248`: Introduces ListBox2D
* :ghissue:`1243`: Add automated GIF recording support for animation examples
* :ghissue:`1250`: Introduces `Card 2d`
* :ghissue:`1253`: Remove legacy custom doc UI and restore default PyData Sphinx theme
* :ghissue:`1228`: Feat: Adding Demos and Examples
* :ghissue:`1251`: Week 2: Fixing Doc Build Warnings and Improving Documentation Pipeline
* :ghissue:`1252`: RF: Camera Animation Port
* :ghissue:`1249`: Update pre-commit hooks
* :ghissue:`1246`: RF: Initial Animation Port from vtk to pygfx.
* :ghissue:`1245`: BF: Fixed missing post and release notes.
* :ghissue:`1236`: Tab UI Introduced
* :ghissue:`1239`: ImageContainer2D Introduced
* :ghissue:`1227`: Feat: Adding Solar System Animation Example
* :ghissue:`1244`: Enables strict documentation builds with warnings treated as errors
* :ghissue:`1233`: Integrate Version Switcher, Add Local Docs Server and Fix Version Redirection
* :ghissue:`1242`: Week 1: Fixing Doc Build Warnings and Introducing RingSlider
* :ghissue:`1240`: Update pre-commit hooks
* :ghissue:`1235`: Week 0 Blog
* :ghissue:`1237`: Fix GL01 and address SS06 warnings (after removing as exceptions from toml)
* :ghissue:`1234`: Update pre-commit hooks
* :ghissue:`1230`: MTN: Update the master with 0.12.0 release notes
* :ghissue:`1232`: Fix: `fetch_viz_icons` for icomoon icons
* :ghissue:`1225`: Fix: Block UI Events to not propagate to 3D Controllers
* :ghissue:`1164`: Fix documentation build warnings
* :ghissue:`1229`: MTN: Cleanup for the v2 branch.
* :ghissue:`1121`: Ports RingSlider2D to v2 architecture
* :ghissue:`1226`: Update pre-commit hooks
* :ghissue:`1142`: UI: Adding PlaybackPanel UI
* :ghissue:`1171`: ENH: Add normalize_colors() utility for standardized color input (#1089)
* :ghissue:`1170`: ENH: Support per-instance geometry parameters for actors
* :ghissue:`1220`: Update pre-commit hooks
* :ghissue:`1156`: Fix: Panel2D crashes when used without borders (#1155)
* :ghissue:`1101`:  Allow create_text to accept multiple texts and materials fixes #1012
* :ghissue:`1215`: NF: Vector Field Actor chunked based on device limits.
* :ghissue:`1217`: Update pre-commit hooks
* :ghissue:`1219`: Bump conda-incubator/setup-miniconda from 3 to 4 in the actions group
* :ghissue:`1213`: Update pre-commit hooks
* :ghissue:`1212`: Bump peter-evans/create-pull-request from 8.1.0 to 8.1.1 in the actions group
* :ghissue:`1211`: Update pre-commit hooks
* :ghissue:`1181`: fix(docs): Implement exponential backoff for GitHub API to resolve HTTP 429/403 crashes
* :ghissue:`1208`: REL: release prep for 2.0.0a7

Issues (24):

* :ghissue:`1122`: Re-implement UI Event Recorder and Counter
* :ghissue:`1173`: BUG: fury.testing imports distutils.version.LooseVersion, breaking tests on Python 3.12+
* :ghissue:`1188`: Documentation build issues and missing dependencies
* :ghissue:`1247`: Animation module need update
* :ghissue:`1140`: [BUG] load_image() returns wrong shape for 16-bit grayscale PNG
* :ghissue:`1110`: Port `ComboBox2D` Component to v2
* :ghissue:`1084`: VTK version constraint prevents installation on fresh environments
* :ghissue:`1139`: BLD: Sphinx doc build crashes on Windows due to missing encoding in apigen.py
* :ghissue:`1109`: Port `TextBox2D` Component to v2
* :ghissue:`1115`: Port `LineDoubleSlider2D` Component to v2
* :ghissue:`1117`: Port `RangeSlider` Component to v2
* :ghissue:`1111`: Port `ListBox2D` Component to v2
* :ghissue:`1113`: Port `Card2D` Component to v2
* :ghissue:`1107`: Port `TabUI` Component to v2
* :ghissue:`1108`: Port `ImageContainer2D` Component to v2
* :ghissue:`1116`: Port `RingSlider2D` Component to v2
* :ghissue:`1221`: `fetch_viz_icons()`: `icomoon.tar.gz` cannot be downloaded from UW server
* :ghissue:`1224`: UI Interactions should not interfere with 3D Scenes
* :ghissue:`1089`: Standardize Color usage throughout Codebase
* :ghissue:`1091`: Update actors to accept individual or ndarray for actor features
* :ghissue:`1222`: CI: Authentication failed for `https://github.com/dipy/docs.dipy.org.git/`
* :ghissue:`1155`: Panel2D Does not work without borders
* :ghissue:`1012`: Provide Multiple text into one actor.
* :ghissue:`924`: Doc Generation Error - HTTP Error 429
