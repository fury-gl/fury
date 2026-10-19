.. _releasev2.1.0:

==============================
 Release notes v2.1.0
==============================

Quick Overview
--------------

* Added physically based rendering (PBR) and environment maps.
* Added billboard ellipsoids, ellipsoid and tensor slicers, and billboard picking information.
* Ported SpinBox to v2 with integer and floating-point input support.
* Added shader-based rounded rectangles.
* Fixed the documentation version switcher and refreshed installation guidance.
* Raised the minimum supported Python version to 3.12.
* Added seam free sphere texture mapping.

Details
--------

GitHub stats for 2026/08/04 - 2026/10/19 (tag: v2.0.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 9 authors contributed 76 commits.

* Anshu kashyap
* Eleftherios Garyfallidis
* Maharshi Gor
* Medha Bhardwaj
* Mohit Kumar
* Praneeth Shetty
* Rishika Sharma
* Sampat
* Serge Koudoro


We closed a total of 37 issues, 30 pull requests and 7 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (30):

* :ghissue:`1370`: NF: Add seam-safe texture_on_sphere actor
* :ghissue:`1372`: Update pre-commit hooks
* :ghissue:`1369`: Version switcher fix
* :ghissue:`1346`: test: add bounds and center validation for superquadric primitive
* :ghissue:`1341`: NF: Port SpinBox to v2
* :ghissue:`1365`: NF: Tensor slicer added
* :ghissue:`1368`: RF: Update readme to more compliant with v2
* :ghissue:`1366`: MNT: Require Python 3.12 and update interpreter coverage
* :ghissue:`1367`: DOC: Refresh installation and Miniforge contributor guidance
* :ghissue:`1364`: NF: Ellipsoid Slicer Introduced
* :ghissue:`1357`: NF: Billboard ellipsoid introduced
* :ghissue:`1344`: Updated text in window.py
* :ghissue:`1361`: Update pre-commit hooks
* :ghissue:`1353`:  I added strict=True to fix the problem B905 warning
* :ghissue:`1342`: docs: replace Furious Atoms with Polyxios and list friends in README
* :ghissue:`1354`: NF: Pick info provided for billboard based actors
* :ghissue:`1349`: Update pre-commit hooks
* :ghissue:`1343`: Update pre-commit hooks
* :ghissue:`1335`: Update pre-commit hooks
* :ghissue:`1334`: Update pre-commit hooks
* :ghissue:`1329`: Update pre-commit hooks
* :ghissue:`1327`: MNT: Remove obsolete pre-v2 test recordings and count files
* :ghissue:`1326`: PBR support
* :ghissue:`1324`: Update pre-commit hooks
* :ghissue:`1323`: Update pre-commit hooks
* :ghissue:`1320`: Week 10: Finalizing Website Revamp and Performance Optimization
* :ghissue:`1319`: Week 9: Improving Tree UI from Review Feedback and Migration Guide
* :ghissue:`1318`: Week 8: Implementing Rounded Rectangle with Fragment Shaders
* :ghissue:`1321`: Update pre-commit hooks
* :ghissue:`1294`: Adding Rounded Rectangle

Issues (7):

* :ghissue:`1114`: Port `SpinBox` Component to v2
* :ghissue:`1348`: Fix B905 Ruff warnings (missing strict parameter in zip) in data/fetcher.py
* :ghissue:`1355`: save_image rejects documented single-channel arrays
* :ghissue:`1350`: Non-square image textures swap width and height
* :ghissue:`1269`: Docs website revamp todo tracker
* :ghissue:`1347`: Fix B905 Ruff warnings (missing strict parameter in zip) in data/fetcher.py
* :ghissue:`1273`: FURY v2 release readiness
