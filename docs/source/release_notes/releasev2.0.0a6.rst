.. _releasev2.0.0a6:

==============================
 Release notes v2.0.0a6
==============================

Quick Overview
--------------

- Gizmo (Axis Helper) Introduction
- Billboard Sphere Actor (Impostor)
- Streamtubes GPU bug fixes
- Adding TexturedButton2D and TextButton2D
- Adding LineSlider2D
- API to read back from GPU
- Updated to pygfx 0.16.0


Details
--------

GitHub stats for 2026/01/23 - 2026/03/06 (tag: v2.0.0a5)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 8 authors contributed 65 commits.

* Aditya Gupta
* Faris Abouagour
* Maharshi Gor
* Mohamed Agour
* Pedamallu Umesh Gupta
* Praneeth Shetty
* Serge Koudoro
* Your Name


We closed a total of 26 issues, 18 pull requests and 8 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (18):

* :ghissue:`1146`: Fix: Billboard bounding box not accounting for visual size
* :ghissue:`1147`: BF/MTN: Bounding box for gpu streamtube actor
* :ghissue:`1141`: NF: Gizmo Introduction.
* :ghissue:`1120`: Fix: Opacity/alpha transparency not applied in primitive actors
* :ghissue:`1125`: Fix: remove redundant `local.position` offset in `actor_from_primitive` (multiple centers)
* :ghissue:`1056`: UI: Adding TexturedButton2D and TextButton2D
* :ghissue:`1118`: UI: Adding LineSlider2D
* :ghissue:`1103`: fix: correct typo in boys2rgb - z4 should be z2*z2 not z*z2 #857
* :ghissue:`1087`: Stremtubes GPU bug fix for issue #1067
* :ghissue:`1037`: Sphere billboard (Impostor)
* :ghissue:`1097`: Fix: handle None and tuple colors in point actor(#1092)
* :ghissue:`1086`: Fix progress bar to update in-place on Windows terminals
* :ghissue:`1098`: NF: API to read back from gpu
* :ghissue:`1094`: DOC: Fix broken links and typos in Contributing guide and README
* :ghissue:`1079`: RF: Size based splitting of streamtube actor.
* :ghissue:`1052`: UI: Adding TextBlock2D
* :ghissue:`1076`: REL: Release Preparation for 2.0.0a5.
* :ghissue:`1075`: BF: Slicer flickering in the nearest interpolation.

Issues (8):

* :ghissue:`1145`: Billboard actors have incorrect initial camera framing (excessive zoom)
* :ghissue:`1119`: Opacity/alpha transparency not applied in primitive actors
* :ghissue:`1124`: `actor_from_primitive` applies double position offset when using multiple centers
* :ghissue:`1067`: GPU streamtubes does not provide colors in the geometery.
* :ghissue:`857`: Wrong boys2rgb colormap
* :ghissue:`1092`: Point actor fails if the colors are not provided.
* :ghissue:`1042`: Download during `fetch_viz_models` is too verbose
* :ghissue:`1078`: Buffers run out if the size of streamtubes/streamlines is huge.
