.. _releasev0.11.0:

==============================
 Release notes v0.11.0
==============================

Quick Overview
--------------

* New SPEC: Keyword-only adopted.
* New SPEC: Lazy loading adopted.
* New standard for coding style.
* Documentation updated.
* Website updated.

Details
-------

GitHub stats for 2024/02/27 - 2024/07/31 (tag: v0.10.0)

These lists are automatically generated, and may be incomplete or contain duplicates.

The following 7 authors contributed 70 commits.

* Ishan Kamboj
* Jon Haitz Legarreta Gorroño
* Kaustav Deka
* Robin Roy
* Serge Koudoro
* Wachiou BOURAÏMA
* dependabot[bot]


We closed a total of 83 issues, 35 pull requests and 48 regular issues;
this is the full list (generated with the script
:file:`tools/github_stats.py`):

Pull Requests (35):

* :ghpull:`919`: NF: Implementation of lazy_loading in `fury.stream.server`
* :ghpull:`918`: DOCS: simplify  FURY import
* :ghpull:`916`: DOCS: simplify import
* :ghpull:`915`: [DOCS][FIX]: Update variable descriptions in visualization examples
* :ghpull:`907`: NF: Add lazy_loader feature  in FURY
* :ghpull:`914`: DOC: GSoC Blogs Week 6, 7, 8
* :ghpull:`911`: [DOC][FIX] fix typos in blog posts
* :ghpull:`913`: Fix: Replace setuptools.extern.packaging with direct packaging import
* :ghpull:`909`: RF: add keyword arguments decorator (warn_on_args_to_kwargs) in the module: stream
* :ghpull:`902`: RF: Add keyword arguments decorator to module: UI
* :ghpull:`899`: RF: Add keyword arguments to module: animation
* :ghpull:`901`: RF: Add keyword arguments decorator to module: shaders
* :ghpull:`900`: RF: Add keyword arguments to module: data
* :ghpull:`898`: RF: Add keyword arguments to module: actors
* :ghpull:`888`: NF: Add keyword_only decorator to enforce keyword-only arguments
* :ghpull:`908`: DOC: add Wachiou's week5 Blog post
* :ghpull:`906`: [DOC] GSoC week 4 and 5
* :ghpull:`905`: DOC: My weeks 3 and 4 blog post
* :ghpull:`903`: CI: remove 3.8 add 3.12
* :ghpull:`896`: DOC: GSoC Week 2 & 3
* :ghpull:`897`: DOC:  Add wachiou's week 2 blog post
* :ghpull:`892`: [DOC] week 1 blog GSoC
* :ghpull:`894`: DOC: Wachiou Week 1 Blog Post
* :ghpull:`893`: fix: gitignore
* :ghpull:`889`: DOC: Add Wachiou BOURAIMA GSoC'24 first  Blog post
* :ghpull:`890`: [DOC] GSoC Blog: Robin Roy (Community Bonding)
* :ghpull:`891`: [TYPO] Typo fix in code
* :ghpull:`886`: DOC: Document the coding style enforcement framework
* :ghpull:`881`: STYLE: Format code using `ruff`
* :ghpull:`884`: build(deps): bump pre-commit/action from 3.0.0 to 3.0.1 in the actions group
* :ghpull:`885`: Fix pycodestyle stream
* :ghpull:`877`: Fixed Pycodestyle errors in fury/actors/odf_slicer.py, fury/actors/peak.py, fury/actors/tensor.py, and fury/data/fetcher.py
* :ghpull:`855`: Fix #780 : Added top/bottom for Tabs Bar in Tab Panel UI
* :ghpull:`879`: STYLE: Transition to `ruff` to enforce import statement sorting
* :ghpull:`868`: Added Dark mode, Fixed Search Bar for documentation site

Issues (48):

* :ghissue:`917`: `fury.stream.server` is missing lazy_loading
* :ghissue:`919`: NF: Implementation of lazy_loading in `fury.stream.server`
* :ghissue:`918`: DOCS: simplify  FURY import
* :ghissue:`916`: DOCS: simplify import
* :ghissue:`915`: [DOCS][FIX]: Update variable descriptions in visualization examples
* :ghissue:`907`: NF: Add lazy_loader feature  in FURY
* :ghissue:`914`: DOC: GSoC Blogs Week 6, 7, 8
* :ghissue:`911`: [DOC][FIX] fix typos in blog posts
* :ghissue:`912`: Deprecator bug with setuptools >=71.0.3
* :ghissue:`913`: Fix: Replace setuptools.extern.packaging with direct packaging import
* :ghissue:`910`: ENH: Add a GHA workflow to build docs
* :ghissue:`909`: RF: add keyword arguments decorator (warn_on_args_to_kwargs) in the module: stream
* :ghissue:`902`: RF: Add keyword arguments decorator to module: UI
* :ghissue:`899`: RF: Add keyword arguments to module: animation
* :ghissue:`901`: RF: Add keyword arguments decorator to module: shaders
* :ghissue:`900`: RF: Add keyword arguments to module: data
* :ghissue:`898`: RF: Add keyword arguments to module: actors
* :ghissue:`888`: NF: Add keyword_only decorator to enforce keyword-only arguments
* :ghissue:`908`: DOC: add Wachiou's week5 Blog post
* :ghissue:`906`: [DOC] GSoC week 4 and 5
* :ghissue:`905`: DOC: My weeks 3 and 4 blog post
* :ghissue:`903`: CI: remove 3.8 add 3.12
* :ghissue:`896`: DOC: GSoC Week 2 & 3
* :ghissue:`897`: DOC:  Add wachiou's week 2 blog post
* :ghissue:`895`: DOC: Add My week 2 blog post
* :ghissue:`892`: [DOC] week 1 blog GSoC
* :ghissue:`894`: DOC: Wachiou Week 1 Blog Post
* :ghissue:`893`: fix: gitignore
* :ghissue:`889`: DOC: Add Wachiou BOURAIMA GSoC'24 first  Blog post
* :ghissue:`890`: [DOC] GSoC Blog: Robin Roy (Community Bonding)
* :ghissue:`871`: `actor.texture` hides all other actors from the scene
* :ghissue:`891`: [TYPO] Typo fix in code
* :ghissue:`887`: NF: Add keyword_only decorator to enforce keyword-only arguments
* :ghissue:`886`: DOC: Document the coding style enforcement framework
* :ghissue:`881`: STYLE: Format code using `ruff`
* :ghissue:`884`: build(deps): bump pre-commit/action from 3.0.0 to 3.0.1 in the actions group
* :ghissue:`883`: Wassiu contributions
* :ghissue:`885`: Fix pycodestyle stream
* :ghissue:`880`: improved readability in fetcher.py
* :ghissue:`882`: Wassiu contributions
* :ghissue:`876`:   Pycodestyle errors in fury/actors/odf_slicer.py, fury/actors/peak.py, fury/actors/tensor.py, and fury/data/fetcher.py
* :ghissue:`877`: Fixed Pycodestyle errors in fury/actors/odf_slicer.py, fury/actors/peak.py, fury/actors/tensor.py, and fury/data/fetcher.py
* :ghissue:`878`: Docs: Remove '$' from fury/docs/README.md commands
* :ghissue:`780`: Tabs bar positioning in TabUI
* :ghissue:`855`: Fix #780 : Added top/bottom for Tabs Bar in Tab Panel UI
* :ghissue:`879`: STYLE: Transition to `ruff` to enforce import statement sorting
* :ghissue:`868`: Added Dark mode, Fixed Search Bar for documentation site
* :ghissue:`867`: Added Dark mode, Fixed Search Bar for documentation site
