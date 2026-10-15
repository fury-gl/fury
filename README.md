<h1 align="center">
  <br>
  <a href="https://fury.gl/"><img src="https://raw.githubusercontent.com/fury-gl/fury/refs/heads/master/docs/source/_static/images/logo.svg" alt="FURY" width="200"></a>
  <br>Free Unified Rendering in Python<br>

</h1>

<h4 align="center">A software library for scientific visualization in Python.
</h4>

<p align="center">
<a href="https://github.com/fury-gl/fury/actions/workflows/test.yml"><img src="https://github.com/fury-gl/fury/actions/workflows/test.yml/badge.svg?branch=master" alt="FURY Tests"></a>
<a href="https://pypi.org/project/fury/"><img src="https://img.shields.io/pypi/v/fury.svg" alt="PyPI version"></a>
<a href="https://anaconda.org/conda-forge/fury"><img src="https://anaconda.org/conda-forge/fury/badges/version.svg" alt="Conda-forge version"></a>
<a href="https://doi.org/10.21105/joss.03384"><img src="https://joss.theoj.org/papers/10.21105/joss.03384/status.svg" alt="JOSS paper"></a>

</p>

<p align="center">
  <a href="#general-information">General Information</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How to use</a> •
  <a href="#credits">Credits</a> •
  <a href="#friends--support">Friends &amp; Support</a> •
  <a href="#contribute">Contribute</a> •
  <a href="#citing">Citing</a>
</p>

<div align="center">

<table align="center">
  <tr>
    <th align="center">Interactive 3D Flight Simulator</th>
    <th align="center">Rotating 4D Tesseract</th>
  </tr>
  <tr>
    <td align="center"><a href="https://fury.gl/latest/auto_examples/04_demos/viz_flight_simulator.html"><img src="https://fury.gl/latest/_images/sphx_glr_viz_flight_simulator_001.gif" alt="FURY flight simulator with an aircraft, runway, and procedural landscape" width="420"></a></td>
    <td align="center"><a href="https://fury.gl/latest/auto_examples/04_demos/viz_tesseract.html"><img src="https://fury.gl/latest/_images/sphx_glr_viz_tesseract_001.gif" alt="Animated projection of a rotating four-dimensional tesseract in FURY" width="420"></a></td>
  </tr>
</table>

Click a demo to explore its tutorial and source code.

</div>

# General Information

- **Website and Documentation:** https://fury.gl
- **Tutorials and examples:** https://fury.gl/latest/auto_examples/index.html
- **Blog:**  https://fury.gl/latest/blog.html
- **Mailing list:** https://mail.python.org/mailman3/lists/fury.python.org/
- **Official source code repo:** https://github.com/fury-gl/fury
- **Download releases:** https://pypi.org/project/fury/
- **Issue tracker:** https://github.com/fury-gl/fury/issues
- **Free software:** [3-clause BSD license](LICENSE)
- **Community:** Come to chat on [Discord](https://discord.gg/6btFPPj)

# Key Features

- GPU-accelerated scientific visualization with pygfx and WebGPU.
- NumPy-based 3D actors and interactive volume slices.
- Built-in user interface components.
- Keyframe, camera, and timeline animation.
- GPU-based force-directed network visualization.
- Mesh, point, and line loading powered by [Polyxios](https://polyxios.org/), with an extensive range of 3D mesh and geometry file extensions, including `.obj`, `.ply`, `.stl`, `.off`, `.vtk`, `.vtp`, and `.vtu`. See the [full list of Polyxios-supported formats and extensions](https://polyxios.org/stable/formats/index.html) for format details and requirements.

# Installation

The current development version of FURY requires Python 3.12 or newer.
Python 3.12, 3.13, and 3.14 are
supported stable versions, tested in blocking Linux, macOS, and Windows CI.
Python 3.12 support will be removed in an upcoming FURY release.

## Releases

`pip install fury` or `conda install -c conda-forge fury`

Published releases may have different Python requirements; see the package metadata for the release you install.

For optional Qt, Jupyter, and ImGui window backends:

    pip install "fury[window]"

## Development

### Installation from source

**Step 1.** Get the latest source by cloning this repo:

    git clone https://github.com/fury-gl/fury.git
    cd fury

**Step 2.** Install requirements:

    pip install -r requirements/default.txt

**Step 3.** Install fury

As a [local project installation](https://pip.pypa.io/en/stable/topics/local-project-installs/#regular-installs) using:

    pip install .

Or as an ["editable" installation](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) using:

    pip install -e .

**If you are developing fury you should go with editable installation.**

**Step 4:** Enjoy!

For more information, see the [development installation guide](docs/source/installation.rst).

## Testing

From the repository root (the directory containing `pyproject.toml`), install test suite requirements:

    pip install -r requirements/test.txt

Launch the test suite from the same repository root:

    pytest -svv fury


# How to use

Start with Quick Visualization below.

<div align="center">

| Tutorial | Description |
|:--------:|:-----------:|
| [Quick Visualization](https://fury.gl/latest/auto_examples/01_introductory/viz_show.html) | Create and display a first 3D scene |
| [Force-Directed Network Visualization](https://fury.gl/latest/auto_examples/04_demos/viz_network.html) | Explore a GPU-based network layout |
| [3D Image Slicer](https://fury.gl/latest/auto_examples/04_demos/viz_slicer.html) | Display and interact with volume slices |
| [UI Sliders](https://fury.gl/latest/auto_examples/07_ui/viz_sliders.html) | Add interactive controls |
| [Keyframe Animation](https://fury.gl/latest/auto_examples/10_animation/viz_fury_animation.html) | Animate camera movement and opacity |

</div>

Explore the full [tutorials and examples gallery](https://fury.gl/latest/auto_examples/index.html).
Some examples require optional dependencies or download example data; follow the instructions in each tutorial.

Upgrading from VTK-based FURY 0.12.x? See the [FURY 2 migration guide](https://fury.gl/latest/migration_guide_v2.0.html).


# Credits

Please, go to [contributors page](https://github.com/fury-gl/fury/graphs/contributors) to see who have been involved in the development of FURY.


# Friends & Support

- [DIPY](https://dipy.org/)
- [Polyxios](https://polyxios.org/)
- [Intelligent Systems Engineering, IU](https://luddy.iu.edu/academics/units/ise.html)
- [Google Summer of Code](https://summerofcode.withgoogle.com/)


# Contribute

We love contributions!

You've discovered a bug or something else you want to change - excellent! Create an [issue](https://github.com/fury-gl/fury/issues/new)!

See the [contribution guide](CONTRIBUTING.rst) for development setup and contribution guidelines.

# Citing

If you are using FURY in your work then do cite [this paper](https://doi.org/10.21105/joss.03384). By citing FURY, you are helping sustain the FURY ecosystem.

    Eleftherios Garyfallidis, Serge Koudoro, Javier Guaje, Marc-Alexandre Côté, Soham Biswas,
    David Reagan, Nasim Anousheh, Filipi Silva, Geoffrey Fox, and Fury Contributors.
    "FURY: advanced scientific visualization." Journal of Open Source Software 6, no. 64 (2021): 3384.
    https://doi.org/10.21105/joss.03384


```bibtex
    @article{Garyfallidis2021,
        doi = {10.21105/joss.03384},
        url = {https://doi.org/10.21105/joss.03384},
        year = {2021},
        publisher = {The Open Journal},
        volume = {6},
        number = {64},
        pages = {3384},
        author = {Eleftherios Garyfallidis and Serge Koudoro and Javier Guaje and Marc-Alexandre Côté and Soham Biswas and David Reagan and Nasim Anousheh and Filipi Silva and Geoffrey Fox and Fury Contributors},
        title = {FURY: advanced scientific visualization},
        journal = {Journal of Open Source Software}
    }
```
