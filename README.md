<h1 align="center">
  <br>
  <a href="https://www.fury.gl"><img src="https://raw.githubusercontent.com/fury-gl/fury-communication-assets/main/fury-logo.png" alt="FURY" width="200"></a>
  <br>Free Unified Rendering in Python<br>

</h1>

<h4 align="center">A software library for scientific visualization in Python.
</h4>

<p align="center">
<a href="https://dev.azure.com/fury-gl/fury/_build/latest?definitionId=1&branchName=master"><img src="https://dev.azure.com/fury-gl/fury/_apis/build/status/fury-gl.fury?branchName=master">
</a>
<a href="https://pypi.python.org/pypi/fury"><img src="https://img.shields.io/pypi/v/fury.svg"></a>
<a href="https://anaconda.org/conda-forge/fury"><img src="https://anaconda.org/conda-forge/fury/badges/version.svg"></a>
<a href="https://codecov.io/gh/fury-gl/fury"><img src="https://codecov.io/gh/fury-gl/fury/branch/master/graph/badge.svg"></a>
<a href="https://app.codacy.com/app/fury-gl/fury?utm_source=github.com&utm_medium=referral&utm_content=fury-gl/fury&utm_campaign=Badge_Grade_Dashboard"><img src="https://api.codacy.com/project/badge/Grade/922600af9f94445ead5a12423b813576"></a>
<a href="https://doi.org/10.21105/joss.03384"><img src="https://joss.theoj.org/papers/10.21105/joss.03384/status.svg"></a>

</p>

<p align="center">
  <a href="#general-information">General Information</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How to use</a> •
  <a href="#credits">Credits</a> •
  <a href="#contribute">Contribute</a> •
  <a href="#credits">Citing</a>
</p>

|         |         |         |
|:--------|:--------|:--------|
| <a href="#"><img src="https://raw.githubusercontent.com/fury-gl/fury-communication-assets/main/ws_smaller.gif" alt="FURY Networks" width="400px"></a> | <a href="#"><img src="https://raw.githubusercontent.com/fury-gl/fury-communication-assets/main/swarming_simulation.gif" alt="swarming simulation" width="400px"></a> | <a href="#"><img src="https://raw.githubusercontent.com/fury-gl/fury-communication-assets/main/shaders_horse.gif" alt="shaders horse" width="400px"></a> |
| *Network Visualization*          | *Swarming/flocking simulation based on simple boids rules*  |  *Easy shader effect integration.*  |
| <a href="#"><img src="https://raw.githubusercontent.com/fury-gl/fury-communication-assets/main/shaders_sdf.gif" alt="sdf" width="400px"></a>  | <a href="#"><img src="https://raw.githubusercontent.com/fury-gl/fury-communication-assets/main/collides_simulation.gif" alt="Collides simulation" width="400px"></a> | <a href="#"><img src="https://raw.githubusercontent.com/fury-gl/fury-communication-assets/main/physics_bricks_fast.gif" alt="Physics bricks" width="400px"></a> |
| *Ray Marching and Signed Distance Functions* | *Particle collisions* | *Interoperability with the [pyBullet](https://pybullet.org/wordpress/) library.*  |
| <a href="#"><img src="https://raw.githubusercontent.com/fury-gl/fury-communication-assets/main/ui_tab.gif" alt="UI Tabs" width="400px"></a>  | <a href="#"><img src="https://raw.githubusercontent.com/fury-gl/fury-communication-assets/main/shaders_dragon_skybox.gif" alt="Shaders dragon skybox" width="400px"></a>  | <a href="#"><img src="https://raw.githubusercontent.com/fury-gl/fury-communication-assets/main/picking_engine.gif" alt="Picking object" width="400px"></a> |
| *Custom User Interfaces* |  *Shaders and SkyBox integration*  | *Easy picking manager* |


# General Information

- **Website and Documentation:** https://fury.gl
- **Tutorials:** https://fury.gl/latest/auto_tutorials/index.html
- **Demos:** https://fury.gl/latest/auto_examples/index.html
- **Blog:**  https://fury.gl/latest/blog.html
- **Mailing list:** https://mail.python.org/mailman3/lists/fury.python.org
- **Official source code repo:** https://github.com/fury-gl/fury.git
- **Download releases:** https://pypi.org/project/fury/
- **Issue tracker:** https://github.com/fury-gl/fury/issues
- **Free software:** 3-clause BSD license
- **Community:** Come to chat on [Discord](https://discord.gg/6btFPPj)

# Key Features

- Custom User Interfaces
- Physics Engines API
- Custom Shaders
- Interactive local and Remote rendering in Jupyter Notebooks
- Large amount of Tutorials and Examples

# Installation

## Dependencies

FURY requires:

- Numpy (>=1.7.1)
- Vtk (>=8.1.2)
- Scipy (>=1.2.0)
- Pillow>=5.4.1

## Releases

`pip install fury` or `conda install -c conda-forge fury`

## Development

### Installation from source

**Step 1.** Get the latest source by cloning this repo:

    git clone https://github.com/fury-gl/fury.git

**Step 2.** Install requirements:

    pip install -r requirements/default.txt

**Step 3.** Install fury

As a [local project installation](https://pip.pypa.io/en/stable/reference/pip_install/#id44) using:

    pip install .

Or as an ["editable" installation](https://pip.pypa.io/en/stable/reference/pip_install/#id44) using:

    pip install -e .

**If you are developing fury you should go with editable installation.**

**Step 4:** Enjoy!

For more information, see also [installation page on fury.gl](https://fury.gl/latest/installation.html)

## Testing

After installation, you can install test suite requirements:

    pip install -r requirements/test.txt

And to launch test suite:

    pytest -svv fury


# How to use

There are many ways to start using FURY:

- Go to [Getting Started](https://fury.gl/latest/getting_started.html)
- Explore our [Tutorials](https://fury.gl/latest/auto_tutorials/index.html) or [Demos](https://fury.gl/latest/auto_examples/index.html).


# Credits

Please, go to [contributors page](https://github.com/fury-gl/fury/graphs/contributors) to see who have been involved in the development of FURY.


# Contribute

We love contributions!

You've discovered a bug or something else you want to change - excellent! Create an [issue](https://github.com/fury-gl/fury/issues/new)!

# Citing

If you are using FURY in your work then do cite [this paper](https://doi.org/10.21105/joss.03384). By citing FURY, you are helping sustain the FURY ecosystem.

    Eleftherios Garyfallidis, Serge Koudoro, Javier Guaje, Marc-Alexandre Côté, Soham Biswas,
    David Reagan, Nasim Anousheh, Filipi Silva, Geoffrey Fox, and Fury Contributors.
    "FURY: advanced scientific visualization." Journal of Open Source Software 6, no. 64 (2021): 3384.
    https://doi.org/10.21105/joss.03384


```css
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
