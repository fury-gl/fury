# Ray Traced ODF glyphs

This folder includes Fury's implementation of "Ray Tracing Spherical Harmonics Glyphs":
https://momentsingraphics.de/VMV2023.html

The fragment shader is based on: https://www.shadertoy.com/view/dlGSDV
(c) 2023, Christoph Peters

His work is licensed under a CC0 1.0 Universal License. To the extent
possible under law, Christoph Peters has waived all copyright and related or
neighboring rights to the following code. This work is published from
Germany. https://creativecommons.org/publicdomain/zero/1.0/

## Base implementation

The original paper implementation can be found in:
 - [ray_traced_1.0.py](ray_traced_1.0.py) for a single glyph
 - [ray_traced_2.0.py](ray_traced_2.0.py) for multiple glyphs.

> **Note:** We keep these files as they are for comparison purposes.

## FURY's implementation

To better understand the base approach and being able to build on top on it we need a simplified yet functional version of it.

 - [ray_traced_3.0.py](ray_traced_3.0.py) simplifies the illumination model getting rid of additional parameters and adding compatibility with VTK's default lighting model. Here is a comparison between the simplified version and the original one:

| BRDF lighting ([ray_traced_1.0.py](ray_traced_1.0.py)) | Blinn-Phong lighting ([ray_traced_3.0.py](ray_traced_3.0.py)) |
|---|---|
|<img width="350" src="https://github.com/tvcastillod/fury/assets/9929496/5b9d2a1e-9c14-4496-86d1-7d5484e8e038">|<img width="350" src="https://github.com/tvcastillod/fury/assets/9929496/c1f766c7-58d9-419f-b79a-7ae69df18493">|

 - [ray_traced_4.0.py](ray_traced_4.0.py) is an analogous version of [ray_traced_2.0.py](ray_traced_2.0.py) but with the simplified illumination model introduced in [ray_traced_3.0.py](ray_traced_3.0.py). Here is a comparison between the simplified version and the original one:

| BRDF lighting ([ray_traced_2.0.py](ray_traced_2.0.py)) | Blinn-Phong lighting ([ray_traced_4.0.py](ray_traced_4.0.py)) |
|---|---|
|<img width="350" src="https://github.com/tvcastillod/fury/assets/9929496/ab4af59e-2cd5-429a-9616-3511e0857a18">|<img width="350" src="https://github.com/tvcastillod/fury/assets/9929496/0509a424-b925-4f03-b862-f24e7bd17c2b">|
