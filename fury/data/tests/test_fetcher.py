import json
import os
from os.path import join as pjoin

import numpy as np
import numpy.testing as npt

from fury.data import (
    fetch_gltf,
    fetch_viz_lucide_icons,
    list_gltf_sample_models,
    read_viz_gltf,
    read_viz_icons,
)
from fury.io import load_image

if "FURY_HOME" in os.environ:
    fury_home = os.environ["FURY_HOME"]
else:
    fury_home = pjoin(os.path.expanduser("~"), ".fury")

GLTF_DATA_URL = (
    "https://api.github.com/repos/KhronosGroup/glTF-Sample-Models/contents/2.0/"  # noqa
)


def tests_fetch_gltf():
    folder = pjoin(fury_home, "glTF")
    boxtex = pjoin(folder, "BoxTextured")
    boxtex = pjoin(boxtex, "glTF")
    models_list = ["BoxTextured", "Box"]
    if os.path.exists(boxtex):
        for path in os.listdir(boxtex):
            os.remove(pjoin(boxtex, path))
        os.rmdir(boxtex)

    fetch_gltf(name=models_list)
    list_gltf = os.listdir(folder)
    results = [model in list_gltf for model in models_list]

    npt.assert_equal(results, [True, True])
    npt.assert_raises(ValueError, fetch_gltf, name=["duck"])
    npt.assert_raises(ValueError, fetch_gltf, name=["Duck"], mode="GLTF")

    fetch_gltf()
    list_gltf = os.listdir(folder)
    default_list = ["BoxTextured", "Duck", "CesiumMilkTruck", "CesiumMan"]
    results = [model in list_gltf for model in default_list]
    npt.assert_equal(results, [True, True, True, True])

    items = os.listdir(boxtex)
    npt.assert_array_equal(len(items), 3)

    filenames, path = fetch_gltf(name="Box", mode="glTF-Binary")
    npt.assert_equal(len(filenames), 1)
    npt.assert_equal(os.listdir(path), filenames)

    gltf = pjoin(boxtex, "BoxTextured.gltf")
    with open(gltf) as f:
        gltf = json.loads(f.read())
    validate_gltf = gltf.get("asset")
    npt.assert_equal(validate_gltf["version"], str(2.0))


def test_list_gltf_sample_models():
    fetch_names = list_gltf_sample_models()
    default_list = ["BoxTextured", "Duck", "CesiumMilkTruck", "CesiumMan"]
    result = [model in fetch_names for model in default_list]
    npt.assert_equal(result, [True, True, True, True])


def test_read_viz_gltf():
    gltf_dir = pjoin(fury_home, "glTF")
    filenames, path = fetch_gltf(name="Box", mode="glTF-Binary")
    filename = read_viz_gltf("Box", mode="glTF-Binary")
    npt.assert_equal(filename, pjoin(path, filenames[0]))

    npt.assert_raises(ValueError, read_viz_gltf, "FURY", mode="glTF")

    box_gltf = pjoin(gltf_dir, "Box")
    for path in os.listdir(box_gltf):
        mode = pjoin(box_gltf, path)
        for file in os.listdir(mode):
            os.remove(pjoin(mode, file))
        os.rmdir(mode)
    npt.assert_raises(ValueError, read_viz_gltf, "Box")

    filenames, path = fetch_gltf(name="Box")
    out_path = read_viz_gltf("Box").split(os.sep)
    mode = out_path[-2:][0]
    npt.assert_equal(mode, "glTF")


def test_fetch_viz_lucide_icons():
    _, folder = fetch_viz_lucide_icons()
    pack = pjoin(folder, "lucide")

    with open(pjoin(pack, "manifest.json")) as f:
        manifest = json.load(f)
    npt.assert_equal(manifest["name"], "lucide")
    npt.assert_(len(manifest["icons"]) > 0)
    npt.assert_(os.path.isfile(pjoin(pack, "LICENSE")))
    for icon in manifest["icons"]:
        for relpath in (
            f"{icon}.png",
            f"{icon}-pressed.png",
            f"{icon}-filled.png",
            f"svg/{icon}.svg",
        ):
            npt.assert_(os.path.isfile(pjoin(pack, relpath)), relpath)

    npt.assert_equal(
        read_viz_icons(style="lucide", fname="play.png"), pjoin(pack, "play.png")
    )
    npt.assert_equal(read_viz_icons(fname="play.png"), pjoin(pack, "play.png"))
    npt.assert_equal(read_viz_icons(), pjoin(pack, "infinity.png"))
    npt.assert_(os.path.isfile(read_viz_icons()))

    idle = load_image(pjoin(pack, "play.png"))
    pressed = load_image(pjoin(pack, "play-pressed.png"))
    npt.assert_equal(idle.shape, (64, 64, 4))
    npt.assert_equal(pressed.shape, (64, 64, 4))
    npt.assert_array_equal(idle[..., 3], pressed[..., 3])
    painted = idle[..., 3] > 0
    npt.assert_(painted.any())
    npt.assert_(np.all(idle[painted, :3] == (68, 68, 68)))
    npt.assert_(np.all(pressed[painted, :3] == (187, 187, 187)))

    filled = load_image(pjoin(pack, "heart-filled.png"))
    outline = load_image(pjoin(pack, "heart.png"))
    npt.assert_equal(filled.shape, (64, 64, 4))
    npt.assert_(np.all(filled[filled[..., 3] > 0, :3] == (68, 68, 68)))
    npt.assert_(np.count_nonzero(filled[..., 3]) > np.count_nonzero(outline[..., 3]))


def test_read_viz_icons_restores_icons_missing_from_an_old_pack():
    _, folder = fetch_viz_lucide_icons()
    filled = pjoin(folder, "lucide", "heart-filled.png")
    os.remove(filled)

    npt.assert_equal(read_viz_icons(fname="heart-filled.png"), filled)
    npt.assert_(os.path.isfile(filled))
