import json
import os
from os.path import join as pjoin
from urllib.request import urlopen

import numpy.testing as npt

from fury.data import fetch_gltf, list_gltf_sample_models, read_viz_gltf

if 'FURY_HOME' in os.environ:
    fury_home = os.environ['FURY_HOME']
else:
    fury_home = pjoin(os.path.expanduser('~'), '.fury')

GLTF_DATA_URL = \
    "https://api.github.com/repos/KhronosGroup/glTF-Sample-Models/contents/2.0/"  # noqa


def tests_fetch_gltf():
    folder = pjoin(fury_home, 'glTF')
    boxtex = pjoin(folder, 'BoxTextured')
    boxtex = pjoin(boxtex, 'glTF')
    models_list = ['BoxTextured', 'Box']
    if os.path.exists(boxtex):
        for path in os.listdir(boxtex):
            os.remove(pjoin(boxtex, path))
        os.rmdir(boxtex)

    fetch_gltf(models_list)
    list_gltf = os.listdir(folder)
    results = [model in list_gltf for model in models_list]

    npt.assert_equal(results, [True, True])
    npt.assert_raises(ValueError, fetch_gltf, ['duck'])
    npt.assert_raises(ValueError, fetch_gltf, ['Duck'], 'GLTF')

    fetch_gltf()
    list_gltf = os.listdir(folder)
    default_list = ['BoxTextured', 'Duck', 'CesiumMilkTruck', 'CesiumMan']
    results = [model in list_gltf for model in default_list]
    npt.assert_equal(results, [True, True, True, True])

    items = os.listdir(boxtex)
    npt.assert_array_equal(len(items), 3)

    filenames, path = fetch_gltf('Box', 'glTF-Binary')
    npt.assert_equal(len(filenames), 1)
    npt.assert_equal(os.listdir(path), filenames)

    gltf = pjoin(boxtex, 'BoxTextured.gltf')
    with open(gltf, 'r') as f:
        gltf = json.loads(f.read())
    validate_gltf = gltf.get('asset')
    npt.assert_equal(validate_gltf['version'], str(2.0))


def test_list_gltf_sample_models():
    fetch_names = list_gltf_sample_models()
    default_list = ['BoxTextured', 'Duck', 'CesiumMilkTruck', 'CesiumMan']
    result = [model in fetch_names for model in default_list]
    npt.assert_equal(result, [True, True, True, True])


def test_read_viz_gltf():
    gltf_dir = pjoin(fury_home, 'glTF')
    filenames, path = fetch_gltf('Box', 'glTF-Binary')
    filename = read_viz_gltf('Box', 'glTF-Binary')
    npt.assert_equal(filename, pjoin(path, filenames[0]))

    npt.assert_raises(ValueError, read_viz_gltf, 'FURY', 'glTF')

    box_gltf = pjoin(gltf_dir, 'Box')
    for path in os.listdir(box_gltf):
        mode = pjoin(box_gltf, path)
        for file in os.listdir(mode):
            os.remove(pjoin(mode, file))
        os.rmdir(mode)
    npt.assert_raises(ValueError, read_viz_gltf, 'Box')

    filenames, path = fetch_gltf('Box')
    out_path = read_viz_gltf('Box').split(os.sep)
    mode = out_path[-2:][0]
    npt.assert_equal(mode, 'glTF')
