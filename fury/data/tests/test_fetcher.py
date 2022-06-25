import os
from os.path import join as pjoin
import json
from urllib.request import urlopen
import numpy.testing as npt
from aiohttp import InvalidURL
from fury.data import (fetch_gltf, read_viz_gltf, list_gltf_sample_models)

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
    npt.assert_raises(InvalidURL, fetch_gltf, ['duck'])
    npt.assert_raises(InvalidURL, fetch_gltf, ['Duck'], 'GLTF')

    items = os.listdir(boxtex)
    npt.assert_array_equal(len(items), 3)

    filenames, path = fetch_gltf('Box', 'glTF-Binary')
    npt.assert_equal(len(filenames), 1)
    npt.assert_equal(os.listdir(path), filenames)

    filename = read_viz_gltf('Box', 'glTF-Binary')
    npt.assert_equal(filename, pjoin(path, filenames[0]))

    gltf = pjoin(boxtex, 'BoxTextured.gltf')
    with open(gltf, 'r') as f:
        gltf = json.loads(f.read())
    validate_gltf = gltf.get('asset')
    npt.assert_equal(validate_gltf['version'], str(2.0))


def test_list_gltf_sample_models():
    gltf_path = pjoin(fury_home, 'glTF')
    list_json = pjoin(gltf_path, 'list.json')
    if not os.path.exists(list_json):
        json_data = urlopen(f'{GLTF_DATA_URL}').read()
        with open(list_json, 'wb') as f:
            f.write(json_data)

    with open(list_json, 'r') as r:
        data = json.load(r)
    model_names = [model['name'] for model in data if model['size'] == 0]
    fetch_names = list_gltf_sample_models()
    npt.assert_equal(len(fetch_names), len(model_names))
    npt.assert_array_equal(model_names, fetch_names)

    default_list = ['BoxTextured', 'Duck', 'CesiumMilkTruck', 'CesiumMan']
    result = [model in fetch_names for model in default_list]
    npt.assert_equal(result, [True, True, True, True])
