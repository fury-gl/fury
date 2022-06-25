import os
from os.path import join as pjoin
import json
import numpy.testing as npt
from aiohttp import InvalidURL
from fury.data import (fetch_gltf, read_viz_gltf)

if 'FURY_HOME' in os.environ:
    fury_home = os.environ['FURY_HOME']
else:
    fury_home = pjoin(os.path.expanduser('~'), '.fury')


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
    print(filenames)
    print(os.listdir(path))
    npt.assert_equal(os.listdir(path), filenames)

    filename = read_viz_gltf('Box', 'glTF-Binary')
    npt.assert_equal(filename, pjoin(path, filenames[0]))

    gltf = pjoin(boxtex, 'BoxTextured.gltf')
    with open(gltf, 'r') as f:
        gltf = json.loads(f.read())
    validate_gltf = gltf.get('asset')
    npt.assert_equal(validate_gltf['version'], str(2.0))
