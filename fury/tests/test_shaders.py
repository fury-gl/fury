import os

import numpy as np
import numpy.testing as npt
import pytest

from fury import window
import fury.shaders as fs
from fury.lib import PolyDataMapper, Actor, ConeSource


def test_load():
    dummy_file_name = 'dummy.txt'
    dummy_file_contents = 'This is some dummy text.'

    dummy_file = open(os.path.join(fs.SHADERS_DIR, dummy_file_name), 'w')
    dummy_file.write(dummy_file_contents)
    dummy_file.close()

    npt.assert_string_equal(fs.load(dummy_file_name), dummy_file_contents)

    os.remove(os.path.join(fs.SHADERS_DIR, dummy_file_name))


def test_shader_callback():

    cone = ConeSource()
    coneMapper = PolyDataMapper()
    coneMapper.SetInputConnection(cone.GetOutputPort())
    actor = Actor()
    actor.SetMapper(coneMapper)

    test_values = []

    def callbackLow(_caller, _event, calldata=None):
        program = calldata
        if program is not None:
            test_values.append(0)

    id_observer = fs.add_shader_callback(
            actor, callbackLow, 0)

    with pytest.raises(Exception):
        fs.add_shader_callback(actor, callbackLow, priority='str')

    mapper = actor.GetMapper()
    mapper.RemoveObserver(id_observer)

    scene = window.Scene()
    scene.add(actor)

    window.snapshot(scene)
    assert len(test_values) == 0

    test_values = []

    def callbackHigh(_caller, _event, calldata=None):
        program = calldata
        if program is not None:
            test_values.append(999)

    def callbackMean(_caller, _event, calldata=None):
        program = calldata
        if program is not None:
            test_values.append(500)

    fs.add_shader_callback(
            actor, callbackHigh, 999)
    fs.add_shader_callback(
            actor, callbackLow, 0)
    id_mean = fs.add_shader_callback(
            actor, callbackMean, 500)

    # check the priority of each call
    window.snapshot(scene)
    assert np.abs([
        test_values[0]-999, test_values[1]-500, test_values[2]-0]).sum() == 0

    # check if the correct observer was removed
    mapper.RemoveObserver(id_mean)
    test_values = []
    window.snapshot(scene)
    assert np.abs([
        test_values[0]-999, test_values[1]-0]).sum() == 0
