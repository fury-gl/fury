import os

import numpy.testing as npt
import pytest 

import fury.shaders as fs
import vtk


def test_load():
    dummy_file_name = 'dummy.txt'
    dummy_file_contents = 'This is some dummy text.'

    dummy_file = open(os.path.join(fs.SHADERS_DIR, dummy_file_name), 'w')
    dummy_file.write(dummy_file_contents)
    dummy_file.close()

    npt.assert_string_equal(fs.load(dummy_file_name), dummy_file_contents)

    os.remove(os.path.join(fs.SHADERS_DIR, dummy_file_name))


def test_shader_callback():

    cone = vtk.vtkConeSource()
    coneMapper = vtk.vtkPolyDataMapper()
    coneMapper.SetInputConnection(cone.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(coneMapper)

    def callback(_caller, _event, calldata=None):
        program = calldata
        if program is not None:
            pass

    id_observer = fs.add_shader_callback(
            actor, callback, 42)

    # print('\t...invalid priority type should create an exception')
    with pytest.raises(Exception):
        fs.add_shader_callback(actor, callback, priority='str')
    
    mapper = actor.GetMapper()
    if id_observer is not None:
        mapper.RemoveObserver(id_observer)