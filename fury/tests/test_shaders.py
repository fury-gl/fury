import os
import pytest

import numpy.testing as npt

import fury.shaders as fs


def test_load():
    dummy_file_name = 'dummy.txt'
    dummy_file_contents = 'This is some dummy text.'

    dummy_file = open(os.path.join(fs.SHADERS_DIR, dummy_file_name), 'w')
    dummy_file.write(dummy_file_contents)
    dummy_file.close()

    npt.assert_string_equal(fs.load(dummy_file_name), dummy_file_contents)

    os.remove(os.path.join(fs.SHADERS_DIR, dummy_file_name))


def test_uniform_tools():
    # print('\nTest: Uniform tools')
    # print('\t...creating a Uniform obj')
    uniform = fs.Uniform('uniform_name', 'f', 1.)
    assert uniform.vtk_func_uniform == 'SetUniformf'
    assert uniform.value == 1.

    class Dummy_Program:
        def SetUniformf(self, name, value):
            """camel case because we need to simulate the
            program used in vtk with the same attr name
            """
            assert name == 'uniform_name'
            assert value == 1.
            # print(f'\t...uniform set for {name} and value {value}')

    dummy_program = Dummy_Program()
    uniform.execute_program(dummy_program)

    # print('\t...invalid uniform type should create an exception')
    with pytest.raises(Exception):
        fs.Uniform('uniform_name', 'invalid_type', 1.)

    # print('\t...creating an Uniforms object')
    uniforms = fs.Uniforms([uniform])
    # print('\t...invalid uniform list  should create an exception')
    with pytest.raises(Exception):
        fs.Uniforms([1, '--'])

    # print('\t... test __call__ method used in callback')
    uniforms(None, None)
