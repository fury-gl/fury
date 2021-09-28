import os

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
