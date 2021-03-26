import os

import numpy.testing as npt
from fury import convert
from tempfile import TemporaryDirectory
from fury.io import load_image
import matplotlib.pyplot as plt


def test_convert():
    names = ['group_a', 'group_b', 'group_c']
    values = [1, 10, 100]

    fig = plt.figure(figsize=(9, 3))
    # plt.subplot(131)
    # plt.bar(names, values)
    # plt.subplot(132)
    # plt.scatter(names, values)
    # plt.subplot(133)
    # plt.plot(names, values)
    # plt.suptitle('Categorical Plotting')
    arr2 = convert.matplotlib_figure_to_numpy(fig, transparent=True)

    with TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'tmp.png')
        dpi = 100
        fig.savefig(fname, dpi=dpi, transparent=True)
        arr1 = load_image(fname)
        npt.assert_array_equal(arr1, arr2)
