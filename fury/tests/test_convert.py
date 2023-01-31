import os
from tempfile import TemporaryDirectory

import numpy.testing as npt
import pytest

from fury.io import load_image

# Optional packages
from fury.optpkg import optional_package

matplotlib, have_matplotlib, _ = optional_package('matplotlib')

if have_matplotlib:
    import matplotlib.pyplot as plt

    from fury.convert import matplotlib_figure_to_numpy


@pytest.mark.skipif(not have_matplotlib, reason='Requires MatplotLib')
def test_convert():
    names = ['group_a', 'group_b', 'group_c']
    values = [1, 10, 100]

    fig = plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.bar(names, values)
    plt.subplot(132)
    plt.scatter(names, values)
    plt.subplot(133)
    plt.plot(names, values)
    plt.suptitle('Categorical Plotting')
    arr2 = matplotlib_figure_to_numpy(fig, transparent=False, flip_up_down=False)

    with TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'tmp.png')
        dpi = 100
        fig.savefig(fname, transparent=False, bbox_inches='tight', pad_inches=0)
        arr1 = load_image(fname)
        npt.assert_array_equal(arr1, arr2)
