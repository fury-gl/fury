import numpy.testing as npt
from fury import molecular
import numpy as np

def test_periodic_table():
    table = molecular.PeriodicTable()
    npt.assert_equal(table.atomic_number('C'), 6)
    npt.assert_equal(table.element_name(7), 'Nitrogen')
    npt.assert_equal(table.atomic_symbol(8), 'O')
    npt.assert_almost_equal(table.atomic_radius(1, 'VDW'), 1.2)
    npt.assert_almost_equal(table.atomic_radius(1, 'Covalent'), 0.37)
    npt.assert_array_almost_equal(table.atom_color(1), np.array([1, 1, 1]))
