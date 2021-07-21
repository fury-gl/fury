import numpy.testing as npt

from fury import molecular


def test_periodic_table():
    table = molecular.PeriodicTable()
    npt.assert_equal(table.atomic_number('C'), 6)
    npt.assert_equal(table.element_name(7), 'N')
    npt.assert_equal(table.atomic_symbol(8), 'O')
    npt.assert_almost_equal(table.atomic_radius(1, 'VDW'), 1.2)
    npt.assert_almost_equal(table.atomic_radius(1, 'Covalent'), 0.32)
