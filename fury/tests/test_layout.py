import numpy as np
import numpy.testing as npt
from fury import actor
from fury.layout import GridLayout


def test_grid_layout_gets_cell_shape():
    centers = np.asarray([[0, 0, 0], [0, 0, 0]])
    directions = np.asarray([[0, 0, 0], [0, 0, 0]])
    heights = np.asarray([2, 5])
    colors = np.random.rand(2, 3)
    cylinders = actor.cylinder(centers, directions, colors, heights=heights)
    grid = GridLayout()
    grid_square = GridLayout(cell_shape="square")
    grid_diagonal = GridLayout(cell_shape="diagonal")
    shape = grid.get_cells_shape([cylinders])
    shape_square = grid_square.get_cells_shape([cylinders])
    shape_diagonal = grid_diagonal.get_cells_shape([cylinders])
    npt.assert_array_equal(shape, [[0.5, 5]])
    npt.assert_array_equal(shape_square, [[5, 5]])
    npt.assert_array_almost_equal(shape_diagonal, [[5, 5]], 0)


def test_grid_layout_compute_positions():
    centers = np.asarray([[0, 0, 0], [0, 0, 0]])
    directions = np.asarray([[0, 0, 0], [0, 0, 0]])
    heights = np.asarray([2, 5])
    colors = np.random.rand(2, 3)
    cylinders = actor.cylinder(centers, directions, colors, heights=heights)
    grid = GridLayout()
    positions = grid.compute_positions([cylinders])
    p = [[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [1.5, 0, 0], [2, 0, 0]]
    npt.assert_array_equal(positions, p)
