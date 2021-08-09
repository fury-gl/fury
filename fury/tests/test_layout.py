import numpy as np
import numpy.testing as npt
from fury import actor
from fury.layout import GridLayout, HorizontalLayout, Layout, \
    VerticalLayout, XLayout, YLayout, ZLayout


def get_default_cubes(centers=np.asarray([[[0, 0, 0]], [[5, 5, 5]]]),
                      directions=np.asarray([[[0, 0, 0]], [[0, 0, 0]]]),
                      colors=np.random.rand(2, 3), scales=[1, 1.5]):
    """Provides cube actors with default parameters

    Parameters
    ----------
    centers: ndarray, shape (2, 3)
        Cube positions
    directions: ndarray, shape (2, 3)
        The orientation vector of the cube.
    colors: ndarray ndarray (2,3) or (2, 4)
        RGB or RGBA (for opacity)
    scales: list of 2 floats
        Cube Sizes
    """
    cube_first_center, cube_second_center = centers
    cube_first_direction, cube_second_direction = directions
    cube_first_color, cube_second_color = colors
    cube_first_scale, cube_second_scale = scales

    cube_first = actor.cube(cube_first_center, cube_first_direction,
                            cube_first_color, cube_first_scale)

    cube_second = actor.cube(cube_second_center, cube_second_direction,
                             cube_second_color, cube_second_scale)

    return (cube_first, cube_second)


def test_layout_apply():

    cube_first, cube_second = get_default_cubes()

    layout = Layout()
    layout.apply([cube_first, cube_second])

    cube_first_center = cube_first.GetCenter()
    cube_second_center = cube_second.GetCenter()

    npt.assert_array_equal(cube_first_center, [0, 0, 0])
    npt.assert_array_equal(cube_second_center, [5, 5, 5])


def test_layout_compute_postions():

    cube_first, cube_second = get_default_cubes()

    layout = Layout()

    positions = layout.compute_positions([cube_first, cube_second])
    npt.assert_array_equal(positions, [])


def test_grid_layout_get_cell_shape():

    cube_first, cube_second = get_default_cubes()

    grid = GridLayout()
    grid_square = GridLayout(cell_shape="square")
    grid_diagonal = GridLayout(cell_shape="diagonal")
    invalid_gird = GridLayout(cell_shape="invalid")

    shape = grid.get_cells_shape([cube_first, cube_second])
    shape_square = grid_square.get_cells_shape([cube_first, cube_second])
    shape_diagonal = grid_diagonal.get_cells_shape([cube_first, cube_second])
    with npt.assert_raises(ValueError):
        shape_invalid = invalid_gird.get_cells_shape([cube_first, cube_second])

    npt.assert_array_equal(shape, [[1.5, 1.5], [1.5, 1.5]])
    npt.assert_array_equal(shape_square, [[1.5, 1.5], [1.5, 1.5]], 0)
    npt.assert_array_almost_equal(shape_diagonal,
                                  [[2.59, 2.59], [2.59, 2.59]], 0)


def test_grid_layout_compute_positions():

    cube_first, cube_second = get_default_cubes()

    grid = GridLayout()
    grid_square = GridLayout(cell_shape="square")
    grid_diagonal = GridLayout(cell_shape="diagonal")

    position_rect = grid.compute_positions([cube_first, cube_second])
    position_square = grid_square.compute_positions([cube_first, cube_second])
    position_diagonal = grid_diagonal.compute_positions([cube_first,
                                                        cube_second])

    npt.assert_array_equal(position_rect, [[0, 0, 0], [1.5, 0, 0]])
    npt.assert_array_equal(position_square, [[0, 0, 0], [1.5, 0, 0]])
    npt.assert_array_almost_equal(position_diagonal,
                                  [[0, 0, 0], [2.59, 0, 0]], 0)


def test_grid_layout_apply():

    cube_first, cube_second = get_default_cubes()

    grid_diagonal = GridLayout(cell_shape="diagonal")
    grid_diagonal.apply([cube_first, cube_second])

    cube_first_center = cube_first.GetCenter()
    cube_second_center = cube_second.GetCenter()
    npt.assert_array_almost_equal([cube_first_center, cube_second_center],
                                  [[0, 0, 0], [2.59, 0, 0]], 0)


def test_horizontal_layout_compute_positions():

    cube_first, cube_second = get_default_cubes()

    horizontal_rect = HorizontalLayout()
    horizontal_square = HorizontalLayout(cell_shape="square")
    horizontal_diagonal = HorizontalLayout(cell_shape="diagonal")

    position_rect = horizontal_rect.compute_positions([cube_first,
                                                       cube_second])

    position_square = horizontal_square.compute_positions([cube_first,
                                                           cube_second])

    position_diagonal = horizontal_diagonal.compute_positions([cube_first,
                                                               cube_second])

    npt.assert_array_equal(position_rect, [[0, 0, 0], [1.5, 0, 0]])
    npt.assert_array_equal(position_square, [[0, 0, 0], [1.5, 0, 0]])
    npt.assert_array_almost_equal(position_diagonal, [[0, 0, 0], [2.59, 0, 0]],
                                  0)


def test_vertical_layout_compute_positions():
    (cube_first, cube_second) = get_default_cubes()

    vertical_layout_rect = VerticalLayout()
    vertical_layout_square = VerticalLayout(cell_shape='square')
    vertical_layout_diagonal = VerticalLayout(cell_shape='diagonal')

    position_rect = vertical_layout_rect.compute_positions([cube_first,
                                                            cube_second])

    position_square = \
        vertical_layout_square.compute_positions([cube_first, cube_second])

    position_diagonal = \
        vertical_layout_diagonal.compute_positions([cube_first, cube_second])

    npt.assert_array_equal(position_rect, [[0, 0, 0], [0, 1.5, 0]])
    npt.assert_array_equal(position_square, [[0, 0, 0], [0, 1.5, 0]])
    npt.assert_array_almost_equal(position_diagonal, [[0, 0, 0], [0,
                                  2.59, 0]], 0)


def test_x_layout():
    cube_first, cube_second = get_default_cubes()
    actors = [cube_first, cube_second]

    positive_x_layout = XLayout(direction='x+')
    negative_x_layout = XLayout(direction='x-')

    with npt.assert_raises(ValueError):
        _ = XLayout(direction='Invalid direction')

    positive_positions = positive_x_layout.compute_positions(actors)
    negative_positions = negative_x_layout.compute_positions(actors)
    npt.assert_array_equal(positive_positions, [[0, 0, 0], [1.5, 0, 0]])
    npt.assert_array_equal(negative_positions, [[0, 0, 0], [1., 0, 0]])

    positive_x_shapes = positive_x_layout.get_cells_shape(actors)
    negative_x_shapes = negative_x_layout.get_cells_shape(actors)
    npt.assert_array_equal(positive_x_shapes, [[1.5, 1.5], [1.5, 1.5]])
    npt.assert_array_equal(negative_x_shapes, [[1.5, 1.5], [1.5, 1.5]])

    positive_x_layout.apply(actors)
    cube_first_center = cube_first.GetCenter()
    cube_second_center = cube_second.GetCenter()

    npt.assert_array_equal([cube_first_center, cube_second_center],
                           [[0, 0, 0], [1.5, 0, 0]])

    negative_x_layout.apply(actors)
    cube_first_center = cube_first.GetCenter()
    cube_second_center = cube_second.GetCenter()

    npt.assert_array_equal([cube_first_center, cube_second_center],
                           [[1.5, 0, 0], [0, 0, 0]])


def test_y_layout():
    cube_first, cube_second = get_default_cubes()
    actors = [cube_first, cube_second]

    positive_y_layout = YLayout(direction='y+')
    negative_y_layout = YLayout(direction='y-')

    with npt.assert_raises(ValueError):
        _ = YLayout(direction='Invalid direction')

    positive_positions = positive_y_layout.compute_positions(actors)
    negative_positions = negative_y_layout.compute_positions(actors)
    npt.assert_array_equal(positive_positions, [[0, 0, 0], [0, 1.5, 0]])
    npt.assert_array_equal(negative_positions, [[0, 0, 0], [0, 1., 0]])

    positive_y_shapes = positive_y_layout.get_cells_shape(actors)
    negative_y_shapes = negative_y_layout.get_cells_shape(actors)
    npt.assert_array_equal(positive_y_shapes, [[1.5, 1.5], [1.5, 1.5]])
    npt.assert_array_equal(negative_y_shapes, [[1.5, 1.5], [1.5, 1.5]])

    positive_y_layout.apply(actors)
    cube_first_center = cube_first.GetCenter()
    cube_second_center = cube_second.GetCenter()

    npt.assert_array_equal([cube_first_center, cube_second_center],
                           [[0, 0, 0], [0, 1.5, 0]])

    negative_y_layout.apply(actors)
    cube_first_center = cube_first.GetCenter()
    cube_second_center = cube_second.GetCenter()

    npt.assert_array_equal([cube_first_center, cube_second_center],
                           [[0, 1.5, 0], [0, 0, 0]])


def test_z_layout():
    cube_first, cube_second = get_default_cubes()
    actors = [cube_first, cube_second]

    positive_z_layout = ZLayout(direction='z+')
    negative_z_layout = ZLayout(direction='z-')

    with npt.assert_raises(ValueError):
        _ = XLayout(direction='Invalid direction')

    positive_positions = positive_z_layout.compute_positions(actors)
    negative_positions = negative_z_layout.compute_positions(actors)
    npt.assert_array_equal(positive_positions, [[0, 0, 0], [0, 0, 1.5]])
    npt.assert_array_equal(negative_positions, [[0, 0, 0], [0, 0, 1.]])

    positive_z_shapes = positive_z_layout.get_cells_shape(actors)
    negative_z_shapes = negative_z_layout.get_cells_shape(actors)
    npt.assert_array_equal(positive_z_shapes, [1.5, 1.5])
    npt.assert_array_equal(negative_z_shapes, [1.5, 1.5])

    positive_z_layout.apply(actors)
    cube_first_center = cube_first.GetCenter()
    cube_second_center = cube_second.GetCenter()

    npt.assert_array_equal([cube_first_center, cube_second_center],
                           [[0, 0, 0], [0, 0, 1.5]])

    negative_z_layout.apply(actors)
    cube_first_center = cube_first.GetCenter()
    cube_second_center = cube_second.GetCenter()

    npt.assert_array_equal([cube_first_center, cube_second_center],
                           [[0, 0, 1.5], [0, 0, 0]])
