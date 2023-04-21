import numpy as np
import numpy.testing as npt
from fury.utils import Actor

import trees


def test_point2d():
    array = np.array([1, 2])
    point = trees.Point2d(array[0], array[1])
    new_array = np.array([1, 1])
    npt.assert_array_equal(array, point())
    npt.assert_equal(array[0], point.get_x_coord())
    npt.assert_equal(array[1], point.get_y_coord())
    point.set_coord(new_array[0], new_array[1])
    npt.assert_array_equal(new_array, point())
    npt.assert_string_equal(str(new_array), str(point))
    npt.assert_string_equal(repr(new_array), repr(point))


def test_point3d():
    array = np.array([1, 2, 3])
    point = trees.Point3d(array[0], array[1], array[2])
    new_array = np.array([1, 1, 1])
    npt.assert_array_equal(array, point())
    npt.assert_equal(array[2], point.get_z_coord())
    point.set_coord(new_array[0], new_array[1], new_array[2])
    npt.assert_array_equal(new_array, point())


def test_branch2d():
    max_points = 4
    x_size = (0.0, 1.0)
    y_size = (0.0, 1.0)
    branch = trees.Branch2d(max_points,
                            x_size[0],
                            x_size[1],
                            y_size[0],
                            y_size[1])
    positions = np.array([[0.1, 0.2],
                          [0.6, 0.1],
                          [0.4, 0.9],
                          [0.8, 0.7],
                          [0.1, 0.6]])
    mid_x = x_size[0] + (x_size[1] - x_size[0]) / float(2)
    mid_y = y_size[0] + (y_size[1] - y_size[0]) / float(2)

    npt.assert_equal(0, branch.n_points())
    npt.assert_equal(max_points, branch.max_points())
    npt.assert_equal(False, branch.is_divided())
    npt.assert_equal(x_size, branch.x_size())
    npt.assert_equal(y_size, branch.y_size())
    npt.assert_equal(mid_x, branch.x_mid_point())
    npt.assert_equal(mid_y, branch.y_mid_point())

    points = np.array([trees.Point2d(pos[0], pos[1]) for pos in positions])
    for i in range(points.shape[0] - 1):
        branch.add_point(points[i])
    npt.assert_equal(points.shape[0] - 1, branch.n_points())
    npt.assert_equal(
        True, positions[:-1] in np.array([point().tolist() for point in branch.points_list()]))
    branch.add_point(points[-1])

    def all_points_inside(branch):
        return branch.points_list()

    npt.assert_equal(True, positions in np.array(
        [point().tolist() for point in branch.process_branch(all_points_inside)]))
    npt.assert_array_equal([], np.array([point().tolist()
                           for point in branch.points_list()]))
    npt.assert_equal(points.size, branch.total_points())
    npt.assert_equal(True, branch.is_divided())
    npt.assert_equal(0, branch.n_points())
    npt.assert_equal(True, branch._downleft == branch.sub_branch(0))
    npt.assert_equal(True, branch._downright == branch.sub_branch(1))
    npt.assert_equal(True, branch._upleft == branch.sub_branch(2))
    npt.assert_equal(True, branch._upright == branch.sub_branch(3))
    div = 2.0
    new_positions = positions / div
    new_points = np.array([trees.Point2d(pos[0], pos[1])
                          for pos in new_positions])
    new_branch = trees.Branch2d(max_points,
                                x_size[0],
                                x_size[1],
                                y_size[0],
                                y_size[1])
    for i in range(new_points.shape[0]):
        new_branch.add_point(new_points[i])

    def divide(branch, div: float):
        for i in range(branch.points_list().shape[0]):
            update_coord = branch.points_list()[i]() / div
            branch.points_list()[i].set_coord(update_coord[0], update_coord[1])
    branch.process_branch(divide, div)
    branch.update()
    npt.assert_equal(True, branch == new_branch)
    branch.sub_branch(0).sub_branch(0).remove_point(0)
    npt.assert_equal(0, branch.sub_branch(0).sub_branch(0).n_points())
    npt.assert_array_equal([], branch.sub_branch(0).points_list())


def test_branch3d():
    max_points = 4
    x_size = (0.0, 1.0)
    y_size = (0.0, 1.0)
    z_size = (0.0, 1.0)
    branch = trees.Branch3d(max_points,
                            x_size[0],
                            x_size[1],
                            y_size[0],
                            y_size[1],
                            z_size[0],
                            z_size[1])
    positions = np.array([[0.1, 0.2, 0.1],
                          [0.6, 0.1, 0.7],
                          [0.4, 0.9, 0.5],
                          [0.8, 0.7, 0.5],
                          [0.1, 0.6, 0.3]])
    mid_z = z_size[0] + (z_size[1] - z_size[0]) / float(2)
    npt.assert_equal(0, branch.n_points())
    npt.assert_equal(max_points, branch.max_points())
    npt.assert_equal(False, branch.is_divided())
    npt.assert_equal(x_size, branch.x_size())
    npt.assert_equal(y_size, branch.y_size())
    npt.assert_equal(z_size, branch.z_size())
    npt.assert_equal(mid_z, branch.z_mid_point())

    points = np.array([trees.Point3d(pos[0], pos[1], pos[2])
                      for pos in positions])
    for i in range(points.shape[0] - 1):
        branch.add_point(points[i])
    npt.assert_equal(points.shape[0] - 1, branch.n_points())
    npt.assert_equal(
        True, positions[:-1] in np.array([point().tolist() for point in branch.points_list()]))
    branch.add_point(points[-1])

    def all_points_inside(branch):
        return branch.points_list()

    npt.assert_equal(True, positions in np.array(
        [point().tolist() for point in branch.process_branch(all_points_inside)]))
    npt.assert_array_equal([], np.array([point().tolist()
                           for point in branch.points_list()]))
    npt.assert_equal(points.size, branch.total_points())
    npt.assert_equal(True, branch.is_divided())
    npt.assert_equal(0, branch.n_points())
    npt.assert_equal(True, branch._front_down_left == branch.sub_branch(0))
    npt.assert_equal(True, branch._front_down_right == branch.sub_branch(1))
    npt.assert_equal(True, branch._front_up_left == branch.sub_branch(2))
    npt.assert_equal(True, branch._front_up_right == branch.sub_branch(3))
    npt.assert_equal(True, branch._back_down_left == branch.sub_branch(4))
    npt.assert_equal(True, branch._back_down_right == branch.sub_branch(5))
    npt.assert_equal(True, branch._back_up_left == branch.sub_branch(6))
    npt.assert_equal(True, branch._back_up_right == branch.sub_branch(7))
    div = 2.0
    new_positions = positions / div
    new_points = np.array([trees.Point3d(pos[0], pos[1], pos[2])
                          for pos in new_positions])
    new_branch = trees.Branch3d(max_points,
                                x_size[0],
                                x_size[1],
                                y_size[0],
                                y_size[1],
                                z_size[0],
                                z_size[1])
    for i in range(new_points.shape[0]):
        new_branch.add_point(new_points[i])

    def divide(branch, div: float):
        for i in range(branch.points_list().shape[0]):
            update_coord = branch.points_list()[i]() / div
            branch.points_list()[i].set_coord(
                update_coord[0], update_coord[1], update_coord[2])
    branch.process_branch(divide, div)
    branch.update()
    npt.assert_equal(True, branch == new_branch)
    branch.sub_branch(0).sub_branch(0).remove_point(0)
    npt.assert_equal(0, branch.sub_branch(0).sub_branch(0).n_points())
    npt.assert_array_equal([], branch.sub_branch(0).points_list())


def test_tree2d():
    max_points = 4
    x_size = (0.0, 1.0)
    y_size = (0.0, 1.0)
    branch = trees.Branch2d(max_points,
                            x_size[0],
                            x_size[1],
                            y_size[0],
                            y_size[1])
    positions = np.array([[0.1, 0.2],
                          [0.6, 0.1],
                          [0.4, 0.9],
                          [0.8, 0.7],
                          [0.1, 0.6]])
    tree = trees.Tree2d(branch)
    npt.assert_equal(True, tree.root() == tree._root)
    npt.assert_equal(0, tree.n_points())
    npt.assert_equal(x_size, tree.x_size())
    npt.assert_equal(y_size, tree.y_size())
    point = trees.Point2d(positions[0, 0], positions[0, 1])
    tree.add_point(point)
    npt.assert_equal(1, tree.root().n_points())
    npt.assert_equal(True, tree.root().all_points_list()[0] == point)


def test_tree3d():
    max_points = 4
    x_size = (0.0, 1.0)
    y_size = (0.0, 1.0)
    z_size = (0.0, 1.0)
    branch = trees.Branch3d(max_points,
                            x_size[0],
                            x_size[1],
                            y_size[0],
                            y_size[1],
                            z_size[0],
                            z_size[1])
    positions = np.array([[0.1, 0.2, 0.1],
                          [0.6, 0.1, 0.6],
                          [0.4, 0.9, 0.7],
                          [0.8, 0.7, 0.3],
                          [0.1, 0.6, 0.9]])
    tree = trees.Tree3d(branch)
    npt.assert_equal(x_size, tree.x_size())
    npt.assert_equal(y_size, tree.y_size())
    npt.assert_equal(z_size, tree.z_size())
    point = trees.Point3d(positions[0, 0], positions[0, 1], positions[0, 2])
    tree.add_point(point)
    npt.assert_equal(1, tree.root().n_points())
    npt.assert_equal(True, tree.root().all_points_list()[0] == point)


def test_bounding_Box():
    center = [0.0, 0.0, 0.0]
    size = [1.0, 1.0, 1.0]
    color = (1.0, 1.0, 1.0)
    line_width = 1.0
    x_c = center[0]
    y_c = center[1]
    z_c = center[2]

    x_l = size[0] / 2
    y_l = size[1] / 2
    z_l = size[2] / 2
    bounds = [x_c - x_l, x_c + x_l, y_c - y_l, y_c + y_l, z_c - z_l, z_c + z_l]
    lines = trees.bounding_box_3d(center, size, color, line_width)
    npt.assert_equal(center, lines.GetCenter())
    npt.assert_equal(bounds, lines.GetBounds())


def test_actor_from_branch_2d():
    max_points = 4
    x_size = (0.0, 1.0)
    y_size = (0.0, 1.0)
    color = (1.0, 1.0, 1.0)
    line_width = 1.0
    branch = trees.Branch2d(max_points,
                            x_size[0],
                            x_size[1],
                            y_size[0],
                            y_size[1])
    positions = np.array([[0.1, 0.2],
                          [0.6, 0.1],
                          [0.4, 0.9],
                          [0.8, 0.7],
                          [0.1, 0.6]])
    branch = trees.Branch2d(
        max_points,
        x_size[0],
        x_size[1],
        y_size[0],
        y_size[1])
    points = np.array([trees.Point2d(pos[0], pos[1]) for pos in positions])
    for i in range(points.shape[0]):
        branch.add_point(points[i])

    actors = trees.actor_from_branch_2d(branch, color, line_width)

    def get_actors(branch, color, linewidth):
        x_c = branch.x_mid_point()
        y_c = branch.y_mid_point()
        z_c = 0.0

        x_l = (branch.x_size()[1] - branch.x_size()[0])
        y_l = (branch.y_size()[1] - branch.y_size()[0])
        z_l = 0.0

        return trees.bounding_box_3d(
            (x_c, y_c, z_c), (x_l, y_l, z_l), color, linewidth)

    actors_test = np.empty(0, dtype=Actor)
    if branch.is_divided() == True:
        actors_test = np.append(
            actors_test,
            get_actors(
                branch.sub_branch(0),
                color,
                line_width))
        actors_test = np.append(
            actors_test,
            get_actors(
                branch.sub_branch(1),
                color,
                line_width))
        actors_test = np.append(
            actors_test,
            get_actors(
                branch.sub_branch(2),
                color,
                line_width))
        actors_test = np.append(
            actors_test,
            get_actors(
                branch.sub_branch(3),
                color,
                line_width))

    else:
        x_c = branch.x_mid_point()
        y_c = branch.y_mid_point()
        z_c = 0.0

        x_l = (branch.x_size()[1] - branch.x_size()[0])
        y_l = (branch.y_size()[1] - branch.y_size()[0])
        z_l = 0.0

        actors = np.append(actors, trees.bounding_box_3d(
            (x_c, y_c, z_c), (x_l, y_l, z_l), color, line_width))

    for i in range(actors.shape[0]):
        npt.assert_equal(actors_test[i].GetCenter(), actors[i].GetCenter())
        npt.assert_equal(actors_test[i].GetBounds(), actors[i].GetBounds())


def test_actor_from_branch_3d():
    max_points = 4
    x_size = (0.0, 1.0)
    y_size = (0.0, 1.0)
    z_size = (0.0, 1.0)
    color = (1.0, 1.0, 1.0)
    line_width = 1.0
    branch = trees.Branch3d(max_points,
                            x_size[0],
                            x_size[1],
                            y_size[0],
                            y_size[1],
                            z_size[0],
                            z_size[1])
    positions = np.array([[0.1, 0.2, 0.1],
                          [0.6, 0.1, 0.7],
                          [0.4, 0.9, 0.5],
                          [0.8, 0.7, 0.5],
                          [0.1, 0.6, 0.3]])

    branch = trees.Branch3d(
        max_points,
        x_size[0],
        x_size[1],
        y_size[0],
        y_size[1],
        z_size[0],
        z_size[1])
    points = np.array([trees.Point3d(pos[0], pos[1], pos[2])
                      for pos in positions])
    for i in range(points.shape[0]):
        branch.add_point(points[i])
    actors = trees.actor_from_branch_3d(branch, color, line_width)

    def get_actors(branch, color, linewidth):
        x_c = branch.x_mid_point()
        y_c = branch.y_mid_point()
        z_c = branch.z_mid_point()

        x_l = (branch.x_size()[1] - branch.x_size()[0])
        y_l = (branch.y_size()[1] - branch.y_size()[0])
        z_l = (branch.z_size()[1] - branch.z_size()[0])

        return trees.bounding_box_3d(
            (x_c, y_c, z_c), (x_l, y_l, z_l), color, linewidth)

    actors_test = np.empty(0, dtype=Actor)
    if branch.is_divided() == True:
        actors_test = np.append(
            actors_test,
            get_actors(
                branch.sub_branch(0),
                color,
                line_width))
        actors_test = np.append(
            actors_test,
            get_actors(
                branch.sub_branch(1),
                color,
                line_width))
        actors_test = np.append(
            actors_test,
            get_actors(
                branch.sub_branch(2),
                color,
                line_width))
        actors_test = np.append(
            actors_test,
            get_actors(
                branch.sub_branch(3),
                color,
                line_width))
        actors_test = np.append(
            actors_test,
            get_actors(
                branch.sub_branch(4),
                color,
                line_width))
        actors_test = np.append(
            actors_test,
            get_actors(
                branch.sub_branch(5),
                color,
                line_width))
        actors_test = np.append(
            actors_test,
            get_actors(
                branch.sub_branch(6),
                color,
                line_width))
        actors_test = np.append(
            actors_test,
            get_actors(
                branch.sub_branch(7),
                color,
                line_width))

    else:
        x_c = branch.x_mid_point()
        y_c = branch.y_mid_point()
        z_c = branch.z_mid_point()

        x_l = (branch.x_size()[1] - branch.x_size()[0])
        y_l = (branch.y_size()[1] - branch.y_size()[0])
        z_l = (branch.z_size()[1] - branch.z_size()[0])

        actors = np.append(actors, trees.bounding_box_3d(
            (x_c, y_c, z_c), (x_l, y_l, z_l), color, line_width))

    for i in range(actors.shape[0]):
        npt.assert_equal(actors_test[i].GetCenter(), actors[i].GetCenter())
        npt.assert_equal(actors_test[i].GetBounds(), actors[i].GetBounds())