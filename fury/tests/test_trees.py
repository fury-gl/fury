import numpy as np
import numpy.testing as npt
from fury.utils import Actor

import trees


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

    points_dic = {}
    for i in range(positions.shape[0] - 1):
        points_dic[branch.add_point(positions[i])] = positions[i]
    npt.assert_equal(positions.shape[0] - 1, branch.n_points())
    npt.assert_equal(True, np.isin(
        positions[:-1], np.array([point.tolist() for point in branch.points_list()])).all())
    points_dic[branch.add_point(positions[-1])] = positions[-1]

    def all_points_inside(branch):
        return branch.points_list()

    ids = list(points_dic.keys())

    def get_branch(s : str, branch):
        branch = branch
        for c in s:
            if c == 'f':
                break
            branch = branch.sub_branch(int(c))
        return branch

    ref_branch = get_branch(branch.search(points_dic[ids[0]], ids[0]), branch)

    npt.assert_equal(
        True, ref_branch.points_dic()[
            ids[0]] == branch.sub_branch(0).points_dic()[
            ids[0]])
    npt.assert_equal(True, branch.sub_branch(
        0) == branch.branch_from_point(points_dic[ids[0]], ids[0]))
    npt.assert_equal(True, branch.sub_branch(0).points_dic() ==
                     branch.relatives_from_point(points_dic[ids[0]], ids[0]))

    processing_points = np.array([branch.sub_branch(i).points_dic() if len(branch.sub_branch(
        i).points_dic()) != 0 else None for i in range(len(branch.sub_branches))])

    npt.assert_equal(True, np.isin(processing_points, branch.points_to_process()).all())

    npt.assert_equal(True, np.isin(positions, np.array(
        [point.tolist() for point in branch.process_branch(all_points_inside)])).all())
    npt.assert_array_equal([], np.array([point.tolist()
                           for point in branch.points_list()]))
    npt.assert_equal(positions.shape[0], branch.total_points())
    npt.assert_equal(True, branch.is_divided())
    npt.assert_equal(0, branch.n_points())
    npt.assert_equal(True, branch._downleft == branch.sub_branch(0))
    npt.assert_equal(True, branch._downright == branch.sub_branch(1))
    npt.assert_equal(True, branch._upleft == branch.sub_branch(2))
    npt.assert_equal(True, branch._upright == branch.sub_branch(3))
    div = 2.0
    new_positions = positions / div

    new_branch = trees.Branch2d(max_points,
                                x_size[0],
                                x_size[1],
                                y_size[0],
                                y_size[1])
    for i in range(new_positions.shape[0]):
        new_branch.add_point(new_positions[i], new_points=False, existing_key=ids[i])

    def divide(branch, div : float):
        dic = branch.points_dic()
        ids_list = list(dic.keys())
        for i in range(branch.points_list().shape[0]):
            update_coord = branch.points_list()[i]/div
            branch.points_dic()[ids_list[i]] = update_coord
    branch.process_branch(divide, div)
    branch.update()
    npt.assert_equal(True, branch == new_branch)
    branch.sub_branch(0).sub_branch(0).remove_point(0)
    npt.assert_equal(0, branch.sub_branch(0).sub_branch(0).n_points())
    npt.assert_array_equal([], branch.sub_branch(0).points_list().tolist())


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

    points_dic = {}
    for i in range(positions.shape[0] - 1):
        points_dic[branch.add_point(positions[i])] = positions[i]
    npt.assert_equal(positions.shape[0] - 1, branch.n_points())
    npt.assert_equal(True, np.isin(
        positions[:-1], np.array([point.tolist() for point in branch.points_list()])).all())
    points_dic[branch.add_point(positions[-1])] = positions[-1]

    ids = list(points_dic.keys())

    def all_points_inside(branch):
        return branch.points_list()

    npt.assert_equal(True, np.isin(positions, np.array(
        [point.tolist() for point in branch.process_branch(all_points_inside)])).all())
    npt.assert_array_equal([], np.array([point.tolist()
                           for point in branch.points_list()]))
    npt.assert_equal(positions.shape[0], branch.total_points())
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

    new_branch = trees.Branch3d(max_points,
                                x_size[0],
                                x_size[1],
                                y_size[0],
                                y_size[1],
                                z_size[0],
                                z_size[1])
    for i in range(new_positions.shape[0]):
        new_branch.add_point(new_positions[i], new_points=False, existing_key=ids[i])

    def divide(branch, div : float):
        dic = branch.points_dic()
        ids_list = list(dic.keys())
        for i in range(branch.points_list().shape[0]):
            update_coord = branch.points_list()[i]/div
            branch.points_dic()[ids_list[i]] = update_coord
    branch.process_branch(divide, div)
    branch.update()
    npt.assert_equal(True, branch == new_branch)
    branch.sub_branch(0).sub_branch(0).remove_point(0)
    npt.assert_equal(0, branch.sub_branch(0).sub_branch(0).n_points())
    npt.assert_array_equal([], branch.sub_branch(0).points_list().tolist())


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
    tree.add_point(positions[0])
    npt.assert_equal(1, tree.root().n_points())
    npt.assert_equal(True, tree.root().all_points_list()[0] == positions[0])


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
    tree.add_point(positions[0])
    npt.assert_equal(1, tree.root().n_points())
    npt.assert_equal(True, tree.root().all_points_list()[0] == positions[0])


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
    for i in range(positions.shape[0]):
        branch.add_point(positions[i])

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
    if branch.is_divided():
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
    for i in range(positions.shape[0]):
        branch.add_point(positions[i])
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
    if branch.is_divided():
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
