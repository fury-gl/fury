from fury.utils import Actor
from fury.actor import line
import numpy as np


# QUADTREE IMPLEMENTATION
class Point2d:
    """General point class. This class only stores 2d coordinates,
       and is to be used as base for any class that intends to use the quadtree structure.
    """

    def __init__(self, x: float, y: float):
        self.coord = np.array([x, y])

    def __call__(self):
        return self.coord

    def __eq__(self, point):
        return np.allclose(self.coord, point.coord, rtol=10 **
                           (-np.finfo(self.coord[0].dtype).precision))

    def get_x_coord(self) -> float:
        """Returns the x coordinate of the point."""
        return self.coord[0]

    def get_y_coord(self) -> float:
        """Returns the y coordinate of the point."""
        return self.coord[1]

    def set_coord(self, x: float, y: float) -> np.array:
        """Updates the coordinate of the point.
        ## Parameters
           * x : float x position
           * y : float y position"""
        self.coord = np.array([x, y])

    def __str__(self):
        return str(self.coord)

    def __repr__(self):
        return repr(self.coord)


class Branch2d():
    """General branch class. It is the base structure for building a quadtree.\n
    This branch, when divided, has four branches: \n
       1. the down-left
       2. the down-right
       3. the up-left
       4. the up-right

    Parameters
    ----------
       * max_points : int\n
            Defines the maximum number of points this branch stores
       * x_min      : float\n
            Lower limit for x
       * x_max      : float\n
            Upper limit for x
       * y_min      : float\n
            Lower limit for y
       * y_max      : float
            Upper limit for y \n
       """

    def __init__(self, max_points: int, x_min: float,
                 x_max: float, y_min: float, y_max: float):
        self._divided = False
        self._max_points = max_points
        self._n_points = 0
        self._xmin = x_min
        self._xmax = x_max
        self._ymin = y_min
        self._ymax = y_max
        self._points = np.empty(0, dtype=Point2d)
        self._upleft = None
        self._upright = None
        self._downleft = None
        self._downright = None
        self.sub_branches = {0: self._downleft,
                             1: self._downright,
                             2: self._upleft,
                             3: self._upright}

    def __eq__(self, branch):
        if self.is_divided() == True:
            eq = True
            if (branch.is_divided() == True):
                if (self.n_points() == branch.n_points() and
                    self.max_points() == branch.max_points() and
                    self.x_size() == branch.x_size() and
                        self.y_size() == branch.y_size()) == True:

                    eq = eq and True
                else:
                    return False

                eq = eq and self._downleft.__eq__(branch._downleft)
                eq = eq and self._downright.__eq__(branch._downright)
                eq = eq and self._upleft.__eq__(branch._upleft)
                eq = eq and self._upright.__eq__(branch._upright)

                return eq

            else:
                return False

        else:
            if (self.n_points() == 0):
                if (self.n_points() == branch.n_points() and
                    self.max_points() == branch.max_points() and
                    self.x_size() == branch.x_size() and
                        self.y_size() == branch.y_size()) == True:
                    return True
                else:
                    return False
            else:
                if (self.n_points() == branch.n_points() and
                    self.max_points() == branch.max_points() and
                    self.x_size() == branch.x_size() and
                    self.y_size() == branch.y_size() and
                        np.any(self.points_list() == branch.points_list()) == True) == True:
                    return True
                else:
                    return False

    def max_points(self) -> int:
        """Returns the current number of maximum points allowed inside the branch."""
        return self._max_points

    def n_points(self) -> int:
        """Returns the current number of points the branch has.

           NOTE: This method is different from total_points, as it only gives the current points
           stored in the branch, and not the points stored in the subbranches.
           If the branch is divided, it will return 0.
           """
        return self._n_points

    def is_divided(self) -> bool:
        """Returns True if the branch is divided, and False otherwise."""
        return self._divided

    def x_size(self) -> tuple:
        """Returns the x information of the branch, as a tuple."""
        return (self._xmin, self._xmax)

    def y_size(self) -> tuple:
        """Returns the y information of the branch, as a tuple."""
        return (self._ymin, self._ymax)

    def x_mid_point(self) -> float:
        """Returns the middle x point of the branch."""
        return self.x_size()[0] + (self.x_size()[1] -
                                   self.x_size()[0]) / float(2)

    def y_mid_point(self) -> float:
        """Method that returns the middle y point of the branch."""
        return self.y_size()[0] + (self.y_size()[1] -
                                   self.y_size()[0]) / float(2)

    def __str__(self):
        if self.is_divided() == True:
            str = f"number of points = {self.total_points()}\n"
            str += f"[ {self._upleft.n_points()} | {self._upright.n_points()} ]\n"
            str += "[---|---]\n"
            str += f"[ {self._downleft.n_points()} | {self._downright.n_points()} ]\n"
        else:
            str = f"number of points = {self._n_points}\n[       ]\n[       ]\n[       ]\n"
        return str

    def points_list(self):
        """Returns a list containing the points existing in that branch."""
        return self._points

    def sub_branch(self, index: int):
        """Returns the sub branch by index. Below, the index correspondence:
           0 : Down-left branch
           1 : Down-right branch
           2 : Up-left branch
           3 : Up-right branch\n
           Parameters
           ----------
           * index : int\n
                    Index of the branch requested
        """
        if self.is_divided() == True:
            return self.sub_branches[index]
        else:
            raise AttributeError("The branch got no subbranches.")

    def total_points(self) -> int:
        """Returns the total number of points in that branch, including the points in the subranches."""
        if self.is_divided() == True:

            sum = 0
            for i in range(len(self.sub_branches)):
                sum += self.sub_branch(i).total_points()

            return sum

        else:
            return self.n_points()

    def remove_point(self, index: int):
        """Removes the given element of the list of points\n
           Parameters
           ----------
           index : int\n
                Index of the point to be removed.
        """
        if self._n_points != 0:
            self._points = np.delete(self._points, index)
            self._n_points -= 1
        else:
            raise AttributeError("This branch has no point to be removed.")

    # def sub_add_point(self):

    def sub_add_point(self, point: Point2d):
        if self.x_size()[0] <= point()[0] < self.x_mid_point():
            if self.y_size()[0] <= point()[1] < self.y_mid_point():
                self._downleft.add_point(point)
            elif self.y_mid_point() <= point()[1] <= self.y_size()[1]:
                self._upleft.add_point(point)
            else:
                raise ValueError(
                    f"The point {point} is outside the tree's bounds : ({self.x_size()}, {self.y_size()}).")
        elif self.x_mid_point() <= point()[0] <= self.x_size()[1]:
            if self.y_size()[0] <= point()[1] < self.y_mid_point():
                self._downright.add_point(point)
            elif self.y_mid_point() <= point()[1] <= self.y_size()[1]:
                self._upright.add_point(point)
            else:
                raise ValueError(
                    f"The point {point} is outside the tree's bounds : ({self.x_size()}, {self.y_size()}).")
        else:
            raise ValueError(
                f"The point {point} is outside the tree's bounds : ({self.x_size()}, {self.y_size()}).")

    def add_point(self, point: Point2d):
        """Recursively adds a point to the branch.\n
           Parameters
           ----------
           point : np.array\n
                Point to be added.\n

           Below, how the method works:
           1. The branch is first checked if divided. If it is, the coordinates of
           the point is checked to add the point in one of the subbranches.
           2. If it is not, it checks if the total current number of points is lower than
           the branch's defined limit,
           3. If it is, adds the point to the branch.
           4. If it's not ,the branch is then divided and the coordinates of the point are checked to
           add the point in one of the newly-created subbranches.
        """

        if self.is_divided() == True:

            self.sub_add_point(point)

        else:

            if self.n_points() == self._max_points:
                self._downleft = Branch2d(
                    self._max_points,
                    self.x_size()[0],
                    self.x_mid_point(),
                    self.y_size()[0],
                    self.y_mid_point())
                self._downright = Branch2d(
                    self._max_points,
                    self.x_mid_point(),
                    self.x_size()[1],
                    self.y_size()[0],
                    self.y_mid_point())
                self._upleft = Branch2d(
                    self._max_points,
                    self.x_size()[0],
                    self.x_mid_point(),
                    self.y_mid_point(),
                    self.y_size()[1])
                self._upright = Branch2d(
                    self._max_points,
                    self.x_mid_point(),
                    self.x_size()[1],
                    self.y_mid_point(),
                    self.y_size()[1])
                self._divided = True

                self.sub_branches = {0: self._downleft,
                                     1: self._downright,
                                     2: self._upleft,
                                     3: self._upright}

                list = self.points_list()
                for i in range(self.n_points()):
                    self.sub_add_point(list[i])
                    self.remove_point(0)

                self.sub_add_point(point)

            else:
                self._points = np.append(self._points, point)
                self._n_points += 1

    def process_branch(self, function, *args):
        """Recursively process the branch and its subbranches with a given function and its arguments.
           If the function returns any value, it will be returned as the value itself or a list of the
           values returned by each subbranch processed, if the branch is already divided.\n
           Parameters
           ----------
           function : Any\n
                function that has only a branch as input\n
           *args : Any\n
                Arguments of the function in the order requested by the function passed
        """
        if self.is_divided() == True:
            list = np.empty(0)

            for i in range(len(self.sub_branches)):
                list = np.append(
                    list, self.sub_branch(i).process_branch(
                        function, *args))

            return list

        else:
            return function(self, *args)

    def all_points_list(self):
        """Returns a list with all the point inside a branch, including its subbranches."""
        if self.is_divided() == True:
            list = np.empty(0)

            for i in range(len(self.sub_branches)):
                list = np.append(
                    list, self.sub_branch(i).all_points_list())

            return list

        else:
            return self.points_list()

    def update(self):
        """Recursively checks if all the points in the branch and its subbranches still belong there.
           Returns an array with all the points that had positions out of the branch's bounds.
        """
        return_list = np.empty(0)
        if self.is_divided() == True:
            update_points = np.empty(0)
            for i in range(len(self.sub_branches)):
                update_points = np.append(
                    update_points, self.sub_branch(i).update())

            removed = 0
            remain_points = np.copy(update_points)
            for i in range(update_points.size):
                if ((self.x_size()[0] > update_points[i]()[0]) or
                    (update_points[i]()[0] > self.x_size()[1]) or
                    (self.y_size()[0] > update_points[i]()[1]) or
                        (update_points[i]()[1] > self.y_size()[1])):
                    return_list = np.append(return_list, update_points[i])
                    remain_points = np.delete(remain_points, i - removed)
                    removed += 1

            if self.total_points() <= self._max_points:
                remain_points = np.append(
                    remain_points, self.all_points_list())
                self._divided = False
                self._upleft = None
                self._upright = None
                self._downleft = None
                self._downright = None
                self.sub_branches = {0: self._downleft,
                                     1: self._downright,
                                     2: self._upleft,
                                     3: self._upright}

            for i in range(remain_points.shape[0]):
                self.add_point(remain_points[i])

            return return_list

        else:
            list = self.points_list()
            removed = 0
            for i in range(list.size):
                if ((self.x_size()[0] > list[i]()[0]) or
                    (list[i]()[0] > self.x_size()[1]) or
                    (self.y_size()[0] > list[i]()[1]) or
                        (list[i]()[1] > self.y_size()[1])):
                    self.remove_point(i - removed)
                    removed += 1
                    return_list = np.append(return_list, list[i])

            return return_list


class Tree2d():
    """Class that stores the root branch and general 2d tree information.\n
       Parameters
       ----------
       branch : Branch2d\n
            Branch to be the root of the tree.
    """

    def __init__(self, branch: Branch2d):
        self._root = branch
        self._xmin = branch.x_size()[0]
        self._xmax = branch.x_size()[1]
        self._ymin = branch.y_size()[0]
        self._ymax = branch.y_size()[1]
        self._n_points = branch.total_points()
        self._str = ""

    def x_size(self):
        """Returns the x size of the tree, as a tuple."""
        return (self._xmin, self._xmax)

    def y_size(self):
        """Returns the y size of the tree, as a tuple."""
        return (self._ymin, self._ymax)

    def n_points(self):
        """Returns the number of points inside the tree."""
        return self._n_points

    def root(self):
        """Returns the root branch of the tree."""
        return self._root

    def __str__(self):
        if self._root.is_divided() == True:
            points_list = []
            for i in range(len(self.root().sub_branches)):
                points_list.append(self.root().sub_branch(i).total_points())
            length = [len(str(x)) for x in points_list]
            length = [np.max([length[j], length[j + 2]])
                      for j in range(len(points_list) // 2)]
            self._str = f"Number of points = {self._root.total_points()}\n"
            self._str += f"[ {self._root._upleft.total_points()} | {self._root._upright.total_points()} ]\n"
            self._str += "[---" + "-" * (length[0] - 1) + \
                "|---" + "-" * (length[1] - 1) + f"]\n"
            self._str += f"[ {self._root._downleft.total_points()} | {self._root._downright.total_points()} ]\n"
        else:
            self._str = f"Number of points = {self._n_points}\n[       ]\n[       ]\n[       ]\n"

        return self._str

    def add_point(self, point: Point2d):
        """Adds a point to the tree.\n
        Parameters
        ----------
        point : np.array\n
            Point to be added into the tree."""
        self._root.add_point(point)
        self._n_points += 1
# END OF QUADTREE IMPLEMENTATION

# OCTREE IMPLEMENTATION


class Point3d(Point2d):
    """General point class. This class only stores 3d coordinates,
       and is to be used as base for any class that intends to use the octree structure.
    """

    def __init__(self, x: float, y: float, z: float):
        self.coord = np.array([x, y, z])

    def get_z_coord(self):
        """Returns the z coordinate of the point."""
        return self.coord[2]

    def set_coord(self, x: float, y: float, z: float):
        """Updates the coordinate of the point.
        ## Parameters
           * x : float\n
                   x position
           * y : float\n
                   y position
           * z : float\n
                  z position"""
        self.coord = np.array([x, y, z])


class Branch3d(Branch2d):
    """General 3d branch class. It is the base structure for building an octree.\n
       Parameters
       ----------
       max_points : int\n
            Defining the maximum number of points this branch stores\n
       x_min      : float\n
            Lower limit for x\n
       x_max      : float\n
            Upper limit for x\n
       y_min      : float\n
            Lower limit for y\n
       y_max      : float\n
            Upper limit for y\n
       z_min      : float\n
            Front limit for z\n
       z_max      : float\n
            Back limit for z \n
       This branch, when divided, has eight branches: \n
       1. the front-down-left
       2. the front-down-right
       3. the front-up-left
       4. the front-up-right
       5. the back-down-left
       6. the back-down-right
       7. the back-up-left
       8. the back-up-right
    """

    def __init__(self,
                 max_points: int,
                 x_min: float,
                 x_max: float,
                 y_min: float,
                 y_max: float,
                 z_min: float,
                 z_max: float):
        self._divided = False
        self._max_points = max_points
        self._n_points = 0
        self._xmin = x_min
        self._xmax = x_max
        self._ymin = y_min
        self._ymax = y_max
        self._zmin = z_min
        self._zmax = z_max
        self._points = np.empty(0, dtype=Point3d)
        self._front_up_left = None
        self._front_up_right = None
        self._front_down_left = None
        self._front_down_right = None
        self._back_up_left = None
        self._back_up_right = None
        self._back_down_left = None
        self._back_down_right = None
        self.sub_branches = {0: self._front_down_left,
                             1: self._front_down_right,
                             2: self._front_up_left,
                             3: self._front_up_right,
                             4: self._back_down_left,
                             5: self._back_down_right,
                             6: self._back_up_left,
                             7: self._back_up_right}

    def __eq__(self, branch):
        if self.is_divided() == True:
            eq = True
            if (branch.is_divided() == True):
                if (self.n_points() == branch.n_points() and
                    self.max_points() == branch.max_points() and
                    self.x_size() == branch.x_size() and
                    self.y_size() == branch.y_size() and
                        self.z_size() == branch.z_size()) == True:

                    eq = eq and True
                else:
                    return False

                eq = eq and self._front_down_left.__eq__(
                    branch._front_down_left)
                eq = eq and self._front_down_right.__eq__(
                    branch._front_down_right)
                eq = eq and self._front_up_left.__eq__(branch._front_up_left)
                eq = eq and self._front_up_right.__eq__(branch._front_up_right)
                eq = eq and self._back_down_left.__eq__(branch._back_down_left)
                eq = eq and self._back_down_right.__eq__(
                    branch._back_down_right)
                eq = eq and self._back_up_left.__eq__(branch._back_up_left)
                eq = eq and self._back_up_right.__eq__(branch._back_up_right)

                return eq

            else:
                return False

        else:
            if (self.n_points() == 0):
                if (self.n_points() == branch.n_points() and
                    self.max_points() == branch.max_points() and
                    self.x_size() == branch.x_size() and
                    self.y_size() == branch.y_size() and
                        self.z_size() == branch.z_size()) == True:
                    return True
                else:
                    return False
            else:
                if (self.n_points() == branch.n_points() and
                    self.max_points() == branch.max_points() and
                    self.x_size() == branch.x_size() and
                    self.y_size() == branch.y_size() and
                    self.z_size() == branch.z_size() and
                        np.any(self.points_list() == branch.points_list()) == True) == True:
                    return True
                else:
                    return False

    def z_size(self) -> tuple:
        """Returns the y information of the branch, as a tuple."""
        return (self._zmin, self._zmax)

    def z_mid_point(self) -> float:
        """Returns the middle z point of the branch."""
        return self.z_size()[0] + (self.z_size()[1] -
                                   self.z_size()[0]) / float(2)

    def __str__(self):
        if self.is_divided() == True:
            str = f"number of points = {self.total_points()}\n"
            str += "  FRONT\n"
            str += f"[ {self._front_up_left.n_points()} | {self._front_up_right.n_points()} ]\n"
            str += "[---|---]\n"
            str += f"[ {self._front_down_left.n_points()} | {self._front_down_right.n_points()} ]\n"
            str += "| BACK\n"
            str += f"[ {self._back_up_left.n_points()} | {self._back_up_right.n_points()} ]\n"
            str += "[---|---]\n"
            str += f"[ {self._back_down_left.n_points()} | {self._back_down_right.n_points()} ]\n"
        else:
            str = f"number of points = {self._n_points}\n| FRONT\n[       ]\n[       ]\n[       ]\n| BACK\n[       ]\n[       ]\n[       ]\n"
        return str

    def sub_branch(self, index: int):
        """Returns the sub branch by index. Below, the index correspondence:
           0 : Front-down-left branch
           1 : Front-down-right branch
           2 : Front-up-left branch
           3 : Front-up-right branch
           4 : Back-down-left branch
           5 : Back-down-right branch
           6 : Back-up-left branch
           7 : Back-up-right branch\n
           Parameters
           ----------
           index : int\n
                index of the branch requested
        """
        if self.is_divided() == True:
            return self.sub_branches[index]
        else:
            raise AttributeError("The branch got no subbranches.")

    def sub_add_point(self, point: Point3d):
        if self.x_size()[0] <= point()[0] < self.x_mid_point():
            if self.y_size()[0] <= point()[1] < self.y_mid_point():
                if self.z_size()[0] <= point()[2] < self.z_mid_point():
                    self._front_down_left.add_point(point)
                elif self.z_mid_point() <= point()[2] <= self.z_size()[1]:
                    self._back_down_left.add_point(point)
                else:
                    raise ValueError(
                        f"The point {point} is outside the tree's bounds : ({self.x_size()}, {self.y_size()}, {self.z_size()}).")

            elif self.y_mid_point() <= point()[1] <= self.y_size()[1]:
                if self.z_size()[0] <= point()[2] < self.z_mid_point():
                    self._front_up_left.add_point(point)
                elif self.z_mid_point() <= point()[2] <= self.z_size()[1]:
                    self._back_up_left.add_point(point)
            else:
                raise ValueError(
                    f"The point {point} is outside the tree's bounds : ({self.x_size()}, {self.y_size()}, {self.z_size()}).")
        elif self.x_mid_point() <= point()[0] <= self.x_size()[1]:
            if self.y_size()[0] <= point()[1] < self.y_mid_point():
                if self.z_size()[0] <= point()[2] < self.z_mid_point():
                    self._front_down_right.add_point(point)
                elif self.z_mid_point() <= point()[2] <= self.z_size()[1]:
                    self._back_down_right.add_point(point)
                else:
                    raise ValueError(
                        f"The point {point} is outside the tree's bounds : ({self.x_size()}, {self.y_size()}, {self.z_size()}).")
            elif self.y_mid_point() <= point()[1] <= self.y_size()[1]:
                if self.z_size()[0] <= point()[2] < self.z_mid_point():
                    self._front_up_right.add_point(point)
                elif self.z_mid_point() <= point()[2] <= self.z_size()[1]:
                    self._back_up_right.add_point(point)
                else:
                    raise ValueError(
                        f"The point {point} is outside the tree's bounds : ({self.x_size()}, {self.y_size()}, {self.z_size()}).")

        else:
            raise ValueError(
                f"The point {point} is outside the tree's bounds : ({self.x_size()}, {self.y_size()}, {self.z_size()}).")

    def add_point(self, point: Point3d):
        """Recursively adds a point to the branch.\n
           Parameters
           ----------
           point : np.array\n
                Point to be added.\n

           Below, how the method works:
           1. The branch is first checked if divided. If it is, the coordinates of
           the point is checked to add the point in one of the subbranches.
           2. If it is not, it checks if the total current number of points is lower than
           the branch's defined limit,
           3. If it is, adds the point to the branch.
           4. If it's not ,the branch is then divided and the coordinates of the point are checked to
           add the point in one of the newly-created subbranches.
        """
        if self.is_divided() == True:

            self.sub_add_point(point)

        else:

            if self.n_points() == self._max_points:
                self._front_down_left = Branch3d(
                    self._max_points,
                    self.x_size()[0],
                    self.x_mid_point(),
                    self.y_size()[0],
                    self.y_mid_point(),
                    self.z_size()[0],
                    self.z_mid_point())
                self._front_down_right = Branch3d(
                    self._max_points,
                    self.x_mid_point(),
                    self.x_size()[1],
                    self.y_size()[0],
                    self.y_mid_point(),
                    self.z_size()[0],
                    self.z_mid_point())
                self._front_up_left = Branch3d(
                    self._max_points,
                    self.x_size()[0],
                    self.x_mid_point(),
                    self.y_mid_point(),
                    self.y_size()[1],
                    self.z_size()[0],
                    self.z_mid_point())
                self._front_up_right = Branch3d(
                    self._max_points,
                    self.x_mid_point(),
                    self.x_size()[1],
                    self.y_mid_point(),
                    self.y_size()[1],
                    self.z_size()[0],
                    self.z_mid_point())
                self._back_down_left = Branch3d(
                    self._max_points,
                    self.x_size()[0],
                    self.x_mid_point(),
                    self.y_size()[0],
                    self.y_mid_point(),
                    self.z_mid_point(),
                    self.z_size()[1])
                self._back_down_right = Branch3d(
                    self._max_points,
                    self.x_mid_point(),
                    self.x_size()[1],
                    self.y_size()[0],
                    self.y_mid_point(),
                    self.z_mid_point(),
                    self.z_size()[1])
                self._back_up_left = Branch3d(
                    self._max_points,
                    self.x_size()[0],
                    self.x_mid_point(),
                    self.y_mid_point(),
                    self.y_size()[1],
                    self.z_mid_point(),
                    self.z_size()[1])
                self._back_up_right = Branch3d(
                    self._max_points,
                    self.x_mid_point(),
                    self.x_size()[1],
                    self.y_mid_point(),
                    self.y_size()[1],
                    self.z_mid_point(),
                    self.z_size()[1])
                self._divided = True

                self.sub_branches = {0: self._front_down_left,
                                     1: self._front_down_right,
                                     2: self._front_up_left,
                                     3: self._front_up_right,
                                     4: self._back_down_left,
                                     5: self._back_down_right,
                                     6: self._back_up_left,
                                     7: self._back_up_right}

                list = self.points_list()
                for i in range(self.n_points()):

                    self.sub_add_point(list[i])
                    self.remove_point(0)

                self.sub_add_point(point)

            else:
                self._points = np.append(self._points, point)
                self._n_points += 1

    def update(self):
        """Recursively checks if all the points in the branch and its subbranches still belong there.
           Returns an array with all the points that had positions out of the branch's bounds.
        """
        return_list = np.empty(0)
        if self.is_divided() == True:
            update_points = np.empty(0)
            for i in range(len(self.sub_branches)):
                update_points = np.append(
                    update_points, self.sub_branch(i).update())

            removed = 0
            remain_points = np.copy(update_points)
            for i in range(update_points.size):
                if ((self.x_size()[0] > update_points[i]()[0]) or
                    (update_points[i]()[0] > self.x_size()[1]) or
                    (self.y_size()[0] > update_points[i]()[1]) or
                    (update_points[i]()[1] > self.y_size()[1]) or
                    (self.z_size()[0] > update_points[i]()[2]) or
                        (update_points[i]()[2] > self.z_size()[1])):
                    return_list = np.append(return_list, update_points[i])
                    remain_points = np.delete(remain_points, i - removed)
                    removed += 1

            if self.total_points() <= self._max_points:
                remain_points = np.append(
                    remain_points, self.all_points_list())
                self._divided = False
                self._front_down_left = None
                self._front_down_right = None
                self._front_up_left = None
                self._front_up_right = None
                self._back_down_left = None
                self._back_down_right = None
                self._back_up_left = None
                self._back_up_right = None
                self.sub_branches = {0: self._front_down_left,
                                     1: self._front_down_right,
                                     2: self._front_up_left,
                                     3: self._front_up_right,
                                     4: self._back_down_left,
                                     5: self._back_down_right,
                                     6: self._back_up_left,
                                     7: self._back_up_right}

            for i in range(remain_points.shape[0]):
                self.add_point(remain_points[i])

            return return_list

        else:
            list = self.points_list()
            removed = 0
            for i in range(list.size):
                if ((self.x_size()[0] > list[i]()[0]) or
                    (list[i]()[0] > self.x_size()[1]) or
                            (self.y_size()[0] > list[i]()[1]) or
                        (list[i]()[1] > self.y_size()[1]) or
                        (self.z_size()[0] > list[i]()[2]) or
                    (list[i]()[2] > self.z_size()[1])
                    ):
                    self.remove_point(i - removed)
                    removed += 1
                    return_list = np.append(return_list, list[i])

            return return_list


class Tree3d(Tree2d):
    """Class that stores the root branch and general 3d tree information.\n
        Parameters
        ----------
        branch : Branch3d\n
            Branch to be the root of the tree.
    """

    def __init__(self, branch: Branch3d):
        self._root = branch
        self._xmin = branch.x_size()[0]
        self._xmax = branch.x_size()[1]
        self._ymin = branch.y_size()[0]
        self._ymax = branch.y_size()[1]
        self._zmin = branch.z_size()[0]
        self._zmax = branch.z_size()[1]
        self._n_points = branch.total_points()
        self._str = ""

    def z_size(self):
        return (self._zmin, self._zmax)

    def __str__(self):
        if self._root.is_divided() == True:
            points_list = []
            for i in range(len(self.root().sub_branches)):
                points_list.append(self.root().sub_branch(i).total_points())
            length = [len(str(x)) for x in points_list]
            length = [np.max([length[j], length[j + 2]])
                      for j in range(len(points_list) // 2)]
            self._str = f"Number of points = {self._root.total_points()}\n"
            self._str += "| FRONT\n"
            self._str += f"[ {self._root._front_up_left.total_points()} | {self._root._front_up_right.total_points()} ]\n"
            self._str += "[---" + "-" * \
                (length[0] - 1) + "|---" + "-" * (length[1] - 1) + "]\n"
            self._str += f"[ {self._root._front_down_left.total_points()} | {self._root._front_down_right.total_points()} ]\n"
            self._str += "| BACK\n"
            self._str += f"[ {self._root._back_up_left.total_points()} | {self._root._back_up_right.total_points()} ]\n"
            self._str += "[---" + "-" * \
                (length[2] - 1) + "|---" + "-" * (length[3] - 1) + "]\n"
            self._str += f"[ {self._root._back_down_left.total_points()} | {self._root._back_down_right.total_points()} ]\n"
        else:
            self._str = f"Number of points = {self._root.total_points()}\n  FRONT\n\
                        [       ]\n\
                        [       ]\n\
                        [       ]\n\
                        | BACK\n\
                        [       ]\n\
                        [       ]\n\
                        [       ]\n"
        return self._str

# END OF OCTREE IMPLEMENTATION


# GRAPHICAL IMPLEMENTATION
def bounding_box_3d(center: tuple = (0.0, 0.0, 0.0),
                    size: tuple = (1.0, 1.0, 1.0),
                    color: tuple = (1.0, 1.0, 1.0, 1.0),
                    line_width: float = 1.0):
    """Creates a bounding box with the parameters given. The box got only is edges renderized.\n
        Parameters
        ----------
        center : tuple\n
            x, y, and z, that determines where is the center of the box.\n
        size : tuple\n
            x, y, and z, that determines its lateral sizes.\n
        color : tuple\n
            r, g, b and a, that determines the color of the bounding box.\n
        line_width : float\n
            Determines the width of the box lines.
    """
    x_c = center[0]
    y_c = center[1]
    z_c = center[2]

    x_l = size[0] / 2
    y_l = size[1] / 2
    z_l = size[2] / 2

    vertices = np.array([
        [x_c - x_l, y_c - y_l, z_c - z_l],
        [x_c + x_l, y_c - y_l, z_c - z_l],
        [x_c + x_l, y_c + y_l, z_c - z_l],
        [x_c - x_l, y_c + y_l, z_c - z_l],
        [x_c - x_l, y_c - y_l, z_c + z_l],
        [x_c + x_l, y_c - y_l, z_c + z_l],
        [x_c + x_l, y_c + y_l, z_c + z_l],
        [x_c - x_l, y_c + y_l, z_c + z_l]
    ])

    data = np.array([
        [vertices[0], vertices[4]],
        [vertices[3], vertices[7]],
        [vertices[1], vertices[5]],
        [vertices[2], vertices[6]],
        [vertices[0], vertices[1]],
        [vertices[3], vertices[2]],
        [vertices[7], vertices[6]],
        [vertices[4], vertices[5]],
        [vertices[0], vertices[3]],
        [vertices[1], vertices[2]],
        [vertices[4], vertices[7]],
        [vertices[5], vertices[6]]
    ])

    lines = line(data, colors=color)
    lines.GetProperty().SetLineWidth(line_width)
    lines.GetProperty().SetLighting(False)

    return lines


def actor_from_branch_2d(branch: Branch2d, color=(
        1.0, 1.0, 1.0), linewidth=1.0) -> Actor:
    """Recursively creates actors for the branch given. If the branch is divided,
       then the function is run for the subbranches until the function reaches a non-divided branch,
       that creates the actor to be returned. This actor is then appended into a list, that is then returned.
       NOTE: This returns a 3d actor.\n
        Parameters
        ----------
       branch : Branch3d\n
            Branch that will have the actor created.
    """

    if branch.is_divided() == True:
        actors = np.empty(0, dtype=Actor)
        actors = np.append(
            actors,
            actor_from_branch_2d(
                branch.sub_branch(0),
                color,
                linewidth))
        actors = np.append(
            actors,
            actor_from_branch_2d(
                branch.sub_branch(1),
                color,
                linewidth))
        actors = np.append(
            actors,
            actor_from_branch_2d(
                branch.sub_branch(2),
                color,
                linewidth))
        actors = np.append(
            actors,
            actor_from_branch_2d(
                branch.sub_branch(3),
                color,
                linewidth))

        return actors

    else:

        x_c = branch.x_mid_point()
        y_c = branch.y_mid_point()
        z_c = 0.0

        x_l = (branch.x_size()[1] - branch.x_size()[0])
        y_l = (branch.y_size()[1] - branch.y_size()[0])
        z_l = 0.0

        cubeActor = bounding_box_3d(
            (x_c, y_c, z_c), (x_l, y_l, z_l), color, linewidth)

        return cubeActor


def actor_from_branch_3d(branch: Branch3d, color=(
        1.0, 1.0, 1.0), linewidth=1.0) -> Actor:
    """Recursively creates actors for the branch given.
       If the branch is divided, then the function is run for the subbranches
       until the function reaches a non-divided branch, that creates the actor to be returned.
       This actor is then appended into a list, that is then returned.\n
       Parameters
       ----------
       branch : Branch3d\n
            Branch that will have the actor created.
    """

    if branch.is_divided() == True:
        actors = np.empty(0, dtype=Actor)
        actors = np.append(
            actors,
            actor_from_branch_3d(
                branch.sub_branch(0),
                color,
                linewidth))
        actors = np.append(
            actors,
            actor_from_branch_3d(
                branch.sub_branch(1),
                color,
                linewidth))
        actors = np.append(
            actors,
            actor_from_branch_3d(
                branch.sub_branch(2),
                color,
                linewidth))
        actors = np.append(
            actors,
            actor_from_branch_3d(
                branch.sub_branch(3),
                color,
                linewidth))
        actors = np.append(
            actors,
            actor_from_branch_3d(
                branch.sub_branch(4),
                color,
                linewidth))
        actors = np.append(
            actors,
            actor_from_branch_3d(
                branch.sub_branch(5),
                color,
                linewidth))
        actors = np.append(
            actors,
            actor_from_branch_3d(
                branch.sub_branch(6),
                color,
                linewidth))
        actors = np.append(
            actors,
            actor_from_branch_3d(
                branch.sub_branch(7),
                color,
                linewidth))

        return actors

    else:

        x_c = branch.x_mid_point()
        y_c = branch.y_mid_point()
        z_c = branch.z_mid_point()

        x_l = (branch.x_size()[1] - branch.x_size()[0])
        y_l = (branch.y_size()[1] - branch.y_size()[0])
        z_l = (branch.z_size()[1] - branch.z_size()[0])

        cubeActor = bounding_box_3d(
            (x_c, y_c, z_c), (x_l, y_l, z_l), color, linewidth)

        return cubeActor
# END OF GRAPH IMPLEMENTATION
