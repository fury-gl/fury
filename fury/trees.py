from fury.utils import Actor
from fury.actor import line
import numpy as np


# QUADTREE IMPLEMENTATION
class Point2d():
    '''General point class. This class only stores 2d coordinates.'''

    def __init__(self, x : float, y: float):
        self.coord = np.array([x, y])

    def __call__(self):
        return self.coord

    def GetXCoord(self) -> float:
        '''Returns the x coordinate of the point.'''
        return self.coord[0]

    def GetYCoord(self) -> float:
        '''Returns the y coordinate of the point.'''
        return self.coord[1]

    def GetNorm(self) -> float:
        '''Returns the norm of the point in relation to the origin.'''
        return np.sqrt(self.GetXCoord()**2 + self.GetYCoord()**2)

    def SetCoord(self, x : float, y : float) -> np.array:
        '''Updates the coordinate of the point.
           * x : float x position
           * y : float y position'''
        self.coord = np.array([x, y])

    def __str__(self):
        string = "[" + str('%.3f' %
                           self.coord[0]) + ", " + str('%.3f' %
                                                       self.coord[1]) + "]"
        return string

    def __repr__(self):
        string = "[" + str('%.3f' %
                           self.coord[0]) + ", " + str('%.3f' %
                                                       self.coord[1]) + "]"
        return string


class Branch2d():
    '''General branch class. It is the base structure for building a quadtree.\n
       * max_points : An integer defining the maximum number of points this branch stores
       * x_min      : The float lower limit for x
       * x_max      : The float upper limit for x
       * y_min      : The float lower limit for y
       * y_max      : The float upper limit for y \n
       This branch, when divided, has four branches: \n
       1. the down-left
       2. the down-right
       3. the up-left
       4. the up-right'''

    def __init__(self, max_points : int, x_min : float,
                 x_max : float, y_min : float, y_max : float):
        self._DIVIDED = False
        self._MAX_POINTS = max_points
        self._N_POINTS = 0
        self._xmin = x_min
        self._xmax = x_max
        self._ymin = y_min
        self._ymax = y_max
        self._POINTS = np.array([], dtype=Point2d)
        self._UPLEFT = None
        self._UPRIGHT = None
        self._DOWNLEFT = None
        self._DOWNRIGHT = None
        self.subBranches = {0 : self._DOWNLEFT,
                            1 : self._DOWNRIGHT,
                            2 : self._UPLEFT,
                            3 : self._UPRIGHT}

    def GetPointsNumber(self) -> int:
        '''Returns the current number of points the branch has.
           This method is different from GetTotalPoints, as it only gives the current points
           stored in the branch, and not the points stored in the subbranches.
           If the branch is divided, it will return 0.'''
        return self._N_POINTS

    def IsDivided(self) -> bool:
        '''Returns True if the branch is divided, and False otherwise.'''
        return self._DIVIDED

    def GetXSize(self) -> tuple:
        '''Returns the x information of the branch, as a tuple.'''
        return (self._xmin, self._xmax)

    def GetYSize(self) -> tuple:
        '''Returns the y information of the branch, as a tuple.'''
        return (self._ymin, self._ymax)

    def GetXMiddlePoint(self) -> float:
        '''Returns the middle x point of the branch.'''
        return self.GetXSize()[0] + (self.GetXSize()[1] - self.GetXSize()[0])/float(2)

    def GetYMiddlePoint(self) -> float:
        '''Method that returns the middle y point of the branch.'''
        return self.GetYSize()[0] + (self.GetYSize()[1] - self.GetYSize()[0])/float(2)

    def __str__(self):
        if self.IsDivided() == True:
            str = f"number of points = {self.GetTotalPoints()}\n\
                    [ {self._UPLEFT.GetPointsNumber()} | {self._UPRIGHT.GetPointsNumber()} ]\n\
                    [---|---]\n\
                    [ {self._DOWNLEFT.GetPointsNumber()} | {self._DOWNRIGHT.GetPointsNumber()} ]\n"
        else:
            str = f"number of points = {self._N_POINTS}\n[       ]\n[       ]\n[       ]\n"
        return str

    def GetPointsList(self) -> np.array:
        '''Returns a list containing the points existing in that branch.'''
        return self._POINTS

    def GetSubBranch(self, index : int):
        '''Returns the sub branch by index. Below, the index correspondence:
           0 : Down-left branch
           1 : Down-right branch
           2 : Up-left branch
           3 : Up-right branch

           * index : int index of the branch requested'''
        if self.IsDivided() == True:
            return self.subBranches[index]
        else:
            raise AttributeError("The branch got no subbranches.")

    def GetTotalPoints(self) -> int:
        '''Returns the total number of points in that branch, including the points in the subranches.'''
        if self.IsDivided() == True:

            sum = 0
            for i in range(len(self.subBranches)):
                sum += self.GetSubBranch(i).GetTotalPoints()

            return sum

        else:
            return self.GetPointsNumber()

    def RemovePoint(self, index : int):
        '''Removes the given element of the list of points'''
        if self._N_POINTS != 0:
            self._POINTS = np.delete(self._POINTS, index)
            self._N_POINTS -= 1
        else:
            raise AttributeError("This branch has no point to be removed.")

    # def SubAddPoint(self):

    def SubAddPoint(self, point : Point2d):
        if self.GetXSize()[0] <= point.GetXCoord() < self.GetXMiddlePoint():
            if self.GetYSize()[0] <= point.GetYCoord() < self.GetYMiddlePoint():
                self._DOWNLEFT.AddPoint(point)
            elif self.GetYMiddlePoint() <= point.GetYCoord() <= self.GetYSize()[1]:
                self._UPLEFT.AddPoint(point)
            else:
                raise ValueError(
                    f"The point {point()} is outside the tree's bounds : ({self.GetXSize()}, {self.GetYSize()}).")
        elif self.GetXMiddlePoint() <= point.GetXCoord() <= self.GetXSize()[1]:
            if self.GetYSize()[0] <= point.GetYCoord() < self.GetYMiddlePoint():
                self._DOWNRIGHT.AddPoint(point)
            elif self.GetYMiddlePoint() <= point.GetYCoord() <= self.GetYSize()[1]:
                self._UPRIGHT.AddPoint(point)
            else:
                raise ValueError(
                    f"The point {point()} is outside the tree's bounds : ({self.GetXSize()}, {self.GetYSize()}).")
        else:
            raise ValueError(
                f"The point {point()} is outside the tree's bounds : ({self.GetXSize()}, {self.GetYSize()}).")

    def AddPoint(self, point : Point2d):
        '''Recursively adds a point to the branch. \n
           * point : Point2d to be added.\n

           Below, how the method works:
           1. The branch is first checked if divided. If it is, the coordinates of
           the point is checked to add the point in one of the subbranches.
           2. If it is not, it checks if the total current number of points is lower than
           the branch's defined limit,
           3. If it is, adds the point to the branch.
           4. If it's not ,the branch is then divided and the coordinates of the point are checked to
           add the point in one of the newly-created subbranches.'''

        if self.IsDivided() == True:

            self.SubAddPoint(point)

        else:

            if self.GetPointsNumber() == self._MAX_POINTS:
                self._DOWNLEFT = Branch2d(
                    self._MAX_POINTS,
                    self.GetXSize()[0],
                    self.GetXMiddlePoint(),
                    self.GetYSize()[0],
                    self.GetYMiddlePoint())
                self._DOWNRIGHT = Branch2d(
                    self._MAX_POINTS,
                    self.GetXMiddlePoint(),
                    self.GetXSize()[1],
                    self.GetYSize()[0],
                    self.GetYMiddlePoint())
                self._UPLEFT = Branch2d(
                    self._MAX_POINTS,
                    self.GetXSize()[0],
                    self.GetXMiddlePoint(),
                    self.GetYMiddlePoint(),
                    self.GetYSize()[1])
                self._UPRIGHT = Branch2d(
                    self._MAX_POINTS,
                    self.GetXMiddlePoint(),
                    self.GetXSize()[1],
                    self.GetYMiddlePoint(),
                    self.GetYSize()[1])
                self._DIVIDED = True

                self.subBranches = {0 : self._DOWNLEFT,
                                    1 : self._DOWNRIGHT,
                                    2 : self._UPLEFT,
                                    3 : self._UPRIGHT}

                list = self.GetPointsList()
                for i in range(self.GetPointsNumber()):

                    self.SubAddPoint(list[i])
                    self.RemovePoint(0)

                self.SubAddPoint(point)

            else:
                self._POINTS = np.append(self._POINTS, point)
                self._N_POINTS += 1

    def ProcessBranch(self, Function, *args):
        '''Recursively process the branch or its subbranches with a given function and its arguments.
           If the function returns any value, it will be returned as the value itself or a list of the
           values returned by each subbranch processed, if the branch is already divided.
           * Function : Any function that has only a branch as input
           * *args : arguments of the function in the order requested by the funciton passed'''
        if self.IsDivided() == True:
            list = np.array([])

            for i in range(len(self.subBranches)):
                list = np.append(
                    list, self.GetSubBranch(i).ProcessBranch(
                        Function, *args))

            return list

        else:
            return Function(self, *args)


class Tree2d():
    '''Class that stores the root branch and general 2d tree informations.
       * branch : Branch2d to be the root of the tree.'''

    def __init__(self, branch : Branch2d):
        self._ROOT = branch
        self.xmin = branch.GetXSize()[0]
        self.xmax = branch.GetXSize()[1]
        self.ymin = branch.GetYSize()[0]
        self.ymax = branch.GetYSize()[1]
        self._N_POINTS = branch.GetTotalPoints()
        self._STR = ""

    def __str__(self):
        if self._ROOT.IsDivided() == True:
            points_list = []
            for i in range(len(self.GetRoot().subBranches)):
                points_list.append(self.GetRoot().GetSubBranch(i).GetTotalPoints())
            length = [len(str(x)) for x in points_list]
            length = [np.max([length[j], length[j + 2]])
                      for j in range(len(points_list)//2)]
            self._STR = f"number of points = {self._ROOT.GetTotalPoints()}\n"
            self._STR += f"[ {self._ROOT._UPLEFT.GetTotalPoints()} | {self._ROOT._UPRIGHT.GetTotalPoints()} ]\n"
            self._STR += "[---" + "-"*(length[0] - 1) + \
                "|---" + "-"*(length[1] - 1) + f"]\n"
            self._STR += f"[ {self._ROOT._DOWNLEFT.GetTotalPoints()} | {self._ROOT._DOWNRIGHT.GetTotalPoints()} ]\n"
        else:
            self._STR = f"number of points = {self._N_POINTS}\n[       ]\n[       ]\n[       ]\n"

        return self._STR

    def GetRoot(self):
        '''Returns the root branch of the tree.'''
        return self._ROOT

    def AddPoint(self, point : Point2d):
        '''Adds a point to the tree.
          * point : Point2d to be added into the tree.'''
        self._ROOT.AddPoint(point)
        self._N_POINTS += 1
# END OF QUADTREE IMPLEMENTATION

# OCTREE IMPLEMENTATION


class Point3d(Point2d):
    '''General point class. This class only stores 3d coordinates.'''

    def __init__(self, x: float, y: float, z: float):
        self.coord = np.array([x, y, z])

    def __call__(self):
        return self.coord

    def GetZCoord(self):
        '''Returns the z coordinate of the point.'''
        return self.coord[2]

    def GetNorm(self):
        '''Returns the norm of the point in relation to the origin.'''
        return np.sqrt(self.GetXCoord()**2 + self.GetYCoord()**2 + self.GetZCoord()**2)

    def SetCoord(self, x: float, y: float, z : float):
        '''Updates the coordinate of the point.
           * x : float x position
           * y : float y position
           * z : float z position'''
        self.coord = np.array([x, y, z])

    def __str__(self):
        string = "[" + str('%.3f' %
                           self.coord[0]) + ", " + str('%.3f' %
                                                       self.coord[1]) + str('%.3f' %
                                                                            self.coord[2]) + "]"
        return string

    def __repr__(self):
        string = "[" + str('%.3f' %
                           self.coord[0]) + ", " + str('%.3f' %
                                                       self.coord[1]) + str('%.3f' %
                                                                            self.coord[2]) + "]"
        return string


class Branch3d(Branch2d):
    '''General 3d branch class. It is the base structure for building an octree.\n
       * max_points : An integer defining the maximum number of points this branch stores
       * x_min      : The float lower limit for x
       * x_max      : The float upper limit for x
       * y_min      : The float lower limit for y
       * y_max      : The float upper limit for y
       * z_min      : The float front limit for z
       * z_max      : The float back limit for z \n
       This branch, when divided, has eight branches: \n
       1. the front-down-left
       2. the front-down-right
       3. the front-up-left
       4. the front-up-right
       5. the back-down-left
       6. the back-down-right
       7. the back-up-left
       8. the back-up-right'''

    def __init__(self,
                 max_points: int,
                 x_min: float,
                 x_max: float,
                 y_min: float,
                 y_max: float,
                 z_min: float,
                 z_max: float):
        self._DIVIDED = False
        self._MAX_POINTS = max_points
        self._N_POINTS = 0
        self._xmin = x_min
        self._xmax = x_max
        self._ymin = y_min
        self._ymax = y_max
        self._zmin = z_min
        self._zmax = z_max
        self._POINTS = np.array([], dtype=Point3d)
        self._FRONTUPLEFT = None
        self._FRONTUPRIGHT = None
        self._FRONTDOWNLEFT = None
        self._FRONTDOWNRIGHT = None
        self._BACKUPLEFT = None
        self._BACKUPRIGHT = None
        self._BACKDOWNLEFT = None
        self._BACKDOWNRIGHT = None
        self.subBranches = {0 : self._FRONTDOWNLEFT,
                            1 : self._FRONTDOWNRIGHT,
                            2 : self._FRONTUPLEFT,
                            3 : self._FRONTUPRIGHT,
                            4 : self._BACKDOWNLEFT,
                            5 : self._BACKDOWNRIGHT,
                            6 : self._BACKUPLEFT,
                            7 : self._BACKUPRIGHT}

    def GetZSize(self) -> tuple:
        '''Returns the y information of the branch, as a tuple.'''
        return (self._zmin, self._zmax)

    def GetZMiddlePoint(self) -> float:
        '''Returns the middle z point of the branch.'''
        return self.GetZSize()[0] + (self.GetZSize()[1] - self.GetZSize()[0])/float(2)

    def __str__(self):
        if self.IsDivided() == True:
            str = f"number of points = {self.GetTotalPoints()}\n\
                      FRONT\n \
                    [ {self._FRONTUPLEFT.GetPointsNumber()} | {self._FRONTUPRIGHT.GetPointsNumber()} ]\n\
                    [---|---]\n\
                    [ {self._FRONTDOWNLEFT.GetPointsNumber()} | {self._FRONTDOWNRIGHT.GetPointsNumber()} ]\n\
                    | BACK\n \
                    [ {self._BACKUPLEFT.GetPointsNumber()} | {self._BACKUPRIGHT.GetPointsNumber()} ]\n\
                    [---|---]\n\
                    [ {self._BACKDOWNLEFT.GetPointsNumber()} | {self._BACKDOWNRIGHT.GetPointsNumber()} ]\n"
        else:
            str = f"number of points = {self._N_POINTS}\n| FRONT\n[       ]\n[       ]\n[       ]\n| BACK\n[       ]\n[       ]\n[       ]\n"
        return str

    def GetSubBranch(self, index : int):
        '''Returns the sub branch by index. Below, the index correspondence:
           0 : Front-down-left branch
           1 : Front-down-right branch
           2 : Front-up-left branch
           3 : Front-up-right branch
           4 : Back-down-left branch
           5 : Back-down-right branch
           6 : Back-up-left branch
           7 : Back-up-right branch

           * index : int index of the branch requested'''
        if self.IsDivided() == True:
            return self.subBranches[index]
        else:
            raise AttributeError("The branch got no subbranches.")

    def SubAddPoint(self, point : Point3d):
        if self.GetXSize()[0] <= point.GetXCoord() < self.GetXMiddlePoint():
            if self.GetYSize()[0] <= point.GetYCoord() < self.GetYMiddlePoint():
                if self.GetZSize()[0] <= point.GetZCoord() < self.GetZMiddlePoint():
                    self._FRONTDOWNLEFT.AddPoint(point)
                elif self.GetZMiddlePoint() <= point.GetZCoord() <= self.GetZSize()[1]:
                    self._BACKDOWNLEFT.AddPoint(point)
                else:
                    raise ValueError(
                        f"The point {point()} is outside the tree's bounds : ({self.GetXSize()}, {self.GetYSize()}, {self.GetZSize()}).")

            elif self.GetYMiddlePoint() <= point.GetYCoord() <= self.GetYSize()[1]:
                if self.GetZSize()[0] <= point.GetZCoord() < self.GetZMiddlePoint():
                    self._FRONTUPLEFT.AddPoint(point)
                elif self.GetZMiddlePoint() <= point.GetZCoord() <= self.GetZSize()[1]:
                    self._BACKUPLEFT.AddPoint(point)
            else:
                raise ValueError(
                    f"The point {point()} is outside the tree's bounds : ({self.GetXSize()}, {self.GetYSize()}, {self.GetZSize()}).")
        elif self.GetXMiddlePoint() <= point.GetXCoord() <= self.GetXSize()[1]:
            if self.GetYSize()[0] <= point.GetYCoord() < self.GetYMiddlePoint():
                if self.GetZSize()[0] <= point.GetZCoord() < self.GetZMiddlePoint():
                    self._FRONTDOWNRIGHT.AddPoint(point)
                elif self.GetZMiddlePoint() <= point.GetZCoord() <= self.GetZSize()[1]:
                    self._BACKDOWNRIGHT.AddPoint(point)
                else:
                    raise ValueError(
                        f"The point {point()} is outside the tree's bounds : ({self.GetXSize()}, {self.GetYSize()}, {self.GetZSize()}).")
            elif self.GetYMiddlePoint() <= point.GetYCoord() <= self.GetYSize()[1]:
                if self.GetZSize()[0] <= point.GetZCoord() < self.GetZMiddlePoint():
                    self._FRONTUPRIGHT.AddPoint(point)
                elif self.GetZMiddlePoint() <= point.GetZCoord() <= self.GetZSize()[1]:
                    self._BACKUPRIGHT.AddPoint(point)
                else:
                    raise ValueError(
                        f"The point {point()} is outside the tree's bounds : ({self.GetXSize()}, {self.GetYSize()}, {self.GetZSize()}).")

        else:
            raise ValueError(
                f"The point {point()} is outside the tree's bounds : ({self.GetXSize()}, {self.GetYSize()}, {self.GetZSize()}).")

    def AddPoint(self, point : Point3d):
        '''Recursively adds a point to the branch. \n
           * point : Point3d to be added.\n

           Below, how the method works:
           1. The branch is first checked if divided. If it is, the coordinates of
           the point is checked to add the point in one of the subbranches.
           2. If it is not, it checks if the total current number of points is lower than
           the branch's defined limit,
           3. If it is, adds the point to the branch.
           4. If it's not ,the branch is then divided and the coordinates of the point are checked to
           add the point in one of the newly-created subbranches.'''
        if self.IsDivided() == True:

            self.SubAddPoint(point)

        else:

            if self.GetPointsNumber() == self._MAX_POINTS:
                self._FRONTDOWNLEFT = Branch3d(
                    self._MAX_POINTS,
                    self.GetXSize()[0],
                    self.GetXMiddlePoint(),
                    self.GetYSize()[0],
                    self.GetYMiddlePoint(),
                    self.GetZSize()[0],
                    self.GetZMiddlePoint())
                self._FRONTDOWNRIGHT = Branch3d(
                    self._MAX_POINTS,
                    self.GetXMiddlePoint(),
                    self.GetXSize()[1],
                    self.GetYSize()[0],
                    self.GetYMiddlePoint(),
                    self.GetZSize()[0],
                    self.GetZMiddlePoint())
                self._FRONTUPLEFT = Branch3d(
                    self._MAX_POINTS,
                    self.GetXSize()[0],
                    self.GetXMiddlePoint(),
                    self.GetYMiddlePoint(),
                    self.GetYSize()[1],
                    self.GetZSize()[0],
                    self.GetZMiddlePoint())
                self._FRONTUPRIGHT = Branch3d(
                    self._MAX_POINTS,
                    self.GetXMiddlePoint(),
                    self.GetXSize()[1],
                    self.GetYMiddlePoint(),
                    self.GetYSize()[1],
                    self.GetZSize()[0],
                    self.GetZMiddlePoint())
                self._BACKDOWNLEFT = Branch3d(
                    self._MAX_POINTS,
                    self.GetXSize()[0],
                    self.GetXMiddlePoint(),
                    self.GetYSize()[0],
                    self.GetYMiddlePoint(),
                    self.GetZMiddlePoint(),
                    self.GetZSize()[1])
                self._BACKDOWNRIGHT = Branch3d(
                    self._MAX_POINTS,
                    self.GetXMiddlePoint(),
                    self.GetXSize()[1],
                    self.GetYSize()[0],
                    self.GetYMiddlePoint(),
                    self.GetZMiddlePoint(),
                    self.GetZSize()[1])
                self._BACKUPLEFT = Branch3d(
                    self._MAX_POINTS,
                    self.GetXSize()[0],
                    self.GetXMiddlePoint(),
                    self.GetYMiddlePoint(),
                    self.GetYSize()[1],
                    self.GetZMiddlePoint(),
                    self.GetZSize()[1])
                self._BACKUPRIGHT = Branch3d(
                    self._MAX_POINTS,
                    self.GetXMiddlePoint(),
                    self.GetXSize()[1],
                    self.GetYMiddlePoint(),
                    self.GetYSize()[1],
                    self.GetZMiddlePoint(),
                    self.GetZSize()[1])
                self._DIVIDED = True

                self.subBranches = {0 : self._FRONTDOWNLEFT,
                                    1 : self._FRONTDOWNRIGHT,
                                    2 : self._FRONTUPLEFT,
                                    3 : self._FRONTUPRIGHT,
                                    4 : self._BACKDOWNLEFT,
                                    5 : self._BACKDOWNRIGHT,
                                    6 : self._BACKUPLEFT,
                                    7 : self._BACKUPRIGHT}

                list = self.GetPointsList()
                for i in range(self.GetPointsNumber()):

                    self.SubAddPoint(list[i])
                    self.RemovePoint(0)

                self.SubAddPoint(point)

            else:
                self._POINTS = np.append(self._POINTS, point)
                self._N_POINTS += 1


class Tree3d(Tree2d):
    '''Class that stores the root branch and general 3d tree informations.
       * branch : Branch3d to be the root of the tree.'''

    def __init__(self, branch : Branch3d):
        self._ROOT = branch
        self.xmin = branch.GetXSize()[0]
        self.xmax = branch.GetXSize()[1]
        self.ymin = branch.GetYSize()[0]
        self.ymax = branch.GetYSize()[1]
        self.zmin = branch.GetZSize()[0]
        self.zmax = branch.GetZSize()[1]
        self._N_POINTS = branch.GetTotalPoints()
        self._STR = ""

    def __str__(self):
        if self._ROOT.IsDivided() == True:
            points_list = []
            for i in range(len(self.GetRoot().subBranches)):
                points_list.append(self.GetRoot().GetSubBranch(i).GetTotalPoints())
            length = [len(str(x)) for x in points_list]
            length = [np.max([length[j], length[j + 2]])
                      for j in range(len(points_list)//2)]
            self._STR = f"number of points = {self._ROOT.GetTotalPoints()}\n\
                      FRONT\n\
                    [ {self._ROOT._FRONTUPLEFT.GetTotalPoints()} | {self._ROOT._FRONTUPRIGHT.GetTotalPoints()} ]\n\
                    [---" + "-"*(length[0] - 1) + "|---" + "-"*(length[1] - 1) + f"]\n\
                    [ {self._ROOT._FRONTDOWNLEFT.GetTotalPoints()} | {self._ROOT._FRONTDOWNRIGHT.GetTotalPoints()} ]\n\
                    | BACK\n\
                    [ {self._ROOT._BACKUPLEFT.GetTotalPoints()} | {self._ROOT._BACKUPRIGHT.GetTotalPoints()} ]\n\
                    [---" + "-"*(length[2] - 1) + "|---" + "-"*(length[3] - 1) + f"]\n\
                    [ {self._ROOT._BACKDOWNLEFT.GetTotalPoints()} | {self._ROOT._BACKDOWNRIGHT.GetTotalPoints()} ]\n"
        else:
            self._STR = f"number of points = {self._ROOT.GetTotalPoints()}\n  FRONT\n\
                        [       ]\n\
                        [       ]\n\
                        [       ]\n\
                        | BACK\n\
                        [       ]\n\
                        [       ]\n\
                        [       ]\n"
        return self._STR
# END OF OCTREE IMPLEMENTATION


# GRAPHICAL IMPLEMENTATION
def BoundingBox3d(center : tuple = (0.0, 0.0, 0.0),
                  size : tuple = (1.0, 1.0, 1.0),
                  color : tuple = (1.0, 1.0, 1.0, 1.0),
                  line_width : float = 1.0):
    '''Creates a bounding box with the parameters given. The box got only is edges renderized.
       * center : tuple with 3 coordinates, x, y, and z, that determines where is the center of the box.
       * size : tuple with 3 coordinates, x, y, and z, that determines its lateral sizes.
       * color : tuple with 4 coordinates, r, g, b and a, that determines the color of the bounding box.
       * line_width : float that determines the width of the box lines.'''
    x_c = center[0]
    y_c = center[1]
    z_c = center[2]

    x_l = size[0]/2
    y_l = size[1]/2
    z_l = size[2]/2

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


def GetActorFromBranch2d(branch : Branch2d, color=(
        1.0, 1.0, 1.0), linewidth=1.0) -> Actor:
    '''Recursively creates actors for the branch given. If the branch is divided,
       then the function is run for the subbranches until the function reaches a non-divided branch,
       that creates the actor to be returned. This actor is then appended into a list, that is then returned.
       NOTE: This returns a 3d actor.
       * branch : Branch3d that will have the actor created.'''

    if branch.IsDivided() == True:
        actors = np.array([], dtype=Actor)
        actors = np.append(
            actors,
            GetActorFromBranch2d(
                branch.GetSubBranch(0),
                color,
                linewidth))
        actors = np.append(
            actors,
            GetActorFromBranch2d(
                branch.GetSubBranch(1),
                color,
                linewidth))
        actors = np.append(
            actors,
            GetActorFromBranch2d(
                branch.GetSubBranch(2),
                color,
                linewidth))
        actors = np.append(
            actors,
            GetActorFromBranch2d(
                branch.GetSubBranch(3),
                color,
                linewidth))

        return actors

    else:

        data = np.array([])
        data0 = branch.GetPointsList()
        for i in range(data0.shape[0]):
            np.append(data, data0[i]())

        x_c = branch.GetXMiddlePoint()
        y_c = branch.GetYMiddlePoint()
        z_c = 0.0

        x_l = (branch.GetXSize()[1] - branch.GetXSize()[0])
        y_l = (branch.GetYSize()[1] - branch.GetYSize()[0])
        z_l = 0.0

        cubeActor = BoundingBox3d((x_c, y_c, z_c), (x_l, y_l, z_l), color, linewidth)

        return cubeActor


def GetActorFromBranch3d(branch : Branch3d, color=(
        1.0, 1.0, 1.0), linewidth=1.0) -> Actor:
    '''Recursively creates actors for the branch given. If the branch is divided,
       then the function is run for the subbranches until the function reaches a non-divided branch,
       that creates the actor to be returned. This actor is then appended into a list, that is then returned.
       * branch : Branch3d that will have the actor created.'''

    if branch.IsDivided() == True:
        actors = np.array([], dtype=Actor)
        actors = np.append(
            actors,
            GetActorFromBranch3d(
                branch.GetSubBranch(0),
                color,
                linewidth))
        actors = np.append(
            actors,
            GetActorFromBranch3d(
                branch.GetSubBranch(1),
                color,
                linewidth))
        actors = np.append(
            actors,
            GetActorFromBranch3d(
                branch.GetSubBranch(2),
                color,
                linewidth))
        actors = np.append(
            actors,
            GetActorFromBranch3d(
                branch.GetSubBranch(3),
                color,
                linewidth))
        actors = np.append(
            actors,
            GetActorFromBranch3d(
                branch.GetSubBranch(4),
                color,
                linewidth))
        actors = np.append(
            actors,
            GetActorFromBranch3d(
                branch.GetSubBranch(5),
                color,
                linewidth))
        actors = np.append(
            actors,
            GetActorFromBranch3d(
                branch.GetSubBranch(6),
                color,
                linewidth))
        actors = np.append(
            actors,
            GetActorFromBranch3d(
                branch.GetSubBranch(7),
                color,
                linewidth))

        return actors

    else:

        data = np.array([])
        data0 = branch.GetPointsList()
        for i in range(data0.shape[0]):
            np.append(data, data0[i]())

        x_c = branch.GetXMiddlePoint()
        y_c = branch.GetYMiddlePoint()
        z_c = branch.GetZMiddlePoint()

        x_l = (branch.GetXSize()[1] - branch.GetXSize()[0])
        y_l = (branch.GetYSize()[1] - branch.GetYSize()[0])
        z_l = (branch.GetZSize()[1] - branch.GetZSize()[0])

        cubeActor = BoundingBox3d((x_c, y_c, z_c), (x_l, y_l, z_l), color, linewidth)

        return cubeActor
# END OF GRAPH IMPLEMENTATION
