from vtk import vtkExtractEdges, vtkProperty, vtkCubeSource
from fury import utils
import numpy as np


# QUADTREE IMPLEMENTATION
class point2d():
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
        string = "[" + str('%.3f' % self.coord[0]) + ", " + str('%.3f' % self.coord[1]) + "]"
        return string
    
    def __repr__(self):
        string = "[" + str('%.3f' % self.coord[0]) + ", " + str('%.3f' % self.coord[1]) + "]"
        return string
        


class branch2d():
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
    def __init__(self, max_points : int, x_min : float, x_max : float, y_min : float, y_max : float):
        self._DIVIDED = False
        self._MAX_POINTS = max_points
        self._N_POINTS = 0
        self._xmin = x_min
        self._xmax = x_max
        self._ymin = y_min
        self._ymax = y_max
        self._POINTS = np.array([], dtype = point2d)
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
    

    def GetTotalPoints(self) -> int:
        '''Returns the total number of points in that branch, including the points in the subranches.'''
        if self.IsDivided() == True:

            sum = 0
            sum += self._DOWNLEFT.GetTotalPoints()
            sum += self._DOWNRIGHT.GetTotalPoints()
            sum += self._UPLEFT.GetTotalPoints()
            sum += self._UPRIGHT.GetTotalPoints()
            return sum
        
        else:
            return self.GetPointsNumber()
        
    def GetSubBranch(self, index : int):
        '''Returns the sub branch by index. Below, the index correspondence:
           0 : Down-left branch
           1 : Down-right branch
           2 : Up-left branch
           3 : Up-right branch'''
        self.subBranches = {0 : self._DOWNLEFT, 
                            1 : self._DOWNRIGHT, 
                            2 : self._UPLEFT, 
                            3 : self._UPRIGHT}

        
        return self.subBranches[index]


    def RemovePoint(self, index : int):
        '''Removes the given element of the list of points'''
        if self._N_POINTS != 0:
            self._POINTS = np.delete(self._POINTS, index)
            self._N_POINTS -= 1
        else:
            raise("This branch has no point to be removed.")


    def AddPoint(self, point : point2d):
        '''Recursive method that adds a point to the branch. \n
           * point : point2d to be added.\n

           Below, how the method works:
           1. The branch is first checked if divided. If it is, the coordinates of
           the point is checked to add the point in one of the subbranches.
           2. If it is not, it checks if the total current number of points is lower than
           the branch's defined limit, 
           3. If it is, adds the point to the branch.
           4. If it's not ,the branch is then divided and the coordinates of the point are checked to 
           add the point in one of the newly-created subbranches.'''

        if self.IsDivided() == True:

            if point.GetXCoord() < self.GetXMiddlePoint():
                if point.GetYCoord() < self.GetYMiddlePoint():
                    self._DOWNLEFT.AddPoint(point)
                else:
                    self._UPLEFT.AddPoint(point)
            else:
                if point.GetYCoord() < self.GetYMiddlePoint():
                    self._DOWNRIGHT.AddPoint(point)
                else:
                    self._UPRIGHT.AddPoint(point)

        
        else:

            if self.GetPointsNumber() == self._MAX_POINTS:
                self._DOWNLEFT = branch2d(self._MAX_POINTS, self.GetXSize()[0], self.GetXMiddlePoint(), self.GetYSize()[0], self.GetYMiddlePoint())
                self._DOWNRIGHT = branch2d(self._MAX_POINTS, self.GetXMiddlePoint(), self.GetXSize()[1], self.GetYSize()[0], self.GetYMiddlePoint())
                self._UPLEFT = branch2d(self._MAX_POINTS, self.GetXSize()[0], self.GetXMiddlePoint(), self.GetYMiddlePoint(), self.GetYSize()[1])
                self._UPRIGHT = branch2d(self._MAX_POINTS, self.GetXMiddlePoint(), self.GetXSize()[1], self.GetYMiddlePoint(), self.GetYSize()[1])
                self._DIVIDED = True

                list = self.GetPointsList()
                for i in range(self.GetPointsNumber()):

                    if list[i].GetXCoord() < self.GetXMiddlePoint():
                        if list[i].GetYCoord() < self.GetYMiddlePoint():
                            self._DOWNLEFT.AddPoint(list[i])
                        else:
                            self._UPLEFT.AddPoint(list[i])
                            
                    else:
                        if list[i].GetYCoord() < self.GetYMiddlePoint():
                            self._DOWNRIGHT.AddPoint(list[i])
                        else:

                            self._UPRIGHT.AddPoint(list[i])
                    self.RemovePoint(0)

                
                if point.GetXCoord() < self.GetXMiddlePoint():
                    if point.GetYCoord() < self.GetYMiddlePoint():
                        self._DOWNLEFT.AddPoint(point)
                    else:
                        self._UPLEFT.AddPoint(point)
                else:
                    if point.GetYCoord() < self.GetYMiddlePoint():
                        self._DOWNRIGHT.AddPoint(point)
                    else:
                        self._UPRIGHT.AddPoint(point)


            else:
                self._POINTS = np.append(self._POINTS, point)
                self._N_POINTS += 1

    def ProcessBranch(self, Function, *args):
        '''Abstract recursive method that process the branch or its subbranches with a given function and its arguments.
           If the function returns any value, it will be returned as the value itself or a list of the
           values returned by each subbranch processed, if the branch is already divided.
           * Function : Any function that has only a branch as input
           * *args : arguments of the function in the order requested by the funciton passed'''
        if self.IsDivided() == True:
            list = np.array([])

            for i in range(len(self.subBranches)):
                list = np.append(list, self.GetSubBranch(i).ProcessBranch(Function, *args))

            return list
                
        else:
            return Function(self, *args)






class Tree2d():
    '''Class that stores the root branch and general 2d tree informations.
       * branch : branch2d to be the root of the tree.'''
    def __init__(self, branch : branch2d):
        self._ROOT = branch
        self.xmin = branch.GetXSize()[0]
        self.xmax = branch.GetXSize()[1]
        self.ymin = branch.GetYSize()[0]
        self.ymax = branch.GetYSize()[1]
        self._N_POINTS = branch.GetTotalPoints()
        self._STR = ""
    

    def __str__(self):
        if self._ROOT.IsDivided() == True:
            self._STR  = f"number of points = {self._ROOT.GetTotalPoints()}\n"
            self._STR += f"[ {self._ROOT._UPLEFT.GetTotalPoints()} | {self._ROOT._UPRIGHT.GetTotalPoints()} ]\n"
            self._STR += f"[---|---]\n"
            self._STR += f"[ {self._ROOT._DOWNLEFT.GetTotalPoints()} | {self._ROOT._DOWNRIGHT.GetTotalPoints()} ]\n"
        else:
            self._STR = f"number of points = {self._N_POINTS}\n[       ]\n[       ]\n[       ]\n"

        return self._STR
    
    def AddPoint(self, point : point2d):
        '''Method that adds a point to the tree.
          * point : point2d to be added into the tree.'''
        self._ROOT.AddPoint(point)
        self._N_POINTS += 1
# END OF QUADTREE IMPLEMENTATION

# OCTREE IMPLEMENTATION
class point3d(point2d):
    '''General point class. This class only stores 3d coordinates.'''
    def __init__(self, x: float, y: float, z: float):
        self.coord = np.array([x, y, z])

    def __call__(self):
        return self.coord

    def GetZCoord(self):
        '''Method that returns the z coordinate of the point.'''
        return self.coord[2]
    
    def GetNorm(self):
        '''Method that returns the norm of the point in relation to the origin.'''
        return np.sqrt(self.GetXCoord()**2 + self.GetYCoord()**2 + self.GetZCoord()**2)
    
    def SetCoord(self, x: float, y: float, z : float):
        '''Method that updates the coordinate of the point.
           * x : float x position
           * y : float y position
           * z : float z position'''
        self.coord = np.array([x, y, z])

    def __str__(self):
        string = "[" + str('%.3f' % self.coord[0]) + ", " + str('%.3f' % self.coord[1]) + str('%.3f' % self.coord[2]) +"]"
        return string
    
    def __repr__(self):
        string = "[" + str('%.3f' % self.coord[0]) + ", " + str('%.3f' % self.coord[1]) + str('%.3f' % self.coord[2]) + "]"
        return string


class branch3d(branch2d):
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
        self._POINTS = np.array([], dtype = point3d)
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
    
    def GetZSize(self) ->tuple:
        '''Method that returns the y information of the branch, as a tuple.'''
        return (self._zmin, self._zmax)

    def GetZMiddlePoint(self) -> float:
        '''Method that returns the middle z point of the branch.'''
        return self.GetZSize()[0] + (self.GetZSize()[1] - self.GetZSize()[0])/float(2)
    
    def GetTotalPoints(self) -> int:
        '''Returns the total number of points in that branch, including the points in the subranches.'''
        if self.IsDivided() == True:

            sum = 0
            sum += self._FRONTDOWNLEFT.GetTotalPoints()
            sum += self._FRONTDOWNRIGHT.GetTotalPoints()
            sum += self._FRONTUPLEFT.GetTotalPoints()
            sum += self._FRONTUPRIGHT.GetTotalPoints()
            sum += self._BACKDOWNLEFT.GetTotalPoints()
            sum += self._BACKDOWNRIGHT.GetTotalPoints()
            sum += self._BACKUPLEFT.GetTotalPoints()
            sum += self._BACKUPRIGHT.GetTotalPoints()
            return sum
        
        else:
            return self.GetPointsNumber()

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
        
        self.subBranches = {0 : self._FRONTDOWNLEFT, 
                            1 : self._FRONTDOWNRIGHT, 
                            2 : self._FRONTUPLEFT, 
                            3 : self._FRONTUPRIGHT, 
                            4 : self._BACKDOWNLEFT,
                            5 : self._BACKDOWNRIGHT,
                            6 : self._BACKUPLEFT,
                            7 : self._BACKUPRIGHT}
        
        return self.subBranches[index]
    

    def AddPoint(self, point : point3d):
        '''Recursive method that adds a point to the branch. \n
           * point : point3d to be added.\n

           Below, how the method works:
           1. The branch is first checked if divided. If it is, the coordinates of
           the point is checked to add the point in one of the subbranches.
           2. If it is not, it checks if the total current number of points is lower than
           the branch's defined limit, 
           3. If it is, adds the point to the branch.
           4. If it's not ,the branch is then divided and the coordinates of the point are checked to 
           add the point in one of the newly-created subbranches.'''
        if self.IsDivided() == True:

            if point.GetXCoord() < self.GetXMiddlePoint():
                if point.GetYCoord() < self.GetYMiddlePoint():
                    if point.GetZCoord() < self.GetZMiddlePoint():
                        self._FRONTDOWNLEFT.AddPoint(point)
                    else:
                        self._BACKDOWNLEFT.AddPoint(point)
                else:
                    if point.GetZCoord() < self.GetZMiddlePoint():
                        self._FRONTUPLEFT.AddPoint(point)
                    else:
                        self._BACKUPLEFT.AddPoint(point)
            else:
                if point.GetYCoord() < self.GetYMiddlePoint():
                    if point.GetZCoord() < self.GetZMiddlePoint():
                        self._FRONTDOWNRIGHT.AddPoint(point)
                    else:
                        self._BACKDOWNRIGHT.AddPoint(point)
                else:
                    if point.GetZCoord() < self.GetZMiddlePoint():
                        self._FRONTUPRIGHT.AddPoint(point)
                    else:
                        self._BACKUPRIGHT.AddPoint(point)
        
        else:

            if self.GetPointsNumber() == self._MAX_POINTS:
                self._FRONTDOWNLEFT = branch3d(self._MAX_POINTS, self.GetXSize()[0], self.GetXMiddlePoint(), self.GetYSize()[0], self.GetYMiddlePoint(), self.GetZSize()[0], self.GetZMiddlePoint())
                self._FRONTDOWNRIGHT = branch3d(self._MAX_POINTS, self.GetXMiddlePoint(), self.GetXSize()[1], self.GetYSize()[0], self.GetYMiddlePoint(), self.GetZSize()[0], self.GetZMiddlePoint())
                self._FRONTUPLEFT = branch3d(self._MAX_POINTS, self.GetXSize()[0], self.GetXMiddlePoint(), self.GetYMiddlePoint(), self.GetYSize()[1], self.GetZSize()[0], self.GetZMiddlePoint())
                self._FRONTUPRIGHT = branch3d(self._MAX_POINTS, self.GetXMiddlePoint(), self.GetXSize()[1], self.GetYMiddlePoint(), self.GetYSize()[1], self.GetZSize()[0], self.GetZMiddlePoint())
                self._BACKDOWNLEFT = branch3d(self._MAX_POINTS, self.GetXSize()[0], self.GetXMiddlePoint(), self.GetYSize()[0], self.GetYMiddlePoint(), self.GetZMiddlePoint(), self.GetZSize()[1])
                self._BACKDOWNRIGHT = branch3d(self._MAX_POINTS, self.GetXMiddlePoint(), self.GetXSize()[1], self.GetYSize()[0], self.GetYMiddlePoint(), self.GetZMiddlePoint(), self.GetZSize()[1])
                self._BACKUPLEFT = branch3d(self._MAX_POINTS, self.GetXSize()[0], self.GetXMiddlePoint(), self.GetYMiddlePoint(), self.GetYSize()[1], self.GetZMiddlePoint(), self.GetZSize()[1])
                self._BACKUPRIGHT = branch3d(self._MAX_POINTS, self.GetXMiddlePoint(), self.GetXSize()[1], self.GetYMiddlePoint(), self.GetYSize()[1], self.GetZMiddlePoint(), self.GetZSize()[1])
                self._DIVIDED = True

                list = self.GetPointsList()
                for i in range(self.GetPointsNumber()):

                    if list[i].GetXCoord() < self.GetXMiddlePoint():
                        if list[i].GetYCoord() < self.GetYMiddlePoint():
                            if list[i].GetZCoord() < self.GetZMiddlePoint():
                                self._FRONTDOWNLEFT.AddPoint(list[i])
                            else:
                                self._BACKDOWNLEFT.AddPoint(list[i])
                        else:
                            if list[i].GetZCoord() < self.GetZMiddlePoint():
                                self._FRONTUPLEFT.AddPoint(list[i])
                            else:
                                self._BACKUPLEFT.AddPoint(list[i])
                    else:
                        if list[i].GetYCoord() < self.GetYMiddlePoint():
                            if list[i].GetZCoord() < self.GetZMiddlePoint():
                                self._FRONTDOWNRIGHT.AddPoint(list[i])
                            else:
                                self._BACKDOWNRIGHT.AddPoint(list[i])
                        else:
                            if list[i].GetZCoord() < self.GetZMiddlePoint():
                                self._FRONTUPRIGHT.AddPoint(list[i])
                            else:
                                self._BACKUPRIGHT.AddPoint(list[i])
                    self.RemovePoint(0)

                
                if point.GetXCoord() < self.GetXMiddlePoint():
                    if point.GetYCoord() < self.GetYMiddlePoint():
                        if point.GetZCoord() < self.GetZMiddlePoint():
                            self._FRONTDOWNLEFT.AddPoint(point)
                        else:
                            self._BACKDOWNLEFT.AddPoint(point)
                    else:
                        if point.GetZCoord() < self.GetZMiddlePoint():
                            self._FRONTUPLEFT.AddPoint(point)
                        else:
                            self._BACKUPLEFT.AddPoint(point)
                else:
                    if point.GetYCoord() < self.GetYMiddlePoint():
                        if point.GetZCoord() < self.GetZMiddlePoint():
                            self._FRONTDOWNRIGHT.AddPoint(point)
                        else:
                            self._BACKDOWNRIGHT.AddPoint(point)
                    else:
                        if point.GetZCoord() < self.GetZMiddlePoint():
                            self._FRONTUPRIGHT.AddPoint(point)
                        else:
                            self._BACKUPRIGHT.AddPoint(point)


            else:
                self._POINTS = np.append(self._POINTS, point)
                self._N_POINTS += 1


class Tree3d():
    '''Class that stores the root branch and general 3d tree informations.
       * branch : branch3d to be the root of the tree.'''
    def __init__(self, branch : branch3d):
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
            self._STR = f"number of points = {self._ROOT.GetTotalPoints()}\n\
                      FRONT\n\
                    [ {self._ROOT._FRONTUPLEFT.GetTotalPoints()} | {self._ROOT._FRONTUPRIGHT.GetTotalPoints()} ]\n\
                    [---|---]\n\
                    [ {self._ROOT._FRONTDOWNLEFT.GetTotalPoints()} | {self._ROOT._FRONTDOWNRIGHT.GetTotalPoints()} ]\n\
                    | BACK\n\
                    [ {self._ROOT._BACKUPLEFT.GetTotalPoints()} | {self._ROOT._BACKUPRIGHT.GetTotalPoints()} ]\n\
                    [---|---]\n\
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
    
    def GetRoot(self) -> branch3d:
        '''Returns the root branch of the tree.'''
        return self._ROOT
    
    def AddPoint(self, point : point3d):
        '''Adds a point to the tree.
           * point : point3d to be added into the tree.'''
        self._ROOT.AddPoint(point)
        self._N_POINTS += 1
# END OF OCTREE IMPLEMENTATION
    


# GRAPH IMPLEMENTATION
def BoundingBox(center : tuple = (0.0, 0.0, 0.0), 
                size : tuple = (1.0, 1.0, 1.0), 
                color : tuple = (1.0, 1.0, 0.0, 1.0), 
                line_width : float = 1.0):
    '''Creates a bounding box with the parameters given. The box got only is edges renderized.
       * center : tuple with 3 coordinates, x, y, and z, that determines where is the center of the box.
       * size : tuple with 3 coordinates, x, y, and z, that determines its lateral sizes.
       * color : tuple with 4 coordinates, r, g, b and a, that determines the color of the bounding box.
       * line_width : float that determines the width of the box lines.'''
    x_c = center[0]
    y_c = center[1]
    z_c = center[2]

    cube = vtkCubeSource()
    cube.SetXLength(size[0])
    cube.SetYLength(size[1])
    cube.SetZLength(size[2])
    cube.SetCenter(x_c, y_c, z_c)

    edges = vtkExtractEdges()
    edges.SetInputConnection(cube.GetOutputPort())

    cubeMapper = utils.PolyDataMapper()
    cubeMapper.SetInputConnection(edges.GetOutputPort())

    cubeActor = utils.get_actor_from_polymapper(cubeMapper)
    cubeActor.SetProperty(vtkProperty().SetLineWidth(line_width))
    cubeActor.GetProperty().SetColor(color[0], color[1], color[2])
    cubeActor.GetProperty().SetLighting(False)

    return cubeActor


def GetActorFromBranch(branch : branch3d) -> utils.Actor:
    '''Recursive function that creates actors for the branch given. If the branch is divided,
       then the function is run for the subbranches until the function reaches a non-divided branch, 
       that creates the actor to be returned. This actor is then appended into a list, that is then returned.
       * branch : branch3d that will have the actor created.'''

    if branch.IsDivided() == True:
        actors = np.array([], dtype = utils.Actor)
        actors = np.append(actors, GetActorFromBranch(branch.GetSubBranch(0)))
        actors = np.append(actors, GetActorFromBranch(branch.GetSubBranch(1)))
        actors = np.append(actors, GetActorFromBranch(branch.GetSubBranch(2)))
        actors = np.append(actors, GetActorFromBranch(branch.GetSubBranch(3)))
        actors = np.append(actors, GetActorFromBranch(branch.GetSubBranch(4)))
        actors = np.append(actors, GetActorFromBranch(branch.GetSubBranch(5)))
        actors = np.append(actors, GetActorFromBranch(branch.GetSubBranch(6)))
        actors = np.append(actors, GetActorFromBranch(branch.GetSubBranch(7)))


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

        cubeActor = BoundingBox((x_c, y_c, z_c), (x_l, y_l, z_l), (0.5, 0.5, 0.5), 3.0)

        return cubeActor
# END OF GRAPH IMPLEMENTATION