import vtk
import numpy as np
from fury.utils import numpy_support as nps


class PickingManager(object):
    def __init__(self, vertices=True, faces=True, actors=True,
                 world_coords=True):
        """ Picking Manager helps with picking 3D objects

        Parameters
        -----------
        vertices : bool
            If True allows to pick vertex indices.
        faces : bool
            If True allows to pick face indices.
        actors : bool
            If True allows to pick actor indices.
        world_coords : bool
            If True allows to pick xyz position in world coordinates.
        """

        self.pickers = {}
        if vertices:
            self.pickers['vertices'] = vtk.vtkPointPicker()
        if faces:
            self.pickers['faces'] = vtk.vtkCellPicker()
        if actors:
            self.pickers['actors'] = vtk.vtkPropPicker()
        if world_coords:
            self.pickers['world_coords'] = vtk.vtkWorldPointPicker()

    def pick(self, disp_xy, sc):
        """ Pick on display coordinates

        Parameters
        ----------
        disp_xyz : tuple
            Display coordinates x, y.

        sc : Scene
        """

        x, y = disp_xy
        z = 0
        info = {'vertex': None, 'face': None, 'actor': None, 'xyz': None}
        keys = self.pickers.keys()

        if 'vertices' in keys:
            self.pickers['vertices'].Pick(x, y, z, sc)
            info['vertex'] = self.pickers['vertices'].GetPointId()

        if 'faces' in keys:
            self.pickers['faces'].Pick(x, y, z, sc)
            info['vertex'] = self.pickers['faces'].GetPointId()
            info['face'] = self.pickers['faces'].GetCellId()

        if 'actors' in keys:
            self.pickers['actors'].Pick(x, y, z, sc)
            info['actor'] = self.pickers['actors'].GetViewProp()

        if 'world_coords' in keys:
            self.pickers['world_coords'].Pick(x, y, z, sc)
            info['xyz'] = self.pickers['world_coords'].GetPickPosition()

        return info

    def event_position(self, iren):
        """ Returns event display position from interactor

        Parameters
        ----------
        iren : interactor
            The interactor object can be retrieved for example
            using providing ShowManager's iren attribute.
        """
        return iren.GetEventPosition()


class SelectorManager(object):

    def __init__(self, select='faces'):
        self.hsel = vtk.vtkHardwareSelector()
        self.selection_type(select)
        # self.hsel.SetActorPassOnly(True)

    def selection_type(self, select):        
        if select == 'faces' or select == 'edges':
            self.hsel.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
        if select == 'points' or select == 'vertices':
            self.hsel.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS)
       
    def pick(self, disp_xy, sc):
        self.select(disp_xy, sc, area=1)

    def select(self, disp_xy, sc, area=1):
        self.hsel.SetRenderer(sc)
        picking_area = area
        self.hsel.SetArea(disp_xy[0] - picking_area, disp_xy[1] - picking_area,
                          disp_xy[0] + picking_area, disp_xy[1] + picking_area)
        res = self.hsel.Select()
        # print(res)
        num_nodes = res.GetNumberOfNodes()
        if (num_nodes < 1):
            sel_node = None
        else:
            print('Number of Nodes ', num_nodes)
            for i in range(num_nodes):
                print('Node ', i)
                sel_node = res.GetNode(i)
                
                if(sel_node is not None):
                    selected_nodes = set(np.floor(nps.vtk_to_numpy(
                        sel_node.GetSelectionList())).astype(int))
                    
                    print('#>>>>', id(sel_node.GetProperties().Get(sel_node.PROP())))

                    # selected_node = list(selected_nodes)[0]
                    print('Selected Nodes ', selected_nodes)
                    # print('Prop ', sel_node.GetProperties())
                    # print('Prop ID', sel_node.PROP_ID())
                    # print('Prop ', sel_node.PROP())
        
        # selected_actor.text.SetText(str(selected_node))
        # if(selected_node is not None):
        #     if(labels is not None):
        #         selected_actor.text.SetText(labels[selected_node])
        #     else:
        #         selected_actor.text.SetText("#%d" % selected_node)
        #     selected_actor.SetPosition(positions[selected_node])

        # else:
        #     selected_actor.text.SetText("")

    def event_position(self, iren):
        """ Returns event display position from interactor

        Parameters
        ----------
        iren : interactor
            The interactor object can be retrieved for example
            using providing ShowManager's iren attribute.
        """
        return iren.GetEventPosition()
    

    def selectable_on(self, actors):        
        for a in actors:
            a.PickableOn()
    
    def selectable_off(self, actors):
        for a in actors:
            a.PickableOff()