import numpy as np

def whatOrder(verts, tri):
    '''
        :param verts: ndarray of vertices corresponding to shape
        :param tri: ndarray of 3 vertice numbers corresponding to shape
        :return:
            0 if the order is clockwise
            1 if the order is counter-clockwise
        '''
    v1 = verts[tri[0]-1]
    v2 = verts[tri[1]-1]
    v3 = verts[tri[2]-1]

    x1 = v1[0]
    x2 = v2[0]
    x3 = v3[0]

    y1 = v1[1]
    y2 = v2[1]
    y3 = v3[1]

    val = (y2 - y1)*(x3-x2) - (y3 - y2)*(x2 - x1)

    if(val<0):
        return 1
    return 0

def changeOrder(tri):
    '''
    :param tri: ndarray of 3 points in a triangle
    :return: new vector of triangle with correct order
    '''

    newVert = []
    newVert.append(tri[2])
    newVert.append(tri[1])
    newVert.append(tri[0])

    return np.array(newVert)

def checkOrder(vert, triarr):
    '''

    :param vert: ndarray of vertices corresponding to a shape
    :param tri: ndarray of triangle coordinates corresponding to vertices
    :return:
        new triangle ndarray with correct winding order
    '''

    shape = triarr.shape
    correct_vert = np.empty(shape)
    correct_order = whatOrder(vert, triarr[0])
    for nb, i in enumerate(triarr):
        order2 = whatOrder(vert, i)
        if (correct_order != order2):
            temp = changeOrder(i)
            correct_vert[nb] = temp
        else:
            correct_vert[nb] = i
    return correct_vert