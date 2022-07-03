import base64
import os
import numpy as np
from pygltflib import GLTF2
from fury.lib import PNGReader, Texture, JPEGReader, ImageFlip
from fury import window, transform, utils


comp_type = {
    5120: {'size': 1, 'dtype': np.byte},
    5121: {'size': 1, 'dtype': np.ubyte},
    5122: {'size': 2, 'dtype': np.short},
    5123: {'size': 2, 'dtype': np.ushort},
    5125: {'size': 4, 'dtype': np.uint},
    5126: {'size': 4, 'dtype': np.float32}
}

acc_type = {
    'SCALAR': 1,
    'VEC2': 2,
    'VEC3': 3,
    'VEC4': 4
}


class glTF:
    """Read and generate actors from glTF files.

    Parameters
    ----------
    filename : str
        Path of the gltf file
    apply_normals : bool, optional
        If `True` applies normals to the mesh.
    """

    def __init__(self, filename, apply_normals=False):

        self.gltf = GLTF2().load(filename)

        gltf_path = filename.split('/')[-1:][0]
        self.pwd = filename[:-len(gltf_path)]
        self.apply_normals = apply_normals  # temporary variable

        self.cameras = {}
        self.actors = []
        self.materials = []
        self.polydatas = []
        self.init_transform = np.identity(4)
        self.get_nodes(0)

    def get_actors(self):
        """Generates actors from glTF file.

        Returns
        -------
        actors : list
            List of vtkActors with texture.
        """
        for i, polydata in enumerate(self.polydatas):
            actor = utils.get_actor_from_polydata(polydata)

            if self.materials[i] is not None:
                baseColorTexture = self.materials[i]['baseColorTexture']
                actor.SetTexture(baseColorTexture)

            self.actors.append(actor)

        return self.actors

    def get_nodes(self, scene_id=0):
        """Loopes over nodes in a scene.

        Parameters
        ----------
        scene_id : int, optional
            scene index of the the glTF.
        """
        scene = self.gltf.scenes[scene_id]
        nodes = scene.nodes

        for node_id in nodes:
            self.transverse_node(node_id, self.init_transform)

    def transverse_node(self, nextnode_id, matrix):
        """Loads mesh and generates transformation matrix.

        Parameters
        ----------
        nextnode_id : int
            Index of the node
        matrix : ndarray (4, 4)
            Transformation matrix
        """
        node = self.gltf.nodes[nextnode_id]

        matnode = np.identity(4)
        if node.matrix is not None:
            matnode = np.array(node.matrix)
            matnode = matnode.reshape(-1, 4).T
        else:
            if node.translation is not None:
                trans = node.translation
                T = transform.translate(trans)
                matnode = np.dot(matnode, T)

            if node.rotation is not None:
                rot = node.rotation
                R = transform.rotate(rot)
                matnode = np.dot(matnode, R)

            if node.scale is not None:
                scales = node.scale
                S = transform.scale(scales)
                matnode = np.dot(matnode, S)

        next_matrix = np.dot(matrix, matnode)

        if node.mesh is not None:
            mesh_id = node.mesh
            self.load_mesh(mesh_id, next_matrix)

        if node.camera is not None:
            camera_id = node.camera
            self.load_camera(camera_id, nextnode_id)

        if node.children:
            for child_id in node.children:
                self.transverse_node(child_id, next_matrix)

    def load_mesh(self, mesh_id, transform_mat):
        """Loads the mesh data from accessor and applies the transformation.

        Parameters
        ----------
        mesh_id : int
            Mesh index to be loaded
        transform_mat : ndarray (4, 4)
            Transformation matrix.
        """
        primitives = self.gltf.meshes[mesh_id].primitives

        for primitive in primitives:

            attributes = primitive.attributes

            vertices = self.get_acc_data(attributes.POSITION)
            vertices = transform.apply_transfomation(vertices, transform_mat.T)

            polydata = utils.PolyData()
            utils.set_polydata_vertices(polydata, vertices)

            if attributes.NORMAL is not None and self.apply_normals:
                normals = self.get_acc_data(attributes.NORMAL)
                utils.set_polydata_normals(polydata, normals)

            if attributes.TEXCOORD_0 is not None:
                uv = self.get_acc_data(attributes.TEXCOORD_0)
                polydata.GetPointData().SetTCoords(
                    utils.numpy_support.numpy_to_vtk(uv))

            if attributes.COLOR_0 is not None:
                color = self.get_acc_data(attributes.COLOR_0)
                color = color[:, :-1]*255
                utils.set_polydata_colors(polydata, color)

            if primitive.indices is not None:
                indices = self.get_acc_data(primitive.indices).reshape(-1, 3)
                utils.set_polydata_triangles(polydata, indices)

            material = None
            if primitive.material is not None:
                material = self.get_materials(primitive.material)

            self.polydatas.append(polydata)
            self.materials.append(material)

    def get_acc_data(self, acc_id):
        """Gets the correct data from buffer uing accessors and bufferviews.

        Parameters
        ----------
        acc_id : int
            Accessor index

        Returns
        -------
        buffer_array : ndarray
            Numpy array extracted from the buffer.
        """

        accessor = self.gltf.accessors[acc_id]

        buffview_id = accessor.bufferView
        acc_byte_offset = accessor.byteOffset
        count = accessor.count
        d_type = comp_type.get(accessor.componentType)
        d_size = d_type['size']
        a_type = acc_type.get(accessor.type)

        buffview = self.gltf.bufferViews[buffview_id]

        buff_id = buffview.buffer
        byte_offset = buffview.byteOffset
        byte_stride = buffview.byteStride
        byte_stride = byte_stride if byte_stride else (a_type * d_size)
        byte_length = count * d_size * a_type

        total_byte_offset = byte_offset + acc_byte_offset

        return self.get_buff_array(buff_id, d_type['dtype'], byte_length,
                                   total_byte_offset, byte_stride)

    def get_buff_array(self, buff_id, d_type, byte_length,
                       byte_offset, byte_stride):
        """Extracts the mesh data from buffer.

        Parameters
        ----------
        buff_id : int
            Buffer Index
        d_type : type
            Element data type
        byte_length : int
            The lenght of the buffer data
        byte_offset : int
            The offset into the buffer in bytes
        byte_stride : int
            The stride, in bytes

        Returns
        -------
        out_arr : ndarray
            Numpy array of size byte_length from buffer.
        """

        buffer = self.gltf.buffers[buff_id]
        uri = buffer.uri

        if d_type == np.short or d_type == np.ushort:
            byte_length = int(byte_length/2)
            byte_stride = int(byte_stride/2)

        elif d_type == np.float32 or d_type == np.uint16:
            byte_length = int(byte_length/4)
            byte_stride = int(byte_stride/4)

        try:
            if uri.startswith('data:application/octet-stream;base64') or \
                    uri.startswith('data:application/gltf-buffer;base64'):
                buff_data = uri.split(',')[1]
                buff_data = base64.b64decode(buff_data)

            elif uri.endswith('.bin'):
                with open(os.path.join(self.pwd, uri), 'rb') as f:
                    buff_data = f.read(-1)

            out_arr = np.frombuffer(buff_data, dtype=d_type,
                                    count=byte_length, offset=byte_offset)

            out_arr = out_arr.reshape(-1, byte_stride)
            return out_arr

        except IOError:
            print('Failed to read ! Error in opening file')

    def get_materials(self, mat_id):
        """Gets the textures data

        Parameters
        ----------
        mat_id : int
            Material index
        """

        material = self.gltf.materials[mat_id]
        bct, mrt, nt = None, None, None

        pbr = material.pbrMetallicRoughness

        if pbr.baseColorTexture is not None:
            bct = pbr.baseColorTexture.index
            bct = self.get_texture(bct)

        return {'baseColorTexture': bct}

    def get_texture(self, tex_id):
        """Reads and converts image into vtk texture

        Parameters
        ----------
        tex_id : int
            Texture index

        Returns
        -------
        atexture : vtkTexture
            Returns flipped vtk texture from image. 
        """

        texture = self.gltf.textures[tex_id].source
        images = self.gltf.images

        reader_type = {
            '.jpg': JPEGReader,
            '.jpeg': JPEGReader,
            '.png': PNGReader
        }
        file = images[texture].uri

        if file.startswith('data:image'):
            buff_data = file.split(',')[1]
            buff_data = base64.b64decode(buff_data)

            extension = '.png' if file.startswith('data:image/png') else '.jpg'
            image_path = os.path.join(self.pwd, str("b64texture"+extension))
            with open(image_path, "wb") as image_file:
                image_file.write(buff_data)

        else:
            extension = os.path.splitext(os.path.basename(file).lower())[1]
            image_path = os.path.join(self.pwd, file)

        reader = reader_type.get(extension)()
        reader.SetFileName(image_path)
        reader.Update()

        flip = ImageFlip()
        flip.SetInputConnection(reader.GetOutputPort())
        flip.SetFilteredAxis(1)  # flip along Y axis

        atexture = Texture()
        atexture.InterpolateOn()
        atexture.EdgeClampOn()
        atexture.SetInputConnection(flip.GetOutputPort())

        return atexture

    def load_camera(self, node_id):
        """Loads the camera data of a node

        Parameters
        ----------
        node_id : int
            Node index of the camera.
        """
        camera = self.gltf.cameras
        self.cameras[node_id] = camera
