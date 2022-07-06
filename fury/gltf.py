import base64
import os
import numpy as np
from pygltflib import *
from fury.lib import Texture as vtkTexture
from fury.lib import PNGReader, JPEGReader, ImageFlip
from fury import window, transform, utils, actor


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
            self.load_camera(nextnode_id)

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
            vertices = transform.apply_transfomation(vertices, transform_mat)

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
            vtkTexture index

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

        atexture = vtkTexture()
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


def generate_gltf(scene, name='default'):
    """Generate gltf from FURY scene.

    Parameters
    ----------
    scene : Scene
        FURY scene object.
    name : str, optional
        Name of the model to be saved
    """
    gltf = GLTF2()
    name = name.split('.')[0] if name.endswith('.gltf') else name
    buffer_file = open(f'{name}.bin', 'wb')
    primitives = []
    buffer_size = 0
    bview_count = 0

    for actor in scene.GetActors():
        prim, size, count = _connect_primitives(gltf, actor, buffer_file,
                                                buffer_size, bview_count)
        primitives.append(prim)
        buffer_size += size
        bview_count += count

    buffer_file.close()
    add_mesh(gltf, primitives)
    add_buffer(gltf, size, f'{name}.bin')
    camera = scene.camera()
    cam_id = None
    if camera:
        add_camera(gltf, camera)
        cam_id = 0
    add_node(gltf, mesh=0, camera=cam_id)
    add_scene(gltf, 0)
    gltf.save(f'{name}.gltf')


def _connect_primitives(gltf, actor, buff_file, boffset, count):
    """Creates Accessor, BufferViews and writes primitive data to a binary file

    Parameters
    ----------
    gltf : Pygltflib.GLTF2
    actor : Actor
        the fury actor
    buff_file : file
        filename.bin opened in `wb` mode
    count : int
        BufferView count
    
    Returns
    -------
    prim : Pygltflib.Primitive
    byteoffset : int
        Offset size of a primitive
    count : int
        BufferView count after adding the primitive.
    """

    polydata = actor.GetMapper().GetInput()
    colors = utils.colors_from_actor(actor)
    if colors is not None:
        polydata = utils.set_polydata_colors(polydata, colors)

    vertices = utils.get_polydata_vertices(polydata)
    indices = utils.get_polydata_triangles(polydata)
    colors = utils.get_polydata_colors(polydata)
    normals = utils.get_polydata_normals(polydata)
    tcoords = utils.get_polydata_tcoord(polydata)

    vertex, index, normal, tcoord, color = (None, None, None,
                                            None, None)
    if indices is not None:
        indices = indices.reshape((-1, ))
        amax = [np.max(indices)]
        amin = [np.min(indices)]

        indices = indices.astype(np.ushort)
        blength = len(indices)*2
        buff_file.write(indices.tobytes())
        add_bufferview(gltf, 0, boffset, blength)
        add_accessor(gltf, count, 0, UNSIGNED_SHORT,
                     len(indices), SCALAR)
        boffset += blength
        index = count
        count += 1

    if vertices is not None:
        amax = np.max(vertices, 0).tolist()
        amin = np.min(vertices, 0).tolist()
        vertices = vertices.reshape((-1, )).astype(np.float32)
        blength = len(vertices)*4
        buff_file.write(vertices.tobytes())
        add_bufferview(gltf, 0, boffset, blength)
        add_accessor(gltf, count, 0, FLOAT, len(vertices)//3,
                     VEC3, amax, amin)
        boffset += blength
        vertex = count
        count += 1

    if normals is not None:
        amax = np.max(normals, 0).tolist()
        amin = np.min(normals, 0).tolist()
        normals = normals.reshape((-1, ))
        blength = len(normals)*4
        buff_file.write(normals.tobytes())
        add_bufferview(gltf, 0, boffset, blength)
        add_accessor(gltf, count, 0, FLOAT, len(normals)//3,
                     VEC3, amax, amin)
        boffset += blength
        normal = count
        count += 1

    if tcoords is not None:
        amax = np.max(tcoords, 0).tolist()
        amin = np.min(tcoords, 0).tolist()
        tcoords = tcoords.reshape((-1, )).astype(np.float32)
        blength = len(tcoords)*4
        buff_file.write(tcoords.tobytes())
        add_bufferview(gltf, 0, boffset, blength)
        add_accessor(gltf, count, 0, FLOAT, len(tcoords)//2,
                     VEC2)
        boffset += blength
        tcoord = count
        count += 1
        # vtk_image = actor.GetTexture().GetInput()
        add_material(gltf, 0, image_path)

    if colors is not None:
        shape = colors.shape[0]
        colors = np.concatenate((colors, np.full((shape, 1), 255.)), axis=1)
        colors = colors/255
        colors = colors.reshape((-1, )).astype(np.float32)
        blength = len(colors)*4
        buff_file.write(colors.tobytes())
        add_bufferview(gltf, 0, boffset, blength)
        add_accessor(gltf, count, 0, FLOAT, shape, VEC4)
        boffset += blength
        color = count
        count += 1
    material = None if tcoords is None else 0
    prim = add_prim(vertex, index, color, tcoord, normal, material)
    return prim, boffset, count


def add_scene(gltf, *nodes):
    scene = Scene()
    scene.nodes = [node for node in nodes]
    gltf.scenes.append(scene)


def add_node(gltf, mesh=None, camera=None):
    node = Node()
    if mesh is not None:
        node.mesh = mesh
    if camera is not None:
        node.camera = camera
    gltf.nodes.append(node)


def add_mesh(gltf, prims):
    mesh = Mesh()
    for prim in prims:
        mesh.primitives.append(prim)

    gltf.meshes.append(mesh)


def add_camera(gltf, camera, aspec_ratio=1.0):
    orthographic = camera.GetParallelProjection()
    cam = Camera()
    if orthographic:
        cam.type = "orthographic"
    else:
        clip_range = camera.GetClippingRange()
        angle = camera.GetViewAngle()
        ratio = aspec_ratio
        pers = Perspective()
        pers.aspectRatio = aspec_ratio
        pers.znear, pers.zfar = clip_range
        pers.yfov = angle*np.pi/180
        cam.type = "perspective"
        cam.perspective = pers
    gltf.cameras.append(cam)


def add_prim(verts, indices, cols, tcoords, normals, mat):
    # for each actor we'll have a primitive
    prim = Primitive()
    attr = Attributes()
    attr.POSITION = verts
    attr.NORMAL = normals
    attr.TEXCOORD_0 = tcoords
    attr.COLOR_0 = cols
    prim.attributes = attr
    prim.indices = indices
    if mat is not None:
        prim.material = mat
    return prim


def add_material(gltf, bct: int, uri: str):
    material = Material()
    texture = Texture()
    image = Image()
    pbr = PbrMetallicRoughness()
    tinfo = TextureInfo()
    tinfo.index = bct
    pbr.baseColorTexture = tinfo
    pbr.metallicFactor = 0.0  # setting default
    material.pbrMetallicRoughness = pbr
    # texture.sampler = 0
    texture.source = bct
    image.uri = uri
    gltf.materials.append(material)
    gltf.textures.append(texture)
    gltf.images.append(image)


def add_accessor(gltf, bv, bo, ct, cnt, atype, max=None, min=None):
    accessor = Accessor()
    accessor.bufferView = bv
    accessor.byteOffset = bo
    accessor.componentType = ct
    accessor.count = cnt
    accessor.type = atype
    if (max is not None) and (min is not None):
        accessor.max = max
        accessor.min = min
    gltf.accessors.append(accessor)


def add_bufferview(gltf, buff, bo, bl, bs=None, target=None):
    buffer_view = BufferView()
    buffer_view.buffer = buff
    buffer_view.byteOffset = bo
    buffer_view.byteLength = bl
    buffer_view.byteStride = bs
    gltf.bufferViews.append(buffer_view)


def add_buffer(gltf, byte_length, uri):
    buffer = Buffer()
    buffer.uri = uri
    buffer.byteLength = byte_length
    gltf.buffers.append(buffer)
