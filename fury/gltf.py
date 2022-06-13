import numpy as np
import base64
import json as j
import os
from dataclasses import dataclass, asdict, fields, field
from typing import Any, Dict, List, Optional
from fury.lib import PNGReader, Texture, JPEGReader, ImageFlip, PolyData
from fury import window, transform, utils
from io import BytesIO


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


@dataclass
class Node:
    name: Optional[str] = None
    camera: Optional[int] = None
    skin: Optional[int] = None
    mesh: Optional[int] = None
    matrix: Optional[np.ndarray] = None
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None
    children: Optional[np.ndarray] = None


@dataclass
class Material:
    pbrMetallicRoughness: Optional[Dict] = None
    occlusionTexture: Optional[Dict] = None
    normalTexture: Optional[Dict] = None
    emissiveTexture: Optional[Dict] = None
    emissiveFactor: Optional[List[int]] = None
    alphaMode: Optional[str] = None
    alphaCutoff: Optional[float] = None
    doubleSided: Optional[bool] = False
    name: Optional[str] = None
    # vtkTextures and Materials
    baseColorTexture: Optional[Texture] = None
    metallicRoughnessTexture: Optional[Texture] = None


@dataclass
class Primitive:
    attributes: Optional[dict] = None
    indices: Optional[int] = None
    mode: Optional[int] = 4
    targets: Optional[Any] = None
    material: Optional[int] = None
    polydata: Optional[PolyData] = None  # Custom Attribute

    @classmethod
    def set_polydata(cls, polydata):
        cls.polydata = polydata
        return Primitive(polydata=polydata)


@dataclass
class Mesh:
    primitives: List[Primitive] = field(default_factory=list)
    weights: Optional[List[float]] = None
    name: Optional[str] = None


@dataclass
class Perspective:
    aspectRatio: Optional[float] = None
    yfov: Optional[float] = None
    zfar: Optional[float] = None
    znear: Optional[float] = None


@dataclass
class Orthographic:
    xmag: Optional[float] = None
    ymag: Optional[float] = None
    zfar: Optional[float] = None
    znear: Optional[float] = None


@dataclass
class Camera:
    type: str
    perspective: Optional[Perspective] = None
    orthographic: Optional[Orthographic] = None


@dataclass
class Accessor:
    componentType: int
    type: str
    bufferView: Optional[int] = None
    byteOffset: Optional[int] = 0
    normalized: Optional[bool] = False
    count: Optional[int] = None
    max: Optional[List[float]] = field(default_factory=list)
    min: Optional[List[float]] = field(default_factory=list)
    sparse: Optional[Any] = None
    name: Optional[str] = None


@dataclass
class BufferView:
    buffer: int
    byteLength: int
    byteOffset: Optional[int] = 0
    byteStride: Optional[int] = None
    target: Optional[int] = None
    name: Optional[str] = None


@dataclass
class Buffer:
    byteLength: int
    uri: Optional[str] = None
    name: Optional[str] = None


@dataclass
class Scene:
    nodes: List[int] = field(default_factory=list)


@dataclass
class glTF:
    scene: int
    scenes: List[Scene] = field(default_factory=list)
    accessors: Optional[List[Accessor]] = None
    animations: Optional[Dict] = None
    asset: Optional[Dict] = None
    bufferViews: Optional[List[BufferView]] = None
    buffers: Optional[List[Buffer]] = None
    cameras: Optional[List[Camera]] = None
    images: Optional[Dict] = None
    materials: Optional[List[Material]] = None
    meshes: Optional[List[Mesh]] = None
    nodes: Optional[List[Node]] = None
    samplers: Optional[Dict] = None
    skins: Optional[Dict] = None
    textures: Optional[Dict] = None


class glTFImporter:

    def __init__(self, filename, apply_normals=False):

        fp = open(filename)
        self.json = glTF(** j.load(fp))
        fp.close()

        gltf = filename.split('/')[-1:][0]
        self.pwd = filename[:-len(gltf)]
        self.apply_normals = apply_normals

        self.materials = {}
        self.nodes = {}
        self.meshes = {}
        self.primitives = []
        self.cameras = {}
        self.transforms = {}
        self.actors = []
        self.init_transform = np.identity(4)
        self.get_nodes(0)

    def get_actors(self):
        for polydatas in self.primitives:
            actor = utils.get_actor_from_polydata(polydatas.polydata)

            if bool(self.materials):
                baseColorTexture = self.materials[0].baseColorTexture
                actor.SetTexture(baseColorTexture)

            self.actors.append(actor)

        return self.actors

    def get_nodes(self, scene_id=0):
        scene = self.json.scenes[scene_id]
        nodes = scene.get('nodes')

        for node_id in nodes:
            self.transverse_node(node_id, self.init_transform)

    def transverse_node(self, nextnode_id, matrix):
        node = Node(** self.json.nodes[nextnode_id])

        matnode = np.identity(4)
        if not (node.matrix is None):
            matnode = np.array(node.matrix)
            matnode = matnode.reshape(-1, 4).T
        else:
            if not (node.translation is None):
                trans = node.translation
                T = transform.translate(trans)
                matnode = np.dot(matnode, T)

            if not (node.rotation is None):
                rot = node.rotation
                R = transform.rotate(rot)
                matnode = np.dot(matnode, R)

            if not (node.scale is None):
                scales = node.scale
                S = transform.scale(scales)
                matnode = np.dot(matnode, S)

        next_matrix = np.dot(matrix, matnode)

        if not (node.mesh is None):
            mesh_id = node.mesh
            self.meshes[mesh_id] = Mesh(** self.json.meshes[mesh_id])
            self.load_mesh(mesh_id, next_matrix)
            self.transforms[mesh_id] = next_matrix

        if not (node.camera is None):
            camera_id = node.camera
            # Todo -->
            self.load_camera(camera_id, nextnode_id)

        if node.children:
            for child_id in node.children:
                self.transverse_node(child_id, next_matrix)

    def load_mesh(self, mesh_id, transform_mat):
        primitives = self.meshes[mesh_id].primitives

        for primitive in primitives:

            primitive = Primitive(** primitive)
            attributes = primitive.attributes

            position_id = attributes.get('POSITION')
            normal_id = attributes.get('NORMAL')
            texcoord_id = attributes.get('TEXCOORD_0')
            color_id = attributes.get('COLOR_0')
            indices_id = primitive.indices
            material_id = primitive.material

            vertices = self.get_acc_data(position_id)
            vertices = transform.apply_transfomation(vertices, transform_mat.T)

            polydata = utils.PolyData()
            utils.set_polydata_vertices(polydata, vertices)

            if not (normal_id is None) and self.apply_normals:
                normals = self.get_acc_data(normal_id)
                utils.set_polydata_normals(polydata, normals)

            if not (indices_id is None):
                indices = self.get_acc_data(indices_id).reshape(-1, 3)
                utils.set_polydata_triangles(polydata, indices)

            if not (texcoord_id is None):
                uv = self.get_acc_data(texcoord_id)
                polydata.GetPointData().SetTCoords(
                    utils.numpy_support.numpy_to_vtk(uv))

            if not (color_id is None):
                color = self.get_acc_data(color_id)
                color = color[:, :-1]*255
                utils.set_polydata_colors(polydata, color)

            if not (material_id is None):
                material = self.get_materials(material_id)
                self.materials[mesh_id] = material

            prim = Primitive.set_polydata(polydata)
            self.primitives.append(prim)

    def get_acc_data(self, acc_id):

        accessor = Accessor(** self.json.accessors[acc_id])

        buffview_id = accessor.bufferView
        acc_byte_offset = accessor.byteOffset
        count = accessor.count
        d_type = comp_type.get(accessor.componentType)
        d_size = d_type['size']
        a_type = acc_type.get(accessor.type)

        buffview = BufferView(** self.json.bufferViews[buffview_id])

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

        buffer = Buffer(** self.json.buffers[buff_id])
        uri = buffer.uri
        dtype = np.dtype('B')

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

        material = Material(** self.json.materials[mat_id])
        bct, mrt, nt = None, None, None

        pbr = material.pbrMetallicRoughness

        if 'baseColorTexture' in pbr:
            bct = pbr.get('baseColorTexture')['index']
            bct = self.get_texture(bct)

        if 'metallicRoughnesstexture' in pbr:
            mrt = pbr.get('metallicRoughnessTexture')['index']
            mrt = self.get_texture(mrt)

        if not (material.normalTexture is None):
            nt = material.normalTexture['index']
            nt = self.get_texture(nt)

        return Material(baseColorTexture=bct, metallicRoughnessTexture=mrt)

    def get_texture(self, tex_id):

        textures = self.json.textures
        images = self.json.images

        reader_type = {
            '.jpg': JPEGReader,
            '.jpeg': JPEGReader,
            '.png': PNGReader
        }
        file = images[textures[tex_id]['source']]['uri']

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

    def load_camera(self, camera_id, node_id):
        camera = Camera(** self.json.cameras[camera_id])
        self.cameras[node_id] = camera
