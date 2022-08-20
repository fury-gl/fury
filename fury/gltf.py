# TODO: Materials, Lights, Animations
import base64
import os
import numpy as np
import pygltflib as gltflib
from pygltflib.utils import glb2gltf
from PIL import Image
from fury.lib import Texture, Camera
from fury import transform, utils, io
from fury.animation.timeline import Timeline
from fury.animation.interpolator import (linear_interpolator, lerp,
                                         step_interpolator, slerp)
from fury.animation import helpers

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
    'VEC4': 4,
    'MAT4': 16
}
temp = []


class glTF:

    def __init__(self, filename, apply_normals=False):
        """Read and generate actors from glTF files.

        Parameters
        ----------
        filename : str
            Path of the gltf file
        apply_normals : bool, optional
            If `True` applies normals to the mesh.

        """
        if filename in ['', None]:
            raise IOError('Filename cannot be empty or None!')

        name, extension = os.path.splitext(filename)

        if extension == '.glb':
            fname_gltf = f'{name}.gltf'
            if not os.path.exists(fname_gltf):
                glb2gltf(filename)
                filename = fname_gltf

        self.gltf = gltflib.GLTF2().load(filename)

        self.pwd = os.path.dirname(filename)
        self.apply_normals = apply_normals

        self.cameras = {}
        self.actors_list = []
        self.materials = []
        self.nodes = []
        self.transformations = []
        self.polydatas = []
        self.init_transform = np.identity(4)
        self.animations = []
        self.node_transform = []
        self.keyframe_transforms = []
        self.joints_0 = []
        self.weights_0 = []
        self.bones = []
        self.ibms = []
        self.inspect_scene(0)

    def actors(self):
        """Generate actors from glTF file.

        Returns
        -------
        actors : list
            List of vtkActors with texture.

        """
        for i, polydata in enumerate(self.polydatas):
            actor = utils.get_actor_from_polydata(polydata)
            transform_mat = self.transformations[i]
            position, rot, scale = transform.trs_from_matrix(transform_mat)

            actor.SetPosition(position)
            actor.SetScale(scale)
            actor.RotateWXYZ(*rot)

            if self.materials[i] is not None:
                base_col_tex = self.materials[i]['baseColorTexture']
                actor.SetTexture(base_col_tex)

            self.actors_list.append(actor)

        return self.actors_list

    def inspect_scene(self, scene_id=0):
        """Loop over nodes in a scene.

        Parameters
        ----------
        scene_id : int, optional
            scene index of the glTF.

        """
        scene = self.gltf.scenes[scene_id]
        nodes = scene.nodes

        for node_id in nodes:
            self.transverse_node(node_id, self.init_transform)
        for animation in self.gltf.animations:
            self.transverse_channels(animation)

    def transverse_node(self, nextnode_id, matrix, parent=None):
        """Load mesh and generates transformation matrix.

        Parameters
        ----------
        nextnode_id : int
            Index of the node
        matrix : ndarray (4, 4)
            Transformation matrix
        parent : list, optional
            List of indices of parent nodes
            Default: None.

        """
        node = self.gltf.nodes[nextnode_id]
        if parent is None:
            parent = [nextnode_id]
        else:
            parent.append(nextnode_id)
        matnode = np.identity(4)

        if node.matrix is not None:
            matnode = np.array(node.matrix)
            matnode = matnode.reshape(-1, 4).T
        else:
            if node.translation is not None:
                trans = node.translation
                translate = transform.translate(trans)
                matnode = np.dot(matnode, translate)

            if node.rotation is not None:
                rot = node.rotation
                rotate = transform.rotate(rot)
                matnode = np.dot(matnode, rotate)

            if node.scale is not None:
                scales = node.scale
                scale = transform.scale(scales)
                matnode = np.dot(matnode, scale)

        next_matrix = np.dot(matrix, matnode)

        if node.mesh is not None:
            mesh_id = node.mesh
            self.load_mesh(mesh_id, next_matrix, parent)

        if node.skin is not None:
            skin_id = node.skin
            joints, ibms = self.get_skin_data(skin_id)
            self.bones.append(joints)  # for each skin will contain nodes
            self.ibms.append(ibms)
            for joint, ibm_matrix in zip(joints, ibms):
                print(joint, ibm_matrix)
                self.transverse_node(joint, ibm_matrix.T, parent)

        if node.camera is not None:
            camera_id = node.camera
            self.load_camera(camera_id, next_matrix)

        if node.children:
            for child_id in node.children:
                self.transverse_node(child_id, next_matrix, parent)

    def load_mesh(self, mesh_id, transform_mat, parent):
        """Load the mesh data from accessor and applies the transformation.

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
            # vertices = transform.apply_transfomation(vertices, transform_mat)
            self.transformations.append(transform_mat)

            polydata = utils.PolyData()
            utils.set_polydata_vertices(polydata, vertices)

            if attributes.NORMAL is not None and self.apply_normals:
                normals = self.get_acc_data(attributes.NORMAL)
                utils.set_polydata_normals(polydata, normals)

            if attributes.TEXCOORD_0 is not None:
                tcoords = self.get_acc_data(attributes.TEXCOORD_0)
                utils.set_polydata_tcoords(polydata, tcoords)

            if attributes.COLOR_0 is not None:
                color = self.get_acc_data(attributes.COLOR_0)
                color = color[:, :-1]*255
                utils.set_polydata_colors(polydata, color)

            if primitive.indices is not None:
                indices = self.get_acc_data(primitive.indices).reshape(-1, 3)
                utils.set_polydata_triangles(polydata, indices)
            
            if attributes.JOINTS_0 is not None:
                vertex_joints = self.get_acc_data(attributes.JOINTS_0)
                vertex_joints = vertex_joints.reshape(-1, 4)
                # add None if weights and joints do not exist
                self.joints_0.append(vertex_joints)
                vertex_weight = self.get_acc_data(attributes.WEIGHTS_0)
                self.weights_0.append(vertex_weight)

            material = None
            if primitive.material is not None:
                material = self.get_materials(primitive.material)

            self.polydatas.append(polydata)
            self.nodes.append(parent[:])
            self.materials.append(material)

    def get_acc_data(self, acc_id):
        """Get the correct data from buffer using accessors and bufferviews.

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
        """Extract the mesh data from buffer.

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

        except IOError as e:
            print(f'Failed to read ! Error in opening file: {e}')

    def get_materials(self, mat_id):
        """Get the materials data.

        Parameters
        ----------
        mat_id : int
            Material index

        Returns
        -------
        materials : dict
            Dictionary of all textures.

        """
        material = self.gltf.materials[mat_id]
        bct = None

        pbr = material.pbrMetallicRoughness

        if pbr.baseColorTexture is not None:
            bct = pbr.baseColorTexture.index
            bct = self.get_texture(bct)

        return {'baseColorTexture': bct}

    def get_texture(self, tex_id):
        """Read and convert image into vtk texture.

        Parameters
        ----------
        tex_id : int
            Texture index

        Returns
        -------
        atexture : Texture
            Returns flipped vtk texture from image.

        """
        texture = self.gltf.textures[tex_id].source
        image = self.gltf.images[texture]

        file = image.uri
        bv_index = image.bufferView
        if file is None:
            mimetype = image.mimeType

        if file is not None and file.startswith('data:image'):
            buff_data = file.split(',')[1]
            buff_data = base64.b64decode(buff_data)

            extension = '.png' if file.startswith('data:image/png') else '.jpg'
            image_path = os.path.join(self.pwd, str("b64texture"+extension))
            with open(image_path, "wb") as image_file:
                image_file.write(buff_data)

        elif bv_index is not None:
            bv = self.gltf.bufferViews[bv_index]
            buffer = bv.buffer
            bo = bv.byteOffset
            bl = bv.byteLength
            uri = self.gltf.buffers[buffer].uri
            with open(os.path.join(self.pwd, uri), 'rb') as f:
                f.seek(bo)
                img_binary = f.read(bl)
            extension = '.png' if mimetype == 'images/png' else '.jpg'
            image_path = os.path.join(self.pwd, str("bvtexture"+extension))
            with open(image_path, "wb") as image_file:
                image_file.write(img_binary)

        else:
            image_path = os.path.join(self.pwd, file)

        rgb = io.load_image(image_path)
        grid = utils.rgb_to_vtk(rgb)
        atexture = Texture()
        atexture.InterpolateOn()
        atexture.EdgeClampOn()
        atexture.SetInputDataObject(grid)

        return atexture

    def load_camera(self, camera_id, transform_mat):
        """Load the camera data of a node.

        Parameters
        ----------
        camera_id : int
            Camera index of a node.
        transform_mat : ndarray (4, 4)
            Transformation matrix of the camera.

        """
        camera = self.gltf.cameras[camera_id]
        vtk_cam = Camera()
        position = vtk_cam.GetPosition()
        position = np.asarray([position])

        new_position = transform.apply_transfomation(position, transform_mat)
        vtk_cam.SetPosition(tuple(new_position[0]))

        if camera.type == "orthographic":
            orthographic = camera.orthographic
            vtk_cam.ParallelProjectionOn()
            zfar = orthographic.zfar
            znear = orthographic.znear
            vtk_cam.SetClippingRange(znear, zfar)
        else:
            perspective = camera.perspective
            vtk_cam.ParallelProjectionOff()
            zfar = perspective.zfar if perspective.zfar else 1000.0
            znear = perspective.znear
            vtk_cam.SetClippingRange(znear, zfar)
            angle = perspective.yfov*180/np.pi if perspective.yfov else 30.0
            vtk_cam.SetViewAngle(angle)
            if perspective.aspectRatio:
                vtk_cam.SetExplicitAspectRatio(perspective.aspectRatio)

        self.cameras[camera_id] = vtk_cam

    def transverse_channels(self, animation: gltflib.Animation):
        """Loops over animation channels and sets animation data.

        Parameters
        ----------
        animation : glTflib.Animation
            pygltflib animation object.
        """
        name = animation.name
        for channel in animation.channels:
            sampler = animation.samplers[channel.sampler]
            node_id = channel.target.node
            path = channel.target.path
            anim_data = self.get_sampler_data(sampler, node_id, path)
            self.node_transform.append(anim_data)

    def get_sampler_data(self, sampler: gltflib.Sampler, node_id: int,
                         transform_type):
        """Gets the timeline and transformation data from sampler.

        Parameters
        ----------
        sampler : glTFlib.Sampler
            pygltflib sampler object.
        node_id : int
            Node index of the current animation channel.
        transform_type : str
            Property of the node to be transformed.

        Returns
        -------
        sampler_data : dict
            dictionary of data containing timestamps, node transformations and
            interpolation type.
        """
        time_array = self.get_acc_data(sampler.input)
        transform_array = self.get_acc_data(sampler.output)
        interpolation = sampler.interpolation

        return {
            'node': node_id,
            'input': time_array,
            'output': transform_array,
            'interpolation': interpolation,
            'property': transform_type}

    def get_skin_data(self, skin_id):
        skin = self.gltf.skins[skin_id]
        inv_bind_matrix = self.get_acc_data(skin.inverseBindMatrices)
        # print(inv_bind_matrix)
        inv_bind_matrix = inv_bind_matrix.reshape((-1, 4, 4))
        # print(f'ibm:\n{inv_bind_matrix}')
        joint_nodes = skin.joints
        return joint_nodes, inv_bind_matrix

    def generate_tmatrix(self, transf, prop):
        if prop == 'translation':
            matrix = transform.translate(transf)
        elif prop == 'rotation':
            matrix = transform.rotate(transf)
        elif prop == 'scale':
            matrix = transform.scale(transf)
        return matrix

    def apply_skin_matrix(self, vertices,
                          joint_matrices, bones):
        """Applies the skinnig matrix, that transforms the vertices.

        NOTE: vertices has joint_matrix applied already.
        Returns
        -------
        vertices : ndarray
            Modified vertices
        """
        # self.precalculate_transforms()
        # keyframe_transfoms = self.keyframe_transforms
        clone = np.copy(vertices)
        weights = self.weights_0[0]
        joints = self.joints_0[0]

        # vertices = transform.apply_transfomation(vertices, joint_matrix)
        for i, xyz in enumerate(clone):
            vweight = weights[i]
            pos = np.array([0, 0, 0])
            for j, bone in enumerate(bones):
                # print(weights[j])
                # try appply inv bind matrix
                temp = transform.apply_transfomation(
                        np.array([xyz]), joint_matrices[j])[0]
                pos = np.add(pos, temp*vweight[j])
            clone[i] = pos
        return clone

    def get_skin_timelines(self):
        """Returns list of animation timeline.

        Returns
        -------
        timelines : List
            List of timelines containing actors.
        """
        # actors = self.actors()

        timelines = []
        timeline = Timeline(playback_panel=True)
        for transforms in self.node_transform:
            target_node = transforms['node']

            for i, nodes in enumerate(self.bones[0]):
                # timeline = Timeline(playback_panel=True)

                if target_node == nodes:
                    timestamp = transforms['input']
                    transform = transforms['output']
                    prop = transforms['property']
                    for time, trs in zip(timestamp, transform):
                        matrix = self.generate_tmatrix(trs, prop)
                        # print(matrix)
                        timeline.set_keyframe(f'transform{i}', time[0], matrix)
                        # timeline.set_interpolator(f'transform{i}', slerp)
                else:
                    transform = np.identity(4)
                    timeline.set_keyframe(f'transform{i}', 0, transform)

        timelines.append(timeline)
        return timeline

    def get_animation_timelines(self):
        """Returns list of animation timeline.

        Returns
        -------
        timelines : List
            List of timelines containing actors.
        """
        actors = self.actors()
        interpolators = {
            'LINEAR': linear_interpolator,
            'STEP': step_interpolator,
            'CUBICSPLINE': tan_cubic_spline_interpolator
        }

        rotation_interpolators = {
            'LINEAR': slerp,
            'STEP': step_interpolator,
            'CUBICSPLINE': tan_cubic_spline_interpolator
        }

        timelines = []
        for transforms in self.node_transform:
            target_node = transforms['node']
            print(transforms)
            for i, nodes in enumerate(self.nodes):
                timeline = Timeline()

                if target_node in nodes:
                    timeline.add_actor(actors[i])
                    timestamp = transforms['input']
                    transform = transforms['output']
                    prop = transforms['property']
                    interpolation_type = transforms['interpolation']

                    interpolator = interpolators.get(interpolation_type)
                    rot_interp = rotation_interpolators.get(
                                       interpolation_type)
                    timeshape = timestamp.shape
                    transhape = transform.shape
                    if transforms['interpolation'] == 'CUBICSPLINE':
                        transform = transform.reshape(
                            (timeshape[0], -1, transhape[1]))

                    for time, trs in zip(timestamp, transform):
                        in_tan, out_tan = None, None
                        if trs.ndim == 2:
                            cubicspline = trs
                            in_tan = cubicspline[0]
                            trs = cubicspline[1]
                            out_tan = cubicspline[2]

                        if prop == 'rotation':
                            timeline.set_rotation(time[0], trs,
                                                  in_tangent=in_tan,
                                                  out_tangent=out_tan)
                            timeline.set_rotation_interpolator(rot_interp)
                        if prop == 'translation':
                            timeline.set_position(time[0], trs,
                                                  in_tangent=in_tan,
                                                  out_tangent=out_tan)
                            timeline.set_position_interpolator(interpolator)
                        if prop == 'scale':
                            timeline.set_scale(time[0], trs,
                                               in_tangent=in_tan,
                                               out_tangent=out_tan)
                            timeline.set_scale_interpolator(interpolator)
                else:
                    timeline.add_static_actor(actors[i])

                timelines.append(timeline)
        return timelines

    def get_main_timeline(self):
        """Returns main timeline with all animations.
        """
        main_timeline = Timeline(playback_panel=True)
        timelines = self.get_animation_timelines()
        for timeline in timelines:
            main_timeline.add_timeline(timeline)
        return main_timeline

    def get_skin_timeline(self):
        """Returns main timeline with all animations.
        """
        main_timeline = Timeline(playback_panel=True)
        timelines = self.get_skin_timelines()
        for timeline in timelines:
            main_timeline.add_timeline(timeline)
        return main_timeline


def tan_cubic_spline_interpolator(keyframes):

    timestamps = helpers.get_timestamps_from_keyframes(keyframes)
    for time in keyframes:
        data = keyframes.get(time)
        value = data.get('value')
        if data.get('in_tangent') is None:
            data['in_tangent'] = np.zeros_like(value)
        if data.get('in_tangent') is None:
            data['in_tangent'] = np.zeros_like(value)

    def interpolate(t):
        t0 = helpers.get_previous_timestamp(timestamps, t)
        t1 = helpers.get_next_timestamp(timestamps, t)

        dt = helpers.get_time_tau(t, t0, t1)

        time_delta = t1 - t0

        p0 = keyframes.get(t0).get('value')
        tan_0 = keyframes.get(t0).get('out_tangent') * time_delta
        p1 = keyframes.get(t1).get('value')
        tan_1 = keyframes.get(t1).get('in_tangent') * time_delta
        # cubic spline equation using tangents
        t2 = dt * dt
        t3 = t2 * dt
        return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + dt) * tan_0 + (
                -2 * t3 + 3 * t2) * p1 + (t3 - t2) * tan_1
    return interpolate


def transformation_interpolator(keyframes):
    timestamps = helpers.get_timestamps_from_keyframes(keyframes)
    for time in keyframes:
        data = keyframes.get(time)
        if data.get('in_matrix1') is None:
            data['in_matrix1'] = np.identity(4)
        if data.get('out_matrix1') is None:
            data['out_matrix1'] = np.identity(4)
        if data.get('in_matrix2') is None:
            data['in_matrix2'] = np.identity(4)
        if data.get('out_matrix2') is None:
            data['out_matrix2'] = np.identity(4)

    def interpolate(t):
        t0 = helpers.get_previous_timestamp(timestamps, t)
        t1 = helpers.get_next_timestamp(timestamps, t)

        mat_0 = keyframes.get(t0).get('in_matrix1')
        mat_1 = keyframes.get(t1).get('out_matrix1')

        mat_2 = keyframes.get(t0).get('in_matrix2')
        mat_3 = keyframes.get(t1).get('out_matrix2')

        out_1 = lerp(mat_0, mat_1, t0, t1, t)
        out_2 = lerp(mat_2, mat_3, t0, t1, t)

        return (out_1, out_2)
    return interpolate


def export_scene(scene, filename='default.gltf'):
    """Generate gltf from FURY scene.

    Parameters
    ----------
    scene : Scene
        FURY scene object.
    filename : str, optional
        Name of the model to be saved
    """
    gltf_obj = gltflib.GLTF2()
    name, extension = os.path.splitext(filename)

    if extension not in ['.gltf', '.glb']:
        raise IOError('Filename should be .gltf or .glb')

    buffer_file = open(f'{name}.bin', 'wb')
    primitives = []
    buffer_size = 0
    bview_count = 0

    for act in scene.GetActors():
        prim, size, count = _connect_primitives(gltf_obj, act, buffer_file,
                                                buffer_size, bview_count, name)
        primitives.append(prim)
        buffer_size = size
        bview_count = count

    buffer_file.close()
    write_mesh(gltf_obj, primitives)
    write_buffer(gltf_obj, size, f'{name}.bin')
    camera = scene.camera()
    cam_id = None
    if camera:
        write_camera(gltf_obj, camera)
        cam_id = 0
    write_node(gltf_obj, mesh_id=0, camera_id=cam_id)
    write_scene(gltf_obj, [0])

    gltf_obj.save(f'{name}.gltf')
    if extension == '.glb':
        gltf2glb(f'{name}.gltf', destination=filename)


def _connect_primitives(gltf, actor, buff_file, byteoffset, count, name):
    """Create Accessor, BufferViews and writes primitive data to a binary file

    Parameters
    ----------
    gltf : Pygltflib.GLTF2
    actor : Actor
        the fury actor
    buff_file : file
        filename.bin opened in `wb` mode
    byteoffset : int
        offset of the bufferview
    count : int
        BufferView count
    name : str
        Prefix of the gltf filename

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
    colors = utils.get_polydata_colors(polydata)
    normals = utils.get_polydata_normals(polydata)
    tcoords = utils.get_polydata_tcoord(polydata)
    try:
        indices = utils.get_polydata_triangles(polydata)
    except AssertionError as error:
        indices = None
        print(error)

    ispoints = polydata.GetNumberOfVerts()
    islines = polydata.GetNumberOfLines()
    istraingles = polydata.GetNumberOfPolys()

    if ispoints:
        mode = 0
    elif islines:
        mode = 3
    elif istraingles:
        mode = 4

    vertex, index, normal, tcoord, color = (None, None, None,
                                            None, None)
    if indices is not None and len(indices) != 0:
        indices = indices.reshape((-1, ))
        amax = [np.max(indices)]
        amin = [np.min(indices)]

        ctype = comp_type.get(gltflib.UNSIGNED_SHORT)
        atype = acc_type.get(gltflib.SCALAR)

        indices = indices.astype(np.ushort)
        blength = len(indices)*ctype['size']
        buff_file.write(indices.tobytes())
        write_bufferview(gltf, 0, byteoffset, blength)
        write_accessor(gltf, count, 0, gltflib.UNSIGNED_SHORT,
                       len(indices), gltflib.SCALAR)
        byteoffset += blength
        index = count
        count += 1

    if vertices is not None:
        amax = np.max(vertices, 0).tolist()
        amin = np.min(vertices, 0).tolist()

        ctype = comp_type.get(gltflib.FLOAT)
        atype = acc_type.get(gltflib.VEC3)

        vertices = vertices.reshape((-1, )).astype(ctype['dtype'])
        blength = len(vertices)*ctype['size']
        buff_file.write(vertices.tobytes())
        write_bufferview(gltf, 0, byteoffset, blength)
        write_accessor(gltf, count, 0, gltflib.FLOAT, len(vertices)//atype,
                       gltflib.VEC3, amax, amin)
        byteoffset += blength
        vertex = count
        count += 1

    if normals is not None:
        amax = np.max(normals, 0).tolist()
        amin = np.min(normals, 0).tolist()

        ctype = comp_type.get(gltflib.FLOAT)
        atype = acc_type.get(gltflib.VEC3)

        normals = normals.reshape((-1, ))
        blength = len(normals)*ctype['size']
        buff_file.write(normals.tobytes())
        write_bufferview(gltf, 0, byteoffset, blength)
        write_accessor(gltf, count, 0, gltflib.FLOAT, len(normals)//atype,
                       gltflib.VEC3, amax, amin)
        byteoffset += blength
        normal = count
        count += 1

    if tcoords is not None:
        amax = np.max(tcoords, 0).tolist()
        amin = np.min(tcoords, 0).tolist()

        ctype = comp_type.get(gltflib.FLOAT)
        atype = acc_type.get(gltflib.VEC2)

        tcoords = tcoords.reshape((-1, )).astype(ctype['dtype'])
        blength = len(tcoords)*ctype['size']
        buff_file.write(tcoords.tobytes())
        write_bufferview(gltf, 0, byteoffset, blength)
        write_accessor(gltf, count, 0, gltflib.FLOAT, len(tcoords)//atype,
                       gltflib.VEC2)
        byteoffset += blength
        tcoord = count
        count += 1
        vtk_image = actor.GetTexture().GetInput()
        rows, cols, _ = vtk_image.GetDimensions()
        scalars = vtk_image.GetPointData().GetScalars()
        np_im = numpy_support.vtk_to_numpy(scalars)
        np_im = np.reshape(np_im, (rows, cols, -1))

        img = Image.fromarray(np_im)
        image_path = f'{name}BaseColorTexture.png'
        img.save(image_path)
        write_material(gltf, 0, image_path)

    if colors is not None:
        ctype = comp_type.get(gltflib.FLOAT)
        atype = acc_type.get(gltflib.VEC3)

        shape = colors.shape[0]
        colors = np.concatenate((colors, np.full((shape, 1), 255.)), axis=1)
        colors = colors / 255
        colors = colors.reshape((-1, )).astype(ctype['dtype'])
        blength = len(colors)*ctype['size']
        buff_file.write(colors.tobytes())
        write_bufferview(gltf, 0, byteoffset, blength)
        write_accessor(gltf, count, 0, gltflib.FLOAT, shape, gltflib.VEC4)
        byteoffset += blength
        color = count
        count += 1
    material = None if tcoords is None else 0
    prim = get_prim(vertex, index, color, tcoord, normal, material, mode)
    return prim, byteoffset, count


def write_scene(gltf, nodes):
    """Create scene

    Parameters
    ----------
    gltf : GLTF2
        Pygltflib GLTF2 object
    nodes : list
        List of node indices.
    """
    scene = gltflib.Scene()
    scene.nodes = nodes
    gltf.scenes.append(scene)


def write_node(gltf, mesh_id=None, camera_id=None):
    """Create node

    Parameters
    ----------
    gltf : GLTF2
        Pygltflib GLTF2 object
    mesh_id : int, optional
        Mesh index
    camera_id : int, optional
        Camera index.
    """
    node = gltflib.Node()
    if mesh_id is not None:
        node.mesh = mesh_id
    if camera_id is not None:
        node.camera = camera_id
    gltf.nodes.append(node)


def write_mesh(gltf, primitives):
    """Create mesh and add primitive.

    Parameters
    ----------
    gltf : GLTF2
        Pygltflib GLTF2 object.
    primitives : list
        List of Primitive object.
    """
    mesh = gltflib.Mesh()
    for prim in primitives:
        mesh.primitives.append(prim)

    gltf.meshes.append(mesh)


def write_camera(gltf, camera):
    """Create and add camera.

    Parameters
    ----------
    gltf : GLTF2
        Pygltflib GLTF2 object.
    camera : vtkCamera
        scene camera.
    """
    orthographic = camera.GetParallelProjection()
    cam = gltflib.Camera()
    if orthographic:
        cam.type = "orthographic"
    else:
        clip_range = camera.GetClippingRange()
        angle = camera.GetViewAngle()
        ratio = camera.GetExplicitAspectRatio()
        aspect_ratio = ratio if ratio else 1.0
        pers = gltflib.Perspective()
        pers.aspectRatio = aspect_ratio
        pers.znear, pers.zfar = clip_range
        pers.yfov = angle * np.pi/180
        cam.type = "perspective"
        cam.perspective = pers
    gltf.cameras.append(cam)


def get_prim(vertex, index, color, tcoord, normal, material, mode=4):
    """Return a Primitive object.

    Parameters
    ----------
    vertex : int
        Accessor index for the vertices data.
    index : int
        Accessor index for the triangles data.
    color : int
        Accessor index for the colors data.
    tcoord : int
        Accessor index for the texture coordinates data.
    normal : int
        Accessor index for the normals data.
    material : int
        Materials index.
    mode : int, optional
        The topology type of primitives to render.
        Default: 4

    Returns
    -------
    prim : Primitive
        pygltflib primitive object.
    """
    prim = gltflib.Primitive()
    attr = gltflib.Attributes()
    attr.POSITION = vertex
    attr.NORMAL = normal
    attr.TEXCOORD_0 = tcoord
    attr.COLOR_0 = color
    prim.attributes = attr
    prim.indices = index
    if material is not None:
        prim.material = material
    prim.mode = mode
    return prim


def write_material(gltf, basecolortexture: int, uri: str):
    """Write Material, Images and Textures

    Parameters
    ----------
    gltf : GLTF2
        Pygltflib GLTF2 object.
    basecolortexture : int
        BaseColorTexture index.
    uri : str
        BaseColorTexture uri.
    """
    material = gltflib.Material()
    texture = gltflib.Texture()
    image = gltflib.Image()
    pbr = gltflib.PbrMetallicRoughness()
    tinfo = gltflib.TextureInfo()
    tinfo.index = basecolortexture
    pbr.baseColorTexture = tinfo
    pbr.metallicFactor = 0.0
    material.pbrMetallicRoughness = pbr
    texture.source = basecolortexture
    image.uri = uri
    gltf.materials.append(material)
    gltf.textures.append(texture)
    gltf.images.append(image)


def write_accessor(gltf, bufferview, byte_offset, comp_type,
                   count, accssor_type, max=None, min=None):
    """Write accessor in the gltf.

    Parameters
    ----------
    gltf : GLTF2
        Pygltflib GLTF2 objecomp_type

                      bufferview : int
        BufferView Index
    byte_offset : int
        ByteOffset of the accessor
    comp_type : type
        Type of a single component
    count : int
        Elements count of the accessor
    accssor_type : type
        Type of the accessor (SCALAR, VEC2, VEC3, VEC4)
    max : ndarray, optional
        Maximum elements of an array
    min : ndarray, optional
        Minimum elements of an array
    """
    accessor = gltflib.Accessor()
    accessor.bufferView = bufferview
    accessor.byteOffset = byte_offset
    accessor.componentType = comp_type
    accessor.count = count
    accessor.type = accssor_type
    if (max is not None) and (min is not None):
        accessor.max = max
        accessor.min = min
    gltf.accessors.append(accessor)


def write_bufferview(gltf, buffer, byte_offset, byte_length,
                     byte_stride=None):
    """Write bufferview in the gltf.

    Parameters
    ----------
    gltf : GLTF2
        Pygltflib GLTF2 object
    buffer : int
        Buffer index
    byte_offset : int
        Byte offset of the bufferview
    byte_length : int
        Byte length ie, Length of the data we want to get from
        the buffer
    byte_stride : int, optional
        Byte stride of the bufferview.
    """
    buffer_view = gltflib.BufferView()
    buffer_view.buffer = buffer
    buffer_view.byteOffset = byte_offset
    buffer_view.byteLength = byte_length
    buffer_view.byteStride = byte_stride
    gltf.bufferViews.append(buffer_view)


def write_buffer(gltf, byte_length, uri):
    """Write buffer int the gltf

    Parameters
    ----------
    gltf : GLTF2
        Pygltflib GLTF2 object
    byte_length : int
        Length of the buffer
    uri : str
        Path to the external `.bin` file.
    """
    buffer = gltflib.Buffer()
    buffer.uri = uri
    buffer.byteLength = byte_length
    gltf.buffers.append(buffer)
