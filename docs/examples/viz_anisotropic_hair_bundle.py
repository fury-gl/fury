import numpy as np
import os

from dipy.io.streamline import load_tractogram
from dipy.data.fetcher import get_bundle_atlas_hcp842
from fury import actor, ui, window
from fury.data import fetch_viz_cubemaps, read_viz_cubemap
from fury.io import load_cubemap_texture
from fury.shaders import attribute_to_actor, shader_to_actor
from fury.utils import (normals_from_actor,
                        tangents_from_direction_of_anisotropy,
                        tangents_to_actor, vertices_from_actor)


if __name__ == '__main__':
    global doa, light, light_panel, light_params, obj_actor, pbr_panel, \
        pbr_params, size

    fetch_viz_cubemaps()

    #texture_name = 'skybox'
    texture_name = 'brudslojan'
    textures = read_viz_cubemap(texture_name)

    cubemap = load_cubemap_texture(textures)

    """
    img_shape = (1024, 1024)

    # Flip horizontally
    img_grad = np.flip(np.tile(np.linspace(0, 255, num=img_shape[0]),
                               (img_shape[1], 1)).astype(np.uint8), axis=1)
    cubemap_side_img = np.stack((img_grad,) * 3, axis=-1)

    cubemap_top_img = np.ones((img_shape[0], img_shape[1], 3)).astype(
        np.uint8) * 255

    cubemap_bottom_img = np.zeros((img_shape[0], img_shape[1], 3)).astype(
        np.uint8)

    cubemap_imgs = [cubemap_side_img, cubemap_side_img, cubemap_top_img,
                    cubemap_bottom_img, cubemap_side_img, cubemap_side_img]

    cubemap = get_cubemap_from_ndarrays(cubemap_imgs, flip=False)
    """

    #cubemap.RepeatOff()
    #cubemap.EdgeClampOn()

    scene = window.Scene()

    #scene = window.Scene(skybox=cubemap)
    #scene.skybox(gamma_correct=False)

    #scene.background((1, 1, 1))

    # Scene rotation for brudslojan texture
    #scene.yaw(-110)

    atlas, bundles = get_bundle_atlas_hcp842()
    bundles_dir = os.path.dirname(bundles)
    tractograms = ['AC.trk', 'CC_ForcepsMajor.trk', 'CC_ForcepsMinor.trk',
                   'CCMid.trk', 'F_L_R.trk', 'MCP.trk', 'PC.trk', 'SCP.trk',
                   'V.trk']

    # Load tractogram
    tract_file = os.path.join(bundles_dir, tractograms[4])
    sft = load_tractogram(tract_file, 'same', bbox_valid_check=False)
    bundle = sft.streamlines

    # Bundle actor
    #obj_actor = actor.streamtube(bundle, linewidth=.25)

    """
    # Wireframe representation for streamtubes
    obj_actor.GetProperty().SetRepresentationToWireframe()

    scene.add(obj_actor)

    # Streamtube vertices
    vertices = vertices_from_actor(obj_actor)

    # Streamtubes normals
    normals = normals_from_actor(obj_actor)

    normal_len = .5
    normals_endpnts = vertices + normals * normal_len

    # View normals as dots
    #normal_actor = actor.dot(normals_endpnts, colors=(0, 0, 1))
    #scene.add(normal_actor)

    # View normals as lines
    normal_lines = [[vertices[i, :], normals_endpnts[i, :]] for i in
                    range(len(vertices))]
    normal_actor = actor.line(normal_lines, colors=(0, 0, 1))
    scene.add(normal_actor)

    # Streamtube tangents from direction of anisotropy
    doa = [0, 1, .5]
    tangents = tangents_from_direction_of_anisotropy(normals, doa)
    tangents_to_actor(obj_actor, tangents)

    tangent_len = .5
    tangents_endpnts = vertices + tangents * tangent_len

    # View tangents as dots
    #tangent_actor = actor.dot(tangents_endpnts, colors=(1, 0, 0))
    #scene.add(tangent_actor)

    # View tangents as lines
    tangent_lines = [[vertices[i, :], tangents_endpnts[i, :]] for i in
                     range(len(vertices))]
    tangent_actor = actor.line(tangent_lines, colors=(1, 0, 0))
    scene.add(tangent_actor)
    """

    tmp_line_idx = 107  # Shortest line
    #tmp_line_idx = 146  # Longest line
    tmp_line = bundle[tmp_line_idx]

    line_length = len(tmp_line)

    # TODO: Move to function and compute them for all the lines in the bundle
    tangents = np.empty((line_length, 3))
    for i in range(line_length - 1):
        dif = tmp_line[i + 1] - tmp_line[i]
        dist = np.sqrt(np.sum(dif ** 2))
        tangents[i, :] = dif / dist
    tangents[line_length - 1, :] = tangents[line_length - 2, :]

    """
    tangent_len = .1
    tangents_endpnts = tmp_line + tangents * tangent_len

    # View tangents as dots
    #tangent_actor = actor.dot(tangents_endpnts, colors=(1, 0, 0))
    #scene.add(tangent_actor)

    # View tangents as lines
    tangent_lines = [[tmp_line[i, :], tangents_endpnts[i, :]] for i in
                     range(len(tmp_line))]
    tangent_actor = actor.line(tangent_lines, colors=(1, 0, 0))
    scene.add(tangent_actor)
    """

    obj_actor = actor.line([tmp_line], lod=False)
    tangents_to_actor(obj_actor, tangents)
    attribute_to_actor(obj_actor, tangents, 'tangent')

    vs_dec_clip = \
    """
    uniform mat4 MCVCMatrix;
    """
    shader_to_actor(obj_actor, 'vertex', block='clip', decl_code=vs_dec_clip)

    vs_dec_vp = \
    """
    in vec3 tangent;
    
    out vec3 tangentVSOutput;
    
    float square(float x)
    {
        return x * x;
    }
    """
    vs_impl_vp = \
    """ 
    tangentVSOutput = tangent;
    """
    shader_to_actor(obj_actor, 'vertex', block='valuepass',
                    decl_code=vs_dec_vp, impl_code=vs_impl_vp)

    vs_impl_light = \
    """
    vec4 camPos = -MCVCMatrix[3] * MCVCMatrix;
    vec3 lightDir = normalize(vertexMC.xyz - camPos.xyz);
    float dotLN = sqrt(1 - square(dot(lightDir, tangent))); 
    """
    shader_to_actor(obj_actor, 'vertex', block='light',
                    impl_code=vs_impl_light)#, debug=True)
    #shader_to_actor(obj_actor, 'fragment', block='valuepass', debug=True)

    scene.add(obj_actor)

    window.show(scene)
