import os

import numpy as np

from fury import actor, window
from fury.lib import FloatArray, ImageData, Texture, numpy_support
from fury.shaders import (
    attribute_to_actor,
    compose_shader,
    import_fury_shader,
    shader_to_actor,
)
from fury.utils import set_polydata_tcoords, numpy_to_vtk_image_data


def np_array_to_vtk_img(data):
    grid = ImageData()
    grid.SetDimensions(data.shape[1], data.shape[0], 1)
    nd = data.shape[-1] if data.ndim == 3 else 1
    vtkarr = numpy_support.numpy_to_vtk(
        np.flip(data.swapaxes(0, 1), axis=1).reshape((-1, nd), order="F")
    )
    vtkarr.SetName("Image")
    grid.GetPointData().AddArray(vtkarr)
    grid.GetPointData().SetActiveScalars("Image")
    grid.GetPointData().Update()
    return grid


if __name__ == "__main__":
    scene = window.Scene()
    scene.background((1, 1, 1))

    centers = np.array([[-3.2, 0.9, 0.4], [-3.5, -0.5, 1], [-2.1, 0, 0.4]])
    # TODO: Add to texure
    dirs = np.array([[-0.2, 0.9, 0.4], [-0.5, -0.5, 1], [0.9, 0, 0.4]])
    # TODO: Compare against setting texture coordinates
    ids = np.array([1.0, 2.0, 3.0])

    box_actor = actor.box(centers=centers, directions=dirs)

    rep_centers = np.repeat(centers, 8, axis=0)
    rep_directions = np.repeat(dirs, 8, axis=0)
    rep_ids = np.repeat(ids, 8, axis=0)

    attribute_to_actor(box_actor, rep_centers, "center")

    attribute_to_actor(box_actor, rep_directions, "direction")
    attribute_to_actor(box_actor, rep_ids, "id")

    #--------------------------------------------------------------------------
    centers = np.array(
        [[-2.5, -2, 0], [-1, -2, 0], [1, -2, 0]]
    )
    scales = [0.5, 0.5, 0.5]

    texture_actor = actor.billboard(centers, colors=(1, 1, 1), scales=scales)
    actor_pd = texture_actor.GetMapper().GetInput()
    actor_box = box_actor.GetMapper().GetInput()

    uv_vals = np.array(
        [
            #[0, 0], [0, 1], [1, 1], [1, 0],
            [0, 2/3], [0, 1], [1, 1], [1, 2/3],  # Top left color
            [0, 1/3], [0, 2/3], [1, 2/3], [1, 1/3],  # Top right color
            [0, 0], [0, 1/3], [1, 1/3], [1, 0],  # Bottom left color
        ]
    )
    # fmt: on

    num_pnts = uv_vals.shape[0]

    t_coords = FloatArray()
    t_coords.SetNumberOfComponents(2)
    t_coords.SetNumberOfTuples(num_pnts)

    [t_coords.SetTuple(i, uv_vals[i]) for i in range(num_pnts)]
    set_polydata_tcoords(actor_pd, t_coords)

    uv_vals2 = np.array(
        [
            [0, 2 / 3], [0, 1], [1, 1], [1, 2 / 3], [0, 2 / 3], [0, 1], [1, 1], [1, 2 / 3],
            [0, 1 / 3], [0, 2 / 3], [1, 2 / 3], [1, 1 / 3], [0, 1 / 3], [0, 2 / 3], [1, 2 / 3], [1, 1 / 3], # Top right color
            [0, 0], [0, 1 / 3], [1, 1 / 3], [1, 0], [0, 0], [0, 1 / 3], [1, 1 / 3], [1, 0], # Bottom left color
        ]
    )

    num_pnts2 = uv_vals2.shape[0]

    t_coords2 = FloatArray()
    t_coords2.SetNumberOfComponents(2)
    t_coords2.SetNumberOfTuples(num_pnts2)
    [t_coords2.SetTuple(i, uv_vals2[i]) for i in range(num_pnts2)]

    set_polydata_tcoords(actor_box, t_coords2)

    #--------------------------------------------------------------------------

    # We organize the data of color, radius and height of the cylinders in a
    # grid of 3x5 which is then encoded as a 2D texture.
    # Data has to be multiplied by 255 to be in range 0-255, since inside the
    # shader a map is done for the values to be in range 0-1.
    # ?: values are divided by 255.
    # TODO: Verify color range inside the shader.
    arr = (
        np.array(
            #[[.5, .5, .5, .5, .5], [.4, .4, .4, .4, .4], [.3, .3, .3, .3, .3]]
            [[1, 0, 0, 0.5, 0.5], [0, 1, 0, 0.3, 0.75], [0, 0, 1, 0.1, 1]]
        )
        * 255
    )
    # grid = rgb_to_vtk(arr.astype(np.uint8))
    grid = numpy_to_vtk_image_data(arr.astype(np.uint8))

    texture = Texture()
    texture.SetInputDataObject(grid)
    texture.Update()

    box_actor.GetProperty().SetTexture("texture0", texture)
    texture_actor.SetTexture(texture)
    # box_actor.SetTexture(texture)  # Look for actortexture in GLSL

    box_actor.GetShaderProperty().GetFragmentCustomUniforms().SetUniformf("n", 5)

    vs_dec = """
        in vec3 center;
        in vec3 direction;
        in float id;

        out vec4 vertexMCVSOutput;
        out vec3 centerMCVSOutput;
        out vec3 directionVSOutput;
        out float idVSOutput;
        """

    vs_impl = """
        vertexMCVSOutput = vertexMC;
        centerMCVSOutput = center;
        directionVSOutput = direction;
        idVSOutput = id;
        """

    shader_to_actor(box_actor, "vertex", decl_code=vs_dec, impl_code=vs_impl)

    fs_vars_dec = """
        in vec4 vertexMCVSOutput;
        in vec3 centerMCVSOutput;
        in vec3 directionVSOutput;
        in float idVSOutput;

        uniform mat4 MCVCMatrix;
        """

    vec_to_vec_rot_mat = import_fury_shader(
        os.path.join("utils", "vec_to_vec_rot_mat.glsl")
    )

    sd_cylinder = import_fury_shader(os.path.join("sdf", "sd_cylinder.frag"))

    sdf_map = """
        float map(in vec3 position)
        {
            mat4 rot = vec2VecRotMat(
                normalize(directionVSOutput), normalize(vec3(0, 1, 0)));
            vec3 pos = (rot * vec4(position - centerMCVSOutput, 0.0)).xyz;

            // GET RADIUS AND HEIGHT FROM TEXTURE -----------------------------
            // .7 and .9 corresponds to the 4 and 5 column from the 2D texture
            // which have the data of the radius and height respectively.
            // idVSOutput is used to identify each element from the actor,
            // that is, which row from the 2D texture.
            // we subtract .01 from the final y coordinate since the border
            // color corresponds to the adjacent tile.
            
            //vec4 tcolor_0 = texture(texture0, tcoordVCVSOutput); // Read texture color
            
            //float r = texture(texture0, vec2(.7, 1/idVSOutput-.01)).x;
            //float h = texture(texture0, vec2(.9, 1/idVSOutput-.01)).x;
            
            float i = 1/(n*2);
            float r = texture(texture0, vec2(i+3/n, tcoordVCVSOutput.y)).x;
            float h = texture(texture0, vec2(i+4/n, tcoordVCVSOutput.y)).x;
            // ----------------------------------------------------------------
            return sdCylinder(pos, r, h / 2);
        }
        """

    central_diffs_normal = import_fury_shader(
        os.path.join("sdf", "central_diffs.frag")
    )

    cast_ray = import_fury_shader(
        os.path.join("ray_marching", "cast_ray.frag")
    )

    blinn_phong_model = import_fury_shader(
        os.path.join("lighting", "blinn_phong_model.frag")
    )

    fs_dec = compose_shader(
        [
            fs_vars_dec,
            vec_to_vec_rot_mat,
            sd_cylinder,
            sdf_map,
            central_diffs_normal,
            cast_ray,
            blinn_phong_model,
        ]
    )

    shader_to_actor(box_actor, "fragment", decl_code=fs_dec)

    sdf_cylinder_frag_impl = """
        vec3 point = vertexMCVSOutput.xyz;

        vec4 ro = -MCVCMatrix[3] * MCVCMatrix;
        vec3 rd = normalize(point - ro.xyz);
        vec3 ld = normalize(ro.xyz - point);

        ro += vec4((point - ro.xyz), 0);
        float t = castRay(ro.xyz, rd);

        if(t < 20.0)
        {
            vec3 position = ro.xyz + t * rd;
            vec3 normal = centralDiffsNormals(position, .0001);
            float lightAttenuation = dot(ld, normal);

            // GET COLOR FROM TEXTURE -----------------------------------------
            // .1, .3 and .5 corresponds to the 1, 2 and 3 column from the 2D
            // texture which have the data of the RGB color.
            float data = texture(texture0, vec2(.1, 1/2-.01)).x;
            
            
            //float cR = texture(texture0, vec2(.1, 1/idVSOutput-.01)).x;
            //float cG = texture(texture0, vec2(.3, 1/idVSOutput-.01)).x;
            //float cB = texture(texture0, vec2(.5, 1/idVSOutput-.01)).x;
            
            float i = 1/(n*2);
            float cR = texture(texture0, vec2(i, tcoordVCVSOutput.y)).x;
            float cG = texture(texture0, vec2(i+1/n, tcoordVCVSOutput.y)).x;
            float cB = texture(texture0, vec2(i+2/n, tcoordVCVSOutput.y)).x;
            vec3 cylinderColor = vec3(cR, cG, cB);
            // ----------------------------------------------------------------
            vec3 color = blinnPhongIllumModel(
                lightAttenuation, lightColor0, cylinderColor, specularPower,
                specularColor, ambientColor);
            fragOutput0 = vec4(cylinderColor, opacity);
        }
        else
        {
            discard;
        }
        """

    shader_to_actor(
        box_actor, "fragment", impl_code=sdf_cylinder_frag_impl, block="light",
    debug=False)

    scene.add(box_actor)
    scene.add(texture_actor)

    window.show(scene)
