"""
This spript includes TEXTURE experimentation for passing SH coeffients
"""
import numpy as np
import os

from fury.lib import Texture, FloatArray

from fury import actor, window
from fury.shaders import (attribute_to_actor, compose_shader,
                          import_fury_shader, shader_to_actor)
from fury.utils import numpy_to_vtk_image_data, set_polydata_tcoords


if __name__ == '__main__':
    centers = np.array([[0, -1, 0], [1.0, -1, 0], [2.0, -1, 0]])
    centers_2 = np.array([[0, -2, 0], [1.0, -2, 0], [2.0, -2, 0]])
    centers_3 = np.array([[0, -3, 0], [1.0, -3, 0], [2.0, -3, 0]])
    vecs = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    colors = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    scales = np.array([1.0, 2.0, 2.0])
    coeffs = np.array(
        [[0.2820735, 0.15236554, -0.04038717, -0.11270988, -0.04532376,
          0.14921817, 0.00257928, 0.0040734, -0.05313807, 0.03486542,
          0.04083064, 0.02105767, -0.04389586, -0.04302812, 0.1048641],
         [0.28549338, 0.0978267, -0.11544838, 0.12525354, -0.00126003,
          0.00320594, 0.04744155, -0.07141446, 0.03211689, 0.04711322,
          0.08064896, 0.00154299, 0.00086506, 0.00162543, -0.00444893],
         [0.28208936, -0.13133252, -0.04701012, -0.06303016, -0.0468775,
          0.02348355, 0.03991898, 0.02587433, 0.02645416, 0.00668765,
          0.00890633, 0.02189304, 0.00387415, 0.01665629, -0.01427194]])

    box_actor_texture = actor.box(centers=centers, scales=1.0)
    box_actor_uniform_1 = actor.box(centers=np.array([centers_2[0]]), scales=1.0)
    box_actor_uniform_2 = actor.box(centers=np.array([centers_2[1]]), scales=1.0)
    box_actor_uniform_3 = actor.box(centers=np.array([centers_2[2]]), scales=1.0)
    box_actor_template = actor.box(centers=centers_3, scales=1.0)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(box_actor_texture, big_centers, 'center')
    attribute_to_actor(box_actor_template, np.repeat(centers_3, 8, axis=0), 'center')
    attribute_to_actor(box_actor_uniform_1, np.repeat(np.array([centers_2[0]]), 8, axis=0), 'center')
    attribute_to_actor(box_actor_uniform_2, np.repeat(np.array([centers_2[1]]), 8, axis=0), 'center')
    attribute_to_actor(box_actor_uniform_3, np.repeat(np.array([centers_2[2]]), 8, axis=0), 'center')

    big_scales = np.repeat(scales, 8, axis=0)
    attribute_to_actor(box_actor_texture, big_scales, 'scale')
    attribute_to_actor(box_actor_template, big_scales, 'scale')
    attribute_to_actor(box_actor_uniform_1, np.repeat(np.array([scales[0]]), 8, axis=0), 'scale')
    attribute_to_actor(box_actor_uniform_2, np.repeat(np.array([scales[1]]), 8, axis=0), 'scale')
    attribute_to_actor(box_actor_uniform_3, np.repeat(np.array([scales[2]]), 8, axis=0), 'scale')

    box_actor_uniform_1.GetShaderProperty().GetFragmentCustomUniforms(). \
        SetUniform1fv("coeffs", 15, coeffs[0])
    box_actor_uniform_2.GetShaderProperty().GetFragmentCustomUniforms(). \
        SetUniform1fv("coeffs", 15, coeffs[1])
    box_actor_uniform_3.GetShaderProperty().GetFragmentCustomUniforms(). \
        SetUniform1fv("coeffs", 15, coeffs[2])

    actor_box = box_actor_texture.GetMapper().GetInput()

    uv_vals = np.array(
        [
            [0, 2 / 3], [0, 1], [1, 1], [1, 2 / 3], [0, 2 / 3], [0, 1], [1, 1], [1, 2 / 3], #glyph1
            [0, 1 / 3], [0, 2 / 3], [1, 2 / 3], [1, 1 / 3], [0, 1 / 3], [0, 2 / 3], [1, 2 / 3], [1, 1 / 3], #glyph2
            [0, 0], [0, 1 / 3], [1, 1 / 3], [1, 0], [0, 0], [0, 1 / 3], [1, 1 / 3], [1, 0], #glyph3
        ]
    )

    num_pnts = uv_vals.shape[0]

    t_coords = FloatArray()
    t_coords.SetNumberOfComponents(2)
    t_coords.SetNumberOfTuples(num_pnts)
    [t_coords.SetTuple(i, uv_vals[i]) for i in range(num_pnts)]

    set_polydata_tcoords(actor_box, t_coords)

    arr = (
            np.array(
                [[0.2820735, 0.15236554, -0.04038717, -0.11270988, -0.04532376,
                  0.14921817, 0.00257928, 0.0040734, -0.05313807, 0.03486542,
                  0.04083064, 0.02105767, -0.04389586, -0.04302812, 0.1048641],
                 [0.28549338, 0.0978267, -0.11544838, 0.12525354, -0.00126003,
                  0.00320594, 0.04744155, -0.07141446, 0.03211689, 0.04711322,
                  0.08064896, 0.00154299, 0.00086506, 0.00162543, -0.00444893],
                 [0.28208936, -0.13133252, -0.04701012, -0.06303016,
                  -0.0468775, 0.02348355, 0.03991898, 0.02587433, 0.02645416,
                  0.00668765, 0.00890633, 0.02189304, 0.00387415, 0.01665629,
                  -0.01427194]])
    )

    minmax = np.array([arr.min(axis=1), arr.max(axis=1)]).T
    big_minmax = np.repeat(minmax, 8, axis=0)
    attribute_to_actor(box_actor_texture, big_minmax, 'minmax')

    min = arr.min(axis=1)
    max = arr.max(axis=1)
    newmin = 0
    newmax = 1
    arr = np.array([(arr[i] - min[i])*((newmax - newmin) / (max[i] - min[i])) + newmin for i in range(arr.shape[0])])
    arr *= 255
    print(arr.astype(np.uint8))
    grid = numpy_to_vtk_image_data(arr.astype(np.uint8))

    texture = Texture()
    texture.SetInputDataObject(grid)
    texture.Update()

    box_actor_texture.GetProperty().SetTexture("texture0", texture)
    box_actor_texture.GetShaderProperty().GetFragmentCustomUniforms()\
        .SetUniformf("k", 15)  # number of coefficients per glyph
    # =========================================================================

    vs_dec = \
        """
        in vec3 center;
        in float scale;
        in vec2 minmax;

        out vec4 vertexMCVSOutput;
        out vec3 centerMCVSOutput;
        out float scaleVSOutput;
        out vec2 minmaxVSOutput;
        """

    vs_impl = \
        """
        vertexMCVSOutput = vertexMC;
        centerMCVSOutput = center;
        scaleVSOutput = scale;
        minmaxVSOutput = minmax;
        vec3 camPos = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
        """

    shader_to_actor(box_actor_texture, 'vertex', decl_code=vs_dec, impl_code=vs_impl)
    shader_to_actor(box_actor_template, 'vertex', decl_code=vs_dec, impl_code=vs_impl)
    shader_to_actor(box_actor_uniform_1, 'vertex', decl_code=vs_dec, impl_code=vs_impl)
    shader_to_actor(box_actor_uniform_2, 'vertex', decl_code=vs_dec, impl_code=vs_impl)
    shader_to_actor(box_actor_uniform_3, 'vertex', decl_code=vs_dec, impl_code=vs_impl)

    fs_vars_dec = \
        """
        in vec4 vertexMCVSOutput;
        in vec3 centerMCVSOutput;
        in float scaleVSOutput;
        in vec2 minmaxVSOutput;
        uniform samplerCube texture_0;

        uniform mat4 MCVCMatrix;
        """

    sdf_map = \
        """

        #define PI 3.1415926535898

        // Clenshaw Legendre normalized
        float Pgn(int l, int m, float x)
        {
            float p0 = 0., p1 = 0., p2 = 0.;

            for (int k = l; k >= 0; k--)
            {
                float k1 = float(k + 1);
                float m1 = float(2 * m) + k1;
                float m2 = float(2 * (m + k) + 1);

                p2 = p1;
                p1 = p0;

                p0 = 0.;
                if (l == m + k)
                    p0 = 1.;

                float u0 = sqrt(
                    (m2 * (m2 + 2.0)) /
                    (k1 * m1)
                );

                float u1 = sqrt(
                    (k1 * m1 * (m2 + 4.0)) / 
                    ((k1 + 1.0) * (m1 + 1.0) * m2)
                );

                p0 += p1 * u0 * x;
                p0 -= u1 * p2;
            }

            for (int k = 1; k <= m; k++)
            {
                p0 *= sqrt(
                    (1.0 - 0.5/float(k)) * (1.0 - x) * (1.0 + x)
                );
            }

            p0 *= sqrt((0.5 * float(m) + 0.25)/PI);

            return p0;
        }

        float SH( in int l, in int m, in vec3 s ) 
        {
            vec3 ns = normalize(s);

            if (m < 0) {
                m = -m;
            }

            // spherical coordinates
            float thetax = ns.y;
            float phi = atan(ns.z, ns.x)+PI/2.;

            float pl = Pgn(l, m, thetax);

            float r = pow(-1.0, float(m)) * cos(float(m) * phi) * pl;
            if (m != 0) {
                r *= sqrt(2.0);
            }
            return r;
        }
        
        float coef_norm( in float coef)
        {
            float min = 0;
            float max = 1;
            float newmin = minmaxVSOutput.x;//-0.13133252;
            float newmax = minmaxVSOutput.y;//0.28208936;
            return (coef - min) * ((newmax - newmin) / (max - min)) + newmin;
        }
    """

    map_function_tex =  \
        """
        vec3 map( in vec3 p )
        {
            p = p - centerMCVSOutput;
            vec3 p00 = p;

            float r, d; vec3 n, s, res;

            #define SHAPE (vec3(d-abs(r), sign(r),d))
            //#define SHAPE (vec3(d-0.35, -1.0+2.0*clamp(0.5 + 16.0*r,0.0,1.0),d))
            d=length(p00);
            n=p00/d;     
            // ================================================================
            float i = 1/(k*2);
            float c = texture(texture0, vec2(i, tcoordVCVSOutput.y)).x;
            r = coef_norm(c)*SH(0, 0, n);
            
            c = texture(texture0, vec2(i+1/k, tcoordVCVSOutput.y)).x;
            r += coef_norm(c)*SH(2, -2, n);
            
            c = texture(texture0, vec2(i+2/k, tcoordVCVSOutput.y)).x;
            r += coef_norm(c)*SH(2, -1, n);
            
            c = texture(texture0, vec2(i+3/k, tcoordVCVSOutput.y)).x;
            r += coef_norm(c)*SH(2, 0, n);
            
            c = texture(texture0, vec2(i+4/k, tcoordVCVSOutput.y)).x;
            r += coef_norm(c)*SH(2, 1, n);
            
            c = texture(texture0, vec2(i+5/k, tcoordVCVSOutput.y)).x;
            r += coef_norm(c)*SH(2, 2, n);
            
            c = texture(texture0, vec2(i+6/k, tcoordVCVSOutput.y)).x;
            r += coef_norm(c)*SH(4, -4, n);
            
            c = texture(texture0, vec2(i+7/k, tcoordVCVSOutput.y)).x;
            r += coef_norm(c)*SH(4, -3, n);
            
            c = texture(texture0, vec2(i+8/k, tcoordVCVSOutput.y)).x;
            r += coef_norm(c)*SH(4, -2, n);
            
            c = texture(texture0, vec2(i+9/k, tcoordVCVSOutput.y)).x;
            r += coef_norm(c)*SH(4, -1, n);
            
            c = texture(texture0, vec2(i+10/k, tcoordVCVSOutput.y)).x;
            r += coef_norm(c)*SH(4, 0, n);
            
            c = texture(texture0, vec2(i+11/k, tcoordVCVSOutput.y)).x;
            r += coef_norm(c)*SH(4, 1, n);
            
            c = texture(texture0, vec2(i+12/k, tcoordVCVSOutput.y)).x;
            r += coef_norm(c)*SH(4, 2, n);
            
            c = texture(texture0, vec2(i+13/k, tcoordVCVSOutput.y)).x;
            r += coef_norm(c)*SH(4, 3, n);
            
            c = texture(texture0, vec2(i+14/k, tcoordVCVSOutput.y)).x;
            r += coef_norm(c)*SH(4, 4, n);
            
            r *= scaleVSOutput;
            // ================================================================
            s = SHAPE; res = s;
            return vec3( res.x, 0.5+0.5*res.y, res.z );
        }
        """

    map_function_unif = \
        """
        vec3 map( in vec3 p )
        {
            p = p - centerMCVSOutput;vec3
            p00 = p;
            float r, d;
            vec3 n, s, res;
            # define SHAPE (vec3(d-abs(r), sign(r),d))
            d = length(p00);
            n = p00 / d;
            float
            sc = scaleVSOutput;
            r = coeffs[0] * SH(0, 0, n) * sc;
            r += coeffs[1] * SH(2, -2, n) * sc;
            r += coeffs[2] * SH(2, -1, n) * sc;
            r += coeffs[3] * SH(2, 0, n) * sc;
            r += coeffs[4] * SH(2, 1, n) * sc;
            r += coeffs[5] * SH(2, 2, n) * sc;
            r += coeffs[6] * SH(4, -4, n) * sc;
            r += coeffs[7] * SH(4, -3, n) * sc;
            r += coeffs[8] * SH(4, -2, n) * sc;
            r += coeffs[9] * SH(4, -1, n) * sc;
            r += coeffs[10] * SH(4, 0, n) * sc;
            r += coeffs[11] * SH(4, 1, n) * sc;
            r += coeffs[12] * SH(4, 2, n) * sc;
            r += coeffs[13] * SH(4, 3, n) * sc;
            r += coeffs[14] * SH(4, 4, n) * sc;
            s = SHAPE;
            res = s;
            return vec3(res.x, 0.5 + 0.5 * res.y, res.z);
            }
        """

    map_function_templ_1 = \
        """
        vec3 map( in vec3 p )
        {
            p = p - centerMCVSOutput;vec3
            p00 = p;
            float r, d;
            vec3 n, s, res;
            # define SHAPE (vec3(d-abs(r), sign(r),d))
            d = length(p00);
            n = p00 / d;
            float
            sc = scaleVSOutput;
        """

    coeffs_1 = \
        """
        float coeffs[15] = float[15](0.2820735, 0.15236554, -0.04038717,
        -0.11270988, -0.04532376, 0.14921817, 0.00257928, 
        0.0040734, -0.05313807, 0.03486542, 0.04083064, 0.02105767, 
        -0.04389586, -0.04302812, 0.1048641);
        """

    coeffs_2 = \
        """
        float coeffs[15] = float[15](0.28549338, 0.0978267, -0.11544838,
        0.12525354, -0.00126003, 0.00320594, 0.04744155, -0.07141446,
        0.03211689, 0.04711322, 0.08064896, 0.00154299, 0.00086506, 0.00162543,
        -0.00444893);
        """

    coeffs_3 = \
        """
        float coeffs[15] = float[15](0.28208936, -0.13133252, -0.04701012,
        -0.06303016, -0.0468775, 0.02348355, 0.03991898, 0.02587433,
        0.02645416, 0.00668765, 0.00890633, 0.02189304, 0.00387415, 0.01665629,
        -0.01427194);
        """

    map_function_templ_2 = \
        """
            r = coeffs[0] * SH(0, 0, n) * sc;
            r += coeffs[1] * SH(2, -2, n) * sc;
            r += coeffs[2] * SH(2, -1, n) * sc;
            r += coeffs[3] * SH(2, 0, n) * sc;
            r += coeffs[4] * SH(2, 1, n) * sc;
            r += coeffs[5] * SH(2, 2, n) * sc;
            r += coeffs[6] * SH(4, -4, n) * sc;
            r += coeffs[7] * SH(4, -3, n) * sc;
            r += coeffs[8] * SH(4, -2, n) * sc;
            r += coeffs[9] * SH(4, -1, n) * sc;
            r += coeffs[10] * SH(4, 0, n) * sc;
            r += coeffs[11] * SH(4, 1, n) * sc;
            r += coeffs[12] * SH(4, 2, n) * sc;
            r += coeffs[13] * SH(4, 3, n) * sc;
            r += coeffs[14] * SH(4, 4, n) * sc;
            s = SHAPE;
            res = s;
            return vec3(res.x, 0.5 + 0.5 * res.y, res.z);
            }
        """

    central_diffs_normal = \
        """
        vec3 centralDiffsNormals(in vec3 pos)
        {
            //vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
            vec2 e = vec2(0.001, -1.0);
            return normalize( e.xyy*map( pos + e.xyy ).x +
                              e.yyx*map( pos + e.yyx ).x +
                              e.yxy*map( pos + e.yxy ).x +
                              e.xxx*map( pos + e.xxx ).x );
        }
        """

    cast_ray = \
        """
        vec3 castRay(in vec3 ro, vec3 rd)
        {
            vec3 res = vec3(1e10,-1.0, 1.0);

            float maxd = 4.0;
            float h = 1.0;
            float t = 0.0;
            vec2  m = vec2(-1.0);
            for( int i=0; i<2000; i++ )
            {
                if( h<0.01||t>maxd ) break;
                vec3 res = map( ro+rd*t );
                h = res.x;
                m = res.yz;
                t += h*0.1;
            }
            if( t<maxd && t<res.x ) res=vec3(t,m);

            return res;
        }

        """

    blinn_phong_model = import_fury_shader(os.path.join(
        'lighting', 'blinn_phong_model.frag'))

    fs_dec = compose_shader([fs_vars_dec, sdf_map, map_function_tex,
                             central_diffs_normal, cast_ray, blinn_phong_model])
    fs_dec_2 = compose_shader([fs_vars_dec, sdf_map, map_function_unif,
                             central_diffs_normal, cast_ray,
                             blinn_phong_model])
    fs_dec_t1 = compose_shader([fs_vars_dec, sdf_map, map_function_templ_1,
                               coeffs_1, map_function_templ_2,
                               central_diffs_normal, cast_ray,
                               blinn_phong_model])
    fs_dec_t2 = compose_shader([fs_vars_dec, sdf_map, map_function_templ_1,
                                coeffs_2, map_function_templ_2,
                                central_diffs_normal, cast_ray,
                                blinn_phong_model])
    fs_dec_t3 = compose_shader([fs_vars_dec, sdf_map, map_function_templ_1,
                                coeffs_3, map_function_templ_2,
                                central_diffs_normal, cast_ray,
                                blinn_phong_model])

    shader_to_actor(box_actor_texture, 'fragment', decl_code=fs_dec, debug=False)
    shader_to_actor(box_actor_uniform_1, 'fragment', decl_code=fs_dec_2)
    shader_to_actor(box_actor_uniform_2, 'fragment', decl_code=fs_dec_2)
    shader_to_actor(box_actor_uniform_3, 'fragment', decl_code=fs_dec_2)
    shader_to_actor(box_actor_template, 'fragment', decl_code=fs_dec_t3)

    sdf_frag_impl = \
        """
        vec3 pnt = vertexMCVSOutput.xyz;

        // Ray Origin
        // Camera position in world space
        vec3 ro = (-MCVCMatrix[3] * MCVCMatrix).xyz;

        // Ray Direction
        vec3 rd = normalize(pnt - ro);

        // Light Direction
        vec3 ld = normalize(ro - pnt);

        ro += pnt - ro;

        vec3 t = castRay(ro, rd);

        if(t.y > -0.5 )
        {
            vec3 pos = ro + t.y * rd;

            vec3 normal = centralDiffsNormals(pos);

            float occ = clamp( 2.0*t.z, 0.0, 1.0 );
            float sss = pow( clamp( 1.0 + dot(normal, rd), 0.0, 1.0 ), 1.0 );

            // lights
            vec3 lin  = 2.5*occ*vec3(1.0,1.0,1.0)*(0.6+0.4*normal.y);
            lin += 1.0*sss*vec3(1.0,0.95,0.70)*occ;	

            vec3 mater = 0.5*mix( vec3(1.0,1.0,0.0), vec3(1.0,1.0,1.0), t.y); 	

            // ================================================================
            fragOutput0 = vec4( vec3(1,1,0)*lin, 1.0);
            // ================================================================
        }
        else
        {
            discard;
        }


        """

    shader_to_actor(box_actor_texture, 'fragment', impl_code=sdf_frag_impl, block='picking')
    shader_to_actor(box_actor_uniform_1, 'fragment', impl_code=sdf_frag_impl, block='light')
    shader_to_actor(box_actor_uniform_2, 'fragment', impl_code=sdf_frag_impl, block='light')
    shader_to_actor(box_actor_uniform_3, 'fragment', impl_code=sdf_frag_impl, block='light')
    shader_to_actor(box_actor_template, 'fragment', impl_code=sdf_frag_impl, block='light')

    show_manager = window.ShowManager(size=(700, 500))
    show_manager.scene.background([255, 255, 255])
    show_manager.scene.add(box_actor_texture)
    show_manager.scene.add(box_actor_uniform_1)
    show_manager.scene.add(box_actor_uniform_2)
    show_manager.scene.add(box_actor_uniform_3)
    show_manager.scene.add(box_actor_template)

    from dipy.reconst.shm import sh_to_sf
    from dipy.data import get_sphere

    sphere = get_sphere('repulsion724')

    sh_basis = 'descoteaux07'
    sh_order = 4
    tensor_sh = coeffs = np.array(
        [[[[0.2820735, 0.15236554, -0.04038717, -0.11270988, -0.04532376,
            0.14921817, 0.00257928, 0.0040734, -0.05313807, 0.03486542,
            0.04083064, 0.02105767, -0.04389586, -0.04302812, 0.1048641]]],
         [[[0.28549338, 0.0978267, -0.11544838, 0.12525354, -0.00126003,
            0.00320594, 0.04744155, -0.07141446, 0.03211689, 0.04711322,
            0.08064896, 0.00154299, 0.00086506, 0.00162543, -0.00444893]]],
         [[[0.28208936, -0.13133252, -0.04701012, -0.06303016, -0.0468775,
            0.02348355, 0.03991898, 0.02587433, 0.02645416, 0.00668765,
            0.00890633, 0.02189304, 0.00387415, 0.01665629, -0.01427194]]]])
    tensor_sf = sh_to_sf(tensor_sh, sh_order=4, basis_type='descoteaux07',
                         sphere=sphere)

    odf_actor = actor.odf_slicer(tensor_sf, sphere=sphere, scale=0.5,
                                 colormap='plasma')
    show_manager.scene.add(odf_actor)

    show_manager.start()
