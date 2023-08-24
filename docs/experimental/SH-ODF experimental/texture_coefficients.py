"""
This spript includes TEXTURE experimentation for passing SH coeffients
"""
import numpy as np
import os

from fury.lib import Texture

from fury import actor, window
from fury.shaders import (attribute_to_actor, compose_shader,
                          import_fury_shader, shader_to_actor)
from fury.utils import rgb_to_vtk


# =============================================================================
def get_cubemap_from_ndarrays(array):
    texture = Texture()
    texture.CubeMapOn()
    arr = np.ones((4, 15, 3))
    arr[:, :, 0] = array
    arr[:, :, 1] = array
    arr[:, :, 2] = array
    print(arr)
    grid = rgb_to_vtk(arr.astype(np.uint8))
    for i in range(6):
        texture.SetInputDataObject(i, grid)
    return texture
# =============================================================================


if __name__ == '__main__':
    centers = np.array([[0, -1, 0], [1.0, -1, 0], [2.0, -1, 0], [3.0, -1, 0]])
    vecs = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
    colors = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
    vals = np.array([1.0, 4.2, 1.5, 2.0])
    coeffs = np.array(
        [[0.2820735, 0.15236554, -0.04038717, -0.11270988, -0.04532376,
          0.14921817, 0.00257928, 0.0040734, -0.05313807, 0.03486542,
          0.04083064, 0.02105767, -0.04389586, -0.04302812, 0.1048641],
         [2.82094529e-01, 7.05702620e-03, 3.20326265e-02, -2.88333917e-02,
          5.33638381e-03, 1.18306258e-02, -2.21964945e-04, 5.54136434e-04,
          1.25108672e-03, -4.69248914e-03, 4.30155475e-04, -1.15585609e-03,
          -4.69016480e-04, 1.44523500e-03, 3.96346915e-04],
         [0.28549338, 0.0978267, -0.11544838, 0.12525354, -0.00126003,
          0.00320594, 0.04744155, -0.07141446, 0.03211689, 0.04711322,
          0.08064896, 0.00154299, 0.00086506, 0.00162543, -0.00444893],
         [0.28208936, -0.13133252, -0.04701012, -0.06303016, -0.0468775,
          0.02348355, 0.03991898, 0.02587433, 0.02645416, 0.00668765,
          0.00890633, 0.02189304, 0.00387415, 0.01665629, -0.01427194]])

    # coeffs = coeffs / np.max(coeffs)

    box_sd_stg_actor = actor.box(centers=np.array([centers[0]]), scales=1.0)

    big_centers = np.repeat(np.array([centers[0]]), 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_centers, 'center')

    big_scales = np.repeat(np.array([vals[2]]), 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_scales, 'scale')

    box_sd_stg_actor.GetShaderProperty().GetFragmentCustomUniforms(). \
        SetUniform1fv("coeffs", 15, coeffs[2])

    # =========================================================================
    data_tex = np.array(
        [[.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
         [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
         [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
         [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5]])  # *200

    data = get_cubemap_from_ndarrays(np.array(data_tex))

    box_sd_stg_actor.SetTexture(data)
    # =========================================================================

    vs_dec = \
        """
        in vec3 center;
        in float scale;

        out vec4 vertexMCVSOutput;
        out vec3 centerMCVSOutput;
        out float scaleVSOutput;
        """

    vs_impl = \
        """
        vertexMCVSOutput = vertexMC;
        centerMCVSOutput = center;
        scaleVSOutput = scale;
        vec3 camPos = -MCVCMatrix[3].xyz * mat3(MCVCMatrix);
        """

    shader_to_actor(box_sd_stg_actor, 'vertex', decl_code=vs_dec,
                    impl_code=vs_impl)

    fs_vars_dec = \
        """
        in vec4 vertexMCVSOutput;
        in vec3 centerMCVSOutput;
        in float scaleVSOutput;
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

        vec3 map( in vec3 p )
        {
            p = p - centerMCVSOutput;
            vec3 p00 = p;

            float r, d; vec3 n, s, res;

            #define SHAPE (vec3(d-abs(r), sign(r),d))
            //#define SHAPE (vec3(d-0.35, -1.0+2.0*clamp(0.5 + 16.0*r,0.0,1.0),d))
            d=length(p00);
            n=p00/d; 
            float sc = scaleVSOutput;            
            r = coeffs[0]*SH(0, 0, n)*sc;
            r += coeffs[1]*SH(2, -2, n)*sc;
            r += coeffs[2]*SH(2, -1, n)*sc;
            r += coeffs[3]*SH(2, 0, n)*sc;
            r += coeffs[4]*SH(2, 1, n)*sc;
            r += coeffs[5]*SH(2, 2, n)*sc;
            r += coeffs[6]*SH(4, -4, n)*sc;
            r += coeffs[7]*SH(4, -3, n)*sc;
            r += coeffs[8]*SH(4, -2, n)*sc;
            r += coeffs[9]*SH(4, -1, n)*sc;
            r += coeffs[10]*SH(4, 0, n)*sc;
            r += coeffs[11]*SH(4, 1, n)*sc;
            r += coeffs[12]*SH(4, 2, n)*sc;
            r += coeffs[13]*SH(4, 3, n)*sc;
            r += coeffs[14]*SH(4, 4, n)*sc;

            s = SHAPE; res = s;
            return vec3( res.x, 0.5+0.5*res.y, res.z );
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

    fs_dec = compose_shader([fs_vars_dec, sdf_map, central_diffs_normal,
                             cast_ray, blinn_phong_model])

    shader_to_actor(box_sd_stg_actor, 'fragment', decl_code=fs_dec)

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
            // get random coefficient from texture
            vec4 texData1 = texture(texture_0, vec3(.5, .5, .5));
            fragOutput0 = vec4( texData1.xyz, 1.0);
            // ================================================================
        }
        else
        {
            discard;
        }


        """

    shader_to_actor(box_sd_stg_actor, 'fragment', impl_code=sdf_frag_impl,
                    block='light')

    show_manager = window.ShowManager(size=(700, 500))
    show_manager.scene.background([255, 255, 255])
    show_manager.scene.add(box_sd_stg_actor)

    from dipy.reconst.shm import sh_to_sf
    from dipy.data import get_sphere

    sphere = get_sphere('repulsion724')

    sh_basis = 'descoteaux07'
    sh_order = 4
    tensor_sh = coeffs = np.array(
        [[[[0.28549338, 0.0978267, -0.11544838, 0.12525354, -0.00126003,
            0.00320594, 0.04744155, -0.07141446, 0.03211689, 0.04711322,
            0.08064896, 0.00154299, 0.00086506, 0.00162543, -0.00444893]]]])
    tensor_sf = sh_to_sf(tensor_sh, sh_order=4, basis_type='descoteaux07',
                         sphere=sphere)

    odf_actor = actor.odf_slicer(tensor_sf, sphere=sphere, scale=0.5,
                                 colormap='plasma')
    show_manager.scene.add(odf_actor)

    show_manager.start()
