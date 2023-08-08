"""
This spript includes the basic implementation of Spherical Harmonics
"""
'''
import numpy as np
import os

from fury import actor, window, ui
from fury.shaders import (attribute_to_actor, compose_shader,
                          import_fury_shader, shader_to_actor)


class Sphere:
    vertices = None
    faces = None


if __name__ == '__main__':
    centers = np.array([[-1.2, 0, 0], [0, 0, 0], [1.2, 0, 0]])
    vecs = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    colors = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    vals = np.array([1.0, 1.0, 1.0])
    lvals = np.array([6, 5, 3])
    mvals = np.array([-4, 4, 2])

    box_sd_stg_actor = actor.box(centers=centers, directions=vecs,
                                 colors=colors, scales=1.0)

    big_centers = np.repeat(centers, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_centers, 'center')

    big_directions = np.repeat(vecs, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_directions, 'direction')

    big_scales = np.repeat(vals, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_scales, 'scale')

    big_lvals = np.repeat(lvals, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_lvals, 'lval')

    big_mvals = np.repeat(mvals, 8, axis=0)
    attribute_to_actor(box_sd_stg_actor, big_mvals, 'mval')

    vs_dec = \
        """
        in vec3 center;
        in vec3 direction;
        in float scale;
        in float lval;
        in float mval;

        out vec4 vertexMCVSOutput;
        out vec3 centerMCVSOutput;
        out vec3 directionVSOutput;
        out float scaleVSOutput;
        out float lvalVSOutput;
        out float mvalVSOutput;
        """

    vs_impl = \
        """
        vertexMCVSOutput = vertexMC;
        centerMCVSOutput = center;
        directionVSOutput = direction;
        scaleVSOutput = scale;
        lvalVSOutput = lval;
        mvalVSOutput = mval;
        """

    shader_to_actor(box_sd_stg_actor, 'vertex', decl_code=vs_dec,
                    impl_code=vs_impl)

    fs_vars_dec = \
        """
        in vec4 vertexMCVSOutput;
        in vec3 centerMCVSOutput;
        in vec3 directionVSOutput;
        in float scaleVSOutput;
        in float lvalVSOutput;
        in float mvalVSOutput;

        uniform mat4 MCVCMatrix;
        """

    vec_to_vec_rot_mat = import_fury_shader(os.path.join(
        'utils', 'vec_to_vec_rot_mat.glsl'))

    sd_cylinder = \
    """
    float sdSphere(vec3 p, float r)
    {
        return length(p) - r;
    }
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
                float c = ns.x;
                ns.x = ns.z;
                ns.z = c;
                m = -m;
            }
            
            // spherical coordinates
            float thetax = ns.y;
            float phi = atan(ns.z, ns.x)+PI/2.;
            
            float pl = Pgn(l, m, thetax);
            
            float r = (m - (2 * (m/2)) == 0 ? 1. : -1.) * cos(float(m) * phi) * pl;
            
            return r;
        }
        
        vec3 map( in vec3 p )
        {
            p = p - centerMCVSOutput;
            vec3 p00 = p;
            
            float r, d; vec3 n, s, res;

            #define SHAPE (vec3(d-abs(r), sign(r),d))
            
            int l = int(lvalVSOutput);
            int m = int(mvalVSOutput);
            
            d=length(p00); n=p00/d; r = SH(l, m, n)*scaleVSOutput; s = SHAPE; res = s;
            
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

    fs_dec = compose_shader([fs_vars_dec, vec_to_vec_rot_mat, sd_cylinder,
                             sdf_map, central_diffs_normal, cast_ray,
                             blinn_phong_model])

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
            vec3 ref = reflect( rd, normal );
        
        
            float occ = clamp( 2.0*t.z, 0.0, 1.0 );
            float sss = pow( clamp( 1.0 + dot(normal, rd), 0.0, 1.0 ), 1.0 );
        
            // lights
            vec3 lin  = 2.5*occ*vec3(1.0,1.00,1.00)*(0.6+0.4*normal.y);
            lin += 1.0*sss*vec3(1.0,0.95,0.70)*occ;	
        
            vec3 mater = 0.5*mix( vec3(1.0,0.6,0.15), vec3(0.2,0.4,0.5), t.y ); 	
        
            fragOutput0 = vec4( mater * lin , 1.0);
        }
        else
        {
            discard;
        }
        """

    shader_to_actor(box_sd_stg_actor, 'fragment', impl_code=sdf_frag_impl,
                    block='light')

    slider_l = ui.LineSlider2D(
        center=(100, 250),
        length=100,
        initial_value=2,
        orientation='horizontal',
        min_value=0,
        max_value=10,
        text_alignment='bottom',
    )

    slider_m = ui.LineSlider2D(
        center=(100, 200),
        length=100,
        initial_value=2,
        orientation='horizontal',
        min_value=-10,
        max_value=10,
        text_alignment='bottom',
    )

    def changeL(slider):
        l = int(slider.value)
        lvals = np.array([l, 5, 3])
        big_lvals = np.repeat(lvals, 8, axis=0)
        attribute_to_actor(box_sd_stg_actor, big_lvals, 'lval')

    def changeM(slider):
        m = int(slider.value)
        mvals = np.array([m, 4, 2])
        big_mvals = np.repeat(mvals, 8, axis=0)
        attribute_to_actor(box_sd_stg_actor, big_mvals, 'mval')
        slider_m.min_value = int(slider_l.value) * -1
        slider_m.max_value = int(slider_l.value)

    slider_m.on_change = changeM
    slider_l.on_change = changeL

    show_manager = window.ShowManager(size=(700, 700))
    show_manager.scene.add(box_sd_stg_actor)
    show_manager.scene.add(slider_m)
    show_manager.scene.add(slider_l)

    show_manager.start()
'''

import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

import dipy.reconst.dti as dti

from dipy.data import get_fnames

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

data, affine = load_nifti(hardi_fname)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

tenmodel = dti.TensorModel(gtab)

from dipy.data import get_sphere
sphere = get_sphere('repulsion724')

interactive = True
from fury import actor, window
scene = window.Scene()

tensor_odfs = tenmodel.fit(data[45:50, 73:78, 38:39]).odf(sphere)
print(tensor_odfs)
print(tensor_odfs.shape)

odf_actor = actor.odf_slicer(tensor_odfs, sphere=sphere, scale=0.5,
                             colormap=None)
scene.add(odf_actor)

if interactive:
    window.show(scene)
