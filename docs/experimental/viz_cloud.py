"""
In this example, we create a basic simulation of clouds.
We use the raymarching algorithm on the box primtive to
render a volumetric cloud.

"""
from fury import actor, window, primitive as fp, utils
from fury.shaders import attribute_to_actor, shader_to_actor
import vtk
from vtk.util import numpy_support
import numpy as np

scene = window.Scene()
scene.background((1.0, 0.8, 0.8))
verts, faces = fp.prim_box()

centers = np.array([[0, 0, 0], [3, 0, 0], [-3, 0, 0],
                    [0, 3, 0], [3, 3, 0], [-3, 3, 0]])
repeated = fp.repeat_primitive(verts, faces, centers=centers, scales=3)

rep_verts, rep_faces, rep_colors, rep_centers = repeated
box_actor = utils.get_actor_from_primitive(rep_verts, rep_faces, rep_colors)

attribute_to_actor(box_actor, centers, "center")

vertex_dec = \
    """
    in vec3 center;
    out vec4 vertexMCVSOutput;
    out vec3 centerWCVSOutput;
    """

vertex_impl = \
    """
    vertexMCVSOutput = vertexMC;
    centerWCVSOutput = center;
    """

fragment_dec = \
    """
    in vec4 vertexMCVSOutput;
    in vec3 centerWCVSOutput;
    uniform mat4 MCVCMatrix;
    uniform mat4 MCWCMatrix;
    uniform mat3 WCVCMatrix;


    mat2 rot(in float a){float c = cos(a), s = sin(a);return mat2(c,s,-s,c);}
    const mat3 m3 = mat3(0.33338, 0.56034, -0.71817, -0.87887,0.32651,
                         -0.15323, 0.15162, 0.69596, 0.61339)*1.93;
    float mag2(vec2 p){return dot(p,p);}
    float linstep(in float mn, in float mx, in float x)
    {
        return clamp((x - mn)/(mx - mn), 0., 1.);
    }
    float prm1 = 0.;
    vec2 bsMo = vec2(0);

    vec2 disp(float t){ return vec2(sin(t*0.22)*1., cos(t*0.175)*1.)*2.; }

    float getsat(vec3 c)
    {
        float mi = min(min(c.x, c.y), c.z);
        float ma = max(max(c.x, c.y), c.z);
        return (ma - mi)/(ma+ 1e-7);
    }

    vec2 map(vec3 p)
    {
        p = p - centerWCVSOutput;
        vec3 p2 = p;
        p2.xy -= disp(p.z).xy;
        p.xy *= rot(sin(p.z)*(0.1 + prm1*0.05) );
        float cl = mag2(p2.xy);
        float d = 0.;
        p *= .61;
        float z = 1.;
        float trk = 1.;
        float dspAmp = 0.1 + prm1*0.2;
        for(int i = 0; i < 5; i++)
        {
            p += sin(p.zxy * 0.75 * trk )*dspAmp;
            d -= abs(dot(cos(p), sin(p.yzx))*z);
            z *= 0.57;
            trk *= 1.4;
            p = p*m3;
        }
        d = abs(d + prm1*3.)+ prm1*.3 - 2.5 + bsMo.y;
        return vec2(d + cl*.2 + 0.25, cl);
    }

    vec4 castRay(in vec3 ro, vec3 rd)
    {
        vec4 rez = vec4(0);
        const float ldst = 8.;
        vec3 lpos = vec3(disp(ldst)*0.1,  ldst);
        float t = 1.5;
        float fogT = 0.;
        for(int i=0; i<130; i++)
        {
            if(rez.a > 0.99)break;

            vec3 pos = ro + t*rd;
            vec2 mpv = map(pos);
            float den = clamp(mpv.x-0.3,0.,1.)*1.12;
            float dn = clamp((mpv.x + 2.),0.,3.);

            vec4 col = vec4(0);
            if (mpv.x > 0.6)
            {
                col = vec4(sin(vec3(5.,0.4,0.2) + mpv.y*0.1 +
                      sin(pos.z*0.4)*0.5 + 1.8) * 0.5 + 0.5,0.08);
                col *= den*den*den;
                col.rgb *= linstep(4.,-2.5, mpv.x)*2.3;
                float dif =  clamp((den - map(pos+.8).x)/9., 0.001, 1.);
                dif += clamp((den - map(pos+.35).x)/2.5, 0.001, 1.);
                col.xyz *= den * (vec3(0.005,.045,.075) +
                           1.5*vec3(0.033,0.07,0.03)*dif);
            }
            rez = rez + col*(1. - rez.a);
            t += clamp(0.5 - dn*dn*.05, 0.09, 0.3);
        }
        return clamp(rez, 0.0, 1.0);
    }
    """

fragment_impl = \
    """
    vec3 point = vertexMCVSOutput.xyz;

    //ray origin
    vec4 ro = -MCVCMatrix[3] * MCVCMatrix;  // camera position in world space
    //ray direction
    vec3 rd = normalize(point - ro.xyz);

    ro += vec4((point - ro.xyz),0.0);

    vec4 scn = castRay(ro.xyz, rd);
    vec3 col = scn.rgb;

    fragOutput0 = vec4( col, 1.0 );
    """

shader_to_actor(box_actor, "vertex", impl_code=vertex_impl,
                decl_code=vertex_dec)

shader_to_actor(box_actor, "fragment", block="light", impl_code=fragment_impl,
                decl_code=fragment_dec)


scene.add(box_actor)
scene.add(actor.axes())
window.show(scene, size=(1920, 1200))
