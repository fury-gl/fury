import numpy as np
import vtk
from vtk.util import numpy_support

import fury.primitive as fp
from fury import actor, window
from fury.utils import (
    get_actor_from_polydata,
    numpy_to_vtk_colors,
    set_polydata_colors,
    set_polydata_triangles,
    set_polydata_vertices,
)


def test_sh():
    from dipy.core.gradients import gradient_table
    from dipy.data import get_fnames, get_sphere
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.sims.voxel import multi_tensor, multi_tensor_odf, sticks_and_ball

    _, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)

    d = 0.0015
    S, sticks = sticks_and_ball(
        gtab, d=d, S0=1, angles=[(0, 0), (30, 30)], fractions=[60, 40], snr=None
    )

    print(S)
    print(sticks)
    mevals = np.array([[0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]])
    angles = [(0, 0), (60, 0)]
    fractions = [50, 50]
    sphere = get_sphere('repulsion724')
    sphere = sphere.subdivide(2)
    odf = multi_tensor_odf(sphere.vertices, mevals, angles, fractions)

    print(odf)
    ren = window.Scene()

    odf_actor = actor.odf_slicer(
        odf[None, None, None, :], sphere=sphere, colormap='plasma'
    )
    # odf_actor.RotateX(90)

    ren.add(odf_actor)
    # window.show(ren)

    odf_test_dec = """
    // Constants, see here: http://en.wikipedia.org/wiki/Table_of_spherical_harmonics
#define k01 0.2820947918 // sqrt(  1/PI)/2
#define k02 0.4886025119 // sqrt(  3/PI)/2
#define k03 1.0925484306 // sqrt( 15/PI)/2
#define k04 0.3153915652 // sqrt(  5/PI)/4
#define k05 0.5462742153 // sqrt( 15/PI)/4
#define k06 0.5900435860 // sqrt( 70/PI)/8
#define k07 2.8906114210 // sqrt(105/PI)/2
#define k08 0.4570214810 // sqrt( 42/PI)/8
#define k09 0.3731763300 // sqrt(  7/PI)/4
#define k10 1.4453057110 // sqrt(105/PI)/4

// unrolled version of the above
float SH_0_0( in vec3 s ) { vec3 n = s.zxy; return  k01; }
float SH_1_0( in vec3 s ) { vec3 n = s.zxy; return -k02*n.y; }
float SH_1_1( in vec3 s ) { vec3 n = s.zxy; return  k02*n.z; }
float SH_1_2( in vec3 s ) { vec3 n = s.zxy; return -k02*n.x; }
float SH_2_0( in vec3 s ) { vec3 n = s.zxy; return  k03*n.x*n.y; }
float SH_2_1( in vec3 s ) { vec3 n = s.zxy; return -k03*n.y*n.z; }
float SH_2_2( in vec3 s ) { vec3 n = s.zxy; return  k04*(3.0*n.z*n.z-1.0); }
float SH_2_3( in vec3 s ) { vec3 n = s.zxy; return -k03*n.x*n.z; }
float SH_2_4( in vec3 s ) { vec3 n = s.zxy; return  k05*(n.x*n.x-n.y*n.y); }
float SH_3_0( in vec3 s ) { vec3 n = s.zxy; return -k06*n.y*(3.0*n.x*n.x-n.y*n.y); }
float SH_3_1( in vec3 s ) { vec3 n = s.zxy; return  k07*n.z*n.y*n.x; }
float SH_3_2( in vec3 s ) { vec3 n = s.zxy; return -k08*n.y*(5.0*n.z*n.z-1.0); }
float SH_3_3( in vec3 s ) { vec3 n = s.zxy; return  k09*n.z*(5.0*n.z*n.z-3.0); }
float SH_3_4( in vec3 s ) { vec3 n = s.zxy; return -k08*n.x*(5.0*n.z*n.z-1.0); }
float SH_3_5( in vec3 s ) { vec3 n = s.zxy; return  k10*n.z*(n.x*n.x-n.y*n.y); }
float SH_3_6( in vec3 s ) { vec3 n = s.zxy; return -k06*n.x*(n.x*n.x-3.0*n.y*n.y); }

vec3 map( in vec3 p )
{
    vec3 p00 = p - vec3( 0.00, 2.5,0.0);
	vec3 p01 = p - vec3(-1.25, 1.0,0.0);
	vec3 p02 = p - vec3( 0.00, 1.0,0.0);
	vec3 p03 = p - vec3( 1.25, 1.0,0.0);
	vec3 p04 = p - vec3(-2.50,-0.5,0.0);
	vec3 p05 = p - vec3(-1.25,-0.5,0.0);
	vec3 p06 = p - vec3( 0.00,-0.5,0.0);
	vec3 p07 = p - vec3( 1.25,-0.5,0.0);
	vec3 p08 = p - vec3( 2.50,-0.5,0.0);
	vec3 p09 = p - vec3(-3.75,-2.0,0.0);
	vec3 p10 = p - vec3(-2.50,-2.0,0.0);
	vec3 p11 = p - vec3(-1.25,-2.0,0.0);
	vec3 p12 = p - vec3( 0.00,-2.0,0.0);
	vec3 p13 = p - vec3( 1.25,-2.0,0.0);
	vec3 p14 = p - vec3( 2.50,-2.0,0.0);
	vec3 p15 = p - vec3( 3.75,-2.0,0.0);

	float r, d; vec3 n, s, res;

    #ifdef SHOW_SPHERES
	#define SHAPE (vec3(d-0.35, -1.0+2.0*clamp(0.5 + 16.0*r,0.0,1.0),d))
	#else
	#define SHAPE (vec3(d-abs(r), sign(r),d))
	#endif
	d=length(p00); n=p00/d; r = SH_0_0( n ); s = SHAPE; res = s;
	d=length(p01); n=p01/d; r = SH_1_0( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p02); n=p02/d; r = SH_1_1( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p03); n=p03/d; r = SH_1_2( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p04); n=p04/d; r = SH_2_0( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p05); n=p05/d; r = SH_2_1( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p06); n=p06/d; r = SH_2_2( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p07); n=p07/d; r = SH_2_3( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p08); n=p08/d; r = SH_2_4( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p09); n=p09/d; r = SH_3_0( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p10); n=p10/d; r = SH_3_1( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p11); n=p11/d; r = SH_3_2( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p12); n=p12/d; r = SH_3_3( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p13); n=p13/d; r = SH_3_4( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p14); n=p14/d; r = SH_3_5( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p15); n=p15/d; r = SH_3_6( n ); s = SHAPE; if( s.x<res.x ) res=s;

	return vec3( res.x, 0.5+0.5*res.y, res.z );
}

vec3 intersect( in vec3 ro, in vec3 rd )
{
	vec3 res = vec3(1e10,-1.0, 1.0);

	float maxd = 10.0;
    float h = 1.0;
    float t = 0.0;
    vec2  m = vec2(-1.0);
    for( int i=0; i<200; i++ )
    {
        if( h<0.001||t>maxd ) break;
	    vec3 res = map( ro+rd*t );
        h = res.x;
		m = res.yz;
        t += h*0.3;
    }
	if( t<maxd && t<res.x ) res=vec3(t,m);


	return res;
}

vec3 calcNormal( in vec3 pos )
{
    vec3 eps = vec3(0.001,0.0,0.0);

	return normalize( vec3(
           map(pos+eps.xyy).x - map(pos-eps.xyy).x,
           map(pos+eps.yxy).x - map(pos-eps.yxy).x,
           map(pos+eps.yyx).x - map(pos-eps.yyx).x ) );
}

    """

    odf_test_impl = """

        // camera matrix
        vec3 ww = vec3(0.0,0.0,0.0); //MCDCMatrix (2);
        vec3 uu = vec3(0.0,0.0,0.0); //MCDCMatrix (0);
        vec3 vv = vec3(0.0,0.0,0.0); //MCDCMatrix (1);
        vec3 tot = vec3(0.0);
        vec2 p = (-vec2(0.5, 0.5) + (2.0*point,0)) / 0.5;
        vec3 ro = vec3(0.0,0.0,0.0);

        // create view ray
        vec3 rd = normalize( p.x*uu + p.y*vv + 2.0*ww );

        // background
        vec3 col = vec3(0.3) * clamp(1.0-length(point)*0.5,0.0,1.0);

        // raymarch
        vec3 tmat = intersect(ro,rd);
        if( tmat.y>-0.5 )
        {
            // geometry
            vec3 pos = ro + tmat.x*rd;
            vec3 nor = calcNormal(pos);
            vec3 ref = reflect( rd, nor );

            // material
            vec3 mate = 0.5*mix( vec3(1.0,0.6,0.15), vec3(0.2,0.4,0.5), tmat.y );

            float occ = clamp( 2.0*tmat.z, 0.0, 1.0 );
            float sss = pow( clamp( 1.0 + dot(nor,rd), 0.0, 1.0 ), 1.0 );

            // lights
            vec3 lin  = 2.5*occ*vec3(1.0,1.00,1.00)*(0.6+0.4*nor.y);
                 lin += 1.0*sss*vec3(1.0,0.95,0.70)*occ;

            // surface-light interaction
            col = mate.xyz * lin;
        }

        // gamma
        col = pow( clamp(col,0.0,1.0), vec3(0.4545) );
        tot += col;
    fragOutput0 = vec4( tot, 1.0 );
    """

    scene = window.Scene()
    scene.background((0.8, 0.8, 0.8))
    centers = np.array([[2, 0, 0], [0, 0, 0], [-2, 0, 0]])
    # np.random.rand(3, 3) * 3
    # colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    colors = np.random.rand(3, 3) * 255
    scale = 1  # np.random.rand(3) * 5

    # https://www.shadertoy.com/view/MstXWS
    # https://www.shadertoy.com/view/XsX3R4

    fs_dec = """
        uniform mat4 MCDCMatrix;
        uniform mat4 MCVCMatrix;


        float sdRoundBox( vec3 p, vec3 b, float r )
        {
            vec3 q = abs(p) - b;
            return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
        }

        float sdEllipsoid( vec3 p, vec3 r )
        {
        float k0 = length(p/r);
        float k1 = length(p/(r*r));
        return k0*(k0-1.0)/k1;
        }
        float sdCylinder(vec3 p, float h, float r)
        {
            vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
            return min(max(d.x,d.y),0.0) + length(max(d,0.0));
        }
        float sdSphere(vec3 pos, float r)
        {
            float d = length(pos) - r;

            return d;
        }
        float map( in vec3 pos)
        {
            float d = sdSphere(pos-0.5, .2);
            float d1 = sdCylinder(pos+0.5, 0.05, .5);
            float d2 = sdEllipsoid(pos + vec3(-0.5,0.5,0), vec3(0.2,0.3,0.5));
            float d3 = sdRoundBox(pos + vec3(0.5,-0.5,0), vec3(0.2,0.1,0.3), .05);


            //.xy

            return min(min(min(d, d1), d2), d3);
        }

        vec3 calcNormal( in vec3 pos )
        {
            vec2 e = vec2(0.0001,0.0);
            return normalize( vec3(map(pos + e.xyy) - map(pos - e.xyy ),
                                   map(pos + e.yxy) - map(pos - e.yxy),
                                   map(pos + e.yyx) - map(pos - e.yyx)
                                   )
                            );
        }

        float castRay(in vec3 ro, vec3 rd)
        {
            float t = 0.0;
            for(int i=0; i < 100; i++)
            {
                vec3 pos = ro + t * rd;
                vec3 nor = calcNormal(pos);

                float h = map(pos);
                if (h < 0.001) break;

                t += h;
                if (t > 20.0) break;
            }
            return t;
        }
        """

    fake_sphere = """

    vec3 uu = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]); // camera right
    vec3 vv = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]); //  camera up
    vec3 ww = vec3(MCVCMatrix[0][2], MCVCMatrix[1][2], MCVCMatrix[2][2]); // camera direction
    vec3 ro = MCVCMatrix[3].xyz * mat3(MCVCMatrix);  // camera position

    // create view ray
    vec3 rd = normalize( point.x*-uu + point.y*-vv + 7*ww);
    vec3 col = vec3(0.0);

    float t = castRay(ro, rd);
    if (t < 20.0)
    {
        vec3 pos = ro + t * rd;
        vec3 nor = calcNormal(pos);
        vec3 sun_dir = vec3(MCVCMatrix[0][2], MCVCMatrix[1][2], MCVCMatrix[2][2]); //normalize()
        float dif = clamp( dot(nor, sun_dir), 0.0, 1.0);
        //vec3 sun_dif = normalize()
        col = color * dot(color,nor); // (color + diffuseColor + ambientColor + specularColor)*nor.zzz;//vec3(1.0);
        fragOutput0 = vec4(col, 1.0);
    }
    else{
        //fragOutput0 = vec4(0,1,0, 1.0);
        discard;
        }


    /*float radius = 1.;
    if(len > radius)
        {discard;}

    //err, lightColor0 vertexColorVSOutput normalVCVSOutput, ambientIntensity; diffuseIntensity;specularIntensity;specularColorUniform;
    float c = len;
    fragOutput0 =  vec4(c,c,c, 1);




    vec3 normalizedPoint = normalize(vec3(point.xy, sqrt(1. - len)));
    vec3 direction = normalize(vec3(1., 1., 1.));
    float df = max(0, dot(direction, normalizedPoint));
    float sf = pow(df, 24);
    fragOutput0 = vec4(max(df * color, sf * vec3(1)), 1);*/
    """

    billboard_actor = actor.billboard(
        centers,
        colors=colors.astype(np.uint8),
        scale=scale,
        fs_dec=fs_dec,
        fs_impl=fake_sphere,
    )
    scene.add(billboard_actor)
    scene.add(actor.axes())
    scene.camera_info()
    matrix = scene.camera().GetViewTransformMatrix()
    mat = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            mat[i, j] = matrix.GetElement(i, j)
    print(mat)
    print(np.dot(-mat[:3, 3], mat[:3, :3]))  # camera position
    window.show(scene)


def test_spheres_on_canvas():

    scene = window.Scene()
    showm = window.ShowManager(scene, reset_camera=False)

    # colors = 255 * np.array([
    #     [.85, .07, .21], [.56, .14, .85], [.16, .65, .20], [.95, .73, .06],
    #     [.95, .55, .05], [.62, .42, .75], [.26, .58, .85], [.24, .82, .95],
    #     [.95, .78, .25], [.85, .58, .35], [1., 1., 1.]
    # ])
    n_points = 2000000
    colors = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    )   # 255 * np.random.rand(n_points, 3)
    # n_points = colors.shape[0]
    np.random.seed(42)
    centers = np.array(
        [[2, 0, 0], [0, 2, 0], [0, 0, 0]]
    )   # 500 * np.random.rand(n_points, 3) - 250 # np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    radius = [2, 2, 2]   # np.random.rand(n_points) #  [1, 1, 2]

    polydata = vtk.vtkPolyData()

    verts, faces = fp.prim_square()

    big_verts = np.tile(verts, (centers.shape[0], 1))
    big_cents = np.repeat(centers, verts.shape[0], axis=0)

    big_verts += big_cents

    # print(big_verts)

    big_scales = np.repeat(radius, verts.shape[0], axis=0)

    # print(big_scales)

    big_verts *= big_scales[:, np.newaxis]

    # print(big_verts)

    tris = np.array([[0, 1, 2], [2, 3, 0]], dtype='i8')

    big_tris = np.tile(tris, (centers.shape[0], 1))
    shifts = np.repeat(
        np.arange(0, centers.shape[0] * verts.shape[0], verts.shape[0]), tris.shape[0]
    )

    big_tris += shifts[:, np.newaxis]

    # print(big_tris)

    big_cols = np.repeat(colors, verts.shape[0], axis=0)

    # print(big_cols)

    big_centers = np.repeat(centers, verts.shape[0], axis=0)

    # print(big_centers)

    big_centers *= big_scales[:, np.newaxis]

    # print(big_centers)

    set_polydata_vertices(polydata, big_verts)
    set_polydata_triangles(polydata, big_tris)
    set_polydata_colors(polydata, big_cols)

    vtk_centers = numpy_support.numpy_to_vtk(big_centers, deep=True)
    vtk_centers.SetNumberOfComponents(3)
    vtk_centers.SetName('center')
    polydata.GetPointData().AddArray(vtk_centers)

    canvas_actor = get_actor_from_polydata(polydata)
    canvas_actor.GetProperty().BackfaceCullingOff()

    scene.add(canvas_actor)

    mapper = canvas_actor.GetMapper()

    mapper.MapDataArrayToVertexAttribute(
        'center', 'center', vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Vertex,
        '//VTK::ValuePass::Dec',
        True,
        """
        //VTK::ValuePass::Dec
        in vec3 center;

        uniform mat4 Ext_mat;

        out vec3 centeredVertexMC;
        out vec3 cameraPosition;
        out vec3 viewUp;

        """,
        False,
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Vertex,
        '//VTK::ValuePass::Impl',
        True,
        """
        //VTK::ValuePass::Impl
        centeredVertexMC = vertexMC.xyz - center;
        float scalingFactor = 1. / abs(centeredVertexMC.x);
        centeredVertexMC *= scalingFactor;

        vec3 CameraRight_worldspace = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
        vec3 CameraUp_worldspace = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);

        vec3 vertexPosition_worldspace = center + CameraRight_worldspace * 1 * centeredVertexMC.x + CameraUp_worldspace * 1 * centeredVertexMC.y;
        gl_Position = MCDCMatrix * vec4(vertexPosition_worldspace, 1.);

        """,
        False,
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        '//VTK::ValuePass::Dec',
        True,
        """
        //VTK::ValuePass::Dec
        in vec3 centeredVertexMC;
        in vec3 cameraPosition;
        in vec3 viewUp;

        uniform vec3 Ext_camPos;
        uniform vec3 Ext_viewUp;
        """,
        False,
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        '//VTK::Light::Impl',
        True,
        """
        // Renaming variables passed from the Vertex Shader
        vec3 color = vertexColorVSOutput.rgb;
        vec3 point = centeredVertexMC;
        fragOutput0 = vec4(color, 0.7);
        /*
        // Comparing camera position from vertex shader and python
        float dist = distance(cameraPosition, Ext_camPos);
        if(dist < .0001)
            fragOutput0 = vec4(1, 0, 0, 1);
        else
            fragOutput0 = vec4(0, 1, 0, 1);


        // Comparing view up from vertex shader and python
        float dist = distance(viewUp, Ext_viewUp);
        if(dist < .0001)
            fragOutput0 = vec4(1, 0, 0, 1);
        else
            fragOutput0 = vec4(0, 1, 0, 1);
        */
        float len = length(point);
        // VTK Fake Spheres
        float radius = 1.;
        if(len > radius)
          discard;
        vec3 normalizedPoint = normalize(vec3(point.xy, sqrt(1. - len)));
        vec3 direction = normalize(vec3(1., 1., 1.));
        float df = max(0, dot(direction, normalizedPoint));
        float sf = pow(df, 24);
        fragOutput0 = vec4(max(df * color, sf * vec3(1)), 1);
        """,
        False,
    )

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def vtk_shader_callback(caller, event, calldata=None):
        res = scene.size()
        camera = scene.GetActiveCamera()
        cam_pos = camera.GetPosition()
        foc_pnt = camera.GetFocalPoint()
        view_up = camera.GetViewUp()
        # cam_light_mat = camera.GetCameraLightTransformMatrix()
        # comp_proj_mat = camera.GetCompositeProjectionTransformMatrix()
        # exp_proj_mat = camera.GetExplicitProjectionTransformMatrix()
        # eye_mat = camera.GetEyeTransformMatrix()
        # model_mat = camera.GetModelTransformMatrix()
        # model_view_mat = camera.GetModelViewTransformMatrix()
        # proj_mat = camera.GetProjectionTransformMatrix(scene)
        view_mat = camera.GetViewTransformMatrix()
        mat = view_mat
        np.set_printoptions(precision=3, suppress=True)
        np_mat = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                np_mat[i, j] = mat.GetElement(i, j)
        program = calldata
        if program is not None:
            # print("\nCamera position: {}".format(cam_pos))
            # print("Focal point: {}".format(foc_pnt))
            # print("View up: {}".format(view_up))
            # print(mat)
            # print(np_mat)
            # print(np.dot(-np_mat[:3, 3], np_mat[:3, :3]))
            # a = np.array(cam_pos) - np.array(foc_pnt)
            # print(a / np.linalg.norm(a))
            # print(cam_light_mat)
            # #print(comp_proj_mat)
            # print(exp_proj_mat)
            # print(eye_mat)
            # print(model_mat)
            # print(model_view_mat)
            # print(proj_mat)
            # print(view_mat)
            program.SetUniform2f('Ext_res', res)
            program.SetUniform3f('Ext_camPos', cam_pos)
            program.SetUniform3f('Ext_focPnt', foc_pnt)
            program.SetUniform3f('Ext_viewUp', view_up)
            program.SetUniformMatrix('Ext_mat', mat)

    mapper.AddObserver(vtk.vtkCommand.UpdateShaderEvent, vtk_shader_callback)

    global timer
    timer = 0

    def timer_callback(obj, event):
        global timer
        timer += 1.0
        showm.render()
        scene.azimuth(2)
        # scene.elevation(5)
        # scene.roll(5)

    label = vtk.vtkOpenGLBillboardTextActor3D()
    label.SetInput('FURY Rocks!!!')
    label.SetPosition(1.0, 1.0, 1)
    label.GetTextProperty().SetFontSize(40)
    label.GetTextProperty().SetColor(0.5, 0.5, 0.5)
    # TODO: Get Billboard's mapper
    # l_mapper = label.GetActors()

    # scene.add(label)
    scene.add(actor.axes())

    scene.background((1, 1, 1))

    # scene.set_camera(position=(1.5, 2.5, 15), focal_point=(1.5, 2.5, 1.5),
    #                  view_up=(0, 1, 0))
    scene.set_camera(position=(1.5, 2.5, 25), focal_point=(0, 0, 0), view_up=(0, 1, 0))

    showm.add_timer_callback(True, 100, timer_callback)
    showm.start()


def test_fireballs_on_canvas():
    scene = window.Scene()
    showm = window.ShowManager(scene)

    # colors = 255 * np.array([
    #     [.85, .07, .21], [.56, .14, .85], [.16, .65, .20], [.95, .73, .06],
    #     [.95, .55, .05], [.62, .42, .75], [.26, .58, .85], [.24, .82, .95],
    #     [.95, .78, .25], [.85, .58, .35], [1., 1., 1.]
    # ])
    colors = np.random.rand(1000000, 3) * 255
    n_points = colors.shape[0]
    np.random.seed(42)
    centers = 500 * np.random.rand(n_points, 3) - 250

    radius = 0.5 * np.ones(n_points)

    polydata = vtk.vtkPolyData()

    verts = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    verts -= np.array([0.5, 0.5, 0])

    big_verts = np.tile(verts, (centers.shape[0], 1))
    big_cents = np.repeat(centers, verts.shape[0], axis=0)

    big_verts += big_cents

    big_scales = np.repeat(radius, verts.shape[0], axis=0)

    big_verts *= big_scales[:, np.newaxis]

    tris = np.array([[0, 1, 2], [2, 3, 0]], dtype='i8')

    big_tris = np.tile(tris, (centers.shape[0], 1))
    shifts = np.repeat(
        np.arange(0, centers.shape[0] * verts.shape[0], verts.shape[0]), tris.shape[0]
    )

    big_tris += shifts[:, np.newaxis]

    big_cols = np.repeat(colors, verts.shape[0], axis=0)

    big_centers = np.repeat(centers, verts.shape[0], axis=0)

    big_centers *= big_scales[:, np.newaxis]

    set_polydata_vertices(polydata, big_verts)
    set_polydata_triangles(polydata, big_tris)
    set_polydata_colors(polydata, big_cols)

    vtk_centers = numpy_support.numpy_to_vtk(big_centers, deep=True)
    vtk_centers.SetNumberOfComponents(3)
    vtk_centers.SetName('center')
    polydata.GetPointData().AddArray(vtk_centers)

    canvas_actor = get_actor_from_polydata(polydata)
    canvas_actor.GetProperty().BackfaceCullingOff()

    scene.add(canvas_actor)

    mapper = canvas_actor.GetMapper()

    mapper.MapDataArrayToVertexAttribute(
        'center', 'center', vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Vertex,
        '//VTK::ValuePass::Dec',
        True,
        """
        //VTK::ValuePass::Dec
        in vec3 center;
        out vec3 centeredVertexMC;
        """,
        False,
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Vertex,
        '//VTK::ValuePass::Impl',
        True,
        """
        //VTK::ValuePass::Impl
        centeredVertexMC = vertexMC.xyz - center;
        float scalingFactor = 1. / abs(centeredVertexMC.x);
        centeredVertexMC *= scalingFactor;

        vec3 CameraRight_worldspace = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
        vec3 CameraUp_worldspace = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);

        vec3 vertexPosition_worldspace = center + CameraRight_worldspace * .5 * centeredVertexMC.x + CameraUp_worldspace * .5 * centeredVertexMC.y;
        gl_Position = MCDCMatrix * vec4(vertexPosition_worldspace, 1.);
        """,
        False,
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        '//VTK::ValuePass::Dec',
        True,
        """
        //VTK::ValuePass::Dec
        in vec3 centeredVertexMC;
        uniform float time;

        float snoise(vec3 uv, float res) {
            const vec3 s = vec3(1e0, 1e2, 1e3);
            uv *= res;
            vec3 uv0 = floor(mod(uv, res)) * s;
            vec3 uv1 = floor(mod(uv + vec3(1.), res)) * s;
            vec3 f = fract(uv);
            f = f * f * (3. - 2. * f);
            vec4 v = vec4(uv0.x + uv0.y + uv0.z, uv1.x + uv0.y + uv0.z,
                uv0.x + uv1.y + uv0.z, uv1.x + uv1.y + uv0.z);
            vec4 r = fract(sin(v * 1e-1) * 1e3);
            float r0 = mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y);
            r = fract(sin((v + uv1.z - uv0.z) * 1e-1) * 1e3);
            float r1 = mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y);
            return mix(r0, r1, f.z) * 2. - 1.;
        }
        """,
        False,
    )

    mapper.AddShaderReplacement(
        vtk.vtkShader.Fragment,
        '//VTK::Light::Impl',
        True,
        """
        // Renaming variables passed from the Vertex Shader
        vec3 color = vertexColorVSOutput.rgb;
        vec3 point = centeredVertexMC;
        float len = length(point);
        float fColor = 2. - 2. * len;
        vec3 coord = vec3(atan(point.x, point.y) / 6.2832 + .5, len * .4, .5);
        for(int i = 1; i <= 7; i++) {
            float power = pow(2., float(i));
            fColor += (1.5 / power) * snoise(coord +
                vec3(0., -time * .005, time * .001), power * 16.);
        }
        if(fColor < 0) discard;
        //color = vec3(fColor);
        color *= fColor;
        fragOutput0 = vec4(color, 1.);
        """,
        False,
    )

    global timer
    timer = 0

    def timer_callback(obj, event):
        global timer
        timer += 1.0
        showm.render()

    @window.vtk.calldata_type(window.vtk.VTK_OBJECT)
    def vtk_shader_callback(caller, event, calldata=None):
        program = calldata
        global timer
        if program is not None:
            try:
                program.SetUniformf('time', timer)
            except ValueError:
                pass

    mapper.AddObserver(window.vtk.vtkCommand.UpdateShaderEvent, vtk_shader_callback)

    showm.add_timer_callback(True, 100, timer_callback)
    showm.start()


# test_spheres_on_canvas()
# test_fireballs_on_canvas()
test_sh()
