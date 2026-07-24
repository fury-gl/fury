{$ include 'pygfx.std.wgsl' $}
{$ include 'fury.utils.wgsl' $}

struct VertexInput {
    @builtin(vertex_index) index : u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    var varyings: Varyings;

    // PyGfx handles index buffers manually via storage buffers!
    // in.index is a flat counter (0, 1, 2, 3, 4, 5...)
    let flat_index = i32(in.index);
    let face_index = flat_index / 3;
    let sub_index = flat_index % 3;

    // Load the indices for the current face (triangle)
    let vii = load_s_indices(face_index);
    let i0 = i32(vii[sub_index]);

    // Load position from the storage buffer using the correct vertex index
    let raw_pos = load_s_positions(i0);
    let local_pos = raw_pos.xyz;

    // Transform position to world space
    let world_pos = u_wobject.world_transform * vec4<f32>(local_pos, 1.0);
    // Transform position to clip space
    let clip_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    varyings.position = vec4<f32>(clip_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz);

    // Pass the local unscaled position (which FURY scaled via plane_geometry width/height)
    varyings.texcoord_vert = vec2<f32>(local_pos.xy);

    return varyings;
}

// Signed Distance Function for a rounded box
// p = current point position (relative to center 0,0)
// b = half-size of the box (width/2, height/2)
// r = corner radius
fn sd_round_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r, r);
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    {$ include 'pygfx.clipping_planes.wgsl' $}

    let size = u_material.size;
    let radius = u_material.corner_radius;

    // The geometry is created with the exact physical width and height.
    // Therefore, varyings.texcoord_vert (which comes from in.position)
    // already contains the physical pixel offset from the center!
    let p = varyings.texcoord_vert.xy;
    let b = size * 0.5;

    // Evaluate the SDF
    let distance = sd_round_box(p, b, radius);

    // Smooth the edge using fwidth for perfect anti-aliasing
    let smooth_edge = fwidth(distance);
    let alpha = 1.0 - smoothstep(-smooth_edge, smooth_edge, distance);

    // Discard pixels completely outside the rounded corners
    if (alpha <= 0.0) {
        discard;
    }

    // Fetch base color from the material
    let base_color = u_material.color;
    let physical_color = srgb2physical(base_color.rgb);

    // Multiply material opacity by the SDF alpha
    let final_alpha = base_color.a * u_material.opacity * alpha;

    var out: FragmentOutput;
    out.color = vec4<f32>(physical_color, final_alpha);

    $$ if write_pick
    out.pick = (
        pick_pack(u32(u_wobject.global_id), 20) +
        pick_pack(0u, 26) +
        pick_pack(0u, 18)
    );
    $$ endif

    return out;
}
