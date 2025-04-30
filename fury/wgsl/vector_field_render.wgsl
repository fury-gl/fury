{$ include 'pygfx.std.wgsl' $}
{$ include 'fury.utils.wgsl' $}

struct VertexInput {
    @builtin(vertex_index) index : u32,
};

const DATA_SHAPE = vec3<i32>{{ data_shape }};
const NUM_VECTORS = i32({{ num_vectors }});

@vertex
fn vs_main(in: VertexInput) -> Varyings {

    let i0 = i32(in.index);

    let raw_pos = load_s_positions(i0);
    let wpos = u_wobject.world_transform * vec4<f32>(raw_pos.xyz, 1.0);
    let npos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos;

    var varyings: Varyings;
    varyings.position = vec4<f32>(npos);
    varyings.world_pos = vec3<f32>(ndc_to_world_pos(npos));

    // let diff = load_s_diffs(i0);

    let center = flatten_to_3d(i0 / (NUM_VECTORS * 2), DATA_SHAPE);
    let color = load_s_colors(i0);

    // if all(color == vec3<f32>(0.0)) {
    //     varyings.color = vec4<f32>(orient2rgb(diff), 1.0);
    // } else {
        varyings.color = vec4<f32>(color, 1.0);
    // }

    // varyings.color = vec4<f32>(orient2rgb(diff), 1.0);
    varyings.center = vec3<i32>(center);
    varyings.cross_section = vec3<i32>(u_material.cross_section.xyz);

    return varyings;
}

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    {$ include 'pygfx.clipping_planes.wgsl' $}

    let cross_section = varyings.cross_section;
    if !all(cross_section == vec3<i32>(-1)) && !visible_cross_section(varyings.center, cross_section) {
        discard;
    }

    let color = varyings.color;
    let physical_color = srgb2physical(color.rgb);
    let opacity = color.a * u_material.opacity;
    let out_color = vec4<f32>(physical_color, opacity);

    return get_fragment_output(varyings.position, out_color);
}
