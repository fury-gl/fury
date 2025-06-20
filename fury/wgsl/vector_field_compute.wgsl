{{ bindings_code }}
{$ include 'fury.utils.wgsl' $}


const DATA_SHAPE = vec3<i32>{{ data_shape }};
const NUM_VECTORS = i32({{ num_vectors }});

@compute @workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    let voxel_id = i32(global_id.x);
    let center = flatten_to_3d(voxel_id, DATA_SHAPE);

    for (var i: i32 = 0; i < NUM_VECTORS; i++) {

        let vector_id = voxel_id * NUM_VECTORS + i;

        let scale = load_s_scales(vector_id);
        let raw_vector = load_s_vectors(vector_id);
        let vector = raw_vector * vec3<f32>(scale);

        let point_i = vector + vec3<f32>(center);
        let point_e = vec3<f32>(-1.0) * vector + vec3<f32>(center);
        let position_idx = vector_id * 6;

        // Removes vectors that are too small.
        if all(abs(point_i - point_e) < vec3<f32>(1e-6)) {
            s_positions[position_idx] = f32(center.x);
            s_positions[position_idx + 1] = f32(center.y);
            s_positions[position_idx + 2] = f32(center.z);
            s_positions[position_idx + 3] = f32(center.x);
            s_positions[position_idx + 4] = f32(center.y);
            s_positions[position_idx + 5] = f32(center.z);
            continue;
        }

        // Adjust the direction of the vector based on the z-component
        if vector.z < 0.0 {
            s_positions[position_idx] = point_e.x;
            s_positions[position_idx + 1] = point_e.y;
            s_positions[position_idx + 2] = point_e.z;
            s_positions[position_idx + 3] = point_i.x;
            s_positions[position_idx + 4] = point_i.y;
            s_positions[position_idx + 5] = point_i.z;
        } else {
            s_positions[position_idx] = point_i.x;
            s_positions[position_idx + 1] = point_i.y;
            s_positions[position_idx + 2] = point_i.z;
            s_positions[position_idx + 3] = point_e.x;
            s_positions[position_idx + 4] = point_e.y;
            s_positions[position_idx + 5] = point_e.z;
        }

        // Set the colors in the output buffer
        if all(load_s_colors(vector_id * 2) == vec3<f32>(0.0)) {
            let color = orient2rgb(point_e - point_i);
            s_colors[position_idx] = color.x;
            s_colors[position_idx + 1] = color.y;
            s_colors[position_idx + 2] = color.z;
            s_colors[position_idx + 3] = color.x;
            s_colors[position_idx + 4] = color.y;
            s_colors[position_idx + 5] = color.z;
        }
    }

}
