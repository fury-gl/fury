{{ bindings_code }}
{$ include 'fury.utils.wgsl' $}


const DATA_SHAPE = vec3<i32>{{ data_shape }};
const VISIBLE = vec3<i32>{{ cross_section }};
const TOTAL_VECTORS = i32({{ total_vectors }});
const NUM_VECTORS = i32({{ num_vectors }});

@compute @workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = i32(global_id.x);

    if (id >= TOTAL_VECTORS) {
        return;
    }

    // let voxel_id = get_voxel_id(id, DATA_SHAPE, VISIBLE);
    // let first_vector_id = get_flatten_id(voxel_id, DATA_SHAPE, NUM_VECTORS);

    let first_vector_id = id;
    let voxel_id = load_s_centers(first_vector_id);

    // for (var i: i32 = 0; i < NUM_VECTORS; i++) {
        // Load the vector data
        let raw_vector = load_s_directions(first_vector_id);

        // Calculate the position of the points
        let pt0 = raw_vector + vec3<f32>(voxel_id);
        let pt1 = -raw_vector + vec3<f32>(voxel_id);

        // Store the points in the buffer
        let position_idx = (first_vector_id) * 2;

        s_positions[position_idx] = pt0.x;
        s_positions[position_idx + 1] = pt0.y;
        s_positions[position_idx + 2] = pt0.z;
        s_positions[position_idx + 3] = pt1.x;
        s_positions[position_idx + 4] = pt1.y;
        s_positions[position_idx + 5] = pt1.z;

        // Store the color of the points
        let color = orient2rgb(pt1 - pt0);
        s_colors[position_idx] = color.x;
        s_colors[position_idx + 1] = color.y;
        s_colors[position_idx + 2] = color.z;
        s_colors[position_idx + 3] = color.x;
        s_colors[position_idx + 4] = color.y;
        s_colors[position_idx + 5] = color.z;

    // }
}
