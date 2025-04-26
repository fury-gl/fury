{{ bindings_code }}
{$ include 'fury.utils.wgsl' $}


const DATA_SHAPE = vec3<i32>{{ data_shape }};
const VISIBLE = vec3<i32>{{ cross_section }};
const NUM_VECTORS = i32({{ num_vectors }});

@compute @workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // if (global_id.x >= u32(NUM_VECTORS)) {
    //     return;
    // }
    let voxel_id = i32(global_id.x);
    let center = flatten_to_3d(voxel_id, DATA_SHAPE);

    for (var i: i32 = 0; i < NUM_VECTORS; i++) {

        let vector_id = voxel_id * NUM_VECTORS + i;
        
        let vector = load_s_directions(vector_id);
    
        if all(vector == vec3<f32>(0.0)) {
            continue;
        }


        let point_i = vector + vec3<f32>(center);
        let point_e = vec3<f32>(-1.0) * vector + vec3<f32>(center);
        let position_idx = vector_id * 6;


        // Set the positions in the output buffer
        s_positions[position_idx] = point_i.x;
        s_positions[position_idx + 1] = point_i.y;
        s_positions[position_idx + 2] = point_i.z;
        s_positions[position_idx + 3] = point_e.x;
        s_positions[position_idx + 4] = point_e.y;
        s_positions[position_idx + 5] = point_e.z;

        // Set the colors in the output buffer
        let color = orient2rgb(point_e - point_i);
        s_colors[position_idx] = color.x;
        s_colors[position_idx + 1] = color.y;
        s_colors[position_idx + 2] = color.z;
        s_colors[position_idx + 3] = color.x;
        s_colors[position_idx + 4] = color.y;
        s_colors[position_idx + 5] = color.z;
    }

}