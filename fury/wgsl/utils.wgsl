fn orient2rgb(v: vec3<f32>) -> vec3<f32> {
    let r = sqrt(dot(v, v));

    if (r != 0.0) {
      return abs(v / r);
    }
    return vec3<f32>(1.0);
}

fn scaled_color(v: vec3<f32>) -> vec3<f32> {
    return abs(normalize(v));
}

fn visible_cross_section(center: vec3<i32>, cross_section: vec3<i32>) -> bool {
    let xVal = center.x == cross_section.x;
    let yVal = center.y == cross_section.y;
    let zVal = center.z == cross_section.z;

    return xVal || yVal || zVal;
}

fn visible_range(center: vec3<i32>, low_range: vec3<i32>, high_range: vec3<i32>) -> bool {
    let xVal = center.x >= low_range.x && center.x <= high_range.x;
    let yVal = center.y >= low_range.y && center.y <= high_range.y;
    let zVal = center.z >= low_range.z && center.z <= high_range.z;

    return xVal && yVal && zVal;
}

fn get_voxel_id(id: i32, data_shape: vec3<i32>, visible: vec3<i32>) -> vec3<i32> {
    var slice_id = id;

    let z_slice = data_shape.x * data_shape.y;
    let x_slice = data_shape.y * data_shape.z;
    let y_slice = data_shape.x * data_shape.z;

    // Calculate if the voxel_id is within Z slice
    if (slice_id < z_slice) {
        let x = slice_id / data_shape.y;
        let y = slice_id % data_shape.y;
        return vec3<i32>(x, y, visible.z);
    }

    // Calculate if the voxel_id is within Y slice
    slice_id = slice_id - z_slice;
    if (slice_id < y_slice) {
        let x = slice_id / data_shape.z;
        let z = slice_id % data_shape.z;
        return vec3<i32>(x, visible.y, z);
    }

    // Calculate if the voxel_id is within X slice
    slice_id = slice_id - y_slice;
    let y = slice_id / data_shape.z;
    let z = slice_id % data_shape.z;
    return vec3<i32>(visible.x, y, z);

}

fn flatten_from_3d(coord: vec3<i32>, data_shape: vec3<i32>) -> i32 {
    return coord.x * data_shape.y * data_shape.z + coord.y * data_shape.z + coord.z;
}

fn flatten_to_3d(index: i32, data_shape: vec3<i32>) -> vec3<i32> {
    let z = index % data_shape.z;
    let y = (index / data_shape.z) % data_shape.y;
    let x = index / (data_shape.y * data_shape.z);
    return vec3<i32>(x, y, z);
}
