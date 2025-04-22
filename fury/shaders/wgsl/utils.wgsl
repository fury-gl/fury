fn orient2rgb(v: vec3<f32>) -> vec3<f32> {
    let r = sqrt(dot(v, v));

    if (r != 0.0) {
      return abs(v / r);
    }
    return vec3<f32>(1.0);
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
