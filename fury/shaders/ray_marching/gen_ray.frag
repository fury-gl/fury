void gen_ray(out vec3 ro, out vec3 rd, out float t)
{
    // Vertex in Model Coordinates
    vec3 point = vertexMCVSOutput.xyz;

    // Ray Origin
    // Camera position in world space
    ro = (-MCVCMatrix[3] * MCVCMatrix).xyz;

    // Ray Direction
    rd = normalize(point - ro);

    ro += point - ro;

    // Total distance traversed along the ray
    t = castRay(ro, rd);
}
