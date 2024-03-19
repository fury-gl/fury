vec3 gltf_dielectric_brdf(vec3 incoming, vec3 outgoing, vec3 normal, float roughness, vec3 base_color)
{
    float ni = dot(normal, incoming);
    float no = dot(normal, outgoing);
    // Early out if incoming or outgoing direction are below the horizon
    if (ni <= 0.0 || no <= 0.0)
        return vec3(0.0);
    // Save some work by not actually computing the half-vector. If the half-
    // vector were h, ih = dot(incoming, h) and
    // sqrt(nh_ih_2 / ih_2) = dot(normal, h).
    float ih_2 = dot(incoming, outgoing) * 0.5 + 0.5;
    float sum = ni + no;
    float nh_ih_2 = 0.25 * sum * sum;
    float ih = sqrt(ih_2);

    // Evaluate the GGX normal distribution function
    float roughness_2 = roughness * roughness;
    float roughness_4  = roughness_2 * roughness_2;
    float roughness_flip = 1.0 - roughness_4;
    float denominator = ih_2 - nh_ih_2 * roughness_flip;
    float ggx = (roughness_4 * M_INV_PI * ih_2) / (denominator * denominator);
    // Evaluate the "visibility" (i.e. masking-shadowing times geometry terms)
    float vi = ni + sqrt(roughness_4 + roughness_flip * ni * ni);
    float vo = no + sqrt(roughness_4 + roughness_flip * no * no);
    float v = 1.0 / (vi * vo);
    // That completes the specular BRDF
    float specular = v * ggx;

    // The diffuse BRDF is Lambertian
    vec3 diffuse = M_INV_PI * base_color;

    // Evaluate the Fresnel term using the Fresnel-Schlick approximation
    const float ior = 1.5;
    const float f0 = ((1.0 - ior) / (1.0 + ior)) * ((1.0 - ior) / (1.0 + ior));
    float ih_flip = 1.0 - ih;
    float ih_flip_2 = ih_flip * ih_flip;
    float fresnel = f0 + (1.0 - f0) * ih_flip * ih_flip_2 * ih_flip_2;

    // Mix the two components
    return mix(diffuse, vec3(specular), fresnel);
}
