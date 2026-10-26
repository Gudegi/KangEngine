// Deterministic 4x4 Gauss-Legendre integration over the emitting rectangle.
// The scene retains one area light; samples are an OpenGL rendering choice.
// Occlusion and textured emission are not evaluated by this forward pass.
vec3 evaluateRectLights(vec3 N, vec3 V, vec3 position, vec3 albedo,
                        float metallic, float roughness) {
    const float nodes[4] = float[4](-0.861136312, -0.339981044, 0.339981044, 0.861136312);
    const float weights[4] = float[4](0.347854845, 0.652145155, 0.652145155, 0.347854845);
    vec3 result = vec3(0.0);
    for (int i = 0; i < min(lightCounts.z, MAX_RECT_LIGHTS); ++i) {
        vec3 ex = rectEdgeX[i].xyz, ey = rectEdgeY[i].xyz;
        vec3 areaNormal = cross(ey, ex) * rectCenter[i].w;
        float area = length(areaNormal);
        if (area < 1e-10 || dot(rectRadiance[i].rgb, vec3(1)) <= 0.0)
            continue;
        vec3 emitterNormal = areaNormal / area;
        if (dot(emitterNormal, position - rectCenter[i].xyz) <= 0.0)
            continue;
        for (int x = 0; x < 4; ++x) {
            for (int y = 0; y < 4; ++y) {
                vec3 samplePosition = rectCenter[i].xyz + 0.5 * (ex * nodes[x] + ey * nodes[y]);
                vec3 d = samplePosition - position;
                float distanceSquared = max(dot(d, d), 1e-8);
                vec3 L = d * inversesqrt(distanceSquared);
                float nl = max(dot(N, L), 0.0);
                if (nl <= 0.0)
                    continue;
                vec3 H = normalize(V + L);
                float nv = max(dot(N, V), 0.0);
                vec3 F = fresnelSchlick(max(dot(H, V), 0.0), mix(vec3(0.04), albedo, metallic));
                float G = geometrySchlickGGX(nv, roughness) * geometrySchlickGGX(nl, roughness);
                vec3 specular = distributionGGX(N, H, roughness) * G * F / max(4.0 * nv * nl, 0.000001);
                vec3 diffuse = (1.0 - F) * (1.0 - metallic) * albedo / PI;
                float solidAngle = area * 0.25 * weights[x] * weights[y] *
                    max(dot(emitterNormal, -L), 0.0) / distanceSquared;
                result += (diffuse * rectEdgeX[i].w + specular * rectEdgeY[i].w) *
                    rectRadiance[i].rgb * nl * solidAngle;
            }
        }
    }
    return result;
}
