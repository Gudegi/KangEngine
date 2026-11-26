#version 410 core
layout(location = 0) out vec4 outId;

in vec2 vTexCoord;
flat in uint vId;
flat in float vOpacity;
#ifdef KE_INSPECTION
in vec3 vWorldPosition;
in float vDepth;
flat in uint vInstanceId;
#endif

layout(std140) uniform ke_g1_b0 {
    uvec4 ids;
    // x: cutoff (-1 disables), y: red channel, z: material opacity.
    vec4 alphaParams;
};
#ifdef KE_TEXTURED
uniform sampler2D ke_g3_b0;
#endif

// Discard unpickable or transparent fragments and write the packed instance ID.
void main() {
    float alpha = vOpacity * alphaParams.z;
#ifdef KE_TEXTURED
    vec4 texel = texture(ke_g3_b0, vTexCoord);
    alpha *= alphaParams.y > 0.5 ? texel.r : texel.a;
#endif
    if (vOpacity <= 0.0 || alpha <= 0.0 || alpha < alphaParams.x)
        discard;

#ifdef KE_INSPECTION
    // Inspection colors bypass tone mapping; normals are world-space face normals.
    if (ids.y == 1u) {
        outId = vec4(vec3(clamp(vDepth, 0.0, 1.0)), 1.0);
    } else if (ids.y == 2u) {
        vec3 normal = normalize(cross(dFdx(vWorldPosition), dFdy(vWorldPosition)));
        outId = vec4(normal * 0.5 + 0.5, 1.0);
    } else {
        uint id = (ids.z + 1u) * 1664525u + vInstanceId * 1013904223u;
        id ^= id >> 16u;
        outId = vec4(vec3(uvec3(id, id >> 8u, id >> 16u) & 255u) / 255.0, 1.0);
    }
#else
    // Pack the 32-bit ID into four RGBA8 channels.
    outId = vec4(uvec4(vId, vId >> 8u, vId >> 16u, vId >> 24u) & 255u) / 255.0;
#endif
}
