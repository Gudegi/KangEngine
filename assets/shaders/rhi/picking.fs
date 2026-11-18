#version 410 core
layout(location = 0) out vec4 outId;

in vec2 vTexCoord;
flat in uint vId;
flat in float vOpacity;

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

    // Pack the 32-bit ID into four RGBA8 channels.
    outId = vec4(uvec4(vId, vId >> 8u, vId >> 16u, vId >> 24u) & 255u) / 255.0;
}
