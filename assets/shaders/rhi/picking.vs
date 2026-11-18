#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec4 aInstanceTransform0;
layout(location = 4) in vec4 aInstanceTransform1;
layout(location = 5) in vec4 aInstanceTransform2;
layout(location = 6) in vec4 aInstanceTransform3;
layout(location = 7) in float aOpacity;
#ifdef KE_SKINNED
layout(location = 8) in ivec4 aBoneIndices;
layout(location = 9) in vec4 aBoneWeights;
#endif

layout(std140) uniform ke_g0_b0 {
    mat4 viewProjection;
};
layout(std140) uniform ke_g1_b0 {
    uvec4 ids;
    vec4 alphaParams;
};
#ifdef KE_SKINNED
layout(std140) uniform ke_g2_b0 {
    mat4 boneMatrices[128];
};
#endif

out vec2 vTexCoord;
flat out uint vId;
flat out float vOpacity;

// Project the posed mesh and pass its instance ID and opacity to the fragment shader.
void main() {
    vec4 position = vec4(aPos, 1.0);
#ifdef KE_SKINNED
    mat4 skin = mat4(0.0);
    float weightSum = 0.0;
    for (int i = 0; i < 4; ++i) {
        if (aBoneIndices[i] >= 0) {
            skin += boneMatrices[aBoneIndices[i]] * aBoneWeights[i];
            weightSum += aBoneWeights[i];
        }
    }
    if (weightSum > 0.0)
        position = skin * position;
#endif
    mat4 model = mat4(aInstanceTransform0, aInstanceTransform1,
                      aInstanceTransform2, aInstanceTransform3);
    gl_Position = viewProjection * model * position;
    vTexCoord = aTexCoord;
    vId = ids.x + uint(gl_InstanceID);
    vOpacity = aOpacity;
}
