#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec4 aInstanceTransform0;
layout(location = 4) in vec4 aInstanceTransform1;
layout(location = 5) in vec4 aInstanceTransform2;
layout(location = 6) in vec4 aInstanceTransform3;
layout(location = 8) in ivec4 aBoneIndices;
layout(location = 9) in vec4 aBoneWeights;

layout(std140) uniform ke_g0_b0 {
    mat4 view;
    mat4 projection;
};
layout(std140) uniform ke_g2_b0 {
    mat4 boneMatrices[128];
};

out vec2 vTexCoord;

void main() {
    mat4 model = mat4(aInstanceTransform0, aInstanceTransform1,
                      aInstanceTransform2, aInstanceTransform3);
    mat4 skin = mat4(0.0);

    float weightSum = 0.0;
    for (int i = 0; i < 4; ++i) {
        if (aBoneIndices[i] >= 0) {
            skin += aBoneWeights[i] * boneMatrices[aBoneIndices[i]];
            weightSum += aBoneWeights[i];
        }
    }
    if (weightSum <= 0.0)
        skin = mat4(1.0);

    gl_Position = projection * view * model * skin * vec4(aPos, 1.0);
    vTexCoord = aTexCoord;
}
