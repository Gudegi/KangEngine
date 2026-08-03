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
    mat4 lightSpaceMatrices[4];
    vec4 cascadeSplits;
    vec4 cascadeOrthoHalfSizes;
    vec4 cascadeMapSizes;
    vec4 shadowParams;
    vec4 shadowInfo; // x: PCF samples, y: cascade count, z: use CSM
};
layout(std140) uniform ke_g2_b0 {
    mat4 boneMatrices[128];
};

out vec2 vTexCoord;

void main() {
    mat4 model = mat4(aInstanceTransform0, aInstanceTransform1,
                      aInstanceTransform2, aInstanceTransform3);
    mat4 skin = mat4(0.0);

    skin += aBoneWeights.x * boneMatrices[aBoneIndices.x];
    skin += aBoneWeights.y * boneMatrices[aBoneIndices.y];
    skin += aBoneWeights.z * boneMatrices[aBoneIndices.z];
    skin += aBoneWeights.w * boneMatrices[aBoneIndices.w];

    gl_Position = lightSpaceMatrices[0] * model * skin * vec4(aPos, 1.0);
    vTexCoord = aTexCoord;
}
