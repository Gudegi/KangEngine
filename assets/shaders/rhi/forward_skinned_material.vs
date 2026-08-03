#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec4 aInstanceTransform0;
layout(location = 4) in vec4 aInstanceTransform1;
layout(location = 5) in vec4 aInstanceTransform2;
layout(location = 6) in vec4 aInstanceTransform3;
layout(location = 7) in vec4 aInstanceColor;
layout(location = 8) in ivec4 aBoneIndices;
layout(location = 9) in vec4 aBoneWeights;
layout(location = 10) in vec4 aTangent;

layout(std140) uniform ke_g0_b0 {
    mat4 view;
    mat4 projection;
};
layout(std140) uniform ke_g2_b0 {
    mat4 boneMatrices[128];
};

out vec3 Normal;
out vec3 FragPos;
out vec3 WorldPos;
out vec3 WorldNormal;
out vec3 Tangent;
out float TangentHandedness;
out vec2 TexCoord;
out vec4 vColor;

void main() {
    mat4 skin = mat4(0.0);
    float weightSum = 0.0;
    if (aBoneIndices.x >= 0) {
        skin += boneMatrices[aBoneIndices.x] * aBoneWeights.x;
        weightSum += aBoneWeights.x;
    }
    if (aBoneIndices.y >= 0) {
        skin += boneMatrices[aBoneIndices.y] * aBoneWeights.y;
        weightSum += aBoneWeights.y;
    }
    if (aBoneIndices.z >= 0) {
        skin += boneMatrices[aBoneIndices.z] * aBoneWeights.z;
        weightSum += aBoneWeights.z;
    }
    if (aBoneIndices.w >= 0) {
        skin += boneMatrices[aBoneIndices.w] * aBoneWeights.w;
        weightSum += aBoneWeights.w;
    }
    if (weightSum <= 0.0)
        skin = mat4(1.0);
    mat4 model = mat4(aInstanceTransform0, aInstanceTransform1,
                      aInstanceTransform2, aInstanceTransform3);
    vec4 worldPos = model * skin * vec4(aPos, 1.0);
    vec3 localNormal = mat3(skin) * aNormal;
    vec3 localTangent = mat3(skin) * aTangent.xyz;
    gl_Position = projection * view * worldPos;
    FragPos = vec3(view * worldPos);
    Normal = mat3(view * model) * localNormal;
    WorldPos = worldPos.xyz;
    WorldNormal = normalize(mat3(model) * localNormal);
    Tangent = mat3(view * model) * localTangent;
    TangentHandedness = aTangent.w;
    TexCoord = aTexCoord;
    vColor = aInstanceColor;
}
