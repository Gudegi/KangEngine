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

#import "skinning_common.glsl"

layout(std140) uniform cameraUBO {
    mat4 view;
    mat4 projection;
};

out vec3 Normal;
out vec3 FragPos;
out vec3 WorldPos;
out vec3 WorldNormal;
out vec2 TexCoord;
out vec4 vColor;

void main() {
    mat4 model = mat4(aInstanceTransform0, aInstanceTransform1, aInstanceTransform2, aInstanceTransform3);
    mat4 skin = skinMatrix(aBoneIndices, aBoneWeights);
    vec4 localPos = skin * vec4(aPos, 1.0);
    vec3 localNormal = normalize(mat3(skin) * aNormal);
    vec4 worldPos4 = model * localPos;
    gl_Position = projection * view * worldPos4;
    FragPos = vec3(view * worldPos4);
    Normal = mat3(view * model) * localNormal;
    WorldPos = vec3(worldPos4);
    WorldNormal = normalize(mat3(model) * localNormal);
    TexCoord = aTexCoord;
    vColor = aInstanceColor;
}
