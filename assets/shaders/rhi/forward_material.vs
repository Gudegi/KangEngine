#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec4 aInstanceTransform0;
layout(location = 4) in vec4 aInstanceTransform1;
layout(location = 5) in vec4 aInstanceTransform2;
layout(location = 6) in vec4 aInstanceTransform3;
layout(location = 7) in vec4 aInstanceColor;
layout(location = 10) in vec4 aTangent;

layout(std140) uniform ke_g0_b0 {
    mat4 view;
    mat4 projection;
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
    mat4 model = mat4(aInstanceTransform0, aInstanceTransform1,
                      aInstanceTransform2, aInstanceTransform3);
    vec4 worldPos = model * vec4(aPos, 1.0);
    gl_Position = projection * view * worldPos;
    FragPos = vec3(view * worldPos);
    Normal = mat3(view * model) * aNormal;
    WorldPos = worldPos.xyz;
    WorldNormal = normalize(mat3(model) * aNormal);
    Tangent = mat3(view * model) * aTangent.xyz;
    TangentHandedness = aTangent.w;
    TexCoord = aTexCoord;
    vColor = aInstanceColor;
}
