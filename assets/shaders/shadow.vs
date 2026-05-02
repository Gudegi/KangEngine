#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 3) in vec4 aInstanceTransform0;
layout(location = 4) in vec4 aInstanceTransform1;
layout(location = 5) in vec4 aInstanceTransform2;
layout(location = 6) in vec4 aInstanceTransform3;

layout(std140) uniform shadowUBO {
    mat4 lightSpaceMatrix;
    vec4 shadowParams;
    vec4 shadowInfo; // x: shadow_extents, yzw: unused
};

void main() {
    mat4 model = mat4(aInstanceTransform0, aInstanceTransform1,
                      aInstanceTransform2, aInstanceTransform3);
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}
