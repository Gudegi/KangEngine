#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 3) in vec4 aInstanceTransform0;
layout(location = 4) in vec4 aInstanceTransform1;
layout(location = 5) in vec4 aInstanceTransform2;
layout(location = 6) in vec4 aInstanceTransform3;

layout(std140) uniform cameraUBO {
    mat4 view;
    mat4 projection;
};

uniform float outlineWidth;
uniform vec3 outlineCenter;

void main() {
    mat4 model = mat4(aInstanceTransform0, aInstanceTransform1,
                      aInstanceTransform2, aInstanceTransform3);
    vec3 expandedPos = outlineCenter + (aPos - outlineCenter) *
                                           (1.0 + outlineWidth);
    gl_Position = projection * view * model * vec4(expandedPos, 1.0);
}
