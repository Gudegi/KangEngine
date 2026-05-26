#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in float aSize;

layout(std140) uniform cameraUBO {
    mat4 view;
    mat4 projection;
};

out vec4 vColor;

void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
    gl_PointSize = max(aSize, 1.0);
    vColor = aColor;
}
