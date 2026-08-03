#version 410 core
layout(location = 0) in vec3 aPos;

layout(std140) uniform ke_g0_b0 {
    mat4 view;
    mat4 projection;
};
layout(std140) uniform ke_g1_b0 {
    vec4 skyboxParams;
};

out vec3 TexDir;

void main() {
    TexDir = skyboxParams.x > 0.5 ? vec3(aPos.x, aPos.z, -aPos.y) : aPos;
    vec4 pos = projection * mat4(mat3(view)) * vec4(aPos, 1.0);
    gl_Position = pos.xyww;
}
