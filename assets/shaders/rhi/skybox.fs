#version 410 core
layout(location = 0) out vec4 outColor;
in vec3 TexDir;

uniform samplerCube ke_g3_b0;

void main() {
    outColor = texture(ke_g3_b0, TexDir);
}
