#version 410 core
layout(location = 0) out vec4 outColor;
in vec4 vColor;

void main() {
    vec2 centered = gl_PointCoord * 2.0 - 1.0;
    if (dot(centered, centered) > 1.0)
        discard;
    outColor = vColor;
}
