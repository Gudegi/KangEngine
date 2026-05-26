#version 410 core
in vec4 vColor;
out vec4 FragColor;

uniform bool uRoundPoint = false;

void main() {
    if (uRoundPoint) {
        vec2 centered = gl_PointCoord * 2.0 - 1.0;
        if (dot(centered, centered) > 1.0)
            discard;
    }
    FragColor = vColor;
}
