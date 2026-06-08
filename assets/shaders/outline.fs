#version 410 core
out vec4 FragColor;

uniform vec4 outlineColor;

void main() {
    FragColor = outlineColor;
}
