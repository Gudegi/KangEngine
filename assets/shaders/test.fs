#version 410 core

out vec4 FragColor;
uniform vec4 color;
//in vec4 vertexColor;

void main() {
   //FragColor = vertexColor;
   //FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
   FragColor = color;
}