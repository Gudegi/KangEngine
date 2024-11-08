#version 410 core

out vec4 FragColor;
uniform vec4 color;
//in vec4 vertexColor;
in vec3 ourColor;
in vec2 TexCoord;

uniform sampler2D ourTexture;

void main() {
   //FragColor = vertexColor;
   //FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
   //FragColor = color;
   FragColor = texture(ourTexture, TexCoord) * vec4(ourColor, 1.0);
}