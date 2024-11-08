#version 410 core

out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D texture1, texture2;
uniform vec4 color;

void main() {
   //FragColor = vertexColor;
   //FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
   //FragColor = color;
   //FragColor = texture(ourTexture, TexCoord) * vec4(ourColor, 1.0);
   FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), 0.2);
}