#version 410 core

out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D texture1, texture2;
uniform vec4 color;
uniform vec3 lightColor;

void main() {
   //FragColor = vertexColor;
   //FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
   //FragColor = color;
   //FragColor = texture(texture1, TexCoord) * vec4(color);
   //FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), 0.2); //* vec4(color);
   FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), 0.2) * vec4(color) * vec4(lightColor, 1.0f);
}