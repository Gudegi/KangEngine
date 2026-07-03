#version 410 core
out vec4 FragColor;

in vec2 TexCoord;
uniform sampler2D uTexture;
uniform int uAlphaMode;
uniform float uAlphaCutoff;

void main() {
    if(uAlphaMode == 1 && texture(uTexture, TexCoord).a < uAlphaCutoff)
        discard;
    FragColor = vec4(1.0);
}
