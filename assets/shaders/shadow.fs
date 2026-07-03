#version 410 core
in vec2 TexCoord;

uniform sampler2D uTexture;
uniform int uAlphaMode;
uniform float uAlphaCutoff;

void main() {
    // Match the color-pass cutout so transparent atlas regions do not cast a
    // solid rectangular shadow.
    if(uAlphaMode == 1 && texture(uTexture, TexCoord).a < uAlphaCutoff)
        discard;
}
