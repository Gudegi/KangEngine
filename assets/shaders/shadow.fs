#version 410 core
in vec2 TexCoord;

uniform sampler2D uTexture;
uniform int uAlphaMode;
uniform float uAlphaCutoff;
uniform int uAlphaTextureRedChannel;

void main() {
    // Match the color-pass cutout so transparent atlas regions do not cast a
    // solid rectangular shadow.
    vec4 alphaTexel = texture(uTexture, TexCoord);
    float alpha = uAlphaTextureRedChannel != 0 ? alphaTexel.r : alphaTexel.a;
    if(uAlphaMode == 1 && alpha < uAlphaCutoff)
        discard;
}
