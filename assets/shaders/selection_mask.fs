#version 410 core
out vec4 FragColor;

in vec2 TexCoord;
uniform sampler2D uTexture;
uniform int uAlphaMode;
uniform float uAlphaCutoff;
uniform int uAlphaTextureRedChannel;

void main() {
    vec4 alphaTexel = texture(uTexture, TexCoord);
    float alpha = uAlphaTextureRedChannel != 0 ? alphaTexel.r : alphaTexel.a;
    if(uAlphaMode == 1 && alpha < uAlphaCutoff)
        discard;
    FragColor = vec4(1.0);
}
