// references : https://github.com/orangeduck/GenoViewPython/blob/main/resources/basic.fs
#version 410 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
in vec3 WorldPos;
in vec3 WorldNormal;
in vec2 TexCoord;
in vec4 vColor;

layout(std140) uniform lightUBO {
    vec4 lightDir;
    vec4 lightColor;
    vec4 ambient;
};
layout(std140) uniform shadowUBO {
    mat4 lightSpaceMatrix;
    vec4 shadowParams;
    vec4 shadowInfo;
};

uniform sampler2D shadowMap;

#import "shadow_common.glsl"

float grid(vec2 uv, float lineWidth) {
    vec4 uvDDXY = vec4(dFdx(uv), dFdy(uv));
    vec2 uvDeriv = vec2(length(uvDDXY.xz), length(uvDDXY.yw));
    vec2 drawWidth = clamp(vec2(lineWidth), uvDeriv, vec2(0.5));
    vec2 lineAA = uvDeriv * 1.5;
    vec2 gridUV = 1.0 - abs(fract(uv) * 2.0 - 1.0);
    vec2 lines = 1.0 - smoothstep(drawWidth - lineAA, drawWidth + lineAA, gridUV);
    lines *= clamp(lineWidth / drawWidth, 0.0, 1.0);
    return max(lines.x, lines.y);
}

float checker(vec2 uv) {
    vec4 uvDDXY = vec4(dFdx(uv), dFdy(uv));
    vec2 w = vec2(length(uvDDXY.xz), length(uvDDXY.yw));
    vec2 i = 2.0 * (abs(fract((uv - 0.5 * w) * 0.5) - 0.5) -
        abs(fract((uv + 0.5 * w) * 0.5) - 0.5)) / w;
    return 0.5 - 0.5 * i.x * i.y;
}

void main() {
    float fine = grid(TexCoord * 200.0, 0.025);
    float coarse = grid(TexCoord * 20.0, 0.02);
    float check = checker(TexCoord * 20.0);

    float pattern = mix(0.82, 1.08, check);
    pattern = mix(pattern, 0.74, fine);
    pattern = mix(pattern, 1.12, coarse);
    pattern = clamp(pattern, 0.72, 1.15);

    vec3 albedo = clamp(vColor.rgb * pattern, 0.0, 1.0);
    vec3 N = normalize(Normal);
    if(!gl_FrontFacing)
        N = -N;
    vec3 L = normalize(lightDir.xyz);
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor.rgb * albedo;
    vec3 V = normalize(-FragPos);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32) * 0.18;
    vec3 specular = spec * lightColor.rgb;

    float shadow = ShadowCalculation();
    float debugShadow = shadow * 0.6;
    vec3 lit = ambient.rgb * albedo + (1.0 - debugShadow) * (diffuse + specular);
    FragColor = vec4(lit, vColor.a);
}
