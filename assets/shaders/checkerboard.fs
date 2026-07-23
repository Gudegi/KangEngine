#version 410 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
in vec3 WorldPos;
in vec3 WorldNormal;
in vec2 TexCoord;
in vec4 vColor;

uniform vec4 checkerColor1;
uniform vec4 checkerColor2;
uniform bool uShowGrid;
uniform vec4 gridColor;

layout(std140) uniform lightUBO {
    vec4 lightDir;
    vec4 lightColor;
    vec4 ambient;
};
layout(std140) uniform shadowUBO {
    mat4 lightSpaceMatrices[4];
    vec4 cascadeSplits;
    vec4 cascadeOrthoHalfSizes;
    vec4 cascadeMapSizes;
    vec4 shadowParams;
    vec4 shadowInfo; // x: PCF samples, y: cascade count, z: use CSM
};

uniform sampler2D shadowMap0;
uniform sampler2D shadowMap1;
uniform sampler2D shadowMap2;
uniform sampler2D shadowMap3;
uniform int debugCsmCascadeTint;

#import "shadow_common.glsl"

float checker(vec2 uv) {
    return mod(floor(uv.x) + floor(uv.y), 2.0);
}

float checkerAA(vec2 uv) {
    vec2 fw = fwidth(uv);
    vec2 grid = abs(fract(uv - 0.5) - 0.5) / fw;
    vec2 line = clamp(grid, 0.0, 1.0);
    float rawChecker = mod(floor(uv.x) + floor(uv.y), 2.0);
    // blend
    return mix(0.5, rawChecker, min(line.x, line.y));
}

float getGrid(vec2 uv, float spacing, float thickness) {
    vec2 localCoord = fract(uv / spacing); // map to [0.0, 1.0] per spacing
    vec2 dist = min(localCoord, 1.0 - localCoord); // find the closest distance to grid line
    // if r < thickness : 0.0 else 1.0
    vec2 lines = step(dist, vec2(thickness * 0.5));
    return max(lines.x, lines.y);
}

float getGridAA(vec2 uv, float spacing, float thickness) {
    vec2 coord = uv / spacing;
    vec2 grid = abs(fract(coord - 0.5) - 0.5);
    vec2 dcoord = fwidth(coord);
    vec2 line = smoothstep(thickness + dcoord, thickness - dcoord, grid);
    
    // X, Y선 통합 및 가시성 알파 보정
    return clamp(line.x + line.y, 0.0, 1.0);
}

void main() {
    float t = checkerAA(TexCoord);
    vec4 col = mix(checkerColor1, checkerColor2, t);
    
    vec3 N = normalize(Normal);
    vec3 L = normalize(lightDir.xyz);
    float diff = max(dot(N, L), 0.0);
    float shadow = ShadowCalculation();
    vec3 light = ambient.rgb + (1.0 - shadow) * diff * lightColor.rgb;

    vec4 finalCheckerColor = vec4(col.rgb * light * ShadowCascadeTint(), col.a);

    if (uShowGrid) {
        vec4 finalGridColor = vec4(gridColor.rgb, gridColor.a);
        // thikness line per 1.0M 
        float gridMask = getGridAA(TexCoord, 1.0, 0.005);
        FragColor = mix(finalCheckerColor, finalGridColor, gridMask);
    }
    else {
        FragColor = finalCheckerColor;
    }
}