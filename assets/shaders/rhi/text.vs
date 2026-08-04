#version 410 core
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aUv;
layout(location = 2) in vec3 iOrigin;
layout(location = 3) in vec2 iOffset;
layout(location = 4) in vec2 iSize;
layout(location = 5) in vec4 iUvRect;
layout(location = 6) in vec4 iColor;

layout(std140) uniform ke_g0_b0 {
    mat4 view;
    mat4 projection;
};
layout(std140) uniform ke_g1_b0 {
    vec4 textPassParams;
};

out vec2 vUv;
out vec4 vColor;

void main() {
    vec2 viewport = textPassParams.xy;
    if (textPassParams.z > 0.5) {
        vec2 pixel = iOrigin.xy + iOffset + aPosition * iSize;
        gl_Position = vec4(pixel.x * 2.0 / viewport.x - 1.0,
                           1.0 - pixel.y * 2.0 / viewport.y, 0.0, 1.0);
        vUv = vec2(mix(iUvRect.x, iUvRect.z, aUv.x),
                   mix(iUvRect.w, iUvRect.y, aUv.y));
    } else {
        vec4 clipOrigin = projection * view * vec4(iOrigin, 1.0);
        vec2 clipOffset = (iOffset + aPosition * iSize) * 2.0 /
                          viewport * clipOrigin.w;
        gl_Position = clipOrigin + vec4(clipOffset, 0.0, 0.0);
        vUv = mix(iUvRect.xy, iUvRect.zw, aUv);
    }
    vColor = iColor;
}
