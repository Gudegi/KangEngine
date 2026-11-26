// Composite debug points, lines, and text over OVRTX color and depth.
[[vk::binding(0, 1)]] StructuredBuffer<float4> primitives;
[[vk::binding(1, 1)]] StructuredBuffer<int> offsets;
[[vk::binding(2, 1)]] StructuredBuffer<int> ids;
[[vk::binding(3, 1)]] Texture2D<float4> image;
[[vk::binding(4, 1)]] Texture2D<float> depth;
[[vk::binding(5, 1)]] RWTexture2D<float4> result;
[[vk::binding(6, 1)]] StructuredBuffer<float4> glyphs;
[[vk::binding(7, 1)]] StructuredBuffer<uint> font;
[[vk::binding(8, 1)]] StructuredBuffer<float4> viewSettings;

// Unpack one R8 texel after the atlas width and height words.
float fontTexel(int2 position) {
    position = clamp(position, int2(0), int2(font[0], font[1]) - 1);
    const uint index = position.y * font[0] + position.x;
    return float((font[2 + index / 4] >> ((index % 4) * 8)) & 255u) / 255;
}

// Bilinear atlas sampling smooths glyph edges at the requested pixel size.
float fontCoverage(float2 uv) {
    const float2 p = uv * float2(font[0], font[1]) - 0.5;
    const int2 base = int2(floor(p));
    const float2 t = frac(p);
    return lerp(lerp(fontTexel(base), fontTexel(base + int2(1, 0)), t.x),
                lerp(fontTexel(base + int2(0, 1)), fontTexel(base + int2(1, 1)), t.x), t.y);
}

// Convert output color to linear RGB for blending.
float decodeSrgb(float x) {
    return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

// Convert blended linear RGB back to the output color space.
float encodeSrgb(float x) {
    x = saturate(x);
    return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1 / 2.4) - 0.055;
}

// Each thread composites the debug entries covering one output pixel.
[shader("compute")]
[numthreads(16, 16, 1)]
void debugDraw(uint3 tid : SV_DispatchThreadID) {
    // Edge workgroups can extend beyond the output image.
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
        return;

    // Start with OVRTX color and camera-space depth.
    float4 pixel = image.Load(int3(tid.xy, 0));
    const int tile = (tid.y / 16) * ((width + 15) / 16) + tid.x / 16;
    float nearest = depth.Load(int3(tid.xy, 0));
    // Display camera-forward meters; missing surfaces stay black.
    const uint viewMode = uint(viewSettings[0].x);
    if (viewMode == 1) {
        const float value = isfinite(nearest) && nearest > 0
            ? saturate(nearest / viewSettings[0].y) : 0;
        pixel = float4(value, value, value, 1);
    }
    bool changed = false;
    float3 linear = 0;

    // Visit only entries assigned to this 16x16 tile, in draw order.
    for (int j = offsets[tile]; j < offsets[tile + 1]; ++j) {
        const int id = ids[j];
        float4 color;

        // Negative IDs select glyphs; text tests depth without writing it.
        if (id < 0) {
            const int i = (-id - 1) * 4;
            const float4 rect = glyphs[i], uv = glyphs[i + 1], params = glyphs[i + 3];
            const float2 p = (float2(tid.xy) + 0.5 - rect.xy) / rect.zw;
            if (any(p < 0) || any(p >= 1) || (params.y == 0 && params.x >= nearest))
                continue;

            // Atlas coverage controls glyph opacity inside its rectangle.
            const float coverage = fontCoverage(lerp(uv.xy, uv.zw, p));
            if (coverage <= 0.001)
                continue;
            color = glyphs[i + 2];
            color.a *= coverage;
        } else {
            // Nonnegative IDs select lines or points with a pixel-space radius.
            const int i = id * 3;
            const float4 a = primitives[i], b = primitives[i + 1];
            const float2 delta = b.xy - a.xy;
            const float denominator = dot(delta, delta);
            const float t = denominator > 0
                                ? saturate(dot(float2(tid.xy) + 0.5 - a.xy, delta) / denominator)
                                : 0;
            const float2 p = float2(tid.xy) + 0.5 - lerp(a.xy, b.xy, t);

            // Interpolate reciprocal depth for perspective-correct occlusion.
            const float z = 1 / lerp(a.z, b.z, t);
            if (dot(p, p) > a.w * a.w || (b.w == 0 && z >= nearest))
                continue;
            color = primitives[i + 2];
            if (b.w == 0)
                nearest = z;
        }

        // Decode once, then alpha-blend all surviving entries in linear RGB.
        if (!changed) {
            linear = float3(decodeSrgb(pixel.r), decodeSrgb(pixel.g), decodeSrgb(pixel.b));
            changed = true;
        }
        linear = lerp(linear, color.rgb, saturate(color.a));
    }

    // Preserve untouched pixels and encode only pixels that were composited.
    if (changed)
        pixel.rgb = float3(encodeSrgb(linear.r), encodeSrgb(linear.g), encodeSrgb(linear.b));
    result[tid.xy] = pixel;
}
