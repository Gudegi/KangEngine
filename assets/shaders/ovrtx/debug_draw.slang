// Composite world-space debug points and lines over OVRTX color and depth.
[[vk::binding(0, 1)]] StructuredBuffer<float4> primitives;
[[vk::binding(1, 1)]] StructuredBuffer<int> offsets;
[[vk::binding(2, 1)]] StructuredBuffer<int> ids;
[[vk::binding(3, 1)]] Texture2D<float4> image;
[[vk::binding(4, 1)]] Texture2D<float> depth;
[[vk::binding(5, 1)]] RWTexture2D<float4> result;

float decodeSrgb(float x) {
    return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

float encodeSrgb(float x) {
    x = saturate(x);
    return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1 / 2.4) - 0.055;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void debugDraw(uint3 tid : SV_DispatchThreadID) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
        return;

    float4 pixel = image.Load(int3(tid.xy, 0));
    const int tile = (tid.y / 16) * ((width + 15) / 16) + tid.x / 16;
    float nearest = depth.Load(int3(tid.xy, 0));
    bool changed = false;
    float3 linear = 0;

    for (int j = offsets[tile]; j < offsets[tile + 1]; ++j) {
        const int i = ids[j] * 3;
        const float4 a = primitives[i], b = primitives[i + 1];
        const float2 delta = b.xy - a.xy;
        const float denominator = dot(delta, delta);
        const float t = denominator > 0
                            ? saturate(dot(float2(tid.xy) + 0.5 - a.xy, delta) / denominator)
                            : 0;
        const float2 p = float2(tid.xy) + 0.5 - lerp(a.xy, b.xy, t);
        const float z = 1 / lerp(a.z, b.z, t);
        if (dot(p, p) > a.w * a.w || (b.w == 0 && z >= nearest))
            continue;

        if (!changed) {
            linear = float3(decodeSrgb(pixel.r), decodeSrgb(pixel.g), decodeSrgb(pixel.b));
            changed = true;
        }
        const float4 color = primitives[i + 2];
        linear = lerp(linear, color.rgb, saturate(color.a));
        if (b.w == 0)
            nearest = z;
    }

    if (changed)
        pixel.rgb = float3(encodeSrgb(linear.r), encodeSrgb(linear.g), encodeSrgb(linear.b));
    result[tid.xy] = pixel;
}
