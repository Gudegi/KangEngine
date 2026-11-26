// Hash all four words of an OVRTX geometry-instance ID into a display color.
[[vk::binding(0, 1)]] Texture2D<uint4> source;
[[vk::binding(1, 1)]] RWTexture2D<float4> result;

[shader("compute")]
[numthreads(16, 16, 1)]
void display(uint3 tid : SV_DispatchThreadID) {
    uint width, height;
    source.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
        return;
    uint4 words = source.Load(int3(tid.xy, 0));
    uint id = 0;
    for (uint i = 0; i < 4; ++i)
        id = (id ^ words[i]) * 1664525u + 1013904223u;
    id ^= id >> 16u;
    result[tid.xy] = all(words == 0) ? float4(0, 0, 0, 1)
        : float4(float3(uint3(id, id >> 8u, id >> 16u) & 255u) / 255, 1);
}
