// Convert OVRTX world-space surface normals into viewport colors.
[[vk::binding(0, 1)]] Texture2D<float4> source;
[[vk::binding(1, 1)]] RWTexture2D<float4> result;

[shader("compute")]
[numthreads(16, 16, 1)]
void display(uint3 tid : SV_DispatchThreadID) {
    uint width, height;
    source.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
        return;
    float3 normal = source.Load(int3(tid.xy, 0)).xyz;
    result[tid.xy] = dot(normal, normal) > 0
        ? float4(normal * 0.5 + 0.5, 1) : float4(0, 0, 0, 1);
}
