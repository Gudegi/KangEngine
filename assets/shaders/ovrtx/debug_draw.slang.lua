-- SPG owns the debug buffers, output image, and Vulkan scheduling.
function debugDraw(inputs, outputs)
    local image, depth = inputs["Image"], inputs["Depth"]
    assert(image.dtype == slang.uchar4 and depth.dtype == slang.float)
    assert(image.shape[1] == depth.shape[1] and image.shape[2] == depth.shape[2])
    outputs["Color"] = slang.image(image.shape, image.dtype)
    return slang.compute({
        bind = {
            slang.StructuredBuffer(slang.array(inputs["primitives"])),
            slang.StructuredBuffer(slang.array(inputs["offsets"])),
            slang.StructuredBuffer(slang.array(inputs["ids"])),
            slang.Texture2D(image), slang.Texture2D(depth),
            slang.RWTexture2D(outputs["Color"]),
            slang.StructuredBuffer(slang.array(inputs["glyphs"])),
            slang.StructuredBuffer(slang.array(inputs["font"])),
            slang.StructuredBuffer(slang.array(inputs["viewSettings"])),
        },
    })
end
