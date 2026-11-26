-- Keep instance-color conversion and RGBA8 output on the GPU.
function display(inputs, outputs)
    local source = inputs["Source"]
    outputs["Color"] = slang.image(source.shape, slang.uchar4)
    return slang.compute({bind = {
        slang.Texture2D(source), slang.RWTexture2D(outputs["Color"]),
    }})
end
