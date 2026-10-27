#version 450
#extension GL_EXT_shader_image_load_formatted : require

layout(
local_size_x = 8,
local_size_y = 8,
local_size_z = 1
) in;

// Declared without a format: the image view's format decides the component order and depth.
layout(binding = 0) uniform image2D dst;
layout(push_constant, std140) uniform PushConstants {
    vec4 color;
    bool blend;

    uint rectSize;
    ivec4 rects[6];
} params;

void main() {
    uvec2 coord = uvec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
    uvec2 outSize = imageSize(dst);

    if (coord.x >= outSize.x || coord.y >= outSize.y)
        return;

    for (int i = 0; i < params.rectSize; i++) {
        ivec4 rect = params.rects[i];

        if (coord.x >= rect.x && coord.x < (rect.x + rect.z)
                && coord.y >= rect.y && coord.y < (rect.y + rect.w))
        {
            vec4 color = params.color;
            if (params.blend && color.a < 1.0) {
                // Premultiplied "over": src + dst * (1 - src.a).
                color = color + imageLoad(dst, ivec2(coord)) * (1.0 - color.a);
            }
            imageStore(dst, ivec2(coord), color);
            break;
        }
    }
}
