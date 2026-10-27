#version 450
#extension GL_EXT_shader_image_load_formatted : require

layout(
local_size_x = 8,
local_size_y = 8,
local_size_z = 1
) in;

// Declared without a format: the image view's format decides the component order and depth.
layout(binding = 0) uniform image2D dst;
layout(binding = 1) uniform sampler2D tex;
layout(push_constant, std140) uniform PushConstants {
    // Maps a framebuffer pixel position to texture coordinates.
    vec4 uvX;
    vec4 uvY;
    uint reserved;
    float alpha;
    // Bit 0: DebugFlags::TINT. Bit 31: the texture has no alpha channel.
    uint flags;

    uint damageSize;
    ivec4 damage[4];
} params;

const uint DEBUG_TINT = 1u;
const uint OPAQUE = 0x80000000u;

// Custom texture programs (`VulkanTexProgram`) are copies of this shader that replace this
// function. They may add uniforms at the end of the push constant block above, starting at
// byte offset `VulkanRenderer::CUSTOM_UNIFORMS_OFFSET`. `color` is the premultiplied sample;
// the result is multiplied by the element's alpha and blended over the target.
vec4 process(vec4 color) {
    return color;
}

void main() {
    uvec2 coord = uvec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);

    // Sample at the pixel centre. The damage rectangles are already clipped to the element.
    vec3 pos = vec3(vec2(coord) + 0.5, 1.0);
    vec2 uv = vec2(dot(params.uvX.xyz, pos), dot(params.uvY.xyz, pos));

    for (int i = 0; i < params.damageSize; i++) {
        ivec4 rect = params.damage[i];

        if (coord.x >= rect.x && coord.x < (rect.x + rect.z) &&
            coord.y >= rect.y && coord.y < (rect.y + rect.w))
        {
            // Compute shaders have no derivatives: sample LOD 0, the renderer selects the filter.
            vec4 sampled = textureLod(tex, uv, 0.0);
            if ((params.flags & OPAQUE) != 0) {
                // The padding bits of formats such as XRGB8888 are undefined.
                sampled.a = 1.0;
            }
            vec4 color = process(sampled) * params.alpha;
            if (color.a < 1.0) {
                // Premultiplied "over": src + dst * (1 - src.a).
                color = color + imageLoad(dst, ivec2(coord)) * (1.0 - color.a);
            }
            if ((params.flags & DEBUG_TINT) != 0) {
                color = vec4(0.0, 0.2, 0.0, 0.2) + color * 0.8;
            }
            imageStore(dst, ivec2(coord), color);
            break;
        }
    }
}
