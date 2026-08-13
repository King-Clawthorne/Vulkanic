// path_tracer_common.glsl — shared shader header.
//
// Included by path_tracer.comp and (via #include) by sky.comp. This is a
// polarized-sky simulator with no scene geometry, so the header is small:
//   * Descriptor bindings: the output image, the scene UBO (sky parameters),
//     and the precomputed Lorenz–Mie scattering-matrix SSBO.
//   * The push-constant block (camera basis, exposure, polarization filter).
//   * The RNG (Wang hash + LCG), the sun direction, and ACES tonemapping.
// The bindings here MUST match the descriptor-set layout built in
// VulkanPathTracer::CreateDescriptorSetLayout().

#ifndef PATH_TRACER_COMMON_INCLUDED
#define PATH_TRACER_COMMON_INCLUDED

layout(set = 0, binding = 0) uniform writeonly image2D outputImage;
layout(set = 0, binding = 2) uniform SceneData
{
    vec4 skyBetaRayleighBetaM;
    vec4 skyRadiiScaleHeights;
    vec4 skySunRadiance;
    vec4 skySunDirectionRadius;
    uvec4 skySampleCounts;
    // x = sun-disk AA width, y = Rayleigh depolarization,
    // z = Mie table angle bins, w unused.
    vec4 skyVrtParams;
} sceneData;

// Precomputed Lorenz–Mie scattering matrix, baked on the CPU. Each entry is
// (F11, F12, F33, F34) at one scattering angle; entries are stored band-major
// (band * angleBins + bin), bands = R, G, B.
layout(std430, set = 0, binding = 7) readonly buffer MieMatrixBuffer
{
    vec4 entries[];
} mieMatrixBuffer;

layout(push_constant) uniform PushConstants
{
    vec4 cameraPositionFrame;
    vec4 cameraForwardSamples;
    vec4 cameraRightBounces;
    vec4 cameraUpTanHalfFovY;
    // x = exposure, y = viewport aspect ratio, z/w unused.
    vec4 displayParams;
    // Camera polarization filter: x = enabled (0/1), y = major-axis angle
    // in radians (image plane, from the camera right axis), z = ellipticity
    // angle in radians (-pi/4..pi/4; 0 = linear, +/-pi/4 = circular), w unused.
    vec4 polarizer;
    uvec2 imageSize;
} pc;

#include "sky.comp"

// Wang hash — seeds per-pixel RNG state from a tile-friendly integer so
// neighbouring pixels diverge after one mix.
uint Hash(uint x)
{
    x ^= 2747636419u;
    x *= 2654435769u;
    x ^= x >> 16;
    x *= 2654435769u;
    x ^= x >> 16;
    x *= 2654435769u;
    return x;
}

// Numerical-Recipes LCG. Cheap and good enough for Monte Carlo integration
// when seeded by Hash() per pixel.
float NextFloat(inout uint state)
{
    state = 1664525u * state + 1013904223u;
    return float(state & 0x00FFFFFFu) / 16777216.0;
}

vec2 NextFloat2(inout uint state)
{
    return vec2(NextFloat(state), NextFloat(state));
}

vec3 GetSunDirection()
{
    return normalize(sceneData.skySunDirectionRadius.xyz);
}

// Hue-preserving photographic shoulder. The curve is evaluated from the
// brightest channel and the same scale is applied to all three channels, so
// brightness changes across an anti-aliased solar edge cannot rotate its hue.
vec3 HuePreservingToneMap(vec3 colour)
{
    float peak = max(colour.r, max(colour.g, colour.b));
    if (peak <= 0.0)
    {
        return vec3(0.0);
    }
    float mappedPeak = peak / (1.0 + peak);
    return colour * (mappedPeak / peak);
}

vec3 LinearToSrgb(vec3 colour)
{
    bvec3 low = lessThanEqual(colour, vec3(0.0031308));
    vec3 linearSegment = 12.92 * colour;
    vec3 powerSegment = 1.055 * pow(max(colour, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(powerSegment, linearSegment, low);
}

vec3 ToneMap(vec3 colour)
{
    colour = max(colour, vec3(0.0));
    return clamp(LinearToSrgb(HuePreservingToneMap(colour)), 0.0, 1.0);
}

#endif
