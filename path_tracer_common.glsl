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

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D outputImage;
layout(set = 0, binding = 2) uniform SceneData
{
    vec4 skyBetaRayleighBetaM;
    vec4 skyMieEarthAtmosScaleHr;
    vec4 skyScaleHmSunRadiusAa;
    vec4 skySunRadiance;
    vec4 skySunDirection;
    uvec4 skySampleCounts;
    // Vector radiative transfer: x = Rayleigh depolarization, y unused,
    // z = Mie table angle bins, w = ozone layer Gaussian width (m).
    vec4 skyVrtParams;
    // Ozone Chappuis-band absorption: xyz = peak absorption coefficient per
    // RGB band (1/m), w = layer center altitude (m).
    vec4 skyOzoneBeta;
    // xyz = sun limb-darkening coefficient per RGB band, w = atmospheric
    // refraction strength (1 = standard atmosphere, 0 = off).
    vec4 skySunLimbRefraction;
    // Aerosol extras: x = stratospheric background peak extinction (1/m),
    // y = background layer center altitude (m), z = layer Gaussian width (m),
    // w = aerosol single-scattering albedo.
    vec4 skyMieBackground;
} sceneData;

// Precomputed Lorenz–Mie scattering matrix, baked on the CPU. Each entry is
// (F11, F12, F33, F34) at one scattering angle; entries are stored band-major
// (band * angleBins + bin), bands = R,G,B, bin i ↦ theta = i/(bins-1)·π.
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
    vec4 skyBottomExposure;
    vec4 skyTopAspect;
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
    return normalize(sceneData.skySunDirection.xyz);
}

// ACES filmic tone mapping (Narkowicz 2015 fit), then sRGB gamma encode.
vec3 ToneMap(vec3 colour)
{
    colour = max(colour, vec3(0.0));
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    vec3 mapped = clamp((colour * (a * colour + b)) / (colour * (c * colour + d) + e), 0.0, 1.0);
    return pow(mapped, vec3(1.0 / 2.2));
}

#endif
