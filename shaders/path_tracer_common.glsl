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
    // x = Rayleigh extinction at 550 nm, y = Mie extinction,
    // z = solar temperature (K), w = solar radiance at 550 nm.
    vec4 skySpectralParams;
    vec4 skyRadiiScaleHeights;
    vec4 skySunDirectionRadius;
    uvec4 skySampleCounts;
    // x = sun-disk AA width, y = Rayleigh depolarization,
    // z = Mie table angle bins, w = Lambertian ground albedo.
    vec4 skyVrtParams;
    // Local rain ellipsoid and optical/table integration parameters.
    vec4 rainbowCenterEnabled;
    vec4 rainbowRadiiEdge;
    // x/y scattering/extinction (1/m), z angle bins, w view steps.
    vec4 rainbowOptical;
} sceneData;

// Precomputed Lorenz–Mie scattering matrix, baked on the CPU. Each entry is
// (F11, F12, F33, F34) at one scattering angle; entries are stored band-major
// (band * angleBins + bin), 13 bands from 400 to 700 nm.
layout(std430, set = 0, binding = 7) readonly buffer MieMatrixBuffer
{
    vec4 entries[];
} mieMatrixBuffer;

layout(std430, set = 0, binding = 8) readonly buffer RainbowMatrixBuffer
{
    vec4 entries[];
} rainbowMatrixBuffer;

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
#include "rainbow.comp"

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

// Smooth analytic fits to the CIE 1931 2-degree colour-matching functions.
// Spectral transport remains in radiometric units; this conversion occurs
// only once, at the display boundary.
float cie_gaussian(float lambda, float mean, float leftScale, float rightScale)
{
    float scale = lambda < mean ? leftScale : rightScale;
    float x = (lambda - mean) * scale;
    return exp(-0.5 * x * x);
}

vec3 CieXyz(float lambda)
{
    float x = 1.056 * cie_gaussian(lambda, 599.8, 0.0264, 0.0323)
            + 0.362 * cie_gaussian(lambda, 442.0, 0.0624, 0.0374)
            - 0.065 * cie_gaussian(lambda, 501.1, 0.0490, 0.0382);
    float y = 0.821 * cie_gaussian(lambda, 568.8, 0.0213, 0.0247)
            + 0.286 * cie_gaussian(lambda, 530.9, 0.0613, 0.0322);
    float z = 1.217 * cie_gaussian(lambda, 437.0, 0.0845, 0.0278)
            + 0.681 * cie_gaussian(lambda, 459.0, 0.0385, 0.0725);
    return max(vec3(x, y, z), vec3(0.0));
}

vec3 XyzToLinearSrgb(vec3 xyz)
{
    return mat3( 3.2406, -0.9689,  0.0557,
                -1.5372,  1.8758, -0.2040,
                -0.4986,  0.0415,  1.0570) * xyz;
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
