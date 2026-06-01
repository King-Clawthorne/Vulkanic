// path_tracer_common.glsl — shared shader header.
//
// Pulled in by every ray-tracing stage (rgen, rmiss, rchit, shadow.rmiss)
// and by sky.comp via #include. Defines:
//   * The descriptor-set bindings (output image, TLAS, scene UBO, the
//     instance/material/vertex/index SSBOs).
//   * The push-constant block carrying camera + per-frame state.
//   * Common math helpers, the RNG (Wang hash + LCG), the GGX/Fresnel
//     utilities used by the closest-hit shader, and the sun cone-sampling
//     primitives used for direct lighting.
//   * The RayPayload / ShadowPayload structs that flow through traceRayEXT.
// The bindings here MUST match the descriptor-set layout built in
// VulkanPathTracer::CreateDescriptorSetLayout().

#ifndef PATH_TRACER_COMMON_INCLUDED
#define PATH_TRACER_COMMON_INCLUDED

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D outputImage;
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 2) uniform SceneData
{
    vec4 skyBetaRayleighBetaM;
    vec4 skyMieEarthAtmosScaleHr;
    vec4 skyScaleHmSunRadiusAa;
    vec4 skySunRadiance;
    vec4 skySunDirection;
    uvec4 skySampleCounts;
    // Vector radiative transfer: x = Rayleigh depolarization, y = scattering
    // orders, z = Mie table angle bins, w unused.
    vec4 skyVrtParams;
} sceneData;

struct InstanceData
{
    uvec4 materialIndexFirstIndex;
};

struct MaterialData
{
    vec4 albedo;
    vec4 emission;
    vec4 eta;
    vec4 extinction;
};

struct VertexData
{
    vec4 position;
    vec4 normal;
};

layout(std430, set = 0, binding = 3) readonly buffer InstanceBuffer
{
    InstanceData instances[];
} instanceBuffer;

layout(std430, set = 0, binding = 4) readonly buffer MaterialBuffer
{
    MaterialData materials[];
} materialBuffer;

layout(std430, set = 0, binding = 5) readonly buffer VertexBuffer
{
    VertexData vertices[];
} vertexBuffer;

layout(std430, set = 0, binding = 6) readonly buffer IndexBuffer
{
    uint indices[];
} indexBuffer;

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

// Unpacked, shader-friendly material. Built from the std430-packed
// MaterialData by GetInstanceMaterial().
struct Material
{
    vec3 albedo;
    vec3 emission;
    vec3 eta;
    vec3 extinction;
};

// Path-tracing payload sent to traceRayEXT. radiance accumulates light,
// throughput tracks the running BSDF/PDF product, state.x carries the
// RNG seed, state.y carries the current bounce index.
struct RayPayload
{
    vec4 radiance;
    vec4 throughput;
    uvec4 state;
};

// Lightweight payload for shadow rays — set to 1 by the shadow miss
// shader if the path was unoccluded.
struct ShadowPayload
{
    uint visible;
};

// Wang hash — used to seed per-pixel RNG state from a tile-friendly
// integer so neighbouring pixels diverge after one mix.
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

// Numerical-Recipes LCG. Cheap and good enough for Monte Carlo
// integration when seeded by Hash() per pixel.
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

// Evaluate the analytic sky model along a ray direction. Used by the
// primary miss shader for image-based lighting on indirect bounces (intensity
// only, single scattering for speed).
vec3 SampleSky(vec3 direction, inout uint rngState)
{
    RNG rng;
    rng.state = rngState;
    vec3 sky = render_sky_pixel(direction, GetSunDirection(), rng);
    rngState = rng.state;
    return sky * pc.skyBottomExposure.w;
}

// Full polarized vector-radiative-transfer sky for the camera-visible ray.
// Returns the exposure-scaled radiance and, via out parameters: the degree and
// camera-plane axis angle of LINEAR polarization, plus the signed degree of
// CIRCULAR polarization (Stokes V/I; sign = handedness). Circular polarization
// is purely a multiple-scattering, Mie-driven effect here.
vec3 SampleSkyStokes(vec3 direction, inout uint rngState,
                     out float degree, out float axisAngle, out float circularDegree)
{
    RNG rng;
    rng.state = rngState;
    vec3 sumI = vec3(0.0);
    vec3 sumQ = vec3(0.0);
    vec3 sumU = vec3(0.0);
    vec3 sumV = vec3(0.0);
    int n = max(1, SkySamples());
    for (int i = 0; i < n; ++i)
    {
        Stokes s = render_sky_stokes(direction, GetSunDirection(), SkyScatteringOrders(), rng);
        sumI += s.I;
        sumQ += s.Q;
        sumU += s.U;
        sumV += s.V;
    }
    float inv = 1.0 / float(n);
    sumI *= inv;
    sumQ *= inv;
    sumU *= inv;
    sumV *= inv;
    rngState = rng.state;

    const vec3 luma = vec3(0.2126, 0.7152, 0.0722);
    float intensity = dot(sumI, luma);
    float q = dot(sumQ, luma);
    float u = dot(sumU, luma);
    float v = dot(sumV, luma);
    degree = clamp(length(vec2(q, u)) / max(intensity, 1.0e-6), 0.0, 1.0);
    axisAngle = 0.5 * atan(u, q);
    circularDegree = clamp(v / max(intensity, 1.0e-6), -1.0, 1.0);
    return sumI * pc.skyBottomExposure.w;
}

// Solid angle of the sun cone — the area of a spherical cap with the
// configured sun radius. Used to weight direct sun samples.
float SunSolidAngle()
{
    return 2.0 * PI * (1.0 - cos(SkySunRadius()));
}

// Sun radiance attenuated by the atmosphere along the sun direction —
// "what reaches the ground" in the sky model. Reused by every direct
// lighting sample.
vec3 SunIncidentRadiance()
{
    vec3 origin = vec3(0.0, SkyEarthRadius() + 1.0, 0.0);
    vec3 T = transmittance(origin, GetSunDirection(), 16);
    return SkySunRadiance() * T * pc.skyBottomExposure.w;
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

float MaxComponent(vec3 value)
{
    return max(value.r, max(value.g, value.b));
}

// Conductor Fresnel split into its s- and p-polarized reflectance
// components — supports complex IOR (eta + i·extinction) so the same code
// path covers both dielectrics (extinction = 0) and metals. The camera
// polarization filter needs the two components separately; the scalar
// FresnelReflectance() below just averages them.
void FresnelReflectanceSP(float cosThetaI, vec3 eta, vec3 extinction, out vec3 rs, out vec3 rp)
{
    cosThetaI = clamp(cosThetaI, 0.0, 1.0);
    vec3 eta2 = eta * eta;
    vec3 extinction2 = extinction * extinction;
    float cosThetaI2 = cosThetaI * cosThetaI;
    vec3 twoEtaCosTheta = 2.0 * eta * cosThetaI;

    vec3 rsNumerator = eta2 + extinction2 - twoEtaCosTheta + vec3(cosThetaI2);
    vec3 rsDenominator = eta2 + extinction2 + twoEtaCosTheta + vec3(cosThetaI2);
    vec3 rpNumerator = (eta2 + extinction2) * cosThetaI2 - twoEtaCosTheta + vec3(1.0);
    vec3 rpDenominator = (eta2 + extinction2) * cosThetaI2 + twoEtaCosTheta + vec3(1.0);

    rs = clamp(rsNumerator / max(rsDenominator, vec3(1.0e-6)), vec3(0.0), vec3(1.0));
    rp = clamp(rpNumerator / max(rpDenominator, vec3(1.0e-6)), vec3(0.0), vec3(1.0));
}

vec3 FresnelReflectance(float cosThetaI, vec3 eta, vec3 extinction)
{
    vec3 rs;
    vec3 rp;
    FresnelReflectanceSP(cosThetaI, eta, extinction, rs, rp);
    return 0.5 * (rs + rp);
}

// Estimate the linear-polarization state imparted by a reflection off this
// surface and stash it in the payload so the ray-generation shader can
// apply the camera's polarization filter (Malus's law). Only meaningful for
// the first, camera-visible hit. We model the dominant effect: an unpolarized
// incident beam reflecting off a surface becomes partially polarized
// perpendicular to the plane of incidence, with the degree set by the gap
// between the s- and p-Fresnel terms (maximal near Brewster's angle).
//   payload.throughput.w := degree of linear polarization (0..1)
//   payload.radiance.w   := polarization axis angle in the camera image plane
void RecordPolarization(inout RayPayload p, vec3 incomingDir, vec3 normal, Material material)
{
    // The s-polarization axis is perpendicular to the plane of incidence.
    vec3 sAxis = cross(incomingDir, normal);
    float sLen = length(sAxis);
    if (sLen < 1.0e-5)
    {
        return; // Near-normal incidence: no preferred polarization axis.
    }
    sAxis /= sLen;

    float cosThetaI = clamp(dot(-incomingDir, normal), 0.0, 1.0);
    vec3 rs;
    vec3 rp;
    FresnelReflectanceSP(cosThetaI, material.eta, material.extinction, rs, rp);

    // Collapse to luminance-weighted scalars for the degree of polarization.
    const vec3 luma = vec3(0.2126, 0.7152, 0.0722);
    float rsScalar = dot(rs, luma);
    float rpScalar = dot(rp, luma);
    float degree = (rsScalar - rpScalar) / max(rsScalar + rpScalar, 1.0e-4);

    // Orientation of the polarization axis projected into the camera image
    // plane, measured from the camera right axis (matches the filter angle).
    float axisX = dot(sAxis, pc.cameraRightBounces.xyz);
    float axisY = dot(sAxis, pc.cameraUpTanHalfFovY.xyz);

    p.throughput.w = clamp(degree, 0.0, 1.0);
    p.radiance.w = atan(axisY, axisX);
}

Material GetInstanceMaterial(InstanceData instanceData)
{
    Material material;
    MaterialData materialData = materialBuffer.materials[instanceData.materialIndexFirstIndex.x];
    material.albedo = materialData.albedo.xyz;
    material.emission = materialData.emission.xyz;
    material.eta = materialData.eta.xyz;
    material.extinction = materialData.extinction.xyz;
    return material;
}

// Reconstruct an object-space shading normal at a hit by interpolating
// the three vertex normals with barycentrics. Falls back to the face
// normal if interpolated normals are degenerate.
vec3 GetTriangleObjectNormal(InstanceData instanceData, uint primitiveId, vec2 barycentrics)
{
    uint firstIndex = instanceData.materialIndexFirstIndex.y + primitiveId * 3u;
    VertexData v0 = vertexBuffer.vertices[indexBuffer.indices[firstIndex + 0u]];
    VertexData v1 = vertexBuffer.vertices[indexBuffer.indices[firstIndex + 1u]];
    VertexData v2 = vertexBuffer.vertices[indexBuffer.indices[firstIndex + 2u]];

    vec3 bary = vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    vec3 interpolatedNormal = v0.normal.xyz * bary.x + v1.normal.xyz * bary.y + v2.normal.xyz * bary.z;
    if (length(interpolatedNormal) > 1.0e-6)
    {
        return normalize(interpolatedNormal);
    }

    vec3 edge0 = v1.position.xyz - v0.position.xyz;
    vec3 edge1 = v2.position.xyz - v0.position.xyz;
    return normalize(cross(edge0, edge1));
}

// Object-to-world normal transform via the transpose of the inverse
// linear part — the standard rule for normals under non-uniform scale.
vec3 TransformNormalToWorld(vec3 objectNormal, mat3 worldToObjectLinear)
{
    return normalize(transpose(worldToObjectLinear) * objectNormal);
}

// Ray Tracing Gems: A Fast and Robust Method for Avoiding Self-Intersection
vec3 OffsetRay(const vec3 p, const vec3 n)
{
    const float origin = 1.0 / 32.0;
    const float float_scale = 1.0 / 65536.0;
    const float int_scale = 256.0;

    ivec3 of_i = ivec3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    vec3 p_i = vec3(
        intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0.0) ? -of_i.x : of_i.x)),
        intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0.0) ? -of_i.y : of_i.y)),
        intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0.0) ? -of_i.z : of_i.z)));

    return vec3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

#endif
