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
    // Ozone: xyz = per-RGB Chappuis absorption coefficient, w = layer peak altitude.
    vec4 skyOzone;
    // Ground coupling: xyz = Lambertian ground albedo, w = ozone tent half-width.
    vec4 skyGround;
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

// Path-tracing payload sent to traceRayEXT.
//
// Scalar path (polarization filter OFF):
//   radiance.xyz   accumulated radiance (Stokes I per band)
//   throughput.xyz running BSDF/PDF product
// Polarized path (filter ON): full-Stokes transport. radiance.xyz still holds
// the accumulated Stokes I; the extra fields below carry the rest:
//   mueller[0..2]  per-band (R,G,B) 4x4 Mueller throughput, camera -> vertex
//   stokesQ/U/V    accumulated Stokes Q/U/V per band (xyz = RGB)
//   frameX.xyz     current Stokes reference axis (perpendicular to the ray)
// state.x = RNG seed, state.y = current bounce index.
struct RayPayload
{
    vec4 radiance;
    vec4 throughput;
    uvec4 state;
    mat4 mueller[3];
    vec4 stokesQ;
    vec4 stokesU;
    vec4 stokesV;
    vec4 frameX;
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

// ── Complex arithmetic (vec2 = a + i·b) for amplitude (phase) Fresnel ──
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
vec2 cdiv(vec2 a, vec2 b) { float d = max(dot(b, b), 1.0e-20); return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / d; }
vec2 csqrt(vec2 z)
{
    float r = length(z);
    float re = sqrt(max(0.5 * (r + z.x), 0.0));
    float im = sqrt(max(0.5 * (r - z.x), 0.0));
    return vec2(re, z.y < 0.0 ? -im : im);
}

// Complex amplitude reflection coefficients for one band (relative index
// m = eta + i·k), incidence from vacuum at cosThetaI. Their squared magnitudes
// match FresnelReflectanceSP; their relative phase drives the elliptical
// (Mueller M34) term that linear-only models drop.
void FresnelAmplitudesBand(float cosThetaI, float eta, float k, out vec2 rs, out vec2 rp)
{
    cosThetaI = clamp(cosThetaI, 0.0, 1.0);
    vec2 m = vec2(eta, k);
    vec2 ci = vec2(cosThetaI, 0.0);
    float sin2 = max(0.0, 1.0 - cosThetaI * cosThetaI);
    vec2 ct = csqrt(vec2(1.0, 0.0) - cdiv(vec2(sin2, 0.0), cmul(m, m))); // cosThetaT
    vec2 mct = cmul(m, ct);
    vec2 mci = cmul(m, ci);
    rs = cdiv(ci - mct, ci + mct);
    rp = cdiv(mci - ct, mci + ct);
}

// Per-band specular reflection Mueller matrix, in the frame where Q is along
// the s-axis (perpendicular to the plane of incidence):
//   [[A,B,0,0],[B,A,0,0],[0,0,C,S],[0,0,-S,C]]
// A=(Rs+Rp)/2, B=(Rs-Rp)/2, C=Re(rs·conj(rp)), S=Im(rs·conj(rp)). Dielectrics
// (k=0) give S=0 (linear only); metals give S≠0 (elliptical).
void ReflectionMueller(float cosThetaI, vec3 eta, vec3 extinction, out mat4 m[3])
{
    for (int b = 0; b < 3; ++b)
    {
        vec2 rs;
        vec2 rp;
        FresnelAmplitudesBand(cosThetaI, eta[b], extinction[b], rs, rp);
        float rsMag2 = dot(rs, rs);
        float rpMag2 = dot(rp, rp);
        vec2 cross_ = cmul(rs, vec2(rp.x, -rp.y)); // rs · conj(rp)
        float A = 0.5 * (rsMag2 + rpMag2);
        float B = 0.5 * (rsMag2 - rpMag2);
        float C = cross_.x;
        float S = cross_.y;
        mat4 mm = mat4(0.0); // column-major: mm[col][row]
        mm[0][0] = A; mm[1][0] = B;
        mm[0][1] = B; mm[1][1] = A;
        mm[2][2] = C; mm[3][2] = S;
        mm[2][3] = -S; mm[3][3] = C;
        m[b] = mm;
    }
}

// Mueller rotator acting on the Q,U sub-block given cos2α, sin2α.
mat4 MuellerRotator(float c2, float s2)
{
    mat4 m = mat4(1.0);
    m[1][1] = c2; m[2][1] = s2;
    m[1][2] = -s2; m[2][2] = c2;
    return m;
}

// Mueller rotator that re-expresses a Stokes vector from transverse frame
// (fromX) into frame (toX); both are perpendicular to ray direction d.
mat4 FrameRotator(vec3 fromX, vec3 toX, vec3 d)
{
    vec3 fromY = cross(d, fromX);
    float c = dot(toX, fromX);
    float s = dot(toX, fromY);
    return MuellerRotator(c * c - s * s, 2.0 * c * s);
}

// Add a per-band contribution Stokes vector (I,Q,U,V) for band b to the
// payload's Stokes accumulator.
void AccumulateStokes(inout RayPayload p, int b, vec4 contribution)
{
    p.radiance[b] += contribution.x;
    p.stokesQ[b] += contribution.y;
    p.stokesU[b] += contribution.z;
    p.stokesV[b] += contribution.w;
}

// Accumulate the polarized sky into the payload's Stokes accumulator through
// the per-band Mueller throughput. The sky source is fetched in the ray's
// carried frame (payload.frameX) so the throughput composes correctly. Used by
// the miss shader on the polarized path (primary ray, or sky seen in a mirror).
void AccumulatePolarizedSky(inout RayPayload p, vec3 direction)
{
    RNG rng;
    rng.state = p.state.x;
    vec3 frameX = p.frameX.xyz;
    int n = max(1, SkySamples());
    float inv = pc.skyBottomExposure.w / float(n);
    for (int i = 0; i < n; ++i)
    {
        Stokes s = render_sky_stokes_framed(direction, GetSunDirection(), SkyScatteringOrders(),
                                            frameX, direction, rng);
        for (int b = 0; b < 3; ++b)
        {
            vec4 src = vec4(s.I[b], s.Q[b], s.U[b], s.V[b]) * inv;
            AccumulateStokes(p, b, p.mueller[b] * src);
        }
    }
    p.state.x = rng.state;
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
