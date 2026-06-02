// Closest-hit shader — the heart of the path tracer.
//
// On a surface hit we:
//   1. Add emissive contribution; if the surface is a pure light, return.
//   2. Build a tangent-space frame around the world-space hit normal.
//   3. Direct sun lighting via next-event estimation: cone-sample the
//      sun direction and trace a shadow ray; on miss, add the GGX BRDF
//      response weighted by sun radiance and solid angle.
//   4. Russian-roulette path termination once depth ≥ 3.
//   5. Indirect bounce: GGX VNDF sampling (Heitz 2018) for the next
//      direction. Surfaces are always perfectly smooth, so this resolves
//      to a near-mirror reflection.
//   6. Recurse via traceRayEXT into the same hitgroup; the path
//      eventually escapes to the sky miss shader or is terminated by
//      Russian roulette / max-bounce gating in the rgen shader.

#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "path_tracer_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT ShadowPayload shadowPayload;
hitAttributeEXT vec2 hitAttributes;

// Smith-Lambda term for the GGX masking-shadowing function. Returns a
// large finite number for grazing-angle directions so the 1/(1+Λ) form
// never produces a div-by-zero.
float SmithLambda(float ax, float ay, vec3 localDirection)
{
    float z2 = localDirection.z * localDirection.z;
    if (z2 <= 1.0e-6)
    {
        return 1.0e8;
    }
    float slope = (ax * ax * localDirection.x * localDirection.x
                 + ay * ay * localDirection.y * localDirection.y) / z2;
    return 0.5 * (-1.0 + sqrt(1.0 + slope));
}

// Anisotropic GGX normal-distribution function evaluated in the local
// (tangent-space) frame. ax / ay are the per-axis roughness parameters.
float GgxDistribution(float ax, float ay, vec3 H_local)
{
    if (H_local.z <= 0.0)
    {
        return 0.0;
    }
    float sx = H_local.x / ax;
    float sy = H_local.y / ay;
    float sz = H_local.z;
    float denom = sx * sx + sy * sy + sz * sz;
    return 1.0 / max(PI * ax * ay * denom * denom, 1.0e-6);
}

// Evaluate the full anisotropic GGX BRDF for a (V, L) pair in tangent
// space. Used by the direct-sun lighting path; the indirect path uses
// VNDF importance sampling and so collapses some terms.
vec3 EvalGgxBrdf(vec3 V_local, vec3 L_local, float ax, float ay, vec3 eta, vec3 extinction)
{
    if (V_local.z <= 0.0 || L_local.z <= 0.0)
    {
        return vec3(0.0);
    }
    vec3 H_local = normalize(V_local + L_local);
    float D = GgxDistribution(ax, ay, H_local);
    float LambdaV = SmithLambda(ax, ay, V_local);
    float LambdaL = SmithLambda(ax, ay, L_local);
    float G = 1.0 / (1.0 + LambdaV + LambdaL);
    vec3 F = FresnelReflectance(max(dot(H_local, V_local), 0.0), eta, extinction);
    return F * (D * G / max(4.0 * V_local.z * L_local.z, 1.0e-6));
}

void main()
{
    bool polarized = pc.polarizer.x > 0.5;

    // Look up per-instance data using the custom index baked into the
    // TLAS instance record at build time.
    InstanceData instanceData = instanceBuffer.instances[gl_InstanceCustomIndexEXT];
    Material material = GetInstanceMaterial(instanceData);

    // Emissive contribution. Emission is unpolarized; on the polarized path it
    // reaches the camera through the current Mueller throughput (its first
    // column), on the scalar path through the scalar throughput.
    if (polarized)
    {
        for (int b = 0; b < 3; ++b)
        {
            AccumulateStokes(payload, b, material.emission[b] * payload.mueller[b][0]);
        }
    }
    else
    {
        payload.radiance.xyz += payload.throughput.xyz * material.emission;
    }
    if (MaxComponent(material.emission) > 0.0)
    {
        return;
    }

    uint maxBounces = uint(pc.cameraRightBounces.w);
    if (payload.state.y + 1u >= maxBounces)
    {
        return;
    }

    // Reconstruct the world-space hit point from the launch ray origin
    // and the reported hit T.
    vec3 hitPosition = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    mat3 worldToObjectLinear =
        mat3(gl_WorldToObjectEXT[0].xyz, gl_WorldToObjectEXT[1].xyz, gl_WorldToObjectEXT[2].xyz);
    vec3 objectNormal = GetTriangleObjectNormal(instanceData, gl_PrimitiveID, hitAttributes);
    vec3 normal = TransformNormalToWorld(objectNormal, worldToObjectLinear);
    // Flip the shading normal toward the incoming ray for back-facing
    // hits so the BSDF math always operates in the upper hemisphere.
    if (dot(normal, gl_WorldRayDirectionEXT) > 0.0)
    {
        normal = -normal;
    }

    // Albedo tint of this surface's reflection. On the polarized path it is a
    // neutral per-band attenuation of the Mueller throughput (it scales every
    // Stokes component equally, so it changes colour but not polarization).
    if (polarized)
    {
        for (int b = 0; b < 3; ++b)
        {
            payload.mueller[b] *= material.albedo[b];
        }
        float intensityThroughput =
            max(payload.mueller[0][0][0], max(payload.mueller[1][0][0], payload.mueller[2][0][0]));
        if (intensityThroughput < 1.0e-6)
        {
            return;
        }
    }
    else
    {
        payload.throughput.xyz *= material.albedo;
        if (MaxComponent(payload.throughput.xyz) < 1.0e-6)
        {
            return;
        }
    }

    RNG rng;
    rng.state = payload.state.x;

    vec3 V = -gl_WorldRayDirectionEXT;
    vec3 helper = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(helper, normal));
    vec3 bitangent = cross(normal, tangent);

    vec3 V_local = normalize(vec3(dot(V, tangent), dot(V, bitangent), dot(V, normal)));

    // Surfaces are always perfectly smooth (roughness = 0), so the GGX
    // roughness collapses to the clamp floor and the lobe is a tight mirror.
    float ax = 0.001;
    float ay = 0.001;

    // ── Direct sun lighting (next-event estimation, sampled over the sun cone) ──
    {
        float coneAngle = max(SkySunRadius(), 1.0e-4);
        vec3 sunDirection = sample_cone(GetSunDirection(), coneAngle, rng);
        vec3 L_local = vec3(dot(sunDirection, tangent),
                            dot(sunDirection, bitangent),
                            dot(sunDirection, normal));
        if (L_local.z > 0.0 && V_local.z > 0.0)
        {
            shadowPayload.visible = 0u;
            traceRayEXT(topLevelAS,
                        gl_RayFlagsOpaqueEXT
                            | gl_RayFlagsTerminateOnFirstHitEXT
                            | gl_RayFlagsSkipClosestHitShaderEXT,
                        0xFF,
                        0,
                        0,
                        1,
                        OffsetRay(hitPosition, normal),
                        0.001,
                        sunDirection,
                        1e30,
                        1);

            if (shadowPayload.visible != 0u)
            {
                vec3 Li = SunIncidentRadiance();
                float solidAngle = SunSolidAngle();
                if (polarized)
                {
                    // Geometric (non-Fresnel) part of the GGX BRDF; the Fresnel
                    // becomes the reflection Mueller matrix carrying polarization.
                    vec3 Hs = normalize(V_local + L_local);
                    float D = GgxDistribution(ax, ay, Hs);
                    float LambdaV = SmithLambda(ax, ay, V_local);
                    float LambdaL = SmithLambda(ax, ay, L_local);
                    float G = 1.0 / (1.0 + LambdaV + LambdaL);
                    float Dg = D * G / max(4.0 * V_local.z * L_local.z, 1.0e-6);
                    float weight = Dg * L_local.z * solidAngle;

                    float cosThetaHs = clamp(dot(normalize(V + sunDirection), V), 0.0, 1.0);
                    mat4 reflectSun[3];
                    ReflectionMueller(cosThetaHs, material.eta, material.extinction, reflectSun);

                    vec3 sAxisSun = cross(sunDirection, V);
                    float sl = length(sAxisSun);
                    sAxisSun = sl > 1.0e-5 ? sAxisSun / sl : payload.frameX.xyz;
                    mat4 crot = FrameRotator(sAxisSun, payload.frameX.xyz, gl_WorldRayDirectionEXT);

                    for (int b = 0; b < 3; ++b)
                    {
                        // Unpolarized sun (I,0,0,0) -> reflectSun first column.
                        vec4 src = (Li[b] * weight) * reflectSun[b][0];
                        AccumulateStokes(payload, b, payload.mueller[b] * (crot * src));
                    }
                }
                else
                {
                    vec3 brdf = EvalGgxBrdf(V_local, L_local, ax, ay, material.eta, material.extinction);
                    payload.radiance.xyz += payload.throughput.xyz * brdf * Li * (L_local.z * solidAngle);
                }
            }
        }
    }

    // Russian roulette after a few guaranteed bounces — survival probability is
    // tied to the remaining intensity throughput (scalar throughput, or the
    // Mueller M11 transmittance on the polarized path).
    if (payload.state.y >= 3u)
    {
        float intensity = polarized
            ? max(payload.mueller[0][0][0], max(payload.mueller[1][0][0], payload.mueller[2][0][0]))
            : MaxComponent(payload.throughput.xyz);
        float surviveProbability = clamp(intensity, 0.05, 0.95);
        if (NextFloat(rng.state) > surviveProbability)
        {
            payload.state.x = rng.state;
            return;
        }
        if (polarized)
        {
            payload.mueller[0] /= surviveProbability;
            payload.mueller[1] /= surviveProbability;
            payload.mueller[2] /= surviveProbability;
        }
        else
        {
            payload.throughput.xyz /= surviveProbability;
        }
    }

    payload.state.y += 1u;

    vec2 xi = vec2(NextFloat(rng.state), NextFloat(rng.state));

    // Heitz 2018 VNDF sampling (anisotropic)
    vec3 Vh = normalize(vec3(ax * V_local.x, ay * V_local.y, V_local.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);

    float r = sqrt(xi.x);
    float phi = 2.0 * PI * xi.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    vec3 Ne = normalize(vec3(ax * Nh.x, ay * Nh.y, max(0.0, Nh.z)));

    vec3 H = normalize(tangent * Ne.x + bitangent * Ne.y + normal * Ne.z);
    vec3 reflectedSample = reflect(-V, H);

    vec3 L_local = normalize(vec3(dot(reflectedSample, tangent),
                                  dot(reflectedSample, bitangent),
                                  dot(reflectedSample, normal)));

    float cosThetaH = clamp(dot(H, V), 0.0, 1.0);

    if (polarized)
    {
        if (L_local.z <= 0.0)
        {
            payload.mueller[0] = mat4(0.0);
            payload.mueller[1] = mat4(0.0);
            payload.mueller[2] = mat4(0.0);
        }
        else
        {
            // Update the Mueller throughput by this specular reflection: rotate
            // the carried frame into the plane of incidence, then reflect. The
            // reflection's s-axis (perpendicular to the plane of incidence)
            // becomes the new carried frame for the reflected ray.
            mat4 reflectM[3];
            ReflectionMueller(cosThetaH, material.eta, material.extinction, reflectM);

            vec3 sAxis = cross(gl_WorldRayDirectionEXT, H);
            float sl = length(sAxis);
            sAxis = sl > 1.0e-5 ? sAxis / sl : payload.frameX.xyz;
            mat4 crot = FrameRotator(sAxis, payload.frameX.xyz, gl_WorldRayDirectionEXT);

            for (int b = 0; b < 3; ++b)
            {
                payload.mueller[b] = payload.mueller[b] * (crot * reflectM[b]);
            }
            payload.frameX = vec4(sAxis, 0.0);
        }
    }
    else
    {
        if (L_local.z <= 0.0)
        {
            payload.throughput.xyz = vec3(0.0);
        }
        else
        {
            vec3 F = FresnelReflectance(cosThetaH, material.eta, material.extinction);

            float LambdaV = SmithLambda(ax, ay, V_local);
            float LambdaL = SmithLambda(ax, ay, L_local);
            float G2_over_G1 = (1.0 + LambdaV) / (1.0 + LambdaV + LambdaL);

            vec3 single_scattering = F * G2_over_G1;

            // Multiscatter GGX Energy Compensation (Fdez-Aguera approximation)
            float E0 = G2_over_G1;
            vec3 F_avg = F;
            vec3 multi_scattering = F_avg * ((1.0 - E0) * (1.0 - E0)) / (1.0 - F_avg * (1.0 - E0));

            payload.throughput.xyz *= (single_scattering + multi_scattering);
        }
    }

    payload.state.x = rng.state;
    // Recurse along the sampled direction. OffsetRay nudges the origin
    // outside the surface to avoid self-intersection. Note: we
    // intentionally only set gl_RayFlagsOpaqueEXT here — adding back-
    // face culling makes objects render pitch black.
    traceRayEXT(topLevelAS,
    gl_RayFlagsOpaqueEXT,
    0xFF,
    0,
    0,
    0,
    OffsetRay(hitPosition, normal),
    0.0,
    reflectedSample,
    1e30,
    0);
}
