// Primary-ray miss shader.
//
// Hit when a primary or indirect path-tracer ray escapes the scene. The
// sky environment is sampled along the ray direction and added to the
// accumulated radiance, weighted by the path's running throughput. This
// is what terminates a path with sky illumination (Image-Based Lighting)
// instead of returning black.

#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "path_tracer_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main()
{
    uint rngState = payload.state.x;

    if (payload.state.y == 0u)
    {
        // Camera-visible sky: solve the full polarized vector radiative
        // transfer so both the colour and the polarization are physical.
        float degree = 0.0;
        float axisAngle = 0.0;
        float circularDegree = 0.0;
        vec3 skyRadiance = SampleSkyStokes(gl_WorldRayDirectionEXT, rngState, degree, axisAngle, circularDegree);
        payload.radiance.xyz += payload.throughput.xyz * skyRadiance;
        if (pc.polarizer.x > 0.5)
        {
            payload.throughput.w = degree;
            payload.radiance.w = axisAngle;
            payload.state.z = floatBitsToUint(circularDegree);
        }
    }
    else
    {
        // Indirect bounce: cheap intensity-only sky for image-based lighting.
        payload.radiance.xyz += payload.throughput.xyz * SampleSky(gl_WorldRayDirectionEXT, rngState);
    }

    payload.state.x = rngState;
}
