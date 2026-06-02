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
    if (pc.polarizer.x > 0.5)
    {
        // Polarized path: the sky is a polarized source. Fetch its full Stokes
        // vector in the ray's carried frame and push it through the Mueller
        // throughput into the camera-frame Stokes accumulator. Works for the
        // primary ray (identity throughput) and for sky seen in a mirror.
        AccumulatePolarizedSky(payload, gl_WorldRayDirectionEXT);
        return;
    }

    // Scalar path: intensity-only sky for colour and image-based lighting.
    uint rngState = payload.state.x;
    payload.radiance.xyz += payload.throughput.xyz * SampleSky(gl_WorldRayDirectionEXT, rngState);
    payload.state.x = rngState;
}
