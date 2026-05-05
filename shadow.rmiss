// Shadow-ray miss shader.
//
// Used for next-event-estimation toward the sun: the shadow ray is
// dispatched with TerminateOnFirstHit + SkipClosestHit so a hit silently
// kills the ray (visible stays 0 from the caller). Reaching this miss
// shader means the path was unoccluded, and we mark the sun as visible
// so the closest-hit shader can add direct sun contribution.

#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "path_tracer_common.glsl"

layout(location = 1) rayPayloadInEXT ShadowPayload shadowPayload;

void main()
{
    shadowPayload.visible = 1u;
}
