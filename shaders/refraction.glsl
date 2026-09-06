// Lookup of CPU-integrated RK4 eikonal trajectories, not an empirical angular
// refraction formula. Grid constants match AtmosphericOptics.h.
const int AtmosphereHeightCount=48;
const int AtmosphereBranchAngles=128;
const int AtmospherePathSamples=33;
const int AtmosphereRayStride=34;
struct AtmosphereLookup { ivec4 bases; vec2 fraction; bool ground; };

AtmosphereLookup atmosphere_lookup(vec3 origin,vec3 direction)
{
    float radius=length(origin), height=max(radius-SkyEarthRadius(),0.01);
    float x=height/sceneData.atmosphericRefraction.y;
    float expMinusOne=x<0.001 ? -x+0.5*x*x : exp(-x)-1.0;
    float nMinusOne=sceneData.atmosphericRefraction.x*(1.0+expMinusOne);
    // Stable near-ground critical angle from conserved impact parameter n*r.
    float delta=(height*(1.0+nMinusOne)+SkyEarthRadius()*sceneData.atmosphericRefraction.x*expMinusOne)
               /((1.0+nMinusOne)*radius);
    float horizon=-2.0*asin(sqrt(clamp(0.5*delta,0.0,1.0)));
    float elevation=asin(clamp(dot(origin/radius,direction),-1.0,1.0));
    AtmosphereLookup lookup;
    lookup.ground=elevation<horizon;
    float angle=lookup.ground ? 1.0-sqrt(clamp((horizon-elevation)/(0.5*PI+horizon),0.0,1.0))
                             : sqrt(clamp((elevation-horizon)/(0.5*PI-horizon),0.0,1.0));
    float h=clamp(log(1.0+height)/log(1.0+SkyAtmosphereRadius()-SkyEarthRadius()),0.0,1.0)
           *float(AtmosphereHeightCount-1);
    float a=angle*float(AtmosphereBranchAngles-1);
    int h0=min(int(h),AtmosphereHeightCount-2), a0=min(int(a),AtmosphereBranchAngles-2);
    int branch=lookup.ground ? 0 : AtmosphereBranchAngles;
    int base=(h0*2*AtmosphereBranchAngles+branch+a0)*AtmosphereRayStride;
    int nextHeight=2*AtmosphereBranchAngles*AtmosphereRayStride;
    lookup.bases=ivec4(base,base+AtmosphereRayStride,base+nextHeight,base+nextHeight+AtmosphereRayStride);
    lookup.fraction=vec2(h-float(h0),a-float(a0));
    return lookup;
}
vec4 atmosphere_entry(AtmosphereLookup lookup,int sampleIndex)
{
    vec4 low=mix(atmosphereRayBuffer.entries[lookup.bases.x+sampleIndex],
                 atmosphereRayBuffer.entries[lookup.bases.y+sampleIndex],lookup.fraction.y);
    vec4 high=mix(atmosphereRayBuffer.entries[lookup.bases.z+sampleIndex],
                  atmosphereRayBuffer.entries[lookup.bases.w+sampleIndex],lookup.fraction.y);
    return mix(low,high,lookup.fraction.x);
}
void atmosphere_point(AtmosphereLookup lookup,vec3 origin,vec3 direction,float distance,out vec3 p,out vec3 d)
{
    if (sceneData.atmosphericRefraction.x==0.0) { p=origin+direction*distance; d=direction; return; }
    float end=atmosphere_entry(lookup,0).x;
    float f=sqrt(clamp(distance/max(end,1e-6),0.0,1.0))*float(AtmospherePathSamples-1);
    int first=min(int(f),AtmospherePathSamples-2);
    float u0=float(first)/float(AtmospherePathSamples-1);
    float u1=float(first+1)/float(AtmospherePathSamples-1);
    float fraction=clamp((distance/max(end,1e-6)-u0*u0)/(u1*u1-u0*u0),0.0,1.0);
    vec4 point=mix(atmosphere_entry(lookup,1+first),atmosphere_entry(lookup,2+first),fraction);
    vec3 up=normalize(origin);
    vec3 tangent=direction-up*dot(up,direction);
    float tangentLength=length(tangent);
    tangent=tangentLength>1e-6 ? tangent/tangentLength : vec3(1,0,0);
    p=origin+tangent*point.x+up*point.y;
    d=normalize(tangent*point.z+up*point.w);
}
void advance_ray(inout vec3 p,inout vec3 d,float distance)
{
    AtmosphereLookup lookup=atmosphere_lookup(p,d);
    vec3 nextP,nextD; atmosphere_point(lookup,p,d,distance,nextP,nextD);
    p=nextP; d=nextD;
}
vec2 medium_boundary(vec3 origin,vec3 direction,out vec3 exitDirection)
{
    AtmosphereLookup lookup=atmosphere_lookup(origin,direction);
    vec4 entry=atmosphere_entry(lookup,0);
    vec3 endPoint;
    atmosphere_point(lookup,origin,direction,entry.x,endPoint,exitDirection);
    if (sceneData.atmosphericRefraction.x==0.0)
    {
        float ground=ray_sphere(origin,direction,SkyEarthRadius()).x;
        float end=max(ray_sphere(origin,direction,SkyAtmosphereRadius()).y,0.0);
        return ground>0.0 ? vec2(min(ground,end),1.0) : vec2(end,0.0);
    }
    return vec2(entry.x,lookup.ground ? 1.0 : 0.0);
}
