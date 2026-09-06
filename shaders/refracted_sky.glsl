// Refracted counterpart of the original atmosphere integrators.
float air_index(vec3 p)
{
    float h = max(length(p) - SkyEarthRadius(), 0.0);
    return 1.0 + sceneData.atmosphericRefraction.x * exp(-h / sceneData.atmosphericRefraction.y);
}
#include "refraction.glsl"

Stokes transport_frame(Stokes s, vec3 from, vec3 to)
{
    vec3 a, b, targetA, targetB;
    beam_basis(from, a, b); beam_basis(to, targetA, targetB);
    vec3 v = cross(from, to);
    float c = dot(from, to);
    vec3 rotatedA = a + cross(v, a) + cross(v, cross(v, a)) / max(1.0+c, 1e-6);
    vec3 rotatedB = cross(to, rotatedA);
    return stokes_reframe(s, rotatedA, rotatedB, targetA, targetB);
}
vec3 local_sun_direction(vec3 origin, vec3 sun)
{
    if (sceneData.atmosphericRefraction.x == 0.0) return sun;
    vec3 up = normalize(origin);
    float elevation = asin(clamp(dot(up, sun), -1.0, 1.0));
    float h = max(length(origin) - SkyEarthRadius(), 0.0);
    float seed = 0.01 * exp(-h/sceneData.atmosphericRefraction.y)
               * (sceneData.atmosphericRefraction.x / 0.000277)
               * exp(-max(elevation, 0.0)*12.0);
    vec3 local = normalize(sun + (up - sun*dot(up, sun))*seed);
    for (int i = 0; i < 3; ++i)
    {
        vec3 escaped;
        vec2 boundary = medium_boundary(origin, local, escaped);
        if (boundary.y > 0.5) break;
        local = normalize(local + sun - escaped);
    }
    return local;
}
float solar_solid_angle_jacobian(vec3 origin,vec3 localSun)
{
    if (sceneData.atmosphericRefraction.x==0.0) return 1.0;
    vec3 up=normalize(origin);
    float elevation=asin(clamp(dot(up,localSun),-1.0,1.0));
    if (abs(elevation)>1.55) return 1.0;
    vec3 tangent=normalize(localSun-up*dot(up,localSun));
    const float delta=0.0001;
    vec3 lowExit,highExit;
    vec2 low=medium_boundary(origin,tangent*cos(elevation-delta)+up*sin(elevation-delta),lowExit);
    vec2 high=medium_boundary(origin,tangent*cos(elevation+delta)+up*sin(elevation+delta),highExit);
    if (low.y>0.5 || high.y>0.5) return 1.0;
    float lowAngle=asin(clamp(dot(up,lowExit),-1.0,1.0));
    float highAngle=asin(clamp(dot(up,highExit),-1.0,1.0));
    // dOmega_local/dOmega_space for the radially symmetric ray mapping.
    return cos(elevation)/max(cos(0.5*(lowAngle+highAngle)),1e-6)
           *(2.0*delta/max(highAngle-lowAngle,1e-6));
}

float refracted_transmittance(vec3 origin, vec3 direction, int band)
{
    AtmosphereLookup lookup = atmosphere_lookup(origin, direction);
    if (lookup.ground) return 0.0;
    vec4 depth = atmosphere_entry(lookup, 0);
    return exp(-(SkyBetaR(band) * depth.y + SkyBetaM() * depth.z));
}
float refracted_camera_transmittance(vec3 origin, vec3 direction, float distance, int band)
{
    AtmosphereLookup lookup = atmosphere_lookup(origin, direction);
    float tau = 0.0;
    for (int i=0; i<8; ++i)
    {
        vec3 p,d;
        atmosphere_point(lookup,origin,direction,(float(i)+0.5)*distance/8.0,p,d);
        float h=max(length(p)-SkyEarthRadius(),0.0);
        tau+=(SkyBetaR(band)*exp(-h/SkyScaleHeightRayleigh())+mie_extinction(h))*distance/8.0;
    }
    return exp(-tau);
}
Stokes refracted_single_atmosphere(vec3 origin,vec3 viewDir,vec3 sun,int steps,int band)
{
    AtmosphereLookup ray=atmosphere_lookup(origin,viewDir);
    float ds=atmosphere_entry(ray,0).x/float(steps);
    Stokes result=stokes_zero(), sunlight=stokes_zero(); sunlight.I=1.0;
    float cameraT=1.0;
    float solidAngle=2.0*PI*(1.0-cos(SkySunRadius()));
    for (int i=0;i<steps;++i)
    {
        vec3 p,d;
        atmosphere_point(ray,origin,viewDir,(float(i)+0.5)*ds,p,d);
        float h=max(length(p)-SkyEarthRadius(),0.0);
        float rhoR=exp(-h/SkyScaleHeightRayleigh()), extM=mie_extinction(h);
        float slabT=exp(-(SkyBetaR(band)*rhoR+extM)*ds);
        vec3 localSun=local_sun_direction(p,sun);
        float source=SkySunRadiance(band)*solidAngle*refracted_transmittance(p,localSun,band)
                     *solar_solid_angle_jacobian(p,localSun);
        Stokes scattered=scatter_stokes(sunlight,-localSun,-d,rhoR,extM,band);
        scattered=transport_frame(scattered,-d,-viewDir);
        // Solar radiance gains n(p)^2; propagation to the observer contributes
        // n(observer)^2/n(p)^2, leaving the observer factor below.
        result=stokes_add(result,stokes_scale(scattered,cameraT*sqrt(slabT)*ds*source));
        cameraT*=slabT;
    }
    float n=air_index(origin);
    return stokes_make_physical(stokes_scale(result,n*n));
}
Stokes refracted_order_1(vec3 origin,vec3 viewDir,vec3 sun,int steps,int band)
{
    Stokes result=refracted_single_atmosphere(origin,viewDir,sun,steps,band);
    AtmosphereLookup ray=atmosphere_lookup(origin,viewDir);
    if (!ray.ground || SkyGroundAlbedo()<=0.0) return result;
    float distance=atmosphere_entry(ray,0).x;
    vec3 ground,d;
    atmosphere_point(ray,origin,viewDir,distance,ground,d);
    vec3 normal=normalize(ground), surface=normal*(SkyEarthRadius()+1.0);
    vec3 localSun=local_sun_direction(surface,sun);
    float irradiance=SkySunRadiance(band)*(2.0*PI*(1.0-cos(SkySunRadius())))
        *max(dot(normal,localSun),0.0)*refracted_transmittance(surface,localSun,band)
        *solar_solid_angle_jacobian(surface,localSun);
    vec3 a,b; beam_basis(normal,a,b);
    float surfaceN=air_index(surface);
    for (int i=0;i<8;++i)
    {
        float u=(float(i)+0.5)/8.0,phi=2.39996322972865332*float(i);
        vec3 incoming=normalize(a*(sqrt(u)*cos(phi))+b*(sqrt(u)*sin(phi))+normal*sqrt(1.0-u));
        irradiance+=PI/8.0*refracted_single_atmosphere(surface,incoming,sun,max(steps,4),band).I
                    /(surfaceN*surfaceN);
    }
    float n=air_index(origin);
    result.I+=SkyGroundAlbedo()/PI*irradiance*refracted_camera_transmittance(origin,viewDir,distance,band)*n*n;
    return stokes_make_physical(result);
}
#define DEFINE_REFRACTED_ORDER(NAME,PREVIOUS) \
Stokes NAME(vec3 origin,vec3 viewDir,vec3 sun,int steps,int band) \
{ \
    AtmosphereLookup ray=atmosphere_lookup(origin,viewDir); \
    float ds=atmosphere_entry(ray,0).x/float(steps),cameraT=1.0; \
    Stokes result=stokes_zero(); \
    for (int i=0;i<steps;++i) \
    { \
        vec3 p,d; atmosphere_point(ray,origin,viewDir,(float(i)+0.5)*ds,p,d); \
        float h=max(length(p)-SkyEarthRadius(),0.0); \
        float rhoR=exp(-h/SkyScaleHeightRayleigh()),extM=mie_extinction(h); \
        float slabT=exp(-(SkyBetaR(band)*rhoR+extM)*ds); \
        Stokes integral=stokes_zero(); \
        for (int j=0;j<SkySecondarySamples();++j) \
        { \
            vec3 din=secondary_direction(j,SkySecondarySamples(),i); \
            Stokes incident=PREVIOUS(p,-din,sun,SkySamples(),band); \
            integral=stokes_add(integral,scatter_stokes(incident,din,-d,rhoR,extM,band)); \
        } \
        integral=transport_frame(integral,-d,-viewDir); \
        float n=air_index(p); \
        result=stokes_add(result,stokes_scale(integral,cameraT*sqrt(slabT)*ds*4.0*PI/float(SkySecondarySamples())/(n*n))); \
        cameraT*=slabT; \
    } \
    float n=air_index(origin); \
    return stokes_make_physical(stokes_scale(result,n*n)); \
}
DEFINE_REFRACTED_ORDER(refracted_order_2,refracted_order_1)
DEFINE_REFRACTED_ORDER(refracted_order_3,refracted_order_2)
DEFINE_REFRACTED_ORDER(refracted_order_4,refracted_order_3)
#undef DEFINE_REFRACTED_ORDER
Stokes render_refracted_sky(vec3 viewDir,vec3 sun,int band)
{
    vec3 origin=vec3(0.0,SkyEarthRadius()+1.0,0.0);
    Stokes result=refracted_order_1(origin,viewDir,sun,SkyViewSteps(),band);
    if (SkyScatteringOrders()>=2) result=stokes_add(result,refracted_order_2(origin,viewDir,sun,SkyViewSteps(),band));
    if (SkyScatteringOrders()>=3) result=stokes_add(result,refracted_order_3(origin,viewDir,sun,SkyViewSteps(),band));
    if (SkyScatteringOrders()>=4) result=stokes_add(result,refracted_order_4(origin,viewDir,sun,SkyViewSteps(),band));
    vec3 escaped;
    vec2 boundary=medium_boundary(origin,viewDir,escaped);
    if (boundary.y<0.5)
    {
        float n=air_index(origin);
        result.I+=sun_disk(escaped,sun)*SkySunRadiance(band)*refracted_transmittance(origin,viewDir,band)*n*n;
    }
    return stokes_make_physical(result);
}
