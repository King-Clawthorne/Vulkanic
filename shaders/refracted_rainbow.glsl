// The original spherical, single-scattering rainbow evaluated on curved rays.
RainbowPathWeights integrate_refracted_rainbow(vec3 viewDir,vec3 sun)
{
    RainbowPathWeights weights;
    for (int band=0;band<SPECTRAL_BANDS;++band)
    {
        weights.bands[band]=0.0;
        weights.refracted[band]=vec4(0.0);
    }
    if (!RainbowEnabled() || RainbowScattering()<=0.0) return weights;
    vec3 origin=vec3(0.0,SkyEarthRadius()+1.0,0.0);
    AtmosphereLookup ray=atmosphere_lookup(origin,viewDir);
    vec2 hit=ray_ellipsoid(vec3(0.0),viewDir);
    // The local volume supplies a conservative straight-line seed. Evaluate
    // density on the curved trajectory, with padding for its weak bending.
    float start=max(hit.x-250.0,0.0),end=min(hit.y+250.0,atmosphere_entry(ray,0).x);
    if (hit.y<=0.0 || end<=start) return weights;
    int steps=max(1,RainbowViewSteps());
    float ds=(end-start)/float(steps),tau=0.0;
    float solidAngle=2.0*PI*(1.0-cos(SkySunRadius()));
    float observerN=air_index(origin);
    Stokes sunlight=stokes_zero();sunlight.I=1.0;
    for (int i=0;i<steps;++i)
    {
        vec3 p,d;atmosphere_point(ray,origin,viewDir,start+(float(i)+0.5)*ds,p,d);
        float density=rainbow_density(p-origin);
        if (density<=0.0) continue;
        float slabTau=RainbowExtinction()*density*ds;
        float cameraT=exp(-(tau+0.5*slabTau));tau+=slabTau;
        vec3 localSun=local_sun_direction(p,sun);
        AtmosphereLookup sunRay=atmosphere_lookup(p,localSun);
        if (sunRay.ground) continue;
        vec4 airDepth=atmosphere_entry(sunRay,0);
        vec2 rainHit=ray_ellipsoid(p-origin,localSun);
        float sunDistance=min(max(rainHit.y+250.0,0.0),airDepth.x),sunTau=0.0;
        for (int j=0;j<8;++j)
        {
            vec3 q,qd;atmosphere_point(sunRay,p,localSun,(float(j)+0.5)*sunDistance/8.0,q,qd);
            sunTau+=rainbow_density(q-origin)*sunDistance/8.0*RainbowExtinction();
        }
        float weight=solidAngle*RainbowScattering()*density/(4.0*PI)*ds*cameraT*exp(-sunTau)
                    *solar_solid_angle_jacobian(p,localSun)*observerN*observerN;
        for (int band=0;band<SPECTRAL_BANDS;++band)
        {
            float airT=exp(-(SkyBetaR(band)*airDepth.y+SkyBetaM()*airDepth.z));
            Stokes scattered=transport_frame(scatter_rainbow(sunlight,-localSun,-d,band),-d,-viewDir);
            weights.refracted[band]+=vec4(scattered.I,scattered.Q,scattered.U,scattered.V)*weight*airT;
        }
    }
    return weights;
}
