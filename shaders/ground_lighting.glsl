// Primary ground lighting with wavelength-independent work shared across
// all bands. Sample locations and quadrature match single_scatter_stokes.
struct GroundSpectrum { float bands[SPECTRAL_BANDS]; };
// Factor wavelength-dependent Rayleigh extinction out of the path integral.
// Only two accumulated depths are needed, instead of thirteen spectral arrays.
vec2 ground_optical_depth(vec3 origin,vec3 direction,float distance,int steps)
{
    vec2 depth=vec2(0.0);
    float ds=distance/float(steps);
    for (int i=0;i<steps;++i)
    {
        vec3 p=origin+direction*((float(i)+0.5)*ds);
        float h=max(length(p)-SkyEarthRadius(),0.0);
        depth+=exp(-h/vec2(SkyScaleHeightRayleigh(),SkyScaleHeightMie()));
    }
    // Step length and sea-level aerosol extinction are constant on the ray.
    return depth*vec2(ds,SkyBetaM()*ds);
}
float ground_band_depth(vec2 depth,int band) { return SkyBetaR(band)*depth.x+depth.y; }
GroundSpectrum primary_ground_lighting(vec3 viewDir,vec3 sun)
{
    GroundSpectrum result;
    for (int band=0;band<SPECTRAL_BANDS;++band) result.bands[band]=0.0;
    if (SkyGroundAlbedo()<=0.0) return result;
    vec3 origin=vec3(0.0,SkyEarthRadius()+1.0,0.0);
    float distance=ray_sphere(origin,viewDir,SkyEarthRadius()).x;
    if (distance<=0.0) return result;
    vec3 ground=origin+viewDir*distance,normal=normalize(ground),surface=ground+normal;
    float sunCosine=max(dot(normal,sun),0.0);
    float solidAngle=2.0*PI*(1.0-cos(SkySunRadius()));
    GroundSpectrum direct, diffuse;
    for (int band=0;band<SPECTRAL_BANDS;++band) { direct.bands[band]=0.0;diffuse.bands[band]=0.0; }
    if (sunCosine>0.0)
    {
        vec2 depth=ground_optical_depth(surface,sun,max(ray_sphere(surface,sun,SkyAtmosphereRadius()).y,0.0),16);
        for (int band=0;band<SPECTRAL_BANDS;++band)
            direct.bands[band]=SkySunRadiance(band)*solidAngle*sunCosine*exp(-ground_band_depth(depth,band));
    }
    vec3 tangent,bitangent;beam_basis(normal,tangent,bitangent);
    int steps=max(SkyViewSteps(),4);
    for (int j=0;j<8;++j)
    {
        float u=(float(j)+0.5)/8.0,radius=sqrt(u),phi=2.39996322972865332*float(j);
        vec3 direction=normalize(tangent*(radius*cos(phi))+bitangent*(radius*sin(phi))+normal*sqrt(max(1.0-u,0.0)));
        vec2 hit=ray_sphere(surface,direction,SkyAtmosphereRadius());
        if (hit.y<0.0) continue;
        float start=max(hit.x,0.0),end=hit.y;
        float groundHit=ray_sphere(surface,direction,SkyEarthRadius()).x;
        if (groundHit>0.0) end=min(end,groundHit);
        float ds=(end-start)/float(steps);
        vec2 cameraDepth=vec2(0.0);
        GroundSpectrum weightR,weightM;
        for (int band=0;band<SPECTRAL_BANDS;++band)
        { weightR.bands[band]=0.0;weightM.bands[band]=0.0; }
        for (int i=0;i<steps;++i)
        {
            vec3 p=surface+direction*(start+(float(i)+0.5)*ds);
            float h=max(length(p)-SkyEarthRadius(),0.0);
            float rhoR=exp(-h/SkyScaleHeightRayleigh()),extM=mie_extinction(h);
            bool lit=ray_sphere(p,sun,SkyEarthRadius()).x<=0.0;
            vec2 sunDepth=vec2(0.0);
            if (lit) sunDepth=ground_optical_depth(p,sun,max(ray_sphere(p,sun,SkyAtmosphereRadius()).y,0.0),8);
            vec2 slabDepth=vec2(rhoR,extM)*ds;
            // Beer-Lambert factors multiply by adding optical depths. Keep
            // the same midpoint attenuation with one exponential per band.
            vec2 totalDepth=cameraDepth+0.5*slabDepth+sunDepth;
            for (int band=0;band<SPECTRAL_BANDS;++band)
            {
                if (lit)
                {
                    // Sun radiance, solid angle and ds are constant along
                    // this direction; apply them once after integration.
                    float w=exp(-ground_band_depth(totalDepth,band));
                    weightR.bands[band]+=rhoR*w;weightM.bands[band]+=extM*w;
                }
            }
            cameraDepth+=slabDepth;
        }
        // Unpolarized sunlight contributes only the Mueller first-column I
        // entry. Basis rotations and polarization clamps cannot change it.
        float mu=clamp(dot(sun,direction),-1.0,1.0);
        float phaseR=rayleigh_mueller(mu).m11;
        // All wavelengths use the same scattering angle and table coordinates.
        int bins=max(2,SkyMieAngleBins());
        float fbin=acos(mu)/PI*float(bins-1);
        int i0=clamp(int(floor(fbin)),0,bins-1),i1=min(i0+1,bins-1);
        float fraction=fbin-float(i0);
        for (int band=0;band<SPECTRAL_BANDS;++band)
        {
            float phaseM=mix(mieMatrixBuffer.entries[band*bins+i0].x,
                mieMatrixBuffer.entries[band*bins+i1].x,fraction);
            diffuse.bands[band]+=(SkyBetaR(band)*weightR.bands[band]*phaseR
                +weightM.bands[band]*phaseM)*ds;
        }
    }
    vec2 cameraDepth=ground_optical_depth(surface,-viewDir,max(distance-1.0,0.0),8);
    // Combine the cosine-weighted PI/8 quadrature with phase normalization
    // 1/(4*PI), and apply the direction-independent solar factors once.
    for (int band=0;band<SPECTRAL_BANDS;++band)
        result.bands[band]=SkyGroundAlbedo()/PI*(direct.bands[band]+diffuse.bands[band]*(SkySunRadiance(band)*solidAngle/32.0))*exp(-ground_band_depth(cameraDepth,band));
    return result;
}
