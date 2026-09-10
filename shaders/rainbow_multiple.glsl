// Finite-order rain-to-rain transport. Geometry and sampling are shared by
// all thirteen wavelengths; the original single-scattering path stays intact.
struct RainbowMultipleRadiance { vec4 bands[SPECTRAL_BANDS]; };
uint rain_hash(uint x)
{
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
float rain_random(inout uint state)
{
    state=rain_hash(state+0x9e3779b9u);
    return (float(state & 0x007fffffu)+0.5)/8388608.0;
}
float rain_phase_cdf(int i)
{
    return rainbowMatrixBuffer.entries[SPECTRAL_BANDS*RainbowAngleBins()+i].x;
}
vec4 rain_sample_direction(vec3 outgoing,inout uint state)
{
    float branch=rain_random(state),u=rain_random(state),phi=2.0*PI*rain_random(state);
    int bins=RainbowAngleBins();
    float theta;
    if (branch<0.9)
    {
        int lo=0,hi=bins-1;
        while (hi-lo>1)
        {
            int mid=(lo+hi)/2;
            if (rain_phase_cdf(mid)<u) lo=mid; else hi=mid;
        }
        float c0=rain_phase_cdf(lo),c1=rain_phase_cdf(hi);
        theta=(float(lo)+(u-c0)/max(c1-c0,1e-20))*PI/float(bins-1);
    }
    else theta=acos(1.0-2.0*u);
    int bin=min(int(theta/PI*float(bins-1)),bins-2);
    float phasePdf=(rain_phase_cdf(bin+1)-rain_phase_cdf(bin))*float(bins-1)
                   /(2.0*PI*PI*max(sin(theta),1e-8));
    vec3 a,b;beam_basis(outgoing,a,b);
    vec3 direction=normalize(outgoing*cos(theta)+(a*cos(phi)+b*sin(phi))*sin(theta));
    // Uniform support prevents a representative wavelength's zeros from
    // excluding contributions at other wavelengths or polarization states.
    return vec4(direction,0.9*phasePdf+0.1/(4.0*PI));
}
vec2 rain_segment(vec3 origin,vec3 direction)
{
    vec2 hit=ray_ellipsoid(origin,direction);
    hit.x=max(hit.x,0.0);
    vec3 earth=origin+vec3(0.0,SkyEarthRadius()+1.0,0.0);
    float ground=ray_sphere(earth,direction,SkyEarthRadius()).x;
    if (ground>0.0) hit.y=min(hit.y,ground);
    return hit;
}
float rain_segment_tau(vec3 origin,vec3 direction,float start,float end)
{
    int steps=int(sceneData.rainbowMultiple.z);
    float ds=(end-start)/float(steps),sum=0.0;
    for (int i=0;i<steps;++i)
        sum+=rainbow_density(origin+direction*(start+(float(i)+0.5)*ds));
    return RainbowExtinction()*ds*sum;
}
RainbowMultipleRadiance integrate_rainbow_multiple(vec3 viewDir,vec3 sun,uint seed)
{
    RainbowMultipleRadiance result;
    for (int b=0;b<SPECTRAL_BANDS;++b) result.bands[b]=vec4(0.0);
    int orders=int(sceneData.rainbowMultiple.x),samples=int(sceneData.rainbowMultiple.y);
    if (!RainbowEnabled() || orders<=1 || RainbowScattering()<=0.0) return result;
    float extinction=RainbowExtinction();
    float solidAngle=2.0*PI*(1.0-cos(SkySunRadius()));
    for (int sampleIndex=0;sampleIndex<samples;++sampleIndex)
    {
        uint state=rain_hash(seed^uint(sampleIndex)*1973u^0x68bc21ebu);
        vec3 origin=vec3(0.0),direction=viewDir,pathDirections[4];
        float throughput=1.0;
        for (int vertex=0;vertex<orders;++vertex)
        {
            vec2 hit=rain_segment(origin,direction);
            float lengthInRain=hit.y-hit.x;
            if (lengthInRain<=0.0) break;
            float tau=extinction*lengthInRain;
            float mass=tau<0.001 ? tau*(1.0-0.5*tau+tau*tau/6.0) : 1.0-exp(-tau);
            float u=rain_random(state);
            float distance=tau<0.001 ? u*lengthInRain : -log(1.0-u*mass)/extinction;
            float pdf=tau<0.001 ? 1.0/lengthInRain : extinction*exp(-extinction*distance)/mass;
            float t=hit.x+distance;
            vec3 p=origin+direction*t;
            float density=rainbow_density(p);
            if (density<=0.0) break;
            throughput*=RainbowScattering()*density*exp(-rain_segment_tau(origin,direction,hit.x,t))/pdf;
            pathDirections[vertex]=direction;
            // Order one is already integrated by the original deterministic
            // renderer. Each later vertex adds precisely one higher order.
            if (vertex>=1)
            {
                vec3 earth=p+vec3(0.0,SkyEarthRadius()+1.0,0.0);
                if (ray_sphere(earth,sun,SkyEarthRadius()).x<=0.0)
                {
                    vec2 air=rainbow_sun_air_depth(earth,sun);
                    vec2 sunHit=rain_segment(p,sun);
                    float sunT=exp(-rain_segment_tau(p,sun,0.0,max(sunHit.y,0.0)));
                    for (int band=0;band<SPECTRAL_BANDS;++band)
                    {
                        Stokes s=stokes_zero();s.I=1.0;
                        s=stokes_scale(scatter_rainbow(s,-sun,-direction,band),1.0/(4.0*PI));
                        for (int previous=vertex-1;previous>=0;--previous)
                            s=stokes_scale(scatter_rainbow(s,-pathDirections[previous+1],-pathDirections[previous],band),1.0/(4.0*PI));
                        float weight=throughput*solidAngle*sunT*exp(-(SkyBetaR(band)*air.x+air.y))
                                     *SkySunRadiance(band)/float(samples);
                        result.bands[band]+=vec4(s.I,s.Q,s.U,s.V)*weight;
                    }
                }
            }
            if (vertex+1<orders)
            {
                vec4 sampled=rain_sample_direction(-direction,state);
                throughput/=sampled.w;
                origin=p;direction=-sampled.xyz;
            }
        }
    }
    return result;
}
