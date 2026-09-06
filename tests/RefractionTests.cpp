#include "sky/AtmosphericOptics.h"
#include "sky/MieScattering.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
void Check(bool pass, const char *message) { if (!pass) throw std::runtime_error(message); }
int main()
{
    try
    {
        SkySpectralConfig sky;
        const auto rays = ComputeAtmosphereRayTable(sky);
        constexpr int stride = 1 + kAtmosphereRaySamples, angles = 2 * kAtmosphereAnglesPerBranch;
        const double atmosphereHeight = sky.atmosphereRadius - sky.earthRadius;
        for (int h = 0; h < kAtmosphereHeights; h += 5)
        {
            const double altitude =
                std::max(0.01, std::expm1(std::log1p(atmosphereHeight) * h / (kAtmosphereHeights - 1)));
            const double radius = sky.earthRadius + altitude;
            for (int a = 0; a < angles; a += 11)
            {
                const size_t base = (h * angles + a) * stride;
                const auto &first = rays[base + 1];
                const double impact = (1 + sky.seaLevelRefractivity * std::exp(-altitude / sky.refractionScaleHeight)) *
                                      radius * first[2];
                for (int t = 0; t < kAtmosphereRaySamples; ++t)
                {
                    const auto &entry = rays[base + 1 + t];
                    const double x = entry[0], y = radius + entry[1], r = std::hypot(x, y);
                    const double n = 1 + sky.seaLevelRefractivity *
                                             std::exp(-std::max(r - sky.earthRadius, 0.0) / sky.refractionScaleHeight);
                    const double b = n * (y * entry[2] - x * entry[3]);
                    Check(std::isfinite(b) && std::abs(b - impact) < std::max(2.0, std::abs(impact) * 1e-6),
                          "Eikonal n*r*sin(theta) invariant");
                    Check(std::abs(std::hypot(entry[2], entry[3]) - 1) < 1e-6, "Ray tangent is unit length");
                }
            }
        }
        const size_t vertical = (angles - 1) * stride;
        const double expected = sky.scaleHeightRayleigh * (std::exp(-0.01 / sky.scaleHeightRayleigh) -
                                                           std::exp(-atmosphereHeight / sky.scaleHeightRayleigh));
        Check(std::abs(rays[vertical][1] / expected - 1) < 0.001, "Vertical optical depth analytic limit");
        sky.seaLevelRefractivity = 0;
        const auto straight = ComputeAtmosphereRayTable(sky);
        for (int a = kAtmosphereAnglesPerBranch; a < angles; a += 13)
        {
            const size_t base = a * stride;
            const auto &begin = straight[base + 1];
            for (int t = 1; t < kAtmosphereRaySamples; ++t)
            {
                const auto &entry = straight[base + 1 + t];
                Check(std::abs(entry[2] - begin[2]) < 1e-7 && std::abs(entry[3] - begin[3]) < 1e-7,
                      "Zero refractivity straight-ray limit");
            }
        }

        const auto defaults=ParseRuntimeConfig("{}");
        const auto zero=ParseRuntimeConfig(R"({"sky":{"spectralConstants":{"SEA_LEVEL_REFRACTIVITY":0}}})");
        Check(defaults.skySpectral.seaLevelRefractivity>0 && zero.skySpectral.seaLevelRefractivity==0,"Refraction config");
        for (const char* invalid : {R"({"sky":{"spectralConstants":{"SEA_LEVEL_REFRACTIVITY":-1}}})",
                                   R"({"sky":{"spectralConstants":{"REFRACTION_SCALE_HEIGHT":0}}})",
                                   R"({"sky":{"spectralConstants":{"SEA_LEVEL_REFRACTIVITY":0.001,"REFRACTION_SCALE_HEIGHT":1000}}})"})
        {
            bool rejected=false;
            try { (void)ParseRuntimeConfig(invalid); } catch(const std::exception&) { rejected=true; }
            Check(rejected,"Invalid refractive profile rejected");
        }
        std::cout << "PASS: refracted ray invariants, analytic references, and config validation.\n";
        return 0;
    }
    catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
