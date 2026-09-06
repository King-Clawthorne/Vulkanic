#include "config/RuntimeConfig.h"
#include "sky/AtmosphericOptics.h"
#include "sky/DropOptics.h"
#include "sky/RainbowScattering.h"
#include <iostream>
#include <stdexcept>

namespace
{
void Check(bool pass, const char *message)
{
    if (!pass)
        throw std::runtime_error(message);
}
void CheckPhase(const std::vector<RainbowMatrixEntry> &table, int bins)
{
    for (int band = 0; band < kSpectralBandCount; ++band)
    {
        double integral = 0;
        for (int i = 0; i < bins - 1; ++i)
            integral += kPi * kPi / (bins - 1) *
                        (table[band * bins + i].f11 * std::sin(kPi * i / (bins - 1)) +
                         table[band * bins + i + 1].f11 * std::sin(kPi * (i + 1) / (bins - 1)));
        Check(std::abs(integral / (4 * kPi) - 1) < 1e-5, "Phase energy normalization");
        for (int i = 0; i < bins; ++i)
        {
            const auto &m = table[band * bins + i];
            Check(m.cdf >= 0 && m.cdf <= 1.000001f, "Phase sampling CDF range");
            if (i > 0) Check(m.cdf >= table[band * bins + i - 1].cdf, "Phase sampling CDF monotonicity");
            Check(std::isfinite(m.f11) && m.f11 >= 0, "Finite nonnegative phase");
            // Fully polarized incident states sample the boundary of the
            // Stokes cone; the Mueller matrix must preserve it.
            for (int q = 0; q < 32; ++q)
            {
                const double z = 1 - 2 * (q + 0.5) / 32, phi = q * 2.3999632297;
                const double x = std::sqrt(1 - z * z) * std::cos(phi), y = std::sqrt(1 - z * z) * std::sin(phi);
                const double I = m.f11 + m.f12 * x, Q = m.f12 + m.f22 * x;
                const double U = m.f33 * y + m.f34 * z, V = -m.f34 * y + m.f44 * z;
                Check(I >= -1e-8 && Q * Q + U * U + V * V <= I * I + 1e-5 * m.f11 * m.f11 + 1e-12,
                      "Mueller preserves physical polarization");
            }
        }
    }
}
} // namespace
int main()
{
    try
    {
        using namespace drop_optics;
        const Vector normal{0, 0, -1};
        Beam incident;
        const Beam reflected = Interface(incident, normal, 1, 1.333, true);
        const Beam transmitted = Interface(incident, normal, 1, 1.333, false);
        Check(std::abs(Mueller(reflected)[0] + Mueller(transmitted)[0] - 1) < 1e-12, "Fresnel energy");
        incident.direction = Unit(Vector{0.8, 0, 0.6});
        incident.a = Unit(Vector{0.6, 0, -0.8});
        Check(std::abs(Mueller(Interface(incident, normal, 1.333, 1, true))[0] - 1) < 1e-12,
              "Total internal reflection");

        RainbowScatteringParams p;
        p.angleBins = 1025;
        p.raySamples = 32768;
        const auto sphere = ComputeRainbowScatteringTable(p);
        CheckPhase(sphere, p.angleBins);
        auto peak = [&](int band) {
            int best = int(130.0 / 180 * (p.angleBins - 1));
            for (int i = best + 1; i <= int(150.0 / 180 * (p.angleBins - 1)); ++i)
                if (sphere[band * p.angleBins + i].f11 > sphere[band * p.angleBins + best].f11)
                    best = i;
            return best * 180.0 / (p.angleBins - 1);
        };
        Check(peak(12) < peak(0) && peak(12) > 136 && peak(0) < 143, "Spherical primary bow angle / dispersion");
        for (const auto &m : sphere)
            Check(m.f22 == m.f11 && m.f44 == m.f33, "Spherical Mueller identities");

        auto config = ParseRuntimeConfig(
            R"({"rainbow":{"viewSteps":3,"scatteringOrders":2},"sky":{"spectralConstants":{"VIEW_STEPS":7,"SCATTERING_ORDERS":4}}})");
        Check(config.skySpectral.viewSteps == 7 && config.skySpectral.scatteringOrders == 4
                  && config.rainbow.viewSteps == 3 && config.rainbow.scatteringOrders == 2,
              "Independent atmosphere and rain controls");
        const auto defaults = ParseRuntimeConfig("{}");
        const auto skyOnly = ParseRuntimeConfig(
            R"({"sky":{"spectralConstants":{"VIEW_STEPS":9,"SCATTERING_ORDERS":4}}})");
        Check(defaults.rainbow.viewSteps == skyOnly.rainbow.viewSteps
                  && defaults.rainbow.scatteringOrders == skyOnly.rainbow.scatteringOrders,
              "Rain defaults do not inherit sky settings");
        for (const char *invalid : {R"({"rainbow":{"viewSteps":0}})",
                                    R"({"rainbow":{"scatteringOrders":0}})",
                                    R"({"rainbow":{"scatteringOrders":5}})"})
        {
            bool rejected = false;
            try { (void)ParseRuntimeConfig(invalid); }
            catch (const std::exception &) { rejected = true; }
            Check(rejected, "Invalid rain quality rejected");
        }
        const float initialScattering = config.rainbow.scatteringCoefficient,
                    initialExtinction = config.rainbow.extinctionCoefficient;
        SetRainbowScattering(config.rainbow, 5e-4f);
        SetRainbowScattering(config.rainbow, 0);
        SetRainbowScattering(config.rainbow, initialScattering);
        Check(std::abs(config.rainbow.extinctionCoefficient - initialExtinction) < 1e-9,
              "Reversible scattering slider");

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
        std::cout
            << "PASS: Fresnel/TIR, spherical bow dispersion, spherical Mueller positivity and normalization, config "
               "ownership, slider reversibility, refractive invariants, analytic optical depth, straight-ray limit.\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
