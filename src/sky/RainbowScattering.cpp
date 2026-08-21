#include "RainbowScattering.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace
{
constexpr double kPi = std::numbers::pi_v<double>;

// Visible-spectrum liquid-water dispersion (air-relative refractive index),
// a compact Cauchy fit anchored near Fraunhofer visible lines.
double WaterIor(double wavelengthNm)
{
    const double lambdaUm = wavelengthNm * 1.0e-3;
    return 1.32292 + 0.00306 / (lambdaUm * lambdaUm);
}

struct FresnelPower
{
    double reflectS;
    double reflectP;
    double transmitS;
    double transmitP;
};

FresnelPower AirToWaterFresnel(double incidence, double refraction, double n)
{
    const double ci = std::cos(incidence);
    const double cr = std::cos(refraction);
    const double rs = (ci - n * cr) / (ci + n * cr);
    const double rp = (n * ci - cr) / (n * ci + cr);
    const double reflectS = rs * rs;
    const double reflectP = rp * rp;
    // For a complete air->water->air path, using 1-R at both interfaces
    // preserves power without needing the reciprocal radiance factors.
    return {reflectS, reflectP, 1.0 - reflectS, 1.0 - reflectP};
}

void Deposit(std::vector<double>& f11, std::vector<double>& f12,
             std::vector<double>& f33, double theta, double weightS,
             double weightP, double sigma, int bins)
{
    theta = std::acos(std::clamp(std::cos(theta), -1.0, 1.0));
    const double scale = static_cast<double>(bins - 1) / kPi;
    const double centre = theta * scale;
    const double sigmaBins = std::max(sigma * scale, 0.65);
    const int radius = std::max(2, static_cast<int>(std::ceil(4.0 * sigmaBins)));
    const int first = std::max(0, static_cast<int>(std::floor(centre)) - radius);
    const int last = std::min(bins - 1, static_cast<int>(std::floor(centre)) + radius);
    for (int bin = first; bin <= last; ++bin)
    {
        const double x = (static_cast<double>(bin) - centre) / sigmaBins;
        const double kernel = std::exp(-0.5 * x * x);
        const double is = weightS * kernel;
        const double ip = weightP * kernel;
        f11[static_cast<size_t>(bin)] += 0.5 * (is + ip);
        f12[static_cast<size_t>(bin)] += 0.5 * (is - ip);
        f33[static_cast<size_t>(bin)] += std::sqrt(std::max(is * ip, 0.0));
    }
}
} // namespace

std::vector<MieMatrixEntry> ComputeRainbowScatteringTable(const RainbowScatteringParams& params)
{
    if (params.angleBins < 16 || params.effectiveRadiusMicrometers <= 0.0
        || params.effectiveVariance < 0.0 || params.solarAngularRadiusRadians < 0.0)
    {
        throw std::runtime_error("Invalid rainbow scattering-table parameters.");
    }

    const int bins = params.angleBins;
    std::vector<MieMatrixEntry> table(static_cast<size_t>(kSpectralBandCount * bins));
    constexpr int raySamples = 32768;

    for (int band = 0; band < kSpectralBandCount; ++band)
    {
        const double wavelengthNm = kSpectralLambdaMinNm + kSpectralLambdaStepNm * band;
        const double n = WaterIor(wavelengthNm);
        const double radiusM = params.effectiveRadiusMicrometers * 1.0e-6;
        // Airy-caustic angular scale is proportional to (lambda/r)^(2/3).
        // Size variance broadens the caustic and suppresses supernumeraries.
        const double airyWidth = 0.55 * std::pow(wavelengthNm * 1.0e-9 / radiusM, 2.0 / 3.0);
        const double blur = std::sqrt(params.solarAngularRadiusRadians * params.solarAngularRadiusRadians
                                      + airyWidth * airyWidth * (1.0 + 8.0 * params.effectiveVariance));
        std::vector<double> f11(static_cast<size_t>(bins), 0.0);
        std::vector<double> f12(static_cast<size_t>(bins), 0.0);
        std::vector<double> f33(static_cast<size_t>(bins), 0.0);

        for (int sample = 0; sample < raySamples; ++sample)
        {
            // Uniform projected disk area: b^2 is uniform. The histogram
            // therefore already carries the correct incident-power measure.
            const double b = std::sqrt((static_cast<double>(sample) + 0.5) / raySamples);
            const double incidence = std::asin(std::min(b, 1.0));
            const double refraction = std::asin(b / n);
            const FresnelPower fr = AirToWaterFresnel(incidence, refraction, n);

            // One internal reflection (Debye p=2): D = pi + 2i - 4r.
            const double primary = kPi + 2.0 * incidence - 4.0 * refraction;
            const double primaryS = fr.transmitS * fr.transmitS * fr.reflectS;
            const double primaryP = fr.transmitP * fr.transmitP * fr.reflectP;
            Deposit(f11, f12, f33, primary, primaryS, primaryP, blur, bins);

            if (params.includeSecondary)
            {
                // Two internal reflections (Debye p=3). Folding through acos
                // maps the >pi deviation to its physical scattering angle.
                const double secondary = 2.0 * kPi + 2.0 * incidence - 6.0 * refraction;
                const double secondaryS = primaryS * fr.reflectS;
                const double secondaryP = primaryP * fr.reflectP;
                Deposit(f11, f12, f33, secondary, secondaryS, secondaryP, blur, bins);
            }
        }

        double integral = 0.0;
        const double dTheta = kPi / static_cast<double>(bins - 1);
        for (int i = 0; i + 1 < bins; ++i)
        {
            const double a = f11[static_cast<size_t>(i)] * std::sin(i * dTheta);
            const double b = f11[static_cast<size_t>(i + 1)] * std::sin((i + 1) * dTheta);
            integral += 0.5 * (a + b) * dTheta;
        }
        const double normalization = std::max(0.5 * integral, 1.0e-30);
        for (int i = 0; i < bins; ++i)
        {
            table[static_cast<size_t>(band * bins + i)] = {
                static_cast<float>(f11[static_cast<size_t>(i)] / normalization),
                static_cast<float>(f12[static_cast<size_t>(i)] / normalization),
                static_cast<float>(f33[static_cast<size_t>(i)] / normalization),
                0.0f,
            };
        }
    }
    return table;
}
