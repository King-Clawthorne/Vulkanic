#include "RainbowScattering.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace
{
constexpr double kPi = std::numbers::pi_v<double>;

// Visible liquid-water dispersion sampled at the renderer's thirteen bands.
// Values follow the smooth room-temperature Hale-Querry visible trend; linear
// interpolation keeps this routine usable if the spectral grid is refined.
double WaterIor(double wavelengthNm)
{
    constexpr std::array<double, kSpectralBandCount> values = {
        1.34350, 1.34055, 1.33795, 1.33570, 1.33370, 1.33225, 1.33110,
        1.33020, 1.32945, 1.32885, 1.32835, 1.32795, 1.32760,
    };
    const double position = std::clamp((wavelengthNm - kSpectralLambdaMinNm)
                                           / kSpectralLambdaStepNm,
                                       0.0, static_cast<double>(kSpectralBandCount - 1));
    const int lower = static_cast<int>(std::floor(position));
    const int upper = std::min(lower + 1, kSpectralBandCount - 1);
    return std::lerp(values[static_cast<size_t>(lower)], values[static_cast<size_t>(upper)],
                     position - lower);
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
        // Match MieScattering.cpp and rayleigh_mueller(): Q is parallel minus
        // perpendicular intensity, so an s-dominated bow has negative F12.
        f12[static_cast<size_t>(bin)] += 0.5 * (ip - is);
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
    constexpr int raySamples = 16384;
    constexpr int radiusSamples = 9;

    for (int band = 0; band < kSpectralBandCount; ++band)
    {
        const double wavelengthNm = kSpectralLambdaMinNm + kSpectralLambdaStepNm * band;
        const double n = WaterIor(wavelengthNm);
        std::vector<double> f11(static_cast<size_t>(bins), 0.0);
        std::vector<double> f12(static_cast<size_t>(bins), 0.0);
        std::vector<double> f33(static_cast<size_t>(bins), 0.0);

        // Interpret effectiveVariance as a real log-normal number distribution.
        // For that distribution v_eff=exp(sigma_ln^2)-1 and
        // r_eff=M3/M2=r_g*exp(2.5*sigma_ln^2). Each radius is then weighted by
        // its projected cross-section (r^2), not merely by droplet count.
        const double sigmaLn = std::sqrt(std::log1p(params.effectiveVariance));
        const double geometricRadiusUm = params.effectiveRadiusMicrometers
                                       / std::exp(2.5 * sigmaLn * sigmaLn);
        double radiusWeightSum = 0.0;
        std::array<double, radiusSamples> radiiUm{};
        std::array<double, radiusSamples> radiusWeights{};
        for (int radiusIndex = 0; radiusIndex < radiusSamples; ++radiusIndex)
        {
            const double z = sigmaLn > 1.0e-8
                ? -3.5 + 7.0 * (static_cast<double>(radiusIndex) + 0.5) / radiusSamples
                : 0.0;
            const double radiusUm = geometricRadiusUm * std::exp(sigmaLn * z);
            const double numberWeight = sigmaLn > 1.0e-8 ? std::exp(-0.5 * z * z) : 1.0;
            const double weight = numberWeight * radiusUm * radiusUm;
            radiiUm[static_cast<size_t>(radiusIndex)] = radiusUm;
            radiusWeights[static_cast<size_t>(radiusIndex)] = weight;
            radiusWeightSum += weight;
        }

        for (int radiusIndex = 0; radiusIndex < radiusSamples; ++radiusIndex)
        {
            const double radiusM = radiiUm[static_cast<size_t>(radiusIndex)] * 1.0e-6;
            const double radiusWeight = radiusWeights[static_cast<size_t>(radiusIndex)]
                                      / radiusWeightSum;
            // Airy-caustic scale for this actual radius. A uniform solar disk
            // has one-axis variance R^2/4, hence the R/2 Gaussian equivalent.
            const double airyWidth = 0.55 * std::pow(wavelengthNm * 1.0e-9 / radiusM, 2.0 / 3.0);
            const double solarSigma = 0.5 * params.solarAngularRadiusRadians;
            const double blur = std::sqrt(solarSigma * solarSigma + airyWidth * airyWidth);

            for (int sample = 0; sample < raySamples; ++sample)
            {
                // Uniform projected disk area: b^2 is uniform. The histogram
                // therefore carries the correct incident-power measure.
                const double b = std::sqrt((static_cast<double>(sample) + 0.5) / raySamples);
                const double incidence = std::asin(std::min(b, 1.0));
                const double refraction = std::asin(b / n);
                const FresnelPower fr = AirToWaterFresnel(incidence, refraction, n);
                const double primary = kPi + 2.0 * incidence - 4.0 * refraction;
                const double primaryS = radiusWeight * fr.transmitS * fr.transmitS * fr.reflectS;
                const double primaryP = radiusWeight * fr.transmitP * fr.transmitP * fr.reflectP;
                Deposit(f11, f12, f33, primary, primaryS, primaryP, blur, bins);

                if (params.includeSecondary)
                {
                    const double secondary = 2.0 * kPi + 2.0 * incidence - 6.0 * refraction;
                    const double secondaryS = primaryS * fr.reflectS;
                    const double secondaryP = primaryP * fr.reflectP;
                    Deposit(f11, f12, f33, secondary, secondaryS, secondaryP, blur, bins);
                }
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
        if (!std::isfinite(integral) || integral <= 1.0e-20)
            throw std::runtime_error("Rainbow phase table has invalid energy normalization.");
        const double normalization = 0.5 * integral;
        for (int i = 0; i < bins; ++i)
        {
            const double normalizedF11 = f11[static_cast<size_t>(i)] / normalization;
            const double normalizedF12 = f12[static_cast<size_t>(i)] / normalization;
            const double normalizedF33 = f33[static_cast<size_t>(i)] / normalization;
            if (!std::isfinite(normalizedF11) || normalizedF11 < 0.0
                || std::abs(normalizedF12) > normalizedF11 * 1.000001
                || std::abs(normalizedF33) > normalizedF11 * 1.000001)
                throw std::runtime_error("Rainbow phase table contains a non-physical Mueller entry.");
            table[static_cast<size_t>(band * bins + i)] = {
                static_cast<float>(normalizedF11),
                static_cast<float>(normalizedF12),
                static_cast<float>(normalizedF33),
                0.0f,
            };
        }
    }

    // Guard the defining primary-bow dispersion: red must occur at a smaller
    // scattering angle (larger radius from the antisolar point) than violet.
    const int primaryFirst = static_cast<int>(130.0 / 180.0 * (bins - 1));
    const int primaryLast = static_cast<int>(150.0 / 180.0 * (bins - 1));
    const auto primaryPeak = [&](int band)
    {
        int peak = primaryFirst;
        for (int i = primaryFirst + 1; i <= primaryLast; ++i)
            if (table[static_cast<size_t>(band * bins + i)].f11
                > table[static_cast<size_t>(band * bins + peak)].f11)
                peak = i;
        return peak;
    };
    if (primaryPeak(kSpectralBandCount - 1) >= primaryPeak(0))
        throw std::runtime_error("Rainbow table failed primary-bow spectral ordering.");
    return table;
}
