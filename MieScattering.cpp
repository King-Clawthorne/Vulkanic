#include "MieScattering.h"

#include <algorithm>
#include <cmath>
#include <complex>

// Lorenz–Mie implementation following Bohren & Huffman, "Absorption and
// Scattering of Light by Small Particles" (1983), chapter 4. Everything is
// done in double precision on the CPU; the result is a small table the shader
// samples, so cost here is irrelevant (a few hundred thousand term-evaluations
// at startup).

namespace
{
constexpr double kPi = 3.14159265358979323846;
using Complex = std::complex<double>;

// Number of Mie terms to sum (Wiscombe 1980 convergence criterion) plus a
// small safety margin.
int MieTermCount(double x)
{
    const int n = static_cast<int>(x + 4.0 * std::cbrt(std::max(x, 1e-8)) + 2.0);
    return n + 2;
}

// Single-particle Mie coefficients a_n, b_n for size parameter x and relative
// (complex) refractive index m. The logarithmic derivative D_n(mx) is built by
// downward recurrence (stable); the real-argument Riccati–Bessel functions
// psi_n / chi_n by upward recurrence.
void MieCoefficients(double x, Complex m, std::vector<Complex>& a, std::vector<Complex>& b)
{
    const int nmax = MieTermCount(x);
    const Complex mx = m * x;

    // Downward recurrence for D_n(mx): D_{n-1} = n/mx - 1/(D_n + n/mx).
    const int nstart = nmax + 15;
    std::vector<Complex> D(static_cast<size_t>(nstart) + 1u, Complex(0.0, 0.0));
    for (int n = nstart; n > 0; --n)
    {
        const Complex nOverMx = Complex(static_cast<double>(n), 0.0) / mx;
        D[static_cast<size_t>(n) - 1u] = nOverMx - Complex(1.0, 0.0) / (D[static_cast<size_t>(n)] + nOverMx);
    }

    a.assign(static_cast<size_t>(nmax) + 1u, Complex(0.0, 0.0));
    b.assign(static_cast<size_t>(nmax) + 1u, Complex(0.0, 0.0));

    // Riccati–Bessel seeds: psi_{-1}=cos x, psi_0=sin x; chi_{-1}=-sin x, chi_0=cos x.
    double psiPrev = std::cos(x);
    double psi = std::sin(x);
    double chiPrev = -std::sin(x);
    double chi = std::cos(x);

    for (int n = 1; n <= nmax; ++n)
    {
        const double dn = static_cast<double>(n);
        const double psiN = (2.0 * dn - 1.0) / x * psi - psiPrev;
        const double chiN = (2.0 * dn - 1.0) / x * chi - chiPrev;
        const Complex ksiN(psiN, -chiN);    // ksi_n = psi_n - i chi_n
        const Complex ksiPrev(psiPrev, -chiPrev);

        const Complex Dn = D[static_cast<size_t>(n)];
        const Complex ta = Dn / m + Complex(dn / x, 0.0);
        a[static_cast<size_t>(n)] = (ta * psiN - psiPrev) / (ta * ksiN - ksiPrev);
        const Complex tb = Dn * m + Complex(dn / x, 0.0);
        b[static_cast<size_t>(n)] = (tb * psiN - psiPrev) / (tb * ksiN - ksiPrev);

        psiPrev = psi;
        psi = psiN;
        chiPrev = chi;
        chi = chiN;
    }
}
} // namespace

std::vector<MieMatrixEntry> ComputeMieScatteringTable(const MieAerosolParams& params)
{
    const int bins = std::max(2, params.angleBins);
    const Complex m(params.refractiveIndexReal, std::max(0.0, params.refractiveIndexImag));

    // Cache scattering-angle cosines / sines.
    std::vector<double> mu(static_cast<size_t>(bins));
    std::vector<double> sinTheta(static_cast<size_t>(bins));
    for (int i = 0; i < bins; ++i)
    {
        const double theta = kPi * static_cast<double>(i) / static_cast<double>(bins - 1);
        mu[static_cast<size_t>(i)] = std::cos(theta);
        sinTheta[static_cast<size_t>(i)] = std::sin(theta);
    }

    // Log-normal number distribution, integrated in ln(r) so samples are
    // geometrically spaced around the geometric mean radius. Constant
    // prefactors cancel in the F11 phase normalization below, so we omit them.
    const int radiusSamples = 48;
    const double lnSigma = std::log(std::max(1.0001, params.sigma));
    const double lnRg = std::log(std::max(1e-4, params.meanRadiusMicrometers));
    const double lnMin = lnRg - 4.0 * lnSigma;
    const double lnMax = lnRg + 4.0 * lnSigma;
    const double dLn = (lnMax - lnMin) / static_cast<double>(radiusSamples - 1);

    std::vector<MieMatrixEntry> table(static_cast<size_t>(bins) * 3u);

    for (int band = 0; band < 3; ++band)
    {
        const double lambdaUm = params.wavelengthsNmRgb[band] * 1e-3; // nm -> µm
        const double k = 2.0 * kPi / lambdaUm;

        std::vector<double> p11(static_cast<size_t>(bins), 0.0);
        std::vector<double> p12(static_cast<size_t>(bins), 0.0);
        std::vector<double> p33(static_cast<size_t>(bins), 0.0);
        std::vector<double> p34(static_cast<size_t>(bins), 0.0);

        for (int rs = 0; rs < radiusSamples; ++rs)
        {
            const double lnR = lnMin + dLn * static_cast<double>(rs);
            const double z = (lnR - lnRg) / lnSigma;
            const double weight = std::exp(-0.5 * z * z) * dLn;
            if (weight < 1e-12)
            {
                continue;
            }

            const double r = std::exp(lnR); // µm
            const double x = k * r;
            std::vector<Complex> a;
            std::vector<Complex> b;
            MieCoefficients(x, m, a, b);
            const int nmax = static_cast<int>(a.size()) - 1;

            for (int i = 0; i < bins; ++i)
            {
                const double u = mu[static_cast<size_t>(i)];

                // Angular functions: pi_0 = 0, pi_1 = 1, tau_1 = mu.
                double piPrev = 0.0; // pi_{n-1}
                double piCur = 1.0;  // pi_n, starts at pi_1
                Complex s1(0.0, 0.0);
                Complex s2(0.0, 0.0);

                for (int n = 1; n <= nmax; ++n)
                {
                    const double dn = static_cast<double>(n);
                    const double tauCur = dn * u * piCur - (dn + 1.0) * piPrev;
                    const double fn = (2.0 * dn + 1.0) / (dn * (dn + 1.0));
                    s1 += fn * (a[static_cast<size_t>(n)] * piCur + b[static_cast<size_t>(n)] * tauCur);
                    s2 += fn * (a[static_cast<size_t>(n)] * tauCur + b[static_cast<size_t>(n)] * piCur);

                    const double piNext = ((2.0 * dn + 1.0) * u * piCur - (dn + 1.0) * piPrev) / dn;
                    piPrev = piCur;
                    piCur = piNext;
                }

                const double i1 = std::norm(s1); // |S1|^2
                const double i2 = std::norm(s2); // |S2|^2
                const Complex cross = s2 * std::conj(s1);
                p11[static_cast<size_t>(i)] += weight * 0.5 * (i2 + i1);
                p12[static_cast<size_t>(i)] += weight * 0.5 * (i2 - i1);
                p33[static_cast<size_t>(i)] += weight * cross.real();
                p34[static_cast<size_t>(i)] += weight * cross.imag();
            }
        }

        // Normalize so (1/4π)∫F11 dΩ = 1  ⇒  ∫F11 dΩ = 4π, matching the
        // Rayleigh phase-matrix convention used in sky.comp. The same factor
        // scales F12/F33/F34 so the polarization ratios are preserved.
        const double dTheta = kPi / static_cast<double>(bins - 1);
        double integral = 0.0;
        for (int i = 0; i + 1 < bins; ++i)
        {
            const double fL = p11[static_cast<size_t>(i)] * sinTheta[static_cast<size_t>(i)];
            const double fR = p11[static_cast<size_t>(i) + 1u] * sinTheta[static_cast<size_t>(i) + 1u];
            integral += 0.5 * (fL + fR) * dTheta;
        }
        const double norm = std::max(0.5 * integral, 1e-20);

        // Non-sphericity blend in [0,1]. Real non-spherical particles (dust,
        // ice) break the spherical F22=F11 / F44=F33 identities; the departure
        // is largest at side/back scattering. We model that phenomenologically
        // with a backscatter-weighted reduction w(θ) = (1 - cosθ)/2, which
        // produces a linear depolarization ratio δ = (F11-F22)/(F11+F22) that
        // rises from 0 (forward) toward n at backscatter. n=0 recovers spheres.
        const double n = std::clamp(params.nonSphericity, 0.0, 1.0);

        for (int i = 0; i < bins; ++i)
        {
            MieMatrixEntry& entry = table[static_cast<size_t>(band) * static_cast<size_t>(bins) + static_cast<size_t>(i)];
            const double f11 = p11[static_cast<size_t>(i)] / norm;
            const double f33 = p33[static_cast<size_t>(i)] / norm;
            const double w = 0.5 * (1.0 - mu[static_cast<size_t>(i)]); // 0 forward → 1 back
            entry.f11 = static_cast<float>(f11);
            entry.f12 = static_cast<float>(p12[static_cast<size_t>(i)] / norm);
            entry.f33 = static_cast<float>(f33);
            entry.f34 = static_cast<float>(p34[static_cast<size_t>(i)] / norm);
            // Spheres: f22=f11, f44=f33. Non-sphericity damps F22 (linear
            // depolarization) and damps F44 about twice as fast (circular
            // depolarization is typically stronger for irregular particles).
            entry.f22 = static_cast<float>(f11 * (1.0 - n * w));
            entry.f44 = static_cast<float>(f33 * (1.0 - 2.0 * n * w));
            entry.pad0 = 0.0f;
            entry.pad1 = 0.0f;
        }
    }

    return table;
}
