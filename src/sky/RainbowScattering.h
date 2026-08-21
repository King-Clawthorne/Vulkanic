#pragma once

#include "MieScattering.h"

#include <vector>

// CPU-baked polarized large-water-droplet phase table. The baker traces the
// one- and two-internal-reflection Debye rays through a sphere, applies the
// Fresnel s/p powers, and angularly broadens them by the finite solar disk and
// an Airy-scale term. Entries use the same normalized Mueller convention as
// the atmospheric Mie table: integral(F11 dOmega) = 4*pi.
struct RainbowScatteringParams
{
    double effectiveRadiusMicrometers = 500.0;
    double effectiveVariance = 0.08;
    double solarAngularRadiusRadians = 0.00465;
    int angleBins = 4097;
    bool includeSecondary = true;
};

std::vector<MieMatrixEntry> ComputeRainbowScatteringTable(const RainbowScatteringParams& params);
