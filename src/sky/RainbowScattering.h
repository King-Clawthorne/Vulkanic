#pragma once

#include "MieScattering.h"

#include <vector>

// Geometric-optics Mueller table for a population of
// spherical droplets. Includes external reflection, transmission, and primary /
// optional secondary internal-reflection families. Airy-scale broadening is
// an approximation; this is not a full-wave diffraction solver.
struct RainbowScatteringParams
{
    double effectiveRadiusMicrometers = 500.0;
    double effectiveVariance = 0.08;
    int angleBins = 4097;
    int raySamples = 65536;
    bool includeSecondary = true;
};

// Store the spherical Mueller matrix in the shared six-component layout.
// Spheres satisfy F22 = F11 and F44 = F33.
struct RainbowMatrixEntry
{
    float f11, f12, f22, f33;
    float f34, f44, cdf = 0.0f, unused = 0.0f;
};
static_assert(sizeof(RainbowMatrixEntry) == 32);

std::vector<RainbowMatrixEntry> ComputeRainbowScatteringTable(const RainbowScatteringParams &params);
