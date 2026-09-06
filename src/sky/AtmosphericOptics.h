#pragma once
#include "config/RuntimeConfig.h"
#include <array>
#include <vector>

// Computed eikonal trajectories in a radially stratified atmosphere. Separate
// angular grids on each side of the refracted horizon avoid interpolation
// between ground-terminating and escaping rays.
inline constexpr int kAtmosphereHeights = 48;
inline constexpr int kAtmosphereAnglesPerBranch = 128;
inline constexpr int kAtmosphereRaySamples = 33;
using AtmosphereRayEntry = std::array<float, 4>;
std::vector<AtmosphereRayEntry> ComputeAtmosphereRayTable(const SkySpectralConfig &sky);
bool HasAtmosphereGeometryChanged(const SkySpectralConfig &a, const SkySpectralConfig &b);
