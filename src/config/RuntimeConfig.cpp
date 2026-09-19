// RuntimeConfig.cpp — config file handling. glaze reads and writes the JSON
// straight into the RuntimeConfig structs; only Vec3 and the upper-case sky
// constants need an explicit mapping.

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "RuntimeConfig.h"

#include <windows.h>

#include <glaze/glaze.hpp>

#include <algorithm>
#include <array>
#include <format>
#include <fstream>
#include <print>
#include <ranges>
#include <stdexcept>
#include <system_error>

// Search a fixed list of conventional locations for a runtime file
// (config / SPIR-V / etc.). Mirrors ResolveModelFilePath() but uses wide
// strings since these names are baked into the binary as L"..." literals.
std::filesystem::path ResolveRuntimeFilePath(const wchar_t* fileName)
{
    WCHAR exePath[MAX_PATH]{};
    const DWORD pathLen = GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    if (pathLen == 0 || pathLen == MAX_PATH) {
        throw std::runtime_error("Failed to resolve executable path.");
    }

    const auto exeDir = std::filesystem::path(exePath).parent_path();
    const std::array<std::filesystem::path, 3> candidatePaths = {
        exeDir / fileName,
        exeDir.parent_path() / fileName,
        std::filesystem::current_path() / fileName,
    };

    for (const auto& candidate : candidatePaths) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return {};
}

// Slurp a whole file into a std::string. Opened in binary mode so size
// reported by tellg() matches the byte count read; otherwise CRLF
// translation on Windows would silently shrink the buffer.
std::string LoadTextFile(const std::filesystem::path& filePath)
{
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open " + filePath.string());
    }

    const auto fileSize = file.tellg();
    if (fileSize <= 0) {
        throw std::runtime_error(filePath.string() + " is empty.");
    }

    std::string data(static_cast<size_t>(fileSize), '\0');
    file.seekg(0, std::ios::beg);
    file.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!file) {
        throw std::runtime_error("Failed to read " + filePath.string());
    }

    return data;
}

template <>
struct glz::meta<Vec3> {
    static constexpr auto value = glz::array(&Vec3::x, &Vec3::y, &Vec3::z);
};

template <>
struct glz::meta<SkySpectralConfig> {
    using T = SkySpectralConfig;
    static constexpr auto value = glz::object(
        "BETA_R_550", &T::betaRayleigh550, "BETA_M", &T::betaMie,
        "EARTH_R", &T::earthRadius, "ATMOS_R", &T::atmosphereRadius,
        "SCALE_H_R", &T::scaleHeightRayleigh, "SCALE_H_M", &T::scaleHeightMie,
        "SUN_TEMPERATURE_K", &T::sunTemperatureKelvin, "SUN_RADIANCE_550", &T::sunRadiance550,
        "SUN_DIRECTION", &T::sunDirection, "SUN_RADIUS", &T::sunRadius, "SUN_AA", &T::sunAa,
        "secondarySamples", &T::secondarySamples, "VIEW_STEPS", &T::viewSteps, "Samples", &T::samples,
        "SCATTERING_ORDERS", &T::scatteringOrders, "RAYLEIGH_DEPOLARIZATION", &T::rayleighDepolarization,
        "GROUND_ALBEDO", &T::groundAlbedo, "AEROSOL_IOR_REAL", &T::aerosolRefractiveIndexReal,
        "AEROSOL_IOR_IMAG", &T::aerosolRefractiveIndexImag, "AEROSOL_MEAN_RADIUS_UM", &T::aerosolMeanRadiusMicrometers,
        "AEROSOL_SIGMA", &T::aerosolSigma, "MIE_TABLE_ANGLE_BINS", &T::mieTableAngleBins);
};

template <>
struct glz::meta<SkyConfig> {
    static constexpr auto value = glz::object("exposure", &SkyConfig::exposure, "spectralConstants", &SkyConfig::spectral);
};

namespace {
    void FailIf(bool failed, const char* message)
    {
        if (failed) throw std::runtime_error(message);
    }
} // namespace

// Parse, then reject values the renderer cannot use. Unknown keys are ignored
// and missing ones keep their defaults.
RuntimeConfig ParseRuntimeConfig(const std::string& jsonText)
{
    RuntimeConfig config{};
    if (const auto error = glz::read<glz::opts{.error_on_unknown_keys = false}>(config, jsonText)) {
        throw std::runtime_error("Invalid config JSON: " + glz::format_error(error, jsonText));
    }

    const SkySpectralConfig& sky = config.sky.spectral;
    const auto& sunDir = sky.sunDirection;

    FailIf(config.render.width == 0 || config.render.height == 0,
           "\"width\" and \"height\" must be greater than 0.");
    FailIf(config.render.frameCount == 0, "\"frameCount\" must be greater than 0.");
    FailIf(config.render.samplesPerPixel == 0, "\"samplesPerPixel\" must be greater than 0.");
    FailIf(Length(config.camera.initialLookAt - config.camera.initialPosition) <= 0.001f,
           "\"initialPosition\" and \"initialLookAt\" must not be the same.");
    FailIf(config.camera.fovYDegrees <= 1.0f || config.camera.fovYDegrees >= 179.0f,
           "\"fovYDegrees\" must be between 1 and 179.");
    FailIf(config.input.mouseSensitivity < 0.0f || config.input.keyLookSpeed < 0.0f || config.input.polarizerRotateSpeed < 0.0f,
           "\"mouseSensitivity\", \"keyLookSpeed\", and \"polarizerRotateSpeed\" must be non-negative.");
    FailIf(config.camera.maxPitchDegrees <= 0.0f || config.camera.maxPitchDegrees >= 90.0f,
           "\"maxPitchDegrees\" must be greater than 0 and less than 90.");
    FailIf(config.sky.exposure <= 0.0f, "\"exposure\" must be greater than 0.");
    FailIf(sky.betaRayleigh550 < 0.0f || sky.betaMie < 0.0f,
           "Sky scattering coefficients must be non-negative.");
    FailIf(sky.earthRadius <= 0.0f || sky.atmosphereRadius <= 0.0f || sky.atmosphereRadius <= sky.earthRadius,
           "\"EARTH_R\" and \"ATMOS_R\" must be positive, and ATMOS_R must exceed EARTH_R.");
    FailIf(sky.scaleHeightRayleigh <= 0.0f || sky.scaleHeightMie <= 0.0f,
           "\"SCALE_H_R\" and \"SCALE_H_M\" must be greater than 0.");
    FailIf(sky.sunTemperatureKelvin <= 0.0f || sky.sunRadiance550 < 0.0f,
           "\"SUN_TEMPERATURE_K\" must be positive and \"SUN_RADIANCE_550\" non-negative.");
    FailIf(sky.sunRadius <= 0.0f || sky.sunAa < 0.0f,
           "\"SUN_RADIUS\" must be greater than 0 and \"SUN_AA\" must be non-negative.");
    FailIf(sunDir[0] * sunDir[0] + sunDir[1] * sunDir[1] + sunDir[2] * sunDir[2] <= 0.0f,
           "\"SUN_DIRECTION\" must be a non-zero vector.");
    FailIf(sky.secondarySamples == 0 || sky.viewSteps == 0 || sky.samples == 0,
           "\"secondarySamples\", \"VIEW_STEPS\", and \"Samples\" must be greater than 0.");
    FailIf(sky.scatteringOrders < 1 || sky.scatteringOrders > 4,
           "\"SCATTERING_ORDERS\" must be between 1 and 4.");
    FailIf(sky.rayleighDepolarization < 0.0f || sky.rayleighDepolarization >= 1.0f,
           "\"RAYLEIGH_DEPOLARIZATION\" must be in [0, 1).");
    FailIf(sky.groundAlbedo < 0.0f || sky.groundAlbedo > 1.0f,
           "\"GROUND_ALBEDO\" must be in [0, 1].");
    FailIf(sky.aerosolRefractiveIndexReal <= 0.0f || sky.aerosolRefractiveIndexImag < 0.0f,
           "Aerosol refractive index must have positive real part and non-negative imaginary part.");
    FailIf(sky.aerosolMeanRadiusMicrometers <= 0.0f || sky.aerosolSigma <= 1.0f,
           "\"AEROSOL_MEAN_RADIUS_UM\" must be > 0 and \"AEROSOL_SIGMA\" must be > 1.");
    FailIf(sky.mieTableAngleBins < 2, "\"MIE_TABLE_ANGLE_BINS\" must be at least 2.");
    const RainbowConfig& rainbow = config.rainbow;
    FailIf(rainbow.enabled > 1 || rainbow.includeSecondary > 1,
           "Rainbow enabled/includeSecondary must be 0 or 1.");
    FailIf(rainbow.radii.x <= 0.0f || rainbow.radii.y <= 0.0f || rainbow.radii.z <= 0.0f,
           "Rainbow radii must all be greater than 0.");
    FailIf(rainbow.edgeSoftness < 0.0f || rainbow.edgeSoftness >= 1.0f,
           "Rainbow edgeSoftness must be in [0, 1).");
    FailIf(rainbow.scatteringCoefficient < 0.0f || rainbow.extinctionCoefficient < 0.0f || rainbow.scatteringCoefficient > rainbow.extinctionCoefficient,
           "Rainbow coefficients must satisfy 0 <= scattering <= extinction.");
    FailIf(rainbow.effectiveRadiusMicrometers <= 0.0f || rainbow.effectiveVariance < 0.0f,
           "Rainbow effective radius must be positive and variance non-negative.");
    FailIf(rainbow.angleBins < 16 || rainbow.viewSteps == 0,
           "Rainbow angleBins must be at least 16 and viewSteps greater than 0.");
    FailIf(rainbow.scatteringOrders < 1 || rainbow.scatteringOrders > 4 || rainbow.multipleScatteringSamples < 1 || rainbow.multipleScatteringSamples > 64 || rainbow.multipleScatteringSteps < 1 || rainbow.multipleScatteringSteps > 64,
           "Rain orders must be 1-4; multiple scattering samples and steps must be 1-64.");
    return config;
}

std::string SerializeRuntimeConfig(const RuntimeConfig& config)
{
    std::string json;
    // Pretty-printed, but with vectors kept on one line like the hand-written file.
    struct Opts : glz::opts {
        bool new_lines_in_arrays = false;
    };
    if (glz::write<Opts{{.prettify = true}}>(config, json)) {
        throw std::runtime_error("Failed to serialize config.");
    }
    return json + "\n";
}

namespace {
    constexpr const wchar_t* kConfigFileName = L"path_tracer_config.json";

    std::filesystem::file_time_type LastWriteTime(const std::filesystem::path& path)
    {
        std::error_code errorCode;
        const auto time = std::filesystem::last_write_time(path, errorCode);
        if (errorCode) throw std::runtime_error("Failed to read config file timestamp.");
        return time;
    }
} // namespace

RuntimeConfig ConfigFile::Read()
{
    RuntimeConfig config = ParseRuntimeConfig(LoadTextFile(m_path));
    m_lastWriteTime = LastWriteTime(m_path);
    return config;
}

RuntimeConfig ConfigFile::Load()
{
    // Prefer the editable source-tree config when launched from the repo.
    // The build also copies a deployment config beside the executable,
    // but choosing that copy first makes source edits appear to require a
    // rebuild because the hot-reloader watches the copied file instead.
    const auto cwd = std::filesystem::current_path();
    for (const auto& candidate : {cwd / L"config" / kConfigFileName, cwd.parent_path() / L"config" / kConfigFileName}) {
        if (std::filesystem::exists(candidate)) {
            m_path = std::filesystem::absolute(candidate).lexically_normal();
            break;
        }
    }
    if (m_path.empty()) m_path = ResolveRuntimeFilePath(kConfigFileName);
    if (m_path.empty()) throw std::runtime_error("Failed to locate path_tracer_config.json.");

    DiscoverFiles();
    RuntimeConfig config = Read();
    std::println("[Config] Loaded {}", m_path.string());
    std::println("[Config] F2 cycles {} discovered config file(s); F5 saves GUI settings.", m_files.size());
    return config;
}

void ConfigFile::DiscoverFiles()
{
    m_files.clear();
    const auto cwd = std::filesystem::current_path();
    for (const auto& directory : {m_path.parent_path(), cwd / L"config", cwd.parent_path() / L"config"}) {
        std::error_code errorCode;
        if (!std::filesystem::is_directory(directory, errorCode)) continue;
        for (const auto& entry : std::filesystem::directory_iterator(directory, errorCode)) {
            if (errorCode) break;
            if (!entry.is_regular_file() || entry.path().extension() != L".json") continue;
            const auto path = std::filesystem::absolute(entry.path()).lexically_normal();
            if (std::ranges::find(m_files, path) == m_files.end()) m_files.push_back(path);
        }
    }
    std::ranges::sort(m_files);
    if (std::ranges::find(m_files, m_path) == m_files.end()) m_files.push_back(m_path);
    m_index = static_cast<size_t>(std::ranges::find(m_files, m_path) - m_files.begin());
}

std::optional<RuntimeConfig> ConfigFile::ReloadIfChanged()
{
    const auto now = std::chrono::steady_clock::now();
    if (m_path.empty() || now - m_lastPollTime < std::chrono::milliseconds(250)) return std::nullopt;
    m_lastPollTime = now;

    std::error_code errorCode;
    const auto currentWriteTime = std::filesystem::last_write_time(m_path, errorCode);
    if (errorCode || currentWriteTime == m_lastWriteTime) return std::nullopt;

    // Record the new time even on failure so a broken edit is reported once,
    // not every poll.
    m_lastWriteTime = currentWriteTime;
    try {
        RuntimeConfig config = Read();
        std::println("[Config] Reloaded {}", m_path.string());
        return config;
    } catch (const std::exception& error) {
        std::println(stderr, "[Config] Reload failed: {}", error.what());
        return std::nullopt;
    }
}

std::optional<RuntimeConfig> ConfigFile::CycleNext()
{
    DiscoverFiles();
    if (m_files.size() < 2) {
        std::println("[Config] No alternate JSON config files found.");
        return std::nullopt;
    }
    const size_t oldIndex = m_index;
    const std::filesystem::path oldPath = m_path;
    m_index = (m_index + 1) % m_files.size();
    m_path = m_files[m_index];
    try {
        RuntimeConfig config = Read();
        std::println("[Config] Switched to {}", m_path.string());
        return config;
    } catch (const std::exception& error) {
        m_index = oldIndex;
        m_path = oldPath;
        std::println(stderr, "[Config] Failed to switch config: {}", error.what());
        return std::nullopt;
    }
}

void ConfigFile::Save(const RuntimeConfig& config)
{
    try {
        std::ofstream file(m_path, std::ios::binary | std::ios::trunc);
        if (!file) throw std::runtime_error("Failed to open active config for saving.");
        const std::string json = SerializeRuntimeConfig(config);
        file.write(json.data(), static_cast<std::streamsize>(json.size()));
        file.close();
        if (!file) throw std::runtime_error("Failed to save active config.");
        m_lastWriteTime = LastWriteTime(m_path);
        std::println("[Config] Saved {}", m_path.string());
    } catch (const std::exception& error) {
        std::println(stderr, "[Config] Save failed: {}", error.what());
    }
}
