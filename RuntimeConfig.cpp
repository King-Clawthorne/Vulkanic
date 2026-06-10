// RuntimeConfig.cpp — JSON config loader.
//
// Implements a tiny, dependency-free JSON parser (JsonParser) plus a set of
// strongly-typed converters that map the parsed tree onto the
// RuntimeConfig struct. The parser supports the subset of RFC 8259 actually
// used by path_tracer_config.json: objects, arrays, strings (with the
// standard escape set, no \uXXXX), doubles via std::strtod, true/false/null,
// and arbitrary whitespace. All errors carry a human-readable context
// string identifying the offending JSON path.

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "RuntimeConfig.h"

#include <windows.h>

#include <array>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>

// Search a fixed list of conventional locations for a runtime file
// (config / SPIR-V / etc.). Mirrors ResolveModelFilePath() but uses wide
// strings since these names are baked into the binary as L"..." literals.
std::filesystem::path ResolveRuntimeFilePath(const wchar_t* fileName)
{
    WCHAR exePath[MAX_PATH]{};
    const DWORD pathLen = GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    if (pathLen == 0 || pathLen == MAX_PATH)
    {
        throw std::runtime_error("Failed to resolve executable path.");
    }

    const auto exeDir = std::filesystem::path(exePath).parent_path();
    const std::array<std::filesystem::path, 3> candidatePaths = {
        exeDir / fileName,
        exeDir.parent_path() / fileName,
        std::filesystem::current_path() / fileName,
    };

    for (const auto& candidate : candidatePaths)
    {
        if (std::filesystem::exists(candidate))
        {
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
    if (!file)
    {
        throw std::runtime_error("Failed to open text file.");
    }

    const auto fileSize = file.tellg();
    if (fileSize <= 0)
    {
        throw std::runtime_error("Text file is empty.");
    }

    std::string data(static_cast<size_t>(fileSize), '\0');
    file.seekg(0, std::ios::beg);
    file.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!file)
    {
        throw std::runtime_error("Failed to read text file.");
    }

    return data;
}

namespace
{
// Generic JSON node. The std::variant carries the actual payload; the
// AsXxx() helpers fail loudly with a context string when a node has the
// wrong type, which keeps the call sites concise.
struct JsonValue
{
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;

    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> data = nullptr;

    JsonValue() = default;
    explicit JsonValue(std::nullptr_t value) : data(value) {}
    explicit JsonValue(bool value) : data(value) {}
    explicit JsonValue(double value) : data(value) {}
    explicit JsonValue(std::string value) : data(std::move(value)) {}
    explicit JsonValue(Array value) : data(std::move(value)) {}
    explicit JsonValue(Object value) : data(std::move(value)) {}

    const Object& AsObject(const std::string& context) const
    {
        const auto* object = std::get_if<Object>(&data);
        if (object == nullptr)
        {
            throw std::runtime_error(context + " must be an object.");
        }
        return *object;
    }

    const Array& AsArray(const std::string& context) const
    {
        const auto* array = std::get_if<Array>(&data);
        if (array == nullptr)
        {
            throw std::runtime_error(context + " must be an array.");
        }
        return *array;
    }

    const std::string& AsString(const std::string& context) const
    {
        const auto* stringValue = std::get_if<std::string>(&data);
        if (stringValue == nullptr)
        {
            throw std::runtime_error(context + " must be a string.");
        }
        return *stringValue;
    }

    double AsNumber(const std::string& context) const
    {
        const auto* number = std::get_if<double>(&data);
        if (number == nullptr)
        {
            throw std::runtime_error(context + " must be a number.");
        }
        return *number;
    }
};

// Recursive-descent JSON parser over a borrowed std::string_view. The
// parser is single-use: construct, call Parse(), then discard.
class JsonParser
{
public:
    explicit JsonParser(std::string_view text) : m_text(text) {}

    // Parse the entire document and assert no trailing data remains.
    JsonValue Parse()
    {
        JsonValue value = ParseValue();
        SkipWhitespace();
        if (!IsAtEnd())
        {
            throw std::runtime_error("Unexpected trailing characters after JSON document.");
        }
        return value;
    }

private:
    // Dispatch on the first non-whitespace character to one of the typed
    // sub-parsers. Each sub-parser leaves m_position pointing past its
    // last consumed character.
    JsonValue ParseValue()
    {
        SkipWhitespace();
        if (IsAtEnd())
        {
            throw std::runtime_error("Unexpected end of JSON input.");
        }

        const char current = m_text[m_position];
        if (current == '{')
        {
            return ParseObject();
        }
        if (current == '[')
        {
            return ParseArray();
        }
        if (current == '"')
        {
            return JsonValue(ParseString());
        }
        if (current == 't')
        {
            ConsumeLiteral("true");
            return JsonValue(true);
        }
        if (current == 'f')
        {
            ConsumeLiteral("false");
            return JsonValue(false);
        }
        if (current == 'n')
        {
            ConsumeLiteral("null");
            return JsonValue(nullptr);
        }
        if (current == '-' || std::isdigit(static_cast<unsigned char>(current)))
        {
            return JsonValue(ParseNumber());
        }

        throw std::runtime_error("Unexpected token in JSON input.");
    }

    JsonValue ParseObject()
    {
        Consume('{');

        JsonValue::Object object;
        SkipWhitespace();
        if (TryConsume('}'))
        {
            return JsonValue(std::move(object));
        }

        while (true)
        {
            SkipWhitespace();
            if (Peek() != '"')
            {
                throw std::runtime_error("Expected a JSON object key.");
            }

            std::string key = ParseString();
            SkipWhitespace();
            Consume(':');
            JsonValue value = ParseValue();

            auto [_, inserted] = object.emplace(key, std::move(value));
            if (!inserted)
            {
                throw std::runtime_error("Duplicate JSON key \"" + key + "\".");
            }

            SkipWhitespace();
            if (TryConsume('}'))
            {
                break;
            }
            Consume(',');
        }

        return JsonValue(std::move(object));
    }

    JsonValue ParseArray()
    {
        Consume('[');

        JsonValue::Array array;
        SkipWhitespace();
        if (TryConsume(']'))
        {
            return JsonValue(std::move(array));
        }

        while (true)
        {
            array.push_back(ParseValue());
            SkipWhitespace();
            if (TryConsume(']'))
            {
                break;
            }
            Consume(',');
        }

        return JsonValue(std::move(array));
    }

    std::string ParseString()
    {
        Consume('"');

        std::string result;
        while (!IsAtEnd())
        {
            const char current = m_text[m_position++];
            if (current == '"')
            {
                return result;
            }
            if (current == '\\')
            {
                if (IsAtEnd())
                {
                    throw std::runtime_error("Unterminated escape sequence in JSON string.");
                }

                const char escaped = m_text[m_position++];
                switch (escaped)
                {
                case '"':
                case '\\':
                case '/':
                    result.push_back(escaped);
                    break;
                case 'b':
                    result.push_back('\b');
                    break;
                case 'f':
                    result.push_back('\f');
                    break;
                case 'n':
                    result.push_back('\n');
                    break;
                case 'r':
                    result.push_back('\r');
                    break;
                case 't':
                    result.push_back('\t');
                    break;
                default:
                    throw std::runtime_error("Unsupported JSON string escape sequence.");
                }
                continue;
            }
            if (static_cast<unsigned char>(current) < 0x20)
            {
                throw std::runtime_error("Control characters are not allowed in JSON strings.");
            }
            result.push_back(current);
        }

        throw std::runtime_error("Unterminated JSON string.");
    }

    // Defer numeric parsing to std::strtod; it accepts the JSON number
    // grammar (and a little more, but the dispatcher in ParseValue already
    // gates on '-' or a digit so the slack is harmless).
    double ParseNumber()
    {
        const char* begin = m_text.data() + m_position;
        char* end = nullptr;
        const double value = std::strtod(begin, &end);
        if (end == begin)
        {
            throw std::runtime_error("Expected a JSON number.");
        }

        m_position = static_cast<size_t>(end - m_text.data());
        return value;
    }

    void ConsumeLiteral(std::string_view literal)
    {
        if (m_text.substr(m_position, literal.size()) != literal)
        {
            throw std::runtime_error("Invalid JSON literal.");
        }
        m_position += literal.size();
    }

    void Consume(char expected)
    {
        SkipWhitespace();
        if (IsAtEnd() || m_text[m_position] != expected)
        {
            throw std::runtime_error(std::string("Expected '") + expected + "' in JSON input.");
        }
        ++m_position;
    }

    bool TryConsume(char expected)
    {
        SkipWhitespace();
        if (IsAtEnd() || m_text[m_position] != expected)
        {
            return false;
        }
        ++m_position;
        return true;
    }

    char Peek() const
    {
        if (IsAtEnd())
        {
            return '\0';
        }
        return m_text[m_position];
    }

    void SkipWhitespace()
    {
        while (!IsAtEnd() && std::isspace(static_cast<unsigned char>(m_text[m_position])))
        {
            ++m_position;
        }
    }

    bool IsAtEnd() const
    {
        return m_position >= m_text.size();
    }

    std::string_view m_text;
    size_t m_position = 0;
};

std::string Quote(std::string_view value)
{
    return "\"" + std::string(value) + "\"";
}

const JsonValue* FindMember(const JsonValue::Object& object, std::string_view key)
{
    const auto iterator = object.find(std::string(key));
    if (iterator == object.end())
    {
        return nullptr;
    }
    return &iterator->second;
}

// Convert a JSON number to float, rejecting NaN/Inf and out-of-range
// magnitudes that would silently saturate when cast.
float ParseFloatValue(const JsonValue& value, const std::string& context)
{
    const double number = value.AsNumber(context);
    if (!std::isfinite(number)
        || number < -static_cast<double>(std::numeric_limits<float>::max())
        || number > static_cast<double>(std::numeric_limits<float>::max()))
    {
        throw std::runtime_error(context + " must be a finite float.");
    }
    return static_cast<float>(number);
}

// Convert a JSON number to uint32_t, rejecting fractional or negative
// inputs so the caller never has to second-guess floor/round behavior.
uint32_t ParseUint32Value(const JsonValue& value, const std::string& context)
{
    const double number = value.AsNumber(context);
    if (!std::isfinite(number) || number < 0.0 || number > static_cast<double>(std::numeric_limits<uint32_t>::max())
        || std::floor(number) != number)
    {
        throw std::runtime_error(context + " must be a uint32 value.");
    }
    return static_cast<uint32_t>(number);
}

std::array<float, 3> ParseFloat3Value(const JsonValue& value, const std::string& context)
{
    const auto& array = value.AsArray(context);
    if (array.size() != 3)
    {
        throw std::runtime_error(context + " must contain exactly three numeric values.");
    }

    return {
        ParseFloatValue(array[0], context + "[0]"),
        ParseFloatValue(array[1], context + "[1]"),
        ParseFloatValue(array[2], context + "[2]"),
    };
}

void ParseOptionalJsonNumber(const JsonValue::Object& object, std::string_view key, float& target)
{
    const JsonValue* value = FindMember(object, key);
    if (value == nullptr)
    {
        return;
    }
    target = ParseFloatValue(*value, Quote(key));
}

void ParseOptionalJsonUint32(const JsonValue::Object& object, std::string_view key, uint32_t& target)
{
    const JsonValue* value = FindMember(object, key);
    if (value == nullptr)
    {
        return;
    }
    target = ParseUint32Value(*value, Quote(key));
}

void ParseOptionalJsonVec3(const JsonValue::Object& object, std::string_view key, Vec3& target)
{
    const JsonValue* value = FindMember(object, key);
    if (value == nullptr)
    {
        return;
    }

    const auto parsed = ParseFloat3Value(*value, Quote(key));
    target = {parsed[0], parsed[1], parsed[2]};
}

void ParseOptionalJsonFloat3(const JsonValue::Object& object,
                             std::string_view key,
                             std::array<float, 3>& target)
{
    const JsonValue* value = FindMember(object, key);
    if (value == nullptr)
    {
        return;
    }
    target = ParseFloat3Value(*value, Quote(key));
}

bool HasNegativeElement(const std::array<float, 3>& value)
{
    return value[0] < 0.0f || value[1] < 0.0f || value[2] < 0.0f;
}

// Pull the global render / camera / input / sky sections off the root
// object. Each sub-section is optional, and within a section every field
// is optional, so partial configs are valid.
void ParseSections(const JsonValue::Object& root, RuntimeConfig& config)
{
    if (const JsonValue* renderValue = FindMember(root, "render"))
    {
        const auto& render = renderValue->AsObject("\"render\"");
        ParseOptionalJsonUint32(render, "width", config.width);
        ParseOptionalJsonUint32(render, "height", config.height);
        ParseOptionalJsonUint32(render, "frameCount", config.frameCount);
        ParseOptionalJsonUint32(render, "samplesPerPixel", config.samplesPerPixel);
    }

    if (const JsonValue* cameraValue = FindMember(root, "camera"))
    {
        const auto& camera = cameraValue->AsObject("\"camera\"");
        ParseOptionalJsonVec3(camera, "initialPosition", config.initialPosition);
        ParseOptionalJsonVec3(camera, "initialLookAt", config.initialLookAt);
        ParseOptionalJsonNumber(camera, "fovYDegrees", config.fovYDegrees);
        ParseOptionalJsonNumber(camera, "maxPitchDegrees", config.maxPitchDegrees);
    }

    if (const JsonValue* inputValue = FindMember(root, "input"))
    {
        const auto& input = inputValue->AsObject("\"input\"");
        ParseOptionalJsonNumber(input, "mouseSensitivity", config.mouseSensitivity);
        ParseOptionalJsonNumber(input, "keyLookSpeed", config.keyLookSpeed);
        ParseOptionalJsonNumber(input, "polarizerRotateSpeed", config.polarizerRotateSpeed);
    }

    if (const JsonValue* skyValue = FindMember(root, "sky"))
    {
        const auto& sky = skyValue->AsObject("\"sky\"");
        ParseOptionalJsonFloat3(sky, "bottomColor", config.skyBottomColor);
        ParseOptionalJsonNumber(sky, "exposure", config.skyExposure);
        ParseOptionalJsonFloat3(sky, "topColor", config.skyTopColor);

        if (const JsonValue* spectralValue = FindMember(sky, "spectralConstants"))
        {
            const auto& spectral = spectralValue->AsObject("\"sky.spectralConstants\"");
            ParseOptionalJsonFloat3(spectral, "BETA_R", config.skySpectral.betaRayleigh);
            ParseOptionalJsonNumber(spectral, "BETA_M", config.skySpectral.betaMie);
            ParseOptionalJsonNumber(spectral, "MIE_G", config.skySpectral.mieG);
            ParseOptionalJsonNumber(spectral, "EARTH_R", config.skySpectral.earthRadius);
            ParseOptionalJsonNumber(spectral, "ATMOS_R", config.skySpectral.atmosphereRadius);
            ParseOptionalJsonNumber(spectral, "SCALE_H_R", config.skySpectral.scaleHeightRayleigh);
            ParseOptionalJsonNumber(spectral, "SCALE_H_M", config.skySpectral.scaleHeightMie);
            ParseOptionalJsonFloat3(spectral, "SUN_RADIANCE", config.skySpectral.sunRadiance);
            ParseOptionalJsonFloat3(spectral, "SUN_DIRECTION", config.skySpectral.sunDirection);
            ParseOptionalJsonNumber(spectral, "SUN_RADIUS", config.skySpectral.sunRadius);
            ParseOptionalJsonNumber(spectral, "SUN_AA", config.skySpectral.sunAa);
            ParseOptionalJsonFloat3(spectral, "BETA_O3", config.skySpectral.betaOzone);
            ParseOptionalJsonNumber(spectral, "OZONE_CENTER", config.skySpectral.ozoneCenterAltitude);
            ParseOptionalJsonNumber(spectral, "OZONE_WIDTH", config.skySpectral.ozoneLayerWidth);
            ParseOptionalJsonFloat3(spectral, "SUN_LIMB_DARKENING", config.skySpectral.sunLimbDarkening);
            ParseOptionalJsonNumber(spectral, "REFRACTION", config.skySpectral.refractionStrength);
            ParseOptionalJsonNumber(spectral, "MIE_BG_BETA", config.skySpectral.mieBackgroundBeta);
            ParseOptionalJsonNumber(spectral, "MIE_BG_CENTER", config.skySpectral.mieBackgroundCenter);
            ParseOptionalJsonNumber(spectral, "MIE_BG_WIDTH", config.skySpectral.mieBackgroundWidth);
            ParseOptionalJsonNumber(spectral, "MIE_ALBEDO", config.skySpectral.mieSingleScatterAlbedo);
            ParseOptionalJsonUint32(spectral, "secondarySamples", config.skySpectral.secondarySamples);
            ParseOptionalJsonUint32(spectral, "VIEW_STEPS", config.skySpectral.viewSteps);
            ParseOptionalJsonUint32(spectral, "Samples", config.skySpectral.samples);

            // Polarized vector radiative transfer controls.
            ParseOptionalJsonNumber(spectral, "RAYLEIGH_DEPOLARIZATION",
                                    config.skySpectral.rayleighDepolarization);
            ParseOptionalJsonNumber(spectral, "AEROSOL_IOR_REAL",
                                    config.skySpectral.aerosolRefractiveIndexReal);
            ParseOptionalJsonNumber(spectral, "AEROSOL_IOR_IMAG",
                                    config.skySpectral.aerosolRefractiveIndexImag);
            ParseOptionalJsonNumber(spectral, "AEROSOL_MEAN_RADIUS_UM",
                                    config.skySpectral.aerosolMeanRadiusMicrometers);
            ParseOptionalJsonNumber(spectral, "AEROSOL_SIGMA", config.skySpectral.aerosolSigma);
            ParseOptionalJsonFloat3(spectral, "AEROSOL_WAVELENGTHS_NM",
                                    config.skySpectral.aerosolWavelengthsNmRgb);
            ParseOptionalJsonUint32(spectral, "MIE_TABLE_ANGLE_BINS",
                                    config.skySpectral.mieTableAngleBins);
        }
    }
}

} // namespace

// Public entry point: parse, populate, then run an exhaustive set of
// semantic checks. Throws on the first problem found rather than
// accumulating errors; the renderer cannot meaningfully degrade past most
// of these issues so fail-fast is the simpler contract.
RuntimeConfig ParseRuntimeConfig(const std::string& jsonText)
{
    RuntimeConfig config{};
    const JsonValue rootValue = JsonParser(std::string_view(jsonText)).Parse();
    const auto& root = rootValue.AsObject("Root config");

    ParseSections(root, config);

    if (config.width == 0 || config.height == 0)
    {
        throw std::runtime_error("\"width\" and \"height\" must be greater than 0.");
    }
    if (config.frameCount == 0)
    {
        throw std::runtime_error("\"frameCount\" must be greater than 0.");
    }
    if (config.samplesPerPixel == 0)
    {
        throw std::runtime_error("\"samplesPerPixel\" must be greater than 0.");
    }
    if (Length(config.initialLookAt - config.initialPosition) <= 0.001f)
    {
        throw std::runtime_error("\"initialPosition\" and \"initialLookAt\" must not be the same.");
    }
    if (config.fovYDegrees <= 1.0f || config.fovYDegrees >= 179.0f)
    {
        throw std::runtime_error("\"fovYDegrees\" must be between 1 and 179.");
    }
    if (config.mouseSensitivity < 0.0f || config.keyLookSpeed < 0.0f || config.polarizerRotateSpeed < 0.0f)
    {
        throw std::runtime_error("\"mouseSensitivity\", \"keyLookSpeed\", and \"polarizerRotateSpeed\" must be non-negative.");
    }
    if (config.maxPitchDegrees <= 0.0f || config.maxPitchDegrees >= 90.0f)
    {
        throw std::runtime_error("\"maxPitchDegrees\" must be greater than 0 and less than 90.");
    }
    if (config.skyExposure <= 0.0f)
    {
        throw std::runtime_error("\"exposure\" must be greater than 0.");
    }
    if (HasNegativeElement(config.skySpectral.betaRayleigh) || config.skySpectral.betaMie < 0.0f)
    {
        throw std::runtime_error("Sky scattering coefficients must be non-negative.");
    }
    if (config.skySpectral.mieG <= -1.0f || config.skySpectral.mieG >= 1.0f)
    {
        throw std::runtime_error("\"MIE_G\" must be between -1 and 1.");
    }
    if (config.skySpectral.earthRadius <= 0.0f || config.skySpectral.atmosphereRadius <= 0.0f
        || config.skySpectral.atmosphereRadius <= config.skySpectral.earthRadius)
    {
        throw std::runtime_error("\"EARTH_R\" and \"ATMOS_R\" must be positive, and ATMOS_R must exceed EARTH_R.");
    }
    if (config.skySpectral.scaleHeightRayleigh <= 0.0f || config.skySpectral.scaleHeightMie <= 0.0f)
    {
        throw std::runtime_error("\"SCALE_H_R\" and \"SCALE_H_M\" must be greater than 0.");
    }
    if (HasNegativeElement(config.skySpectral.sunRadiance))
    {
        throw std::runtime_error("\"SUN_RADIANCE\" values must be non-negative.");
    }
    if (config.skySpectral.sunRadius <= 0.0f || config.skySpectral.sunAa < 0.0f)
    {
        throw std::runtime_error("\"SUN_RADIUS\" must be greater than 0 and \"SUN_AA\" must be non-negative.");
    }
    if (HasNegativeElement(config.skySpectral.betaOzone))
    {
        throw std::runtime_error("\"BETA_O3\" values must be non-negative.");
    }
    if (config.skySpectral.ozoneCenterAltitude < 0.0f || config.skySpectral.ozoneLayerWidth <= 0.0f)
    {
        throw std::runtime_error("\"OZONE_CENTER\" must be non-negative and \"OZONE_WIDTH\" must be greater than 0.");
    }
    if (HasNegativeElement(config.skySpectral.sunLimbDarkening)
        || config.skySpectral.sunLimbDarkening[0] > 1.0f
        || config.skySpectral.sunLimbDarkening[1] > 1.0f
        || config.skySpectral.sunLimbDarkening[2] > 1.0f)
    {
        throw std::runtime_error("\"SUN_LIMB_DARKENING\" values must be in [0, 1].");
    }
    if (config.skySpectral.refractionStrength < 0.0f)
    {
        throw std::runtime_error("\"REFRACTION\" must be non-negative.");
    }
    if (config.skySpectral.mieBackgroundBeta < 0.0f || config.skySpectral.mieBackgroundCenter < 0.0f
        || config.skySpectral.mieBackgroundWidth <= 0.0f)
    {
        throw std::runtime_error("\"MIE_BG_BETA\" and \"MIE_BG_CENTER\" must be non-negative and \"MIE_BG_WIDTH\" must be greater than 0.");
    }
    if (config.skySpectral.mieSingleScatterAlbedo < 0.0f || config.skySpectral.mieSingleScatterAlbedo > 1.0f)
    {
        throw std::runtime_error("\"MIE_ALBEDO\" must be in [0, 1].");
    }
    {
        const auto& d = config.skySpectral.sunDirection;
        if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= 0.0f)
        {
            throw std::runtime_error("\"SUN_DIRECTION\" must be a non-zero vector.");
        }
    }
    // secondarySamples is currently unused, so any value is accepted.
    if (config.skySpectral.viewSteps == 0 || config.skySpectral.samples == 0)
    {
        throw std::runtime_error("\"VIEW_STEPS\" and \"Samples\" must be greater than 0.");
    }
    if (config.skySpectral.rayleighDepolarization < 0.0f || config.skySpectral.rayleighDepolarization >= 1.0f)
    {
        throw std::runtime_error("\"RAYLEIGH_DEPOLARIZATION\" must be in [0, 1).");
    }
    if (config.skySpectral.aerosolRefractiveIndexReal <= 0.0f
        || config.skySpectral.aerosolRefractiveIndexImag < 0.0f)
    {
        throw std::runtime_error("Aerosol refractive index must have positive real part and non-negative imaginary part.");
    }
    if (config.skySpectral.aerosolMeanRadiusMicrometers <= 0.0f || config.skySpectral.aerosolSigma <= 1.0f)
    {
        throw std::runtime_error("\"AEROSOL_MEAN_RADIUS_UM\" must be > 0 and \"AEROSOL_SIGMA\" must be > 1.");
    }
    if (HasNegativeElement(config.skySpectral.aerosolWavelengthsNmRgb)
        || config.skySpectral.aerosolWavelengthsNmRgb[0] <= 0.0f
        || config.skySpectral.aerosolWavelengthsNmRgb[1] <= 0.0f
        || config.skySpectral.aerosolWavelengthsNmRgb[2] <= 0.0f)
    {
        throw std::runtime_error("\"AEROSOL_WAVELENGTHS_NM\" values must be greater than 0.");
    }
    if (config.skySpectral.mieTableAngleBins < 2)
    {
        throw std::runtime_error("\"MIE_TABLE_ANGLE_BINS\" must be at least 2.");
    }
    return config;
}
