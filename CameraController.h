#pragma once

// CameraController — owns the look-only camera and the polarization-filter
// analyzer state, and translates raw Win32 keyboard / mouse input into them.
//
// The sky is directional, so the camera never moves positionally; it only
// rotates (yaw/pitch) plus an optional R reset. The polarizer state (enabled,
// linear vs. elliptical, axis angle / ellipticity) lives here too because it
// is driven entirely by the same per-frame input poll. The renderer reads the
// resulting camera basis and polarizer parameters when it builds push
// constants; it owns no camera/input state of its own.

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "RuntimeConfig.h"

class CameraController
{
public:
    // Snap the camera to the config's initialPosition / initialLookAt and
    // recompute yaw/pitch. Safe to call before the window exists.
    void Reset(const RuntimeConfig& config);

    // Re-clamp the pitch into the (possibly changed) maxPitchDegrees range
    // after a config hot-reload that did not reset the camera.
    void ClampPitch(const RuntimeConfig& config);

    // Integrate look + polarizer input over deltaSeconds. Input is ignored
    // unless `window` is the foreground window.
    void Update(double deltaSeconds, HWND window, const RuntimeConfig& config);

    [[nodiscard]] Vec3 Position() const { return m_position; }
    [[nodiscard]] Vec3 Forward() const;

    [[nodiscard]] bool PolarizerEnabled() const { return m_polarizerEnabled; }
    [[nodiscard]] float PolarizerAngleRadians() const { return m_polarizerAngleRadians; }
    // Ellipticity only applies in elliptical mode; linear mode reports 0.
    [[nodiscard]] float PolarizerEllipticityRadians() const
    {
        return m_polarizerElliptical ? m_polarizerEllipticityRadians : 0.0f;
    }

private:
    void UpdateMouseLook(HWND window, const RuntimeConfig& config);

    Vec3 m_position{};
    float m_yaw = 0.0f;
    float m_pitch = 0.0f;

    bool m_resetKeyDown = false;
    bool m_mouseLookActive = false;
    POINT m_lastMousePosition{};

    // Camera polarization filter. P toggles it on/off; C switches between a
    // linear analyzer and an elliptical one. In linear mode [ and ] rotate the
    // major axis (measured in the image plane from camera-right, positive
    // toward camera-down on screen — counterclockwise facing the incoming
    // beam); in elliptical mode [ and ] adjust ellipticity (-45..+45 degrees).
    bool m_polarizerEnabled = false;
    float m_polarizerAngleRadians = 0.0f;
    bool m_polarizerToggleKeyDown = false;
    bool m_polarizerElliptical = false;
    bool m_polarizerModeKeyDown = false;
    float m_polarizerEllipticityRadians = kPi * 0.125f;
};
