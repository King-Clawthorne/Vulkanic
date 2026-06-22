#include "CameraController.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace
{
// True while the given virtual-key is held this frame (high bit of the async
// key state). Used for the look / polarizer controls.
bool IsKeyDown(int virtualKey)
{
    return (GetAsyncKeyState(virtualKey) & 0x8000) != 0;
}

float MaxPitchRadians(const RuntimeConfig& config)
{
    return config.maxPitchDegrees * kPi / 180.0f;
}
} // namespace

void CameraController::Reset(const RuntimeConfig& config)
{
    m_position = config.initialPosition;
    const Vec3 initialForward = Normalize(config.initialLookAt - m_position);
    m_yaw = std::atan2(initialForward.x, initialForward.z);
    m_pitch = std::asin(std::clamp(initialForward.y, -1.0f, 1.0f));
}

void CameraController::ClampPitch(const RuntimeConfig& config)
{
    const float maxPitch = MaxPitchRadians(config);
    m_pitch = std::clamp(m_pitch, -maxPitch, maxPitch);
}

Vec3 CameraController::Forward() const
{
    const float cosPitch = std::cos(m_pitch);
    return Normalize({
        std::sin(m_yaw) * cosPitch,
        std::sin(m_pitch),
        std::cos(m_yaw) * cosPitch,
    });
}

// Consume accumulated mouse-delta input and rotate the camera. Pitch is
// clamped to ±maxPitchDegrees so the camera never goes upside down and the yaw
// axis remains world-up.
void CameraController::UpdateMouseLook(HWND window, const RuntimeConfig& config)
{
    const bool windowFocused = GetForegroundWindow() == window;
    const bool wantsMouseLook = windowFocused && IsKeyDown(VK_RBUTTON);
    if (!wantsMouseLook)
    {
        if (m_mouseLookActive && GetCapture() == window)
        {
            ReleaseCapture();
        }
        m_mouseLookActive = false;
        return;
    }

    POINT cursorPosition{};
    if (!GetCursorPos(&cursorPosition))
    {
        return;
    }

    if (!m_mouseLookActive)
    {
        m_mouseLookActive = true;
        m_lastMousePosition = cursorPosition;
        SetCapture(window);
        return;
    }

    const float mouseDeltaX = static_cast<float>(cursorPosition.x - m_lastMousePosition.x);
    const float mouseDeltaY = static_cast<float>(cursorPosition.y - m_lastMousePosition.y);
    m_lastMousePosition = cursorPosition;

    const float maxPitch = MaxPitchRadians(config);
    m_yaw += mouseDeltaX * config.mouseSensitivity;
    m_pitch = std::clamp(m_pitch - mouseDeltaY * config.mouseSensitivity, -maxPitch, maxPitch);
}

// Integrate look input over deltaSeconds: mouse-look, arrow-key look, and the
// R reset. The sky is directional, so the camera only rotates — there is no
// positional movement.
void CameraController::Update(double deltaSeconds, HWND window, const RuntimeConfig& config)
{
    const float deltaTime = static_cast<float>(std::min(deltaSeconds, 0.1));
    UpdateMouseLook(window, config);

    const bool windowFocused = GetForegroundWindow() == window;
    const bool resetCameraDown = windowFocused && IsKeyDown('R');
    if (resetCameraDown && !m_resetKeyDown)
    {
        Reset(config);
    }
    m_resetKeyDown = resetCameraDown;

    // Polarization filter: P toggles it (edge-triggered so one press is one
    // toggle), [ and ] rotate the filter axis while held.
    const bool polarizerToggleDown = windowFocused && IsKeyDown('P');
    if (polarizerToggleDown && !m_polarizerToggleKeyDown)
    {
        m_polarizerEnabled = !m_polarizerEnabled;
        std::printf("[Polarizer] %s (%s)\n",
                    m_polarizerEnabled ? "ON" : "OFF",
                    m_polarizerElliptical ? "elliptical" : "linear");
    }
    m_polarizerToggleKeyDown = polarizerToggleDown;

    // C toggles between a linear analyzer and an elliptical one.
    const bool polarizerModeDown = windowFocused && IsKeyDown('C');
    if (polarizerModeDown && !m_polarizerModeKeyDown)
    {
        m_polarizerElliptical = !m_polarizerElliptical;
        std::printf("[Polarizer] mode: %s\n", m_polarizerElliptical ? "elliptical" : "linear");
    }
    m_polarizerModeKeyDown = polarizerModeDown;

    if (windowFocused)
    {
        if (m_polarizerElliptical)
        {
            // [ / ] adjust ellipticity; +/-45 degrees reaches circular.
            if (IsKeyDown(VK_OEM_4))
            {
                m_polarizerEllipticityRadians -= config.polarizerRotateSpeed * deltaTime;
            }
            if (IsKeyDown(VK_OEM_6))
            {
                m_polarizerEllipticityRadians += config.polarizerRotateSpeed * deltaTime;
            }
            m_polarizerEllipticityRadians = std::clamp(m_polarizerEllipticityRadians, -kPi * 0.25f, kPi * 0.25f);
        }
        else
        {
            if (IsKeyDown(VK_OEM_4)) // '[' rotates the filter axis one way
            {
                m_polarizerAngleRadians -= config.polarizerRotateSpeed * deltaTime;
            }
            if (IsKeyDown(VK_OEM_6)) // ']' rotates it the other way
            {
                m_polarizerAngleRadians += config.polarizerRotateSpeed * deltaTime;
            }
        }
    }

    if (!windowFocused)
    {
        return;
    }

    const float maxPitch = MaxPitchRadians(config);
    if (IsKeyDown(VK_LEFT))
    {
        m_yaw -= config.keyLookSpeed * deltaTime;
    }
    if (IsKeyDown(VK_RIGHT))
    {
        m_yaw += config.keyLookSpeed * deltaTime;
    }
    if (IsKeyDown(VK_UP))
    {
        m_pitch += config.keyLookSpeed * deltaTime;
    }
    if (IsKeyDown(VK_DOWN))
    {
        m_pitch -= config.keyLookSpeed * deltaTime;
    }
    m_pitch = std::clamp(m_pitch, -maxPitch, maxPitch);
}
