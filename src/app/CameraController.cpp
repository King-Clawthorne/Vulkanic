#include "CameraController.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <print>

namespace {
    // True while the given key is held this frame. Used for the look /
    // polarizer controls.
    bool IsKeyDown(GLFWwindow* window, int key)
    {
        return glfwGetKey(window, key) == GLFW_PRESS;
    }

    bool IsFocused(GLFWwindow* window)
    {
        return glfwGetWindowAttrib(window, GLFW_FOCUSED) == GLFW_TRUE;
    }

    float MaxPitchRadians(const RuntimeConfig& config)
    {
        return config.camera.maxPitchDegrees * kPi / 180.0f;
    }
} // namespace

void CameraController::Reset(const RuntimeConfig& config)
{
    m_position = config.camera.initialPosition;
    const Vec3 initialForward = Normalize(config.camera.initialLookAt - m_position);
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
void CameraController::UpdateMouseLook(GLFWwindow* window, const RuntimeConfig& config)
{
    // GLFW keeps reporting cursor motion outside the window while a mouse
    // button is held, so no explicit capture is needed.
    const bool wantsMouseLook = IsFocused(window) && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (!wantsMouseLook) {
        m_mouseLookActive = false;
        return;
    }

    double cursorX = 0.0;
    double cursorY = 0.0;
    glfwGetCursorPos(window, &cursorX, &cursorY);

    if (!m_mouseLookActive) {
        m_mouseLookActive = true;
        m_lastMouseX = cursorX;
        m_lastMouseY = cursorY;
        return;
    }

    const float mouseDeltaX = static_cast<float>(cursorX - m_lastMouseX);
    const float mouseDeltaY = static_cast<float>(cursorY - m_lastMouseY);
    m_lastMouseX = cursorX;
    m_lastMouseY = cursorY;

    const float maxPitch = MaxPitchRadians(config);
    m_yaw += mouseDeltaX * config.input.mouseSensitivity;
    m_pitch = std::clamp(m_pitch - mouseDeltaY * config.input.mouseSensitivity, -maxPitch, maxPitch);
}

// Integrate look input over deltaSeconds: mouse-look, arrow-key look, and the
// R reset. The sky is directional, so the camera only rotates — there is no
// positional movement.
void CameraController::Update(double deltaSeconds, GLFWwindow* window, const RuntimeConfig& config)
{
    const float deltaTime = static_cast<float>(std::min(deltaSeconds, 0.1));
    UpdateMouseLook(window, config);

    const bool windowFocused = IsFocused(window);
    const bool resetCameraDown = windowFocused && IsKeyDown(window, GLFW_KEY_R);
    if (resetCameraDown && !m_resetKeyDown) {
        Reset(config);
    }
    m_resetKeyDown = resetCameraDown;

    // Polarization filter: P toggles it (edge-triggered so one press is one
    // toggle), [ and ] rotate the filter axis while held.
    const bool polarizerToggleDown = windowFocused && IsKeyDown(window, GLFW_KEY_P);
    if (polarizerToggleDown && !m_polarizerToggleKeyDown) {
        m_polarizerEnabled = !m_polarizerEnabled;
        std::println("[Polarizer] {} ({})",
                     m_polarizerEnabled ? "ON" : "OFF",
                     m_polarizerElliptical ? "elliptical" : "linear");
    }
    m_polarizerToggleKeyDown = polarizerToggleDown;

    // C toggles between a linear analyzer and an elliptical one.
    const bool polarizerModeDown = windowFocused && IsKeyDown(window, GLFW_KEY_C);
    if (polarizerModeDown && !m_polarizerModeKeyDown) {
        m_polarizerElliptical = !m_polarizerElliptical;
        std::println("[Polarizer] mode: {}", m_polarizerElliptical ? "elliptical" : "linear");
    }
    m_polarizerModeKeyDown = polarizerModeDown;

    if (windowFocused) {
        if (m_polarizerElliptical) {
            // [ / ] adjust ellipticity; +/-45 degrees reaches circular.
            if (IsKeyDown(window, GLFW_KEY_LEFT_BRACKET)) {
                m_polarizerEllipticityRadians -= config.input.polarizerRotateSpeed * deltaTime;
            }
            if (IsKeyDown(window, GLFW_KEY_RIGHT_BRACKET)) {
                m_polarizerEllipticityRadians += config.input.polarizerRotateSpeed * deltaTime;
            }
            m_polarizerEllipticityRadians = std::clamp(m_polarizerEllipticityRadians, -kPi * 0.25f, kPi * 0.25f);
        } else {
            if (IsKeyDown(window, GLFW_KEY_LEFT_BRACKET)) // '[' rotates the filter axis one way
            {
                m_polarizerAngleRadians -= config.input.polarizerRotateSpeed * deltaTime;
            }
            if (IsKeyDown(window, GLFW_KEY_RIGHT_BRACKET)) // ']' rotates it the other way
            {
                m_polarizerAngleRadians += config.input.polarizerRotateSpeed * deltaTime;
            }
        }
    }

    if (!windowFocused) {
        return;
    }

    const float maxPitch = MaxPitchRadians(config);
    if (IsKeyDown(window, GLFW_KEY_LEFT)) {
        m_yaw -= config.input.keyLookSpeed * deltaTime;
    }
    if (IsKeyDown(window, GLFW_KEY_RIGHT)) {
        m_yaw += config.input.keyLookSpeed * deltaTime;
    }
    if (IsKeyDown(window, GLFW_KEY_UP)) {
        m_pitch += config.input.keyLookSpeed * deltaTime;
    }
    if (IsKeyDown(window, GLFW_KEY_DOWN)) {
        m_pitch -= config.input.keyLookSpeed * deltaTime;
    }
    m_pitch = std::clamp(m_pitch, -maxPitch, maxPitch);
}
