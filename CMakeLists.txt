cmake_minimum_required(VERSION 3.25)
project(vulkan_path_tracer)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>")

find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)

include(FetchContent)
FetchContent_Declare(vk_bootstrap GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git GIT_TAG v1.3.302)
# VMA-Hpp bundles the matching VulkanMemoryAllocator as a submodule.
FetchContent_Declare(vma_hpp GIT_REPOSITORY https://github.com/YaaZ/VulkanMemoryAllocator-Hpp.git GIT_TAG v3.3.0+3 GIT_SHALLOW TRUE)
FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.92.9)
FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4)
FetchContent_Declare(glaze GIT_REPOSITORY https://github.com/stephenberry/glaze.git GIT_TAG v8.4.0 GIT_SHALLOW TRUE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(spirv_reflect GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Reflect.git GIT_TAG vulkan-sdk-1.4.321.0 GIT_SHALLOW TRUE)
set(SPIRV_REFLECT_EXECUTABLE OFF CACHE BOOL "" FORCE)
set(SPIRV_REFLECT_INSTALL OFF CACHE BOOL "" FORCE)
set(SPIRV_REFLECT_STATIC_LIB ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(vk_bootstrap vma_hpp imgui glfw glaze spirv_reflect)

set(IMGUI_SOURCES imgui.cpp imgui_draw.cpp imgui_tables.cpp imgui_widgets.cpp
        backends/imgui_impl_glfw.cpp backends/imgui_impl_vulkan.cpp)
list(TRANSFORM IMGUI_SOURCES PREPEND "${imgui_SOURCE_DIR}/")
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/$<CONFIG>")
file(GLOB SHADER_DEPENDENCIES CONFIGURE_DEPENDS "${SHADER_DIR}/*.comp" "${SHADER_DIR}/*.glsl")

set(SHADER_SPV_OUTPUTS)
foreach(SHADER IN ITEMS path_tracer.comp post_process.comp)
    set(SHADER_OUTPUT "${SHADER_OUT_DIR}/${SHADER}.spv")
    add_custom_command(
            OUTPUT "${SHADER_OUTPUT}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUT_DIR}"
            COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.4 -O
            -I "${SHADER_DIR}" -o "${SHADER_OUTPUT}" "${SHADER_DIR}/${SHADER}"
            DEPENDS ${SHADER_DEPENDENCIES}
            VERBATIM COMMAND_EXPAND_LISTS)
    list(APPEND SHADER_SPV_OUTPUTS "${SHADER_OUTPUT}")
endforeach()
add_custom_target(vulkan_shader ALL DEPENDS ${SHADER_SPV_OUTPUTS})

add_executable(Vulkanic
        src/main.cpp
        src/config/RuntimeConfig.cpp
        src/sky/MieScattering.cpp
        src/sky/RainbowScattering.cpp
        src/app/CameraController.cpp
        src/renderer/VmaUsage.cpp
        src/renderer/VulkanPathTracer.cpp)
target_include_directories(Vulkanic PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(Vulkanic PRIVATE Vulkan::Vulkan vk-bootstrap::vk-bootstrap GPUOpen::VulkanMemoryAllocator VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp imgui glfw glaze::glaze spirv-reflect-static)
add_dependencies(Vulkanic vulkan_shader)

add_custom_command(TARGET Vulkanic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/config/path_tracer_config.json" "$<TARGET_FILE_DIR:Vulkanic>")
