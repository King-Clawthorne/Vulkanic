cmake_minimum_required(VERSION 3.25)
project(vulkan_path_tracer)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    # CMake does not yet expose CXX26 as a strict MSVC dialect, but its
    # non-strict fallback selects MSVC's C++26-capable /std:c++latest mode.
    set(CMAKE_CXX_STANDARD_REQUIRED OFF)

    foreach(flag_var
            CMAKE_C_FLAGS_DEBUG
            CMAKE_CXX_FLAGS_DEBUG
            CMAKE_C_FLAGS_RELWITHDEBINFO
            CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(DEFINED ${flag_var})
            string(REPLACE "/Zi" "/Z7" ${flag_var} "${${flag_var}}")
        endif()
    endforeach()
else()
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_program(SCCACHE_EXECUTABLE NAMES sccache)
if(SCCACHE_EXECUTABLE)
    if(NOT DEFINED CMAKE_C_COMPILER_LAUNCHER)
        set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_EXECUTABLE}" CACHE FILEPATH "C compiler launcher")
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_EXECUTABLE}" CACHE FILEPATH "C++ compiler launcher")
    endif()
endif()

find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)

include(FetchContent)
FetchContent_Declare(
        vk_bootstrap
        GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
        GIT_TAG v1.3.302)
FetchContent_Declare(
        vma
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG v3.1.0)
FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.92.9)
FetchContent_MakeAvailable(vk_bootstrap vma imgui)

add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan)

set(SHADER_SOURCES
        shaders/path_tracer.comp)

set(SHADER_SHARED_DEPENDENCIES
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/path_tracer_common.glsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sky.comp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rainbow.comp")

set(SHADER_SPV_OUTPUTS)
foreach(SHADER_SOURCE IN LISTS SHADER_SOURCES)
    set(SHADER_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}")
    get_filename_component(SHADER_SOURCE_NAME "${SHADER_SOURCE}" NAME)
    set(SHADER_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SOURCE_NAME}.spv")

    add_custom_command(
            OUTPUT "${SHADER_OUTPUT}"
            COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -g -O0 -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            -o "${SHADER_OUTPUT}" "${SHADER_INPUT}"
            DEPENDS "${SHADER_INPUT}" ${SHADER_SHARED_DEPENDENCIES}
            VERBATIM)

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
add_dependencies(Vulkanic vulkan_shader)
target_include_directories(Vulkanic PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(Vulkanic PRIVATE Vulkan::Vulkan vk-bootstrap::vk-bootstrap GPUOpen::VulkanMemoryAllocator imgui)

foreach(SHADER_OUTPUT IN LISTS SHADER_SPV_OUTPUTS)
    get_filename_component(SHADER_FILE_NAME "${SHADER_OUTPUT}" NAME)
    add_custom_command(TARGET Vulkanic POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SHADER_OUTPUT}"
            "$<TARGET_FILE_DIR:Vulkanic>/${SHADER_FILE_NAME}")
endforeach()

add_custom_command(TARGET Vulkanic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/config/path_tracer_config.json"
        "$<TARGET_FILE_DIR:Vulkanic>/path_tracer_config.json")
