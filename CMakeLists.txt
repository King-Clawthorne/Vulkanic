cmake_minimum_required(VERSION 3.20)
project(vulkan_path_tracer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    foreach(flag_var
            CMAKE_C_FLAGS_DEBUG
            CMAKE_CXX_FLAGS_DEBUG
            CMAKE_C_FLAGS_RELWITHDEBINFO
            CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(DEFINED ${flag_var})
            string(REPLACE "/Zi" "/Z7" ${flag_var} "${${flag_var}}")
        endif()
    endforeach()
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
FetchContent_MakeAvailable(vk_bootstrap vma)

set(SHADER_SOURCES
        path_tracer.comp)

set(SHADER_SHARED_DEPENDENCIES
        "${CMAKE_CURRENT_SOURCE_DIR}/path_tracer_common.glsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/sky.comp")

set(SHADER_SPV_OUTPUTS)
foreach(SHADER_SOURCE IN LISTS SHADER_SOURCES)
    set(SHADER_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}")
    set(SHADER_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SOURCE}.spv")

    add_custom_command(
            OUTPUT "${SHADER_OUTPUT}"
            COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -g -O0 -I "${CMAKE_CURRENT_SOURCE_DIR}"
            -o "${SHADER_OUTPUT}" "${SHADER_INPUT}"
            DEPENDS "${SHADER_INPUT}" ${SHADER_SHARED_DEPENDENCIES}
            VERBATIM)

    list(APPEND SHADER_SPV_OUTPUTS "${SHADER_OUTPUT}")
endforeach()

add_custom_target(vulkan_shader ALL DEPENDS ${SHADER_SPV_OUTPUTS})

add_executable(Vulkanic
        main.cpp
        RuntimeConfig.cpp
        MieScattering.cpp
        CameraController.cpp
        VmaUsage.cpp
        VulkanPathTracer.cpp)
add_dependencies(Vulkanic vulkan_shader)
target_link_libraries(Vulkanic PRIVATE Vulkan::Vulkan vk-bootstrap::vk-bootstrap GPUOpen::VulkanMemoryAllocator)

foreach(SHADER_OUTPUT IN LISTS SHADER_SPV_OUTPUTS)
    get_filename_component(SHADER_FILE_NAME "${SHADER_OUTPUT}" NAME)
    add_custom_command(TARGET Vulkanic POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SHADER_OUTPUT}"
            "$<TARGET_FILE_DIR:Vulkanic>/${SHADER_FILE_NAME}")
endforeach()
