cmake_minimum_required(VERSION 3.30)
project(vulkan_path_tracer)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" REQUIRED)

include(FetchContent)
FetchContent_Declare(vk_bootstrap GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git GIT_TAG v1.3.302 GIT_SHALLOW TRUE)
FetchContent_Declare(vma_hpp GIT_REPOSITORY https://github.com/YaaZ/VulkanMemoryAllocator-Hpp.git GIT_TAG v3.4.0+2 GIT_SHALLOW TRUE)
FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4 GIT_SHALLOW TRUE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(vk_bootstrap vma_hpp glfw)

set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(GLOB SHADER_DEPENDENCIES CONFIGURE_DEPENDS "${SHADER_DIR}/*.comp" "${SHADER_DIR}/*.glsl")
file(MAKE_DIRECTORY "${SHADER_OUT_DIR}")

set(SHADER_SPV_OUTPUTS)
foreach(SHADER IN ITEMS path_tracer.comp)
    set(SHADER_OUTPUT "${SHADER_OUT_DIR}/${SHADER}.spv.inc")
    add_custom_command(
            OUTPUT "${SHADER_OUTPUT}"
            COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.4 -O -mfmt=num
            -I "${SHADER_DIR}" -o "${SHADER_OUTPUT}" "${SHADER_DIR}/${SHADER}"
            DEPENDS ${SHADER_DEPENDENCIES}
            VERBATIM)
    list(APPEND SHADER_SPV_OUTPUTS "${SHADER_OUTPUT}")
endforeach()
add_custom_target(vulkan_shader ALL DEPENDS ${SHADER_SPV_OUTPUTS})

add_executable(Vulkanic src/SkyTables.cpp src/VulkanPathTracer.cpp)
target_include_directories(Vulkanic PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SHADER_OUT_DIR}")
target_link_libraries(Vulkanic PRIVATE Vulkan::Vulkan vk-bootstrap::vk-bootstrap GPUOpen::VulkanMemoryAllocator VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp glfw)
add_dependencies(Vulkanic vulkan_shader)
target_compile_options(Vulkanic PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wno-nullability-completeness>)
