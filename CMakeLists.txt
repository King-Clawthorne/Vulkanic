cmake_minimum_required(VERSION 3.30)
project(vulkan_path_tracer)

set(CMAKE_CXX_STANDARD 26)

# Require glslc from the Vulkan SDK; the compute shader is compiled as part of
# the normal build rather than relying on a checked-in binary shader.
find_package(Vulkan REQUIRED COMPONENTS glslc)

# Fetch pinned library sources; the Vulkan SDK provides the loader and headers.
include(FetchContent)
FetchContent_Declare(
    vma_hpp
    GIT_REPOSITORY https://github.com/YaaZ/VulkanMemoryAllocator-Hpp.git
    GIT_TAG v3.4.0+2
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
    GIT_SHALLOW TRUE
    # GLM is used through headers; skip its legacy CMake project.
    SOURCE_SUBDIR headers_only
)
FetchContent_Declare(
    vk_bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG v1.4.357
    GIT_SHALLOW TRUE
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)
set(VK_BOOTSTRAP_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(vma_hpp glfw glm vk_bootstrap)
add_library(glm_header_only INTERFACE)
add_library(glm::glm ALIAS glm_header_only)
target_include_directories(glm_header_only INTERFACE "${glm_SOURCE_DIR}")

set(SPV "${CMAKE_BINARY_DIR}/pathTracer.comp.spv.inc")
# Emit numeric SPIR-V words as a C++ include, allowing the executable to create
# its shader module from the embedded array at runtime.
add_custom_command(
    OUTPUT "${SPV}"
    DEPENDS shaders/pathTracer.comp shaders/sobol.glsl
    COMMAND Vulkan::glslc --target-env=vulkan1.4 -O -mfmt=num -o "${SPV}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/pathTracer.comp"
)

add_executable(Vulkanic src/SkyTables.cpp src/VulkanPathTracer.cpp "${SPV}")
# The generated include lives in the build tree and is included by the renderer.
target_include_directories(Vulkanic PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(
    Vulkanic
    PRIVATE Vulkan::Vulkan
            GPUOpen::VulkanMemoryAllocator
            VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp
            glfw
            glm::glm
            vk-bootstrap::vk-bootstrap
)
target_compile_options(Vulkanic PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wno-nullability-completeness>)
