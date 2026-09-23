cmake_minimum_required(VERSION 3.30)
project(vulkan_path_tracer)

set(CMAKE_CXX_STANDARD 26)

find_package(Vulkan REQUIRED COMPONENTS glslc)

include(FetchContent)
FetchContent_Declare(vma_hpp GIT_REPOSITORY https://github.com/YaaZ/VulkanMemoryAllocator-Hpp.git GIT_TAG v3.4.0+2 GIT_SHALLOW TRUE)
FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4 GIT_SHALLOW TRUE)
FetchContent_Declare(glm GIT_REPOSITORY https://github.com/g-truc/glm.git GIT_TAG 1.0.1 GIT_SHALLOW TRUE)
FetchContent_Declare(vk_bootstrap GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git GIT_TAG v1.4.357 GIT_SHALLOW TRUE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)
set(VK_BOOTSTRAP_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(vma_hpp glfw glm vk_bootstrap)

set(SPV "${CMAKE_BINARY_DIR}/pathTracer.comp.spv.inc")
add_custom_command(OUTPUT "${SPV}" DEPENDS shaders/pathTracer.comp
    COMMAND Vulkan::glslc --target-env=vulkan1.4 -O -mfmt=num -o "${SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/pathTracer.comp")

add_executable(Vulkanic src/SkyTables.cpp src/VulkanPathTracer.cpp "${SPV}")
target_include_directories(Vulkanic PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(Vulkanic PRIVATE Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp glfw glm::glm vk-bootstrap::vk-bootstrap)
target_compile_options(Vulkanic PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wno-nullability-completeness>)
