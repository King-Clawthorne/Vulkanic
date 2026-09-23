cmake_minimum_required(VERSION 3.30)
project(vulkan_path_tracer)

set(CMAKE_CXX_STANDARD 26)

find_package(Vulkan REQUIRED COMPONENTS glslc)

include(FetchContent)
FetchContent_Declare(vk_bootstrap GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git GIT_TAG v1.3.302 GIT_SHALLOW TRUE)
FetchContent_Declare(vma_hpp GIT_REPOSITORY https://github.com/YaaZ/VulkanMemoryAllocator-Hpp.git GIT_TAG v3.4.0+2 GIT_SHALLOW TRUE)
FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4 GIT_SHALLOW TRUE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glm GIT_REPOSITORY https://github.com/g-truc/glm.git GIT_TAG 1.0.1 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(vk_bootstrap vma_hpp glfw glm)

set(SPV "${CMAKE_BINARY_DIR}/pathTracer.comp.spv.inc")
add_custom_command(OUTPUT "${SPV}" DEPENDS shaders/pathTracer.comp
    COMMAND Vulkan::glslc --target-env=vulkan1.4 -O -mfmt=num -o "${SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/pathTracer.comp")

add_executable(Vulkanic src/SkyTables.cpp src/VulkanPathTracer.cpp "${SPV}")
target_include_directories(Vulkanic PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(Vulkanic PRIVATE Vulkan::Vulkan vk-bootstrap::vk-bootstrap GPUOpen::VulkanMemoryAllocator VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp glfw glm::glm)
target_compile_options(Vulkanic PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wno-nullability-completeness>)
