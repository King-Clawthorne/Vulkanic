cmake_minimum_required(VERSION 4.4)
project(vulkan_path_tracer)
set(CMAKE_CXX_STANDARD 26)

find_package(Vulkan REQUIRED COMPONENTS glslc)

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

set(SPV "${CMAKE_BINARY_DIR}/Main.comp.spv.inc")
add_custom_command(
    OUTPUT "${SPV}"
    DEPENDS shaders/Main.comp
    COMMAND Vulkan::glslc --target-env=vulkan1.4 -O -mfmt=num -o "${SPV}"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/Main.comp"
)

add_executable(Vulkanic src/App.cpp src/Baker.cpp "${SPV}")
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

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    doxygen_add_docs(
        docs
        CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Generating Vulkanic API and shader documentation"
    )
else()
    message(STATUS "Doxygen was not found. The optional docs target is unavailable")
endif()
