## @file CMakeLists.txt
## @brief Configures the Vulkanic renderer, shader, and documentation targets.
## @details Requires CMake 4.4 or newer, C++26 support, and a Vulkan SDK with
## `glslc`. The Windows preset uses Ninja and Clang 20 or newer with `clang-cl`.
## the compute shader targets Vulkan 1.4.

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
# The Doxyfile enables Graphviz, so locate dot even when it is outside PATH.
find_program(
    VULKANIC_DOT_EXECUTABLE
    NAMES dot
    HINTS
        "$ENV{ProgramW6432}/Graphviz/bin"
        "$ENV{ProgramFiles}/Graphviz/bin"
)

if(DOXYGEN_FOUND AND VULKANIC_DOT_EXECUTABLE)
    get_filename_component(VULKANIC_DOT_DIRECTORY "${VULKANIC_DOT_EXECUTABLE}" DIRECTORY)
    set(VULKANIC_DOCS_PATH "$ENV{PATH}")
    if(VULKANIC_DOCS_PATH)
        set(VULKANIC_DOCS_PATH "${VULKANIC_DOT_DIRECTORY};${VULKANIC_DOCS_PATH}")
    else()
        set(VULKANIC_DOCS_PATH "${VULKANIC_DOT_DIRECTORY}")
    endif()

    add_custom_target(
        docs
        COMMAND "${CMAKE_COMMAND}" -E env "PATH=${VULKANIC_DOCS_PATH}"
                "$<TARGET_FILE:Doxygen::doxygen>" Doxyfile
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/assets"
                "${CMAKE_CURRENT_SOURCE_DIR}/build/docs/html/assets"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Generating Vulkanic API and shader documentation"
        VERBATIM
    )
else()
    message(STATUS "Doxygen or Graphviz dot was not found. The optional docs target is unavailable")
endif()
