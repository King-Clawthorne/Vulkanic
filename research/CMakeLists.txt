cmake_minimum_required(VERSION 4.4)
project(PolarizedIntegrationResearch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED COMPONENTS glslc)
find_path(VMA_INCLUDE_DIR NAMES vma/vk_mem_alloc.h HINTS ${Vulkan_INCLUDE_DIRS} REQUIRED)

include(FetchContent)
FetchContent_Declare(vk_bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG 556b79b165386f6c1a18362d30f2a076fdaa2778
)
FetchContent_MakeAvailable(vk_bootstrap)

FetchContent_Declare(vma_hpp
    GIT_REPOSITORY https://github.com/YaaZ/VulkanMemoryAllocator-Hpp.git
    GIT_TAG f84c0c983fb751b50050503536676ccf8e74ed73
    SOURCE_SUBDIR include
)
FetchContent_MakeAvailable(vma_hpp)

set(RESEARCH_SPV "${CMAKE_CURRENT_BINARY_DIR}/Research.comp.spv")
add_custom_command(
    OUTPUT "${RESEARCH_SPV}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Research.comp"
    COMMAND Vulkan::glslc --target-env=vulkan1.4 -O -DVULKANIC_RESEARCH=1
            -o "${RESEARCH_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/Research.comp"
    VERBATIM
)

add_executable(PolarizedResearchHarness gpu_harness.cpp "${RESEARCH_SPV}")
target_include_directories(PolarizedResearchHarness PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(PolarizedResearchHarness PRIVATE ${VMA_INCLUDE_DIR}/vma)
target_link_libraries(PolarizedResearchHarness PRIVATE Vulkan::Vulkan vk-bootstrap::vk-bootstrap VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp)
