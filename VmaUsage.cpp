// Single translation unit that compiles the Vulkan Memory Allocator
// implementation. Every other file includes vk_mem_alloc.h as a plain header;
// only this one defines VMA_IMPLEMENTATION so the definitions land here once.
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
