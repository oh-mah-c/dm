/*
 * dm_gpu_internal.h — shared internals between gpu_bpe.c, gpu_sinkhorn.c,
 * and gpu_unigram_em.c.  Not part of the public API.
 */

#ifndef DM_GPU_INTERNAL_H
#define DM_GPU_INTERNAL_H

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  define DL_OPEN(name)   ((void *)LoadLibraryA(name))
#  define DL_SYM(h, sym)  ((void *)GetProcAddress((HMODULE)(h), (sym)))
#  define DL_CLOSE(h)     FreeLibrary((HMODULE)(h))
#else
#  include <dlfcn.h>
#  define DL_OPEN(name)   dlopen((name), RTLD_NOW | RTLD_LOCAL)
#  define DL_SYM(h, sym)  dlsym((h), (sym))
#  define DL_CLOSE(h)     dlclose(h)
#endif

#include "vulkan/vulkan.h"
#include "gpu/dm_gpu.h"

/* -------------------------------------------------------------------------
 * Device-local buffer (GPU-accessible, may also be host-visible)
 * ---------------------------------------------------------------------- */
typedef struct {
    VkBuffer       buf;
    VkDeviceMemory mem;
    VkDeviceSize   size;
    void          *mapped;   /* non-NULL if memory is persistently mapped */
} GpuBuf;

/* -------------------------------------------------------------------------
 * One compute pipeline (descriptor layout + pipeline layout + pipeline)
 * ---------------------------------------------------------------------- */
typedef struct {
    VkDescriptorSetLayout dsl;
    VkPipelineLayout      pl;
    VkPipeline            pipe;
    uint32_t              n_bindings;     /* storage buffers                */
    uint32_t              pc_bytes;       /* push-constant range size       */
} GpuPipeline;

/* -------------------------------------------------------------------------
 * Vulkan function table
 * Instance-level functions are loaded once into a global table.
 * Device-level functions are stored per-context.
 * ---------------------------------------------------------------------- */

/* Global: filled by vk_load_global() on first dm_gpu_create() call. */
typedef struct {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
    PFN_vkCreateInstance      vkCreateInstance;
} VkGlobal;

extern VkGlobal g_vk;

/* Per-context function table (device-level). */
typedef struct {
    PFN_vkDestroyInstance                       vkDestroyInstance;
    PFN_vkEnumeratePhysicalDevices              vkEnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceProperties           vkGetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties     vkGetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties;
    PFN_vkEnumerateDeviceExtensionProperties    vkEnumerateDeviceExtensionProperties;
    PFN_vkCreateDevice                          vkCreateDevice;
    PFN_vkGetDeviceProcAddr                     vkGetDeviceProcAddr;
    /* device-level */
    PFN_vkDestroyDevice                         vkDestroyDevice;
    PFN_vkGetDeviceQueue                        vkGetDeviceQueue;
    PFN_vkCreateCommandPool                     vkCreateCommandPool;
    PFN_vkDestroyCommandPool                    vkDestroyCommandPool;
    PFN_vkAllocateCommandBuffers                vkAllocateCommandBuffers;
    PFN_vkBeginCommandBuffer                    vkBeginCommandBuffer;
    PFN_vkEndCommandBuffer                      vkEndCommandBuffer;
    PFN_vkCmdBindPipeline                       vkCmdBindPipeline;
    PFN_vkCmdBindDescriptorSets                 vkCmdBindDescriptorSets;
    PFN_vkCmdDispatch                           vkCmdDispatch;
    PFN_vkCmdPipelineBarrier                    vkCmdPipelineBarrier;
    PFN_vkCmdPushConstants                      vkCmdPushConstants;
    PFN_vkCmdFillBuffer                         vkCmdFillBuffer;
    PFN_vkQueueSubmit                           vkQueueSubmit;
    PFN_vkQueueWaitIdle                         vkQueueWaitIdle;
    PFN_vkCreateBuffer                          vkCreateBuffer;
    PFN_vkDestroyBuffer                         vkDestroyBuffer;
    PFN_vkGetBufferMemoryRequirements           vkGetBufferMemoryRequirements;
    PFN_vkAllocateMemory                        vkAllocateMemory;
    PFN_vkFreeMemory                            vkFreeMemory;
    PFN_vkBindBufferMemory                      vkBindBufferMemory;
    PFN_vkMapMemory                             vkMapMemory;
    PFN_vkUnmapMemory                           vkUnmapMemory;
    PFN_vkFlushMappedMemoryRanges               vkFlushMappedMemoryRanges;
    PFN_vkInvalidateMappedMemoryRanges          vkInvalidateMappedMemoryRanges;
    PFN_vkCreateShaderModule                    vkCreateShaderModule;
    PFN_vkDestroyShaderModule                   vkDestroyShaderModule;
    PFN_vkCreateDescriptorSetLayout             vkCreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout            vkDestroyDescriptorSetLayout;
    PFN_vkCreatePipelineLayout                  vkCreatePipelineLayout;
    PFN_vkDestroyPipelineLayout                 vkDestroyPipelineLayout;
    PFN_vkCreateComputePipelines                vkCreateComputePipelines;
    PFN_vkDestroyPipeline                       vkDestroyPipeline;
    PFN_vkCreateDescriptorPool                  vkCreateDescriptorPool;
    PFN_vkDestroyDescriptorPool                 vkDestroyDescriptorPool;
    PFN_vkAllocateDescriptorSets                vkAllocateDescriptorSets;
    PFN_vkFreeDescriptorSets                    vkFreeDescriptorSets;
    PFN_vkUpdateDescriptorSets                  vkUpdateDescriptorSets;
    PFN_vkCreateFence                           vkCreateFence;
    PFN_vkDestroyFence                          vkDestroyFence;
    PFN_vkWaitForFences                         vkWaitForFences;
    PFN_vkResetFences                           vkResetFences;
    PFN_vkResetCommandBuffer                    vkResetCommandBuffer;
} VkFn;

/* -------------------------------------------------------------------------
 * Pipeline tags (index into DmGpuCtx.pipes[])
 * ---------------------------------------------------------------------- */
#define PIPE_BPE_PAIR_COUNT  0   /* bpe_pair_count.spv   — BPE histogram    */
#define PIPE_SINKHORN_SPMV   1   /* sinkhorn_spmv.spv    — SpMV K@v / K^T@u */
#define PIPE_SINKHORN_ROWSUM 2   /* sinkhorn_rowsum.spv  — u[i]*(K@v)[i]    */
#define PIPE_UNIGRAM_WORD    3   /* unigram_word.spv     — fwd+bwd per word  */
#define PIPE_COUNT           4

static const char *const PIPE_SHADER_NAMES[PIPE_COUNT] = {
    "bpe_pair_count.spv",
    "sinkhorn_spmv.spv",
    "sinkhorn_rowsum.spv",
    "unigram_word.spv",
};

/* -------------------------------------------------------------------------
 * The opaque GPU context (definition hidden from public callers)
 * ---------------------------------------------------------------------- */
struct DmGpuCtx {
    int      ready;            /* 1 if all shaders loaded and device usable  */
    int      has_atomic_float; /* VK_EXT_shader_atomic_float supported       */
    char     device_name[256];
    uint64_t vram_bytes;

    VkFn vk;   /* all Vulkan function pointers */

    VkInstance                   instance;
    VkPhysicalDevice             phys;
    VkDevice                     dev;
    VkQueue                      queue;
    uint32_t                     queue_family;
    VkPhysicalDeviceMemoryProperties mem_props;

    VkCommandPool    cmd_pool;
    VkCommandBuffer  cmd;
    VkFence          fence;
    VkDescriptorPool dpool;

    GpuPipeline pipes[PIPE_COUNT];
};

/* -------------------------------------------------------------------------
 * Internal helpers declared here, defined in dm_gpu.c
 * ---------------------------------------------------------------------- */

/* Allocate a HOST_VISIBLE | HOST_COHERENT buffer and map it. */
int  gpu_buf_alloc(DmGpuCtx *ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                   GpuBuf *out);
void gpu_buf_free(DmGpuCtx *ctx, GpuBuf *b);

/* Upload n bytes from host src into an already-allocated mapped buffer. */
void gpu_buf_upload(GpuBuf *b, const void *src, size_t n);

/* Download n bytes from a mapped buffer into host dst. */
void gpu_buf_download(GpuBuf *b, void *dst, size_t n);

/* Zero a buffer using vkCmdFillBuffer. */
void gpu_buf_zero_cmd(DmGpuCtx *ctx, GpuBuf *b);

/* Begin a one-shot command buffer. */
int  gpu_cmd_begin(DmGpuCtx *ctx);

/* End, submit, and wait for a one-shot command buffer. */
int  gpu_cmd_end_submit(DmGpuCtx *ctx);

/* Full compute barrier between dispatches. */
void gpu_barrier(DmGpuCtx *ctx);

/*
 * Allocate a descriptor set for pipeline p, bind n_bindings storage buffers,
 * and return the set.  Caller frees with vk.vkFreeDescriptorSets().
 */
VkDescriptorSet gpu_bind_buffers(DmGpuCtx *ctx, int pipe_idx,
                                 GpuBuf *bufs[], uint32_t n_bufs);

/* Dispatch pipeline pipe_idx with push constants pc[pc_bytes]. */
void gpu_dispatch(DmGpuCtx *ctx, int pipe_idx,
                  VkDescriptorSet ds,
                  const void *pc, uint32_t pc_bytes,
                  uint32_t groups_x);

#endif /* DM_GPU_INTERNAL_H */
