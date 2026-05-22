/*
 * dm_gpu.c — Vulkan GPU context: init, teardown, helpers, CPU fallbacks.
 *
 * Compile with:
 *   -I include -I include/gpu
 *   -I third_party/Vulkan-Hpp/Vulkan-Headers/include
 */

#include "dm_gpu_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Global Vulkan loader state (shared across all contexts)
 * ====================================================================== */

VkGlobal g_vk = {0};

static void *g_vk_lib = NULL;

/* Names to try when opening the Vulkan loader library */
static const char *const VK_LIB_NAMES[] = {
#ifdef _WIN32
    "vulkan-1.dll",
#elif defined(__APPLE__)
    "libvulkan.1.dylib",
    "libMoltenVK.dylib",
#else
    "libvulkan.so.1",
    "libvulkan.so",
#endif
    NULL
};

static int vk_load_global(void) {
    if (g_vk.vkGetInstanceProcAddr) return 0; /* already loaded */

    for (int i = 0; VK_LIB_NAMES[i]; i++) {
        g_vk_lib = DL_OPEN(VK_LIB_NAMES[i]);
        if (g_vk_lib) break;
    }
    if (!g_vk_lib) return -1;

    g_vk.vkGetInstanceProcAddr =
        (PFN_vkGetInstanceProcAddr)DL_SYM(g_vk_lib, "vkGetInstanceProcAddr");
    if (!g_vk.vkGetInstanceProcAddr) return -1;

    g_vk.vkCreateInstance =
        (PFN_vkCreateInstance)g_vk.vkGetInstanceProcAddr(NULL, "vkCreateInstance");
    return g_vk.vkCreateInstance ? 0 : -1;
}

/* =========================================================================
 * Macro helpers for loading function pointers
 * ====================================================================== */

#define LOAD_INST(ctx, name) do { \
    (ctx)->vk.name = (PFN_##name) \
        g_vk.vkGetInstanceProcAddr((ctx)->instance, #name); \
    if (!(ctx)->vk.name) { \
        fprintf(stderr, "[gpu] missing instance fn: " #name "\n"); \
        return NULL; \
    } \
} while (0)

#define LOAD_DEV(ctx, name) do { \
    (ctx)->vk.name = (PFN_##name) \
        (ctx)->vk.vkGetDeviceProcAddr((ctx)->dev, #name); \
    if (!(ctx)->vk.name) { \
        fprintf(stderr, "[gpu] missing device fn: " #name "\n"); \
        return NULL; \
    } \
} while (0)

#define VK_OK(call) ((call) == VK_SUCCESS)

/* =========================================================================
 * SPIR-V shader loading
 * ====================================================================== */

static uint32_t *load_spirv(const char *dir, const char *name, size_t *words_out) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (sz <= 0 || sz % 4 != 0) { fclose(f); return NULL; }

    uint32_t *buf = (uint32_t *)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if ((long)fread(buf, 1, (size_t)sz, f) != sz) { free(buf); fclose(f); return NULL; }

    fclose(f);
    *words_out = (size_t)sz / 4;
    return buf;
}

/* Try multiple candidate directories to find compiled shaders. */
static char *resolve_shader_dir(const char *hint) {
    static char resolved[4096];

    const char *candidates[8];
    int nc = 0;

    if (hint) candidates[nc++] = hint;

    /* Paths relative to CWD */
    candidates[nc++] = "shaders";
    candidates[nc++] = "bin/shaders";
    candidates[nc++] = "src/gpu/shaders";

#ifdef _WIN32
    /* Try exe directory */
    static char exe_dir[4096];
    GetModuleFileNameA(NULL, exe_dir, sizeof(exe_dir));
    char *sep = strrchr(exe_dir, '\\');
    if (sep) {
        *sep = '\0';
        static char exe_shaders[4096];
        snprintf(exe_shaders, sizeof(exe_shaders), "%s\\shaders", exe_dir);
        candidates[nc++] = exe_shaders;
    }
#endif

    for (int i = 0; i < nc; i++) {
        /* Check if bpe_pair_count.spv exists here */
        char probe[4096];
        snprintf(probe, sizeof(probe), "%s/%s", candidates[i], PIPE_SHADER_NAMES[0]);
        FILE *f = fopen(probe, "rb");
        if (f) {
            fclose(f);
            strncpy(resolved, candidates[i], sizeof(resolved) - 1);
            return resolved;
        }
    }
    return NULL;
}

/* =========================================================================
 * Pipeline creation
 * ====================================================================== */

/* Binding counts and push-constant sizes for each pipeline */
static const uint32_t PIPE_N_BINDINGS[PIPE_COUNT] = { 5, 6, 4, 9 };
static const uint32_t PIPE_PC_BYTES  [PIPE_COUNT] = { 8, 8, 8, 16 };

static int make_pipeline(DmGpuCtx *ctx, int idx,
                          const uint32_t *spirv, size_t spirv_words) {
    VkFn *vk = &ctx->vk;
    GpuPipeline *p = &ctx->pipes[idx];
    VkResult r;

    p->n_bindings = PIPE_N_BINDINGS[idx];
    p->pc_bytes   = PIPE_PC_BYTES[idx];

    /* Descriptor set layout: all bindings are storage buffers */
    VkDescriptorSetLayoutBinding bindings[16] = {0};
    for (uint32_t b = 0; b < p->n_bindings; b++) {
        bindings[b].binding         = b;
        bindings[b].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl_ci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        NULL, 0, p->n_bindings, bindings
    };
    if (!VK_OK(r = vk->vkCreateDescriptorSetLayout(ctx->dev, &dsl_ci, NULL, &p->dsl)))
        return (int)r;

    /* Pipeline layout with push constants */
    VkPushConstantRange pc_range = {
        VK_SHADER_STAGE_COMPUTE_BIT, 0, p->pc_bytes
    };
    VkPipelineLayoutCreateInfo pl_ci = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        NULL, 0, 1, &p->dsl, 1, &pc_range
    };
    if (!VK_OK(r = vk->vkCreatePipelineLayout(ctx->dev, &pl_ci, NULL, &p->pl)))
        return (int)r;

    /* Shader module */
    VkShaderModuleCreateInfo sm_ci = {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        NULL, 0, spirv_words * 4, spirv
    };
    VkShaderModule sm;
    if (!VK_OK(r = vk->vkCreateShaderModule(ctx->dev, &sm_ci, NULL, &sm)))
        return (int)r;

    /* Compute pipeline */
    VkComputePipelineCreateInfo cp_ci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
        p->pl, VK_NULL_HANDLE, -1
    };
    r = vk->vkCreateComputePipelines(ctx->dev, VK_NULL_HANDLE, 1, &cp_ci, NULL, &p->pipe);
    vk->vkDestroyShaderModule(ctx->dev, sm, NULL);
    return VK_OK(r) ? 0 : (int)r;
}

/* =========================================================================
 * dm_gpu_create / dm_gpu_destroy
 * ====================================================================== */

DmGpuCtx *dm_gpu_create(int device_index, const char *shader_dir) {
    if (vk_load_global() != 0) return NULL;

    DmGpuCtx *ctx = (DmGpuCtx *)calloc(1, sizeof(DmGpuCtx));
    if (!ctx) return NULL;

    VkResult r;

    /* --- Instance --- */
    VkApplicationInfo app = {
        VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL,
        "dm-tokenizer", 1, "dm-gpu", 1, VK_API_VERSION_1_1
    };
    VkInstanceCreateInfo inst_ci = {
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL
    };
    if (!VK_OK(g_vk.vkCreateInstance(&inst_ci, NULL, &ctx->instance))) goto fail;

    /* Load instance-level functions */
    LOAD_INST(ctx, vkDestroyInstance);
    LOAD_INST(ctx, vkEnumeratePhysicalDevices);
    LOAD_INST(ctx, vkGetPhysicalDeviceProperties);
    LOAD_INST(ctx, vkGetPhysicalDeviceMemoryProperties);
    LOAD_INST(ctx, vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD_INST(ctx, vkEnumerateDeviceExtensionProperties);
    LOAD_INST(ctx, vkCreateDevice);
    LOAD_INST(ctx, vkGetDeviceProcAddr);

    /* --- Physical device selection --- */
    uint32_t n_phys = 0;
    ctx->vk.vkEnumeratePhysicalDevices(ctx->instance, &n_phys, NULL);
    if (n_phys == 0) { fprintf(stderr, "[gpu] no Vulkan devices\n"); goto fail; }

    VkPhysicalDevice *devs = (VkPhysicalDevice *)malloc(n_phys * sizeof(*devs));
    ctx->vk.vkEnumeratePhysicalDevices(ctx->instance, &n_phys, devs);

    /* Prefer discrete GPU; honour explicit device_index if given */
    ctx->phys = devs[0];
    if (device_index >= 0 && (uint32_t)device_index < n_phys) {
        ctx->phys = devs[device_index];
    } else {
        for (uint32_t i = 0; i < n_phys; i++) {
            VkPhysicalDeviceProperties props;
            ctx->vk.vkGetPhysicalDeviceProperties(devs[i], &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                ctx->phys = devs[i]; break;
            }
        }
    }
    free(devs);

    {
        VkPhysicalDeviceProperties props;
        ctx->vk.vkGetPhysicalDeviceProperties(ctx->phys, &props);
        strncpy(ctx->device_name, props.deviceName, sizeof(ctx->device_name) - 1);
        ctx->vram_bytes = 0; /* queried from memory properties below */
    }
    ctx->vk.vkGetPhysicalDeviceMemoryProperties(ctx->phys, &ctx->mem_props);

    /* Detect VK_EXT_shader_atomic_float */
    {
        uint32_t n_ext = 0;
        ctx->vk.vkEnumerateDeviceExtensionProperties(ctx->phys, NULL, &n_ext, NULL);
        VkExtensionProperties *exts =
            (VkExtensionProperties *)malloc(n_ext * sizeof(*exts));
        ctx->vk.vkEnumerateDeviceExtensionProperties(ctx->phys, NULL, &n_ext, exts);
        for (uint32_t i = 0; i < n_ext; i++) {
            if (strcmp(exts[i].extensionName, "VK_EXT_shader_atomic_float") == 0)
                ctx->has_atomic_float = 1;
        }
        free(exts);
    }

    /* --- Compute queue family --- */
    uint32_t n_qf = 0;
    ctx->vk.vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys, &n_qf, NULL);
    VkQueueFamilyProperties *qfps =
        (VkQueueFamilyProperties *)malloc(n_qf * sizeof(*qfps));
    ctx->vk.vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys, &n_qf, qfps);
    ctx->queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < n_qf; i++) {
        if (qfps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx->queue_family = i; break;
        }
    }
    free(qfps);
    if (ctx->queue_family == UINT32_MAX) {
        fprintf(stderr, "[gpu] no compute queue\n"); goto fail;
    }

    /* --- Logical device --- */
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0,
        ctx->queue_family, 1, &prio
    };

    const char *dev_exts[2] = { "VK_EXT_shader_atomic_float", NULL };
    uint32_t n_dev_exts = ctx->has_atomic_float ? 1 : 0;

    VkDeviceCreateInfo dev_ci = {
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0,
        1, &qci, 0, NULL, n_dev_exts, dev_exts, NULL
    };
    if (!VK_OK(ctx->vk.vkCreateDevice(ctx->phys, &dev_ci, NULL, &ctx->dev)))
        goto fail;

    /* Load device-level functions */
    LOAD_DEV(ctx, vkDestroyDevice);
    LOAD_DEV(ctx, vkGetDeviceQueue);
    LOAD_DEV(ctx, vkCreateCommandPool);
    LOAD_DEV(ctx, vkDestroyCommandPool);
    LOAD_DEV(ctx, vkAllocateCommandBuffers);
    LOAD_DEV(ctx, vkBeginCommandBuffer);
    LOAD_DEV(ctx, vkEndCommandBuffer);
    LOAD_DEV(ctx, vkCmdBindPipeline);
    LOAD_DEV(ctx, vkCmdBindDescriptorSets);
    LOAD_DEV(ctx, vkCmdDispatch);
    LOAD_DEV(ctx, vkCmdPipelineBarrier);
    LOAD_DEV(ctx, vkCmdPushConstants);
    LOAD_DEV(ctx, vkCmdFillBuffer);
    LOAD_DEV(ctx, vkQueueSubmit);
    LOAD_DEV(ctx, vkQueueWaitIdle);
    LOAD_DEV(ctx, vkCreateBuffer);
    LOAD_DEV(ctx, vkDestroyBuffer);
    LOAD_DEV(ctx, vkGetBufferMemoryRequirements);
    LOAD_DEV(ctx, vkAllocateMemory);
    LOAD_DEV(ctx, vkFreeMemory);
    LOAD_DEV(ctx, vkBindBufferMemory);
    LOAD_DEV(ctx, vkMapMemory);
    LOAD_DEV(ctx, vkUnmapMemory);
    LOAD_DEV(ctx, vkFlushMappedMemoryRanges);
    LOAD_DEV(ctx, vkInvalidateMappedMemoryRanges);
    LOAD_DEV(ctx, vkCreateShaderModule);
    LOAD_DEV(ctx, vkDestroyShaderModule);
    LOAD_DEV(ctx, vkCreateDescriptorSetLayout);
    LOAD_DEV(ctx, vkDestroyDescriptorSetLayout);
    LOAD_DEV(ctx, vkCreatePipelineLayout);
    LOAD_DEV(ctx, vkDestroyPipelineLayout);
    LOAD_DEV(ctx, vkCreateComputePipelines);
    LOAD_DEV(ctx, vkDestroyPipeline);
    LOAD_DEV(ctx, vkCreateDescriptorPool);
    LOAD_DEV(ctx, vkDestroyDescriptorPool);
    LOAD_DEV(ctx, vkAllocateDescriptorSets);
    LOAD_DEV(ctx, vkFreeDescriptorSets);
    LOAD_DEV(ctx, vkUpdateDescriptorSets);
    LOAD_DEV(ctx, vkCreateFence);
    LOAD_DEV(ctx, vkDestroyFence);
    LOAD_DEV(ctx, vkWaitForFences);
    LOAD_DEV(ctx, vkResetFences);
    LOAD_DEV(ctx, vkResetCommandBuffer);

    ctx->vk.vkGetDeviceQueue(ctx->dev, ctx->queue_family, 0, &ctx->queue);

    /* --- Command pool + buffer + fence --- */
    VkCommandPoolCreateInfo cp_ci = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, ctx->queue_family
    };
    if (!VK_OK(ctx->vk.vkCreateCommandPool(ctx->dev, &cp_ci, NULL, &ctx->cmd_pool)))
        goto fail;

    VkCommandBufferAllocateInfo cb_ai = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL,
        ctx->cmd_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1
    };
    if (!VK_OK(ctx->vk.vkAllocateCommandBuffers(ctx->dev, &cb_ai, &ctx->cmd)))
        goto fail;

    VkFenceCreateInfo fence_ci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
    if (!VK_OK(ctx->vk.vkCreateFence(ctx->dev, &fence_ci, NULL, &ctx->fence)))
        goto fail;

    /* --- Descriptor pool (generous limits to cover all pipelines + temps) --- */
    VkDescriptorPoolSize pool_size = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256
    };
    VkDescriptorPoolCreateInfo dp_ci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        32, 1, &pool_size
    };
    if (!VK_OK(ctx->vk.vkCreateDescriptorPool(ctx->dev, &dp_ci, NULL, &ctx->dpool)))
        goto fail;

    /* --- Load SPIR-V shaders and build compute pipelines --- */
    char *sdir = resolve_shader_dir(shader_dir);
    if (!sdir) {
        fprintf(stderr, "[gpu] shaders not found. Run scripts/compile_shaders.ps1 first.\n");
        goto fail;
    }

    for (int i = 0; i < PIPE_COUNT; i++) {
        /* Skip PIPE_UNIGRAM_WORD if no atomic-float support */
        if (i == PIPE_UNIGRAM_WORD && !ctx->has_atomic_float) continue;

        size_t words = 0;
        uint32_t *spirv = load_spirv(sdir, PIPE_SHADER_NAMES[i], &words);
        if (!spirv) {
            fprintf(stderr, "[gpu] failed to load %s from %s\n",
                    PIPE_SHADER_NAMES[i], sdir);
            goto fail;
        }
        int rc = make_pipeline(ctx, i, spirv, words);
        free(spirv);
        if (rc != 0) {
            fprintf(stderr, "[gpu] failed to build pipeline %d (rc=%d)\n", i, rc);
            goto fail;
        }
    }

    fprintf(stderr, "[gpu] ready: %s%s\n", ctx->device_name,
            ctx->has_atomic_float ? " (+atomic_float)" : "");
    ctx->ready = 1;
    return ctx;

fail:
    dm_gpu_destroy(ctx);
    return NULL;
}

void dm_gpu_destroy(DmGpuCtx *ctx) {
    if (!ctx) return;
    VkFn *vk = &ctx->vk;

    if (ctx->dev) {
        for (int i = 0; i < PIPE_COUNT; i++) {
            if (ctx->pipes[i].pipe) vk->vkDestroyPipeline(ctx->dev, ctx->pipes[i].pipe, NULL);
            if (ctx->pipes[i].pl)   vk->vkDestroyPipelineLayout(ctx->dev, ctx->pipes[i].pl, NULL);
            if (ctx->pipes[i].dsl)  vk->vkDestroyDescriptorSetLayout(ctx->dev, ctx->pipes[i].dsl, NULL);
        }
        if (ctx->dpool)    vk->vkDestroyDescriptorPool(ctx->dev, ctx->dpool, NULL);
        if (ctx->fence)    vk->vkDestroyFence(ctx->dev, ctx->fence, NULL);
        if (ctx->cmd_pool) vk->vkDestroyCommandPool(ctx->dev, ctx->cmd_pool, NULL);
        vk->vkDestroyDevice(ctx->dev, NULL);
    }
    if (ctx->instance && vk->vkDestroyInstance)
        vk->vkDestroyInstance(ctx->instance, NULL);

    free(ctx);
}

int dm_gpu_ready(const DmGpuCtx *ctx) {
    return ctx && ctx->ready;
}

int dm_gpu_device_name(const DmGpuCtx *ctx, char *buf, size_t buf_len) {
    if (!ctx || !buf) return DM_GPU_ERR_NO_DEVICE;
    strncpy(buf, ctx->device_name, buf_len - 1);
    buf[buf_len - 1] = '\0';
    return DM_GPU_OK;
}

/* =========================================================================
 * Internal buffer helpers
 * ====================================================================== */

static uint32_t find_mem_type(DmGpuCtx *ctx, uint32_t type_bits,
                               VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < ctx->mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (ctx->mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return UINT32_MAX;
}

int gpu_buf_alloc(DmGpuCtx *ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                  GpuBuf *out) {
    memset(out, 0, sizeof(*out));
    out->size = size;

    VkBufferCreateInfo bci = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, usage,
        VK_SHARING_MODE_EXCLUSIVE, 0, NULL
    };
    if (!VK_OK(ctx->vk.vkCreateBuffer(ctx->dev, &bci, NULL, &out->buf)))
        return DM_GPU_ERR_OOM;

    VkMemoryRequirements req;
    ctx->vk.vkGetBufferMemoryRequirements(ctx->dev, out->buf, &req);

    VkMemoryPropertyFlags mem_flags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t mt = find_mem_type(ctx, req.memoryTypeBits, mem_flags);
    if (mt == UINT32_MAX) {
        ctx->vk.vkDestroyBuffer(ctx->dev, out->buf, NULL);
        return DM_GPU_ERR_OOM;
    }

    VkMemoryAllocateInfo mai = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, mt
    };
    if (!VK_OK(ctx->vk.vkAllocateMemory(ctx->dev, &mai, NULL, &out->mem))) {
        ctx->vk.vkDestroyBuffer(ctx->dev, out->buf, NULL);
        return DM_GPU_ERR_OOM;
    }
    ctx->vk.vkBindBufferMemory(ctx->dev, out->buf, out->mem, 0);
    ctx->vk.vkMapMemory(ctx->dev, out->mem, 0, req.size, 0, &out->mapped);
    return DM_GPU_OK;
}

void gpu_buf_free(DmGpuCtx *ctx, GpuBuf *b) {
    if (!b || !b->buf) return;
    if (b->mapped) ctx->vk.vkUnmapMemory(ctx->dev, b->mem);
    ctx->vk.vkDestroyBuffer(ctx->dev, b->buf, NULL);
    ctx->vk.vkFreeMemory(ctx->dev, b->mem, NULL);
    memset(b, 0, sizeof(*b));
}

void gpu_buf_upload(GpuBuf *b, const void *src, size_t n) {
    memcpy(b->mapped, src, n);
}

void gpu_buf_download(GpuBuf *b, void *dst, size_t n) {
    memcpy(dst, b->mapped, n);
}

void gpu_buf_zero_cmd(DmGpuCtx *ctx, GpuBuf *b) {
    ctx->vk.vkCmdFillBuffer(ctx->cmd, b->buf, 0, b->size, 0u);
}

/* =========================================================================
 * Internal command helpers
 * ====================================================================== */

int gpu_cmd_begin(DmGpuCtx *ctx) {
    ctx->vk.vkResetCommandBuffer(ctx->cmd, 0);
    VkCommandBufferBeginInfo bi = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL
    };
    return VK_OK(ctx->vk.vkBeginCommandBuffer(ctx->cmd, &bi)) ? 0 : -1;
}

int gpu_cmd_end_submit(DmGpuCtx *ctx) {
    ctx->vk.vkEndCommandBuffer(ctx->cmd);
    ctx->vk.vkResetFences(ctx->dev, 1, &ctx->fence);
    VkSubmitInfo si = {
        VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL,
        0, NULL, NULL, 1, &ctx->cmd, 0, NULL
    };
    if (!VK_OK(ctx->vk.vkQueueSubmit(ctx->queue, 1, &si, ctx->fence))) return -1;
    ctx->vk.vkWaitForFences(ctx->dev, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
    return 0;
}

void gpu_barrier(DmGpuCtx *ctx) {
    VkMemoryBarrier mb = {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
    };
    ctx->vk.vkCmdPipelineBarrier(ctx->cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, NULL, 0, NULL);
}

/* =========================================================================
 * Descriptor set binding and dispatch
 * ====================================================================== */

VkDescriptorSet gpu_bind_buffers(DmGpuCtx *ctx, int pipe_idx,
                                  GpuBuf *bufs[], uint32_t n_bufs) {
    VkDescriptorSetAllocateInfo ai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL,
        ctx->dpool, 1, &ctx->pipes[pipe_idx].dsl
    };
    VkDescriptorSet ds;
    if (!VK_OK(ctx->vk.vkAllocateDescriptorSets(ctx->dev, &ai, &ds)))
        return VK_NULL_HANDLE;

    VkWriteDescriptorSet writes[16];
    VkDescriptorBufferInfo buf_infos[16];
    for (uint32_t b = 0; b < n_bufs; b++) {
        buf_infos[b] = (VkDescriptorBufferInfo){ bufs[b]->buf, 0, VK_WHOLE_SIZE };
        writes[b] = (VkWriteDescriptorSet){
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL,
            ds, b, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            NULL, &buf_infos[b], NULL
        };
    }
    ctx->vk.vkUpdateDescriptorSets(ctx->dev, n_bufs, writes, 0, NULL);
    return ds;
}

void gpu_dispatch(DmGpuCtx *ctx, int pipe_idx, VkDescriptorSet ds,
                  const void *pc, uint32_t pc_bytes, uint32_t groups_x) {
    GpuPipeline *p = &ctx->pipes[pipe_idx];
    ctx->vk.vkCmdBindPipeline(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
    ctx->vk.vkCmdBindDescriptorSets(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     p->pl, 0, 1, &ds, 0, NULL);
    if (pc && pc_bytes)
        ctx->vk.vkCmdPushConstants(ctx->cmd, p->pl,
                                    VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes, pc);
    ctx->vk.vkCmdDispatch(ctx->cmd, groups_x, 1, 1);
}

/* =========================================================================
 * CPU fallbacks
 * ====================================================================== */

void dm_gpu_bpe_pair_count_cpu(const DmGpuBpeInput *in, uint32_t *pair_counts) {
    uint32_t V = in->vocab_size;
    for (size_t w = 0; w < in->n_words; w++) {
        uint32_t off  = in->word_starts[w];
        uint32_t len  = in->word_lens[w];
        uint32_t freq = in->word_freqs[w];
        for (uint32_t i = 0; i + 1 < len; i++) {
            uint32_t a = in->sym_ids[off + i];
            uint32_t b = in->sym_ids[off + i + 1];
            pair_counts[a * V + b] += freq;
        }
    }
}

void dm_gpu_sinkhorn_cpu(const DmGpuCSR *K, const DmGpuCSR *Kt,
                          const float *p_tok, const float *p_char,
                          int max_iter, float tol, float *row_sums_out) {
    uint32_t n_tok  = K->n_rows;
    uint32_t n_char = K->n_cols;

    float *u  = (float *)malloc(n_tok  * sizeof(float));
    float *v  = (float *)malloc(n_char * sizeof(float));
    float *kv = (float *)malloc(n_tok  * sizeof(float));

    for (uint32_t i = 0; i < n_tok;  i++) u[i] = 1.0f;
    for (uint32_t j = 0; j < n_char; j++) v[j] = 1.0f;

    for (int iter = 0; iter < max_iter; iter++) {
        /* u[i] = p_tok[i] / (K @ v)[i] */
        for (uint32_t i = 0; i < n_tok; i++) {
            float dot = 0.0f;
            for (uint32_t k = K->row_ptr[i]; k < K->row_ptr[i + 1]; k++)
                dot += K->vals[k] * v[K->col_idx[k]];
            kv[i] = dot;
            u[i] = (dot > 1e-30f) ? p_tok[i] / dot : 0.0f;
        }
        /* v[j] = p_char[j] / (K^T @ u)[j] */
        float *ktu = (float *)calloc(n_char, sizeof(float));
        for (uint32_t j = 0; j < n_char; j++) {
            float dot = 0.0f;
            for (uint32_t k = Kt->row_ptr[j]; k < Kt->row_ptr[j + 1]; k++)
                dot += Kt->vals[k] * u[Kt->col_idx[k]];
            v[j] = (dot > 1e-30f) ? p_char[j] / dot : 0.0f;
        }
        free(ktu);

        /* Convergence: check max change in u */
        float max_d = 0.0f;
        for (uint32_t i = 0; i < n_tok; i++) {
            float new_kv = 0.0f;
            for (uint32_t k = K->row_ptr[i]; k < K->row_ptr[i + 1]; k++)
                new_kv += K->vals[k] * v[K->col_idx[k]];
            float new_u = (new_kv > 1e-30f) ? p_tok[i] / new_kv : 0.0f;
            float d = fabsf(new_u - u[i]);
            if (d > max_d) max_d = d;
        }
        if (max_d < tol) break;
    }

    /* row_sums[i] = u[i] * (K @ v)[i] */
    for (uint32_t i = 0; i < n_tok; i++) {
        float dot = 0.0f;
        for (uint32_t k = K->row_ptr[i]; k < K->row_ptr[i + 1]; k++)
            dot += K->vals[k] * v[K->col_idx[k]];
        row_sums_out[i] = u[i] * dot;
    }

    free(u); free(v); free(kv);
}

/* =========================================================================
 * CSR build helpers
 * ====================================================================== */

void dm_gpu_build_csr(const uint32_t *coo_row, const uint32_t *coo_col,
                       const float *coo_val, uint32_t nnz,
                       uint32_t n_rows, uint32_t n_cols,
                       uint32_t **row_ptr_out, uint32_t **col_idx_out,
                       float **vals_out) {
    (void)n_cols;
    uint32_t *rp = (uint32_t *)calloc(n_rows + 1, sizeof(uint32_t));
    for (uint32_t k = 0; k < nnz; k++) rp[coo_row[k] + 1]++;
    for (uint32_t i = 0; i < n_rows; i++) rp[i + 1] += rp[i];

    uint32_t *ci = (uint32_t *)malloc(nnz * sizeof(uint32_t));
    float    *cv = (float    *)malloc(nnz * sizeof(float));
    uint32_t *pos = (uint32_t *)calloc(n_rows, sizeof(uint32_t));

    for (uint32_t k = 0; k < nnz; k++) {
        uint32_t r   = coo_row[k];
        uint32_t idx = rp[r] + pos[r]++;
        ci[idx] = coo_col[k];
        cv[idx] = coo_val[k];
    }
    free(pos);
    *row_ptr_out = rp;
    *col_idx_out = ci;
    *vals_out    = cv;
}

void dm_gpu_csr_transpose(const DmGpuCSR *K,
                           uint32_t **kt_row_ptr_out,
                           uint32_t **kt_col_idx_out,
                           float    **kt_vals_out) {
    uint32_t n_rows = K->n_rows, n_cols = K->n_cols, nnz = K->nnz;
    uint32_t *rp = (uint32_t *)calloc(n_cols + 1, sizeof(uint32_t));

    for (uint32_t k = 0; k < nnz; k++) rp[K->col_idx[k] + 1]++;
    for (uint32_t j = 0; j < n_cols; j++) rp[j + 1] += rp[j];

    uint32_t *ci = (uint32_t *)malloc(nnz * sizeof(uint32_t));
    float    *cv = (float    *)malloc(nnz * sizeof(float));
    uint32_t *pos = (uint32_t *)calloc(n_cols, sizeof(uint32_t));

    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t k = K->row_ptr[i]; k < K->row_ptr[i + 1]; k++) {
            uint32_t j   = K->col_idx[k];
            uint32_t idx = rp[j] + pos[j]++;
            ci[idx] = i;
            cv[idx] = K->vals[k];
        }
    }
    free(pos);
    *kt_row_ptr_out = rp;
    *kt_col_idx_out = ci;
    *kt_vals_out    = cv;
}

/* =========================================================================
 * Public dispatch wrappers (forward to gpu_bpe.c / gpu_sinkhorn.c / etc.)
 * ====================================================================== */

/* Forward declarations of GPU implementations */
int _gpu_bpe_pair_count(DmGpuCtx *ctx, const DmGpuBpeInput *in,
                         uint32_t *pair_counts);
int _gpu_sinkhorn(DmGpuCtx *ctx, const DmGpuCSR *K, const DmGpuCSR *Kt,
                   const float *p_tok, const float *p_char,
                   int max_iter, float tol, float *row_sums_out);
int _gpu_unigram_em_step(DmGpuCtx *ctx, const DmGpuUnigramModel *model,
                          const DmGpuUnigramCorpus *corpus,
                          float *new_counts, float *expected_total);

int dm_gpu_bpe_pair_count(DmGpuCtx *ctx, const DmGpuBpeInput *in,
                           uint32_t *pair_counts) {
    if (in->vocab_size > DM_GPU_BPE_MAX_VOCAB)
        return DM_GPU_ERR_VOCAB_TOO_LARGE;
    if (!dm_gpu_ready(ctx)) {
        dm_gpu_bpe_pair_count_cpu(in, pair_counts);
        return DM_GPU_OK;
    }
    return _gpu_bpe_pair_count(ctx, in, pair_counts);
}

int dm_gpu_sinkhorn(DmGpuCtx *ctx,
                     const DmGpuCSR *K, const DmGpuCSR *Kt,
                     const float *p_tok, const float *p_char,
                     int max_iter, float tol, float *row_sums_out) {
    if (!dm_gpu_ready(ctx)) {
        dm_gpu_sinkhorn_cpu(K, Kt, p_tok, p_char, max_iter, tol, row_sums_out);
        return DM_GPU_OK;
    }
    return _gpu_sinkhorn(ctx, K, Kt, p_tok, p_char, max_iter, tol, row_sums_out);
}

int dm_gpu_unigram_em_step(DmGpuCtx *ctx,
                            const DmGpuUnigramModel *model,
                            const DmGpuUnigramCorpus *corpus,
                            float *new_counts, float *expected_total) {
    if (!dm_gpu_ready(ctx) || !ctx->has_atomic_float)
        return DM_GPU_ERR_NO_ATOMIC_FLOAT;
    return _gpu_unigram_em_step(ctx, model, corpus, new_counts, expected_total);
}
