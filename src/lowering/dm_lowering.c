#include "lowering/dm_lowering.h"
#include "lowering/dm_lower_cpu.h"
#include "lowering/dm_lower_vulkan.h"
#include "lowering/dm_lower_tf.h"

int dm_lower_block(const DM_Block *src, DM_Block *dst, DM_Backend target_backend, DM_LowerPolicy policy) {
    if (!src || !dst) return -1;
    
    // Fast path: already on target backend
    if (src->backend == target_backend) {
        *dst = *src;
        if (policy == DM_LOWER_VIEW) {
            dst->owns_data = 0;
            dst->owns_handle = 0;
        } else if (policy == DM_LOWER_MOVE) {
            dst->owns_data = src->owns_data;
            dst->owns_handle = src->owns_handle;
            // The caller is responsible for clearing the src ownership to prevent double free.
        } else { // COPY
            // TODO: implement deep copy
            return -1;
        }
        return 0;
    }

    switch (target_backend) {
        case DM_BACKEND_CPU:
            if (src->backend == DM_BACKEND_TENSORFLOW) {
                if (policy == DM_LOWER_VIEW) return -1; // CPU cannot directly view TF memory zero-copy in our current model
                return dm_tf_tensor_copy_to_block(src->handle, dst);
            }
            return dm_block_to_cpu(src, dst);
        case DM_BACKEND_VULKAN_COMPUTE:
        case DM_BACKEND_VULKAN_COOP_MAT:
            return dm_block_to_vulkan_buffer(src, dst);
        case DM_BACKEND_TENSORFLOW:
            {
                if (src->backend != DM_BACKEND_CPU) {
                    // TF from Vulkan etc requires a copy/transfer step not implemented here
                    if (policy == DM_LOWER_VIEW) return -1; 
                }

                DM_TF_Tensor *tf_tensor = NULL;
                
                // Persistent Backend Handle Caching
                DM_Block *mut_src = (DM_Block *)src;
                if (src->backend == DM_BACKEND_CPU && src->handle != NULL) {
                    if (src->dirty) {
                        // Invalidate cached handle
                        if (src->owns_handle && src->handle_destructor) {
                            src->handle_destructor(src->handle);
                        }
                        mut_src->handle = NULL;
                        mut_src->owns_handle = 0;
                    } else {
                        tf_tensor = (DM_TF_Tensor *)src->handle;
                    }
                }
                
                int rc = 0;
                if (!tf_tensor) {
                    rc = dm_block_to_tf_tensor(src, &tf_tensor);
                    if (rc == 0 && src->backend == DM_BACKEND_CPU) {
                        mut_src->handle = tf_tensor;
                        mut_src->owns_handle = 1;
                        mut_src->handle_destructor = dm_tf_tensor_free;
                        mut_src->dirty = 0;
                    }
                }
                if (rc == 0) {
                    *dst = *src;
                    dst->backend = DM_BACKEND_TENSORFLOW;
                    dst->kind = DM_KIND_EXTERNAL;
                    dst->layout = DM_LAYOUT_TF_HANDLE;
                    dst->handle = tf_tensor;
                    dst->handle_destructor = dm_tf_tensor_free;
                    dst->data = NULL;
                    
                    if (policy == DM_LOWER_VIEW) {
                        dst->owns_data = 0;
                        dst->owns_handle = 0; // The source block caches and owns the handle
                    } else if (policy == DM_LOWER_MOVE) {
                        dst->owns_data = src->owns_data;
                        dst->owns_handle = src->owns_handle;
                        mut_src->owns_handle = 0; // Transfer ownership
                        mut_src->handle = NULL;
                    } else { // DM_LOWER_COPY
                        // dm_block_to_tf_tensor is currently zero-copy (wraps CPU pointer).
                        // If strict COPY is requested, we should ideally allocate new memory or a new tensor.
                        // For now we will return an error since true copy isn't implemented.
                        return -1;
                    }
                }
                return rc;
            }
        default:
            return -1;
    }
}

int dm_raise_block(const DM_Block *src, DM_Block *dst, DM_Backend target_backend, DM_LowerPolicy policy) {
    if (!src || !dst) return -1;
    
    // Fast path
    if (src->backend == target_backend) {
        *dst = *src;
        if (policy == DM_LOWER_VIEW) {
            dst->owns_data = 0;
            dst->owns_handle = 0;
        } else if (policy == DM_LOWER_MOVE) {
            dst->owns_data = src->owns_data;
            dst->owns_handle = src->owns_handle;
        } else {
            return -1; // TODO: deep copy
        }
        return 0;
    }

    // Reverse operation: bringing data back from a backend to CPU usually
    if (src->backend == DM_BACKEND_TENSORFLOW && target_backend == DM_BACKEND_CPU) {
        return dm_tf_tensor_to_block(src->handle, dst, src->kind, src->role);
    } else if (src->backend == DM_BACKEND_VULKAN_COMPUTE && target_backend == DM_BACKEND_CPU) {
        return dm_vulkan_buffer_to_block(src, dst);
    } else if (src->backend == DM_BACKEND_CPU && target_backend == DM_BACKEND_CPU) {
        return dm_cpu_to_block(src, dst);
    }
    
    return -1;
}