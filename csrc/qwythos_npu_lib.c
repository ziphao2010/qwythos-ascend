/*
 * Qwythos NPU Runtime - shared library for Ascend 310.
 * Callable from Python via ctypes.
 *
 * Compile:
 *   gcc -shared -fPIC qwythos_npu_lib.c -o libqwythos_npu.so \\
 *       -I/usr/local/Ascend/ascend-toolkit/latest/include \\
 *       -L/usr/local/Ascend/ascend-toolkit/latest/lib64 \\
 *       -lascendcl -Wl,-rpath,/usr/local/Ascend/ascend-toolkit/latest/lib64
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "acl/acl.h"

#define EXPORT __attribute__((visibility("default")))

/* Initialize NPU. Returns 0 on success. */
EXPORT int npu_init(int device_id) {
    aclError ret = aclInit(NULL);
    if (ret != ACL_SUCCESS && ret != 500000) return -1;
    ret = aclrtSetDevice(device_id);
    if (ret != ACL_SUCCESS) return -2;
    return 0;
}

/* Allocate NPU memory. Returns pointer or NULL. */
EXPORT void* npu_malloc(size_t size) {
    void *ptr = NULL;
    aclError ret = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return NULL;
    return ptr;
}

/* Free NPU memory. Returns 0 on success. */
EXPORT int npu_free(void *ptr) {
    return aclrtFree(ptr) == ACL_SUCCESS ? 0 : -1;
}

/* Host -> Device memcpy. Returns 0 on success. */
EXPORT int npu_h2d(void *dev_dst, const void *host_src, size_t size) {
    return aclrtMemcpy(dev_dst, size, (void*)host_src, size,
                       ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS ? 0 : -1;
}

/* Device -> Host memcpy. Returns 0 on success. */
EXPORT int npu_d2h(void *host_dst, const void *dev_src, size_t size) {
    return aclrtMemcpy(host_dst, size, (void*)dev_src, size,
                       ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS ? 0 : -1;
}

/* Get total NPU memory. Returns value in bytes, or -1 on error. */
EXPORT size_t npu_get_total_mem(void) {
    size_t free_mem = 0, total_mem = 0;
    aclError ret = aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem);
    if (ret != ACL_SUCCESS) return -1;
    return total_mem;
}

/* Get free NPU memory. */
EXPORT size_t npu_get_free_mem(void) {
    size_t free_mem = 0, total_mem = 0;
    aclError ret = aclrtGetMemInfo(ACL_HBM_MEM, &free_mem, &total_mem);
    if (ret != ACL_SUCCESS) return -1;
    return free_mem;
}

/* Shutdown NPU. */
EXPORT void npu_shutdown(void) {
    aclrtResetDevice(0);
    aclFinalize();
}

/* Run a compiled TBE kernel.
 * kernel_name: name of the registered kernel
 * args: array of void* pointers to NPU memory
 * arg_count: number of arguments
 */
EXPORT int npu_run_kernel(const char *kernel_name, void **args, int arg_count) {
    // This requires op registration - simplified for now
    // Full implementation would use aclopExecute or runtime API
    fprintf(stderr, "Kernel execution not yet implemented\n");
    return -1;
}
