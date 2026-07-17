/*
 * Load & execute ATC-compiled .om model on NPU via aclmdl.
 *
 * Compile:
 *   source /usr/local/Ascend/ascend-toolkit/set_env.sh
 *   gcc run_om_model.c -o /tmp/run_om \
 *       -I/usr/local/Ascend/ascend-toolkit/7.0.0/include \
 *       -L/usr/local/Ascend/ascend-toolkit/7.0.0/x86_64-linux/lib64 \
 *       -lascendcl -Wl,-rpath,.../lib64
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "acl/acl.h"

int main() {
    aclError ret;

    // Init
    ret = aclInit(NULL);
    if (ret != ACL_SUCCESS && ret != 500000) return -1;
    ret = aclrtSetDevice(0);
    printf("Init OK\n");

    // Load .om model
    uint32_t model_id;
    ret = aclmdlLoadFromFile("/root/qwythos_engine/matmul_simple.om", &model_id);
    printf("aclmdlLoad: %d (model_id=%u)\n", ret, model_id);

    if (ret == ACL_SUCCESS) {
        // Get model info
        aclmdlDesc *desc = aclmdlCreateDesc();
        aclmdlGetDesc(desc, model_id);

        // Get input/output sizes
        size_t in_sz0 = aclmdlGetInputSizeByIndex(desc, 0);
        size_t in_sz1 = aclmdlGetInputSizeByIndex(desc, 1);
        size_t out_sz = aclmdlGetOutputSizeByIndex(desc, 0);
        printf("Input sizes: %zu, %zu, Output: %zu\n", in_sz0, in_sz1, out_sz);

        // Allocate NPU memory
        void *in0, *in1, *out;
        aclrtMalloc(&in0, in_sz0, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&in1, in_sz1, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&out, out_sz, ACL_MEM_MALLOC_HUGE_FIRST);

        // Fill with test data (all 2.0 and 3.0 in fp16)
        uint16_t *h0 = malloc(in_sz0);
        uint16_t *h1 = malloc(in_sz1);
        int n0 = in_sz0 / 2, n1 = in_sz1 / 2;
        for (int i = 0; i < n0; i++) h0[i] = 0x4000;  // 2.0 fp16
        for (int i = 0; i < n1; i++) h1[i] = 0x4200;  // 3.0 fp16
        aclrtMemcpy(in0, in_sz0, h0, in_sz0, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(in1, in_sz1, h1, in_sz1, ACL_MEMCPY_HOST_TO_DEVICE);
        free(h0); free(h1);

        // Prepare input/output datasets
        aclmdlDataset *input_dataset = aclmdlCreateDataset();
        aclmdlDataset *output_dataset = aclmdlCreateDataset();
        aclmdlAddDatasetBuffer(input_dataset, aclCreateDataBuffer(in0, in_sz0));
        aclmdlAddDatasetBuffer(input_dataset, aclCreateDataBuffer(in1, in_sz1));
        aclmdlAddDatasetBuffer(output_dataset, aclCreateDataBuffer(out, out_sz));

        // Execute model
        ret = aclmdlExecute(model_id, input_dataset, output_dataset);
        printf("aclmdlExecute: %d\n", ret);

        // Read output
        if (ret == ACL_SUCCESS) {
            float *h_out = malloc(out_sz);
            aclrtMemcpy(h_out, out_sz, out, out_sz, ACL_MEMCPY_DEVICE_TO_HOST);
            // Interpret as fp16
            uint16_t *h16 = (uint16_t*)h_out;
            int exp = (h16[0]>>10)&0x1F, mant=h16[0]&0x3FF;
            float f = exp==0 ? 0 : (1.0f+mant/1024.0f)*(1<<(exp-15));
            printf("C[0,0] = %0.1f (expected 96.0)\n", f);
            free(h_out);
        }

        aclrtFree(in0); aclrtFree(in1); aclrtFree(out);
        aclmdlDestroyDataset(input_dataset);
        aclmdlDestroyDataset(output_dataset);
        aclmdlDestroyDesc(desc);
        aclmdlUnload(model_id);
    }

    aclrtResetDevice(0);
    aclFinalize();
    printf("DONE\n");
    return ret;
}
