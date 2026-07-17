/*
 * FINAL: Register TBE .o kernel via aclopCreateKernel + execute via aclopExecute.
 * This is the definitive NPU execution test for Ascend 310.
 *
 * Kernel: ssm_state_update.o (6 inputs, 2 outputs, 32 float16 heads)
 *
 * Compile:
 *   source /usr/local/Ascend/ascend-toolkit/set_env.sh
 *   gcc final_npu_test.c -o /tmp/final_npu_test \
 *       -I/usr/local/Ascend/ascend-toolkit/latest/include \
 *       -L/usr/local/Ascend/ascend-toolkit/latest/lib64 \
 *       -lascendcl -Wl,-rpath,/usr/local/Ascend/ascend-toolkit/latest/lib64
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "acl/acl.h"
#include "acl/acl_op.h"

int main() {
    aclError ret;
    int NH = 32, SL = 16;

    // Init NPU
    ret = aclInit(NULL);
    if (ret != ACL_SUCCESS && ret != 500000) { printf("Init failed: %d\n", ret); return -1; }
    ret = aclrtSetDevice(0);
    printf("Init OK\n");

    // ========== 1. Load compiled kernel ==========
    const char *kernel_path = "/root/qwythos_engine/kernel_meta/ssm_state_update.o";
    const char *kernel_name = "ssm_state_update__kernel0";  // from JSON
    const char *op_type = "SSMStateUpdateCustom";
    const char *kernel_id = "k0";

    FILE *f = fopen(kernel_path, "rb");
    if (!f) { printf("Can't open %s\n", kernel_path); return -1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);
    void *kernel_data = malloc(fsize);
    fread(kernel_data, 1, fsize, f);
    fclose(f);
    printf("Loaded kernel: %s (%ld bytes)\n", kernel_path, fsize);

    // Register kernel
    ret = aclopCreateKernel(op_type, kernel_id, kernel_name,
                             kernel_data, (int)fsize,
                             ACL_ENGINE_AICORE, NULL);
    if (ret != ACL_SUCCESS) {
        printf("aclopCreateKernel failed: %d\n", ret);
        free(kernel_data);
        return -1;
    }
    printf("Kernel registered: %s\n", op_type);

    // ========== 2. Create NPU data ==========
    size_t sz_vec = (size_t)NH * 2;    // 32 float16 = 64 bytes
    size_t sz_mat = (size_t)SL * NH * 2;  // 16x32 float16

    void *d_state, *d_alog, *d_dtb, *d_B, *d_C, *d_dt, *d_y, *d_sout;
    aclrtMalloc(&d_state, sz_vec, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_alog,  sz_vec, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_dtb,   sz_vec, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_B,    sz_mat, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_C,    sz_mat, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_dt,   sz_mat, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_y,    (size_t)SL*2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_sout, sz_vec, ACL_MEM_MALLOC_HUGE_FIRST);

    // Fill with test data (all zeros for simple test)
    aclrtMemset(d_state, sz_vec, 0, sz_vec);
    aclrtMemset(d_alog,  sz_vec, 0, sz_vec);
    aclrtMemset(d_dtb,   sz_vec, 0, sz_vec);
    aclrtMemset(d_B,    sz_mat, 0, sz_mat);
    aclrtMemset(d_C,    sz_mat, 0, sz_mat);
    aclrtMemset(d_dt,   sz_mat, 0, sz_mat);
    aclrtMemset(d_y,    (size_t)SL*2, 0, (size_t)SL*2);
    aclrtMemset(d_sout, sz_vec, 0, sz_vec);
    printf("NPU memory allocated and zeroed\n");

    // ========== 3. Execute kernel ==========
    // Create tensor descriptors
    int64_t dim1[1] = {NH};
    int64_t dim2[2] = {SL, NH};
    int64_t dimY[1] = {SL};
    int64_t dimSO[1] = {NH};

    aclTensorDesc *in_desc[6];
    aclTensorDesc *out_desc[2];
    aclDataBuffer *in_buf[6];
    aclDataBuffer *out_buf[2];

    // Inputs: state(32), alog(32), dtb(32), B(SL,32), C(SL,32), dt(SL,32)
    in_desc[0] = aclCreateTensorDesc(ACL_FLOAT16, 1, dim1, ACL_FORMAT_ND);
    in_desc[1] = aclCreateTensorDesc(ACL_FLOAT16, 1, dim1, ACL_FORMAT_ND);
    in_desc[2] = aclCreateTensorDesc(ACL_FLOAT16, 1, dim1, ACL_FORMAT_ND);
    in_desc[3] = aclCreateTensorDesc(ACL_FLOAT16, 2, dim2, ACL_FORMAT_ND);
    in_desc[4] = aclCreateTensorDesc(ACL_FLOAT16, 2, dim2, ACL_FORMAT_ND);
    in_desc[5] = aclCreateTensorDesc(ACL_FLOAT16, 2, dim2, ACL_FORMAT_ND);

    // Outputs: y(SL), sout(32)
    out_desc[0] = aclCreateTensorDesc(ACL_FLOAT16, 1, dimY, ACL_FORMAT_ND);
    out_desc[1] = aclCreateTensorDesc(ACL_FLOAT16, 1, dimSO, ACL_FORMAT_ND);

    in_buf[0] = aclCreateDataBuffer(d_state, sz_vec);
    in_buf[1] = aclCreateDataBuffer(d_alog,  sz_vec);
    in_buf[2] = aclCreateDataBuffer(d_dtb,   sz_vec);
    in_buf[3] = aclCreateDataBuffer(d_B,    sz_mat);
    in_buf[4] = aclCreateDataBuffer(d_C,    sz_mat);
    in_buf[5] = aclCreateDataBuffer(d_dt,   sz_mat);

    out_buf[0] = aclCreateDataBuffer(d_y,    (size_t)SL*2);
    out_buf[1] = aclCreateDataBuffer(d_sout, sz_vec);

    // Create stream and execute
    aclrtStream stream;
    aclrtCreateStream(&stream);

    ret = aclopExecute(op_type, 6, (const aclTensorDesc**)in_desc,
                       (const aclDataBuffer**)in_buf,
                       2, (const aclTensorDesc**)out_desc,
                       out_buf, NULL, stream);
    aclrtSynchronizeStream(stream);

    if (ret == ACL_SUCCESS) {
        printf("✅ KERNEL EXECUTED SUCCESSFULLY!\n");

        // Read results
        uint16_t *h_y = malloc((size_t)SL * 2);
        uint16_t *h_sout = malloc(sz_vec);
        aclrtMemcpy(h_y, (size_t)SL*2, d_y, (size_t)SL*2, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(h_sout, sz_vec, d_sout, sz_vec, ACL_MEMCPY_DEVICE_TO_HOST);

        printf("  y[0..4]     =");
        for (int i = 0; i < 5 && i < SL; i++) printf(" %04x", h_y[i]);
        printf("\n  state[0..4] =");
        for (int i = 0; i < 5 && i < NH; i++) printf(" %04x", h_sout[i]);
        printf("\n");

        free(h_y);
        free(h_sout);
    } else {
        const char *msg = aclGetRecentErrMsg();
        printf("❌ Kernel execution failed: ret=%d %s\n", ret, msg ? msg : "");
    }

    // ========== 4. Cleanup ==========
    for (int i = 0; i < 6; i++) {
        aclDestroyTensorDesc(in_desc[i]);
        aclDestroyDataBuffer(in_buf[i]);
    }
    for (int i = 0; i < 2; i++) {
        aclDestroyTensorDesc(out_desc[i]);
        aclDestroyDataBuffer(out_buf[i]);
    }

    aclrtFree(d_state); aclrtFree(d_alog); aclrtFree(d_dtb);
    aclrtFree(d_B); aclrtFree(d_C); aclrtFree(d_dt);
    aclrtFree(d_y); aclrtFree(d_sout);
    aclrtDestroyStream(stream);
    free(kernel_data);
    aclrtResetDevice(0);
    aclFinalize();

    printf("DONE\n");
    return ret == ACL_SUCCESS ? 0 : -1;
}
