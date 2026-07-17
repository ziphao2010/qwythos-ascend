"""
Tik matmul for Ascend 310: use FP32 output (Cube Unit limitation).
Ascend 310 Cube: f16xf16->f16 NOT supported, use f16xf16->f32.
"""
from te import tik

def tik_matmul_fp32(M, N, K, kernel_name="tik_matmul_fp32"):
    tik_inst = tik.Tik()
    BM, BN, BK = 16, 16, 16

    gm_a = tik_inst.Tensor("float16", (M, K), name="A", scope=tik.scope_gm)
    gm_b = tik_inst.Tensor("float16", (K, N), name="B", scope=tik.scope_gm)
    gm_c = tik_inst.Tensor("float32", (M, N), name="C", scope=tik.scope_gm)

    for m in range(M // BM):
        for n in range(N // BN):
            c_l0c = tik_inst.Tensor("float32", (BM, BN),
                                    name="c", scope=tik.scope_cc)

            for k in range(K // BK):
                a_l1 = tik_inst.Tensor("float16", (BM, BK),
                                       name="a", scope=tik.scope_cbuf)
                tik_inst.data_move(a_l1, gm_a[m*BM*K + k*BK],
                                   0, 1, BM*BK//16, 0, 0)

                b_l1 = tik_inst.Tensor("float16", (BK, BN),
                                       name="b", scope=tik.scope_cbuf)
                tik_inst.data_move(b_l1, gm_b[k*BK*N + n*BN],
                                   0, 1, BK*BN//16, 0, 0)

                # FP16 x FP16 -> FP32 matmul
                tik_inst.matmul(c_l0c, a_l1, b_l1, BM, BK, BN)

            # L0C -> UB -> GM (sid 0=normal, 1=buffer)
            c_ub = tik_inst.Tensor("float32", (BM, BN),
                                   name="c_ub", scope=tik.scope_ubuf)
            tik_inst.data_move(c_ub, c_l0c, 1, 1, BM*BN//8, 0, 0)
            tik_inst.data_move(gm_c[m*BM*N + n*BN], c_ub,
                               0, 1, BM*BN//8, 0, 0)

    tik_inst.BuildCCE(
        kernel_name=kernel_name,
        inputs=(gm_a, gm_b),
        outputs=(gm_c,),
        enable_l2=True,
    )
    return tik_inst

tik_matmul_fp32(64, 64, 64, "tik_matmul_fp32")
print("✅ Tik matmul fp32 built!")
