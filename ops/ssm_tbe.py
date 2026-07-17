"""Final TBE kernel: SSM state update with CORRECT tik API."""
from te import tik


def ssm_kernel_core(tik_inst, gm_state, gm_alog, gm_dtb,
                     gm_B, gm_C, gm_dt, gm_y, gm_sout,
                     seq_len, NH=32):
    """Build the SSM state update kernel body."""
    BLK = 16       # mask for float16 (16 elements per instruction)
    nb = NH // 16  # repeats = 2 for 32 elements

    # === UB tensors ===
    ub_s = tik_inst.Tensor("float16", (NH,), name="s", scope=tik.scope_ubuf)
    ub_al = tik_inst.Tensor("float16", (NH,), name="al", scope=tik.scope_ubuf)
    ub_db = tik_inst.Tensor("float16", (NH,), name="db", scope=tik.scope_ubuf)
    ub_B = tik_inst.Tensor("float16", (NH,), name="B", scope=tik.scope_ubuf)
    ub_C = tik_inst.Tensor("float16", (NH,), name="C", scope=tik.scope_ubuf)
    ub_dt = tik_inst.Tensor("float16", (NH,), name="dt", scope=tik.scope_ubuf)

    ub_t0 = tik_inst.Tensor("float16", (NH,), name="t0", scope=tik.scope_ubuf)
    ub_t1 = tik_inst.Tensor("float16", (NH,), name="t1", scope=tik.scope_ubuf)
    ub_t2 = tik_inst.Tensor("float16", (NH,), name="t2", scope=tik.scope_ubuf)
    ub_t3 = tik_inst.Tensor("float16", (NH,), name="t3", scope=tik.scope_ubuf)
    ub_t4 = tik_inst.Tensor("float16", (NH,), name="t4", scope=tik.scope_ubuf)
    ub_one = tik_inst.Tensor("float16", (NH,), name="one", scope=tik.scope_ubuf)
    ub_yh = tik_inst.Tensor("float16", (NH,), name="yh", scope=tik.scope_ubuf)
    ub_wk = tik_inst.Tensor("float16", (NH,), name="wk", scope=tik.scope_ubuf)
    ub_ys = tik_inst.Tensor("float16", (1,), name="ys", scope=tik.scope_ubuf)

    # === Load persistent data ===
    tik_inst.data_move(ub_s, gm_state, 0, 1, nb, 0, 0)
    tik_inst.data_move(ub_al, gm_alog, 0, 1, nb, 0, 0)
    tik_inst.data_move(ub_db, gm_dtb, 0, 1, nb, 0, 0)

    # === Init constants ===
    tik_inst.vec_dup(BLK, ub_one, 1.0, nb, 0)

    # === Pre-compute exp(A_log) into t0 ===
    tik_inst.vec_exp(BLK, ub_t0, ub_al, nb, 0, 0)

    # === Sequence loop ===
    with tik_inst.for_range(0, seq_len, name="sl") as i:
        tik_inst.data_move(ub_B, gm_B[i], 0, 1, nb, 0, 0)
        tik_inst.data_move(ub_C, gm_C[i], 0, 1, nb, 0, 0)
        tik_inst.data_move(ub_dt, gm_dt[i], 0, 1, nb, 0, 0)

        # 1. dt_softplus = ln(1 + exp(dt_bias + dt_input))
        tik_inst.vec_add(BLK, ub_t1, ub_db, ub_dt, nb, 0, 0, 0)   # t1 = bias + dt
        tik_inst.vec_exp(BLK, ub_t2, ub_t1, nb, 0, 0)              # t2 = exp(t1)
        tik_inst.vec_add(BLK, ub_t3, ub_one, ub_t2, nb, 0, 0, 0)   # t3 = 1 + t2
        tik_inst.vec_ln(BLK, ub_t4, ub_t3, nb, 0, 0)               # t4 = ln(t3)

        # 2. A = exp(-exp(A_log) * dt_softplus)
        #    t1 = exp(alog)[t0] * dt[t4]
        tik_inst.vec_mul(BLK, ub_t1, ub_t0, ub_t4, nb, 0, 0, 0)
        #    t2 = -t1
        #    vec_muls: (mask, dst, src, scalar, repeat, dst_s, src_s)
        tik_inst.vec_muls(BLK, ub_t2, ub_t1, -1.0, nb, 0, 0)
        #    t3 = exp(t2) = A
        tik_inst.vec_exp(BLK, ub_t3, ub_t2, nb, 0, 0)

        # 3. state_new = A * state_prev + B
        #    t0 = A * s
        tik_inst.vec_mul(BLK, ub_t0, ub_t3, ub_s, nb, 0, 0, 0)
        #    s = t0 + B  (state update)
        tik_inst.vec_add(BLK, ub_s, ub_t0, ub_B, nb, 0, 0, 0)

        # 4. Reduce: y = sum(C * state_new)
        tik_inst.vec_mul(BLK, ub_yh, ub_C, ub_s, nb, 0, 0, 0)
        #    vec_reduce_add(mask, dst, src, work_tensor, repeat, src_s)
        tik_inst.vec_reduce_add(BLK, ub_ys, ub_yh, ub_wk, nb, 0)

        # 5. Store y[t]
        tik_inst.data_move(gm_y[i], ub_ys, 1, 1, 1, 0, 0)

    # Store final state
    tik_inst.data_move(gm_sout, ub_s, 1, 1, nb, 0, 0)


def build_ssm_kernel(seq_len=16, kernel_name="ssm_state_update"):
    """Build and return the SSM kernel."""
    tik_inst = tik.Tik()
    NH = 32

    # GM tensors
    gm_state = tik_inst.Tensor("float16", (NH,), name="gm_state", scope=tik.scope_gm)
    gm_alog = tik_inst.Tensor("float16", (NH,), name="gm_alog", scope=tik.scope_gm)
    gm_dtb = tik_inst.Tensor("float16", (NH,), name="gm_dtb", scope=tik.scope_gm)
    gm_B = tik_inst.Tensor("float16", (seq_len, NH), name="gm_B", scope=tik.scope_gm)
    gm_C = tik_inst.Tensor("float16", (seq_len, NH), name="gm_C", scope=tik.scope_gm)
    gm_dt = tik_inst.Tensor("float16", (seq_len, NH), name="gm_dt", scope=tik.scope_gm)
    gm_y = tik_inst.Tensor("float16", (seq_len,), name="gm_yout", scope=tik.scope_gm)
    gm_sout = tik_inst.Tensor("float16", (NH,), name="gm_sout", scope=tik.scope_gm)

    ssm_kernel_core(tik_inst, gm_state, gm_alog, gm_dtb,
                     gm_B, gm_C, gm_dt, gm_y, gm_sout,
                     seq_len, NH)

    tik_inst.BuildCCE(
        kernel_name=kernel_name,
        inputs=(gm_state, gm_alog, gm_dtb, gm_B, gm_C, gm_dt),
        outputs=(gm_y, gm_sout),
        enable_l2=True,
    )
    return tik_inst


if __name__ == "__main__":
    print("Compiling SSM TBE kernel...")
    try:
        build_ssm_kernel(seq_len=16, kernel_name="ssm_state_update")
        print("✅ Kernel built successfully!")
    except Exception as e:
        import traceback
        print(f"❌ {e}")
        traceback.print_exc()
