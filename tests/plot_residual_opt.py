import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Breakdown data (qkv_proj K=6144 N=9216, B200, CUDA-graph), ms.
M       = [128,   256,   512,   1024,  2048]
q1      = [0.0021, 0.0027, 0.0037, 0.0055, 0.0084]   # single-level quant (unchanged)
qres_lut= [0.0100, 0.0133, 0.0233, 0.0442, 0.0781]   # residual quant, branchy LUT
qres_cvt= [0.0028, 0.0036, 0.0055, 0.0092, 0.0155]   # residual quant, hw cvt.e2m1x2
res_lut = [0.0210, 0.0300, 0.0513, 0.1017, 0.1954]   # full residual path, LUT
res_cvt = [0.0139, 0.0203, 0.0335, 0.0664, 0.1312]   # full residual path, cvt
fp4     = [0.0107, 0.0118, 0.0180, 0.0313, 0.0596]   # single-level path (reference)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: residual-quant kernel time, before vs after.
ax[0].plot(M, qres_lut, "o-", color="tab:red", label="residual quant — LUT (before)")
ax[0].plot(M, qres_cvt, "s-", color="tab:green", label="residual quant — cvt.e2m1x2 (after)")
ax[0].plot(M, q1, "^--", color="tab:gray", label="single-level quant (reference)")
for x, a, b in zip(M, qres_lut, qres_cvt):
    ax[0].annotate(f"{a/b:.1f}x", (x, b), textcoords="offset points", xytext=(0, -14),
                   ha="center", fontsize=8, color="tab:green")
ax[0].set_xscale("log", base=2); ax[0].set_xticks(M); ax[0].set_xticklabels(M)
ax[0].set_xlabel("M"); ax[0].set_ylabel("residual-quant kernel time (ms)")
ax[0].set_title("Residual-quant kernel: 3.6–5.0x faster"); ax[0].grid(True, alpha=0.3); ax[0].legend()

# Panel 2: full residual path, before vs after (+ single-level reference).
ax[1].plot(M, res_lut, "o-", color="tab:red", label="residual path — LUT (before)")
ax[1].plot(M, res_cvt, "s-", color="tab:green", label="residual path — cvt (after)")
ax[1].plot(M, fp4, "^--", color="tab:blue", label="single-level FP4 path")
for x, a, b in zip(M, res_lut, res_cvt):
    ax[1].annotate(f"{a/b:.2f}x", (x, b), textcoords="offset points", xytext=(0, -14),
                   ha="center", fontsize=8, color="tab:green")
ax[1].set_xscale("log", base=2); ax[1].set_xticks(M); ax[1].set_xticklabels(M)
ax[1].set_xlabel("M"); ax[1].set_ylabel("full path time (ms)")
ax[1].set_title("Full residual path: 1.5x faster end-to-end"); ax[1].grid(True, alpha=0.3); ax[1].legend()

fig.suptitle("Residual-quant optimization: branchy LUT -> hardware cvt.e2m1x2  (B200, qkv_proj, breakdown)", fontsize=12)
fig.tight_layout()
fig.savefig("/tmp/residual_quant_opt.png", dpi=140, bbox_inches="tight")
print("saved")
