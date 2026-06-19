import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K, N = 6144, 9216
M    = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
bf16 = [0.0226, 0.0206, 0.0226, 0.0227, 0.0269, 0.0410, 0.0756, 0.1447, 0.2810, 0.5602]
sgl  = [0.0545, 0.0527, 0.0567, 0.0562, 0.0570, 0.0596, 0.0752, 0.1077, 0.1788, 0.3209]
res  = [0.0570, 0.0600, 0.0621, 0.0626, 0.0639, 0.0797, 0.1132, 0.1812, 0.3165, 0.5864]

def toks(ts):
    return [m / t / 1e3 for m, t in zip(M, ts)]

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

ax[0].plot(M, toks(bf16), "o-", color="tab:gray", label="bf16 (cuBLAS)")
ax[0].plot(M, toks(sgl), "s-", color="tab:blue", label="single W4A4")
ax[0].plot(M, toks(res), "^-", color="tab:red", label="residual W4A4 (2-level)")
ax[0].set_xscale("log", base=2); ax[0].set_xticks(M); ax[0].set_xticklabels(M, rotation=45)
ax[0].set_xlabel("M (decode batch / tokens)"); ax[0].set_ylabel("throughput  (M tokens / s)")
ax[0].set_title("Throughput  (EAGER, no CUDA graph)"); ax[0].grid(True, alpha=0.3); ax[0].legend()

sf = [b / f for b, f in zip(bf16, sgl)]
sr = [b / r for b, r in zip(bf16, res)]
ax[1].axhline(1.0, color="tab:gray", ls="--", label="bf16 baseline")
ax[1].plot(M, sf, "s-", color="tab:blue", label="single W4A4")
ax[1].plot(M, sr, "^-", color="tab:red", label="residual W4A4")
ax[1].set_xscale("log", base=2); ax[1].set_xticks(M); ax[1].set_xticklabels(M, rotation=45)
ax[1].set_xlabel("M (decode batch / tokens)"); ax[1].set_ylabel("speedup vs bf16  (x)")
ax[1].set_title("Speedup over bf16  (>1 = faster; EAGER)"); ax[1].grid(True, alpha=0.3); ax[1].legend()

fig.suptitle("EAGER (no CUDA graph): residual W4A4 stays below bf16; single W4A4 crosses bf16 at M~1024\n"
             "B200, qkv_proj K=6144 N=9216  -  launch-overhead floor dominates the FP4 paths", fontsize=11)
fig.tight_layout()
fig.savefig("/tmp/compare_eager.png", dpi=140, bbox_inches="tight")
print("saved")
