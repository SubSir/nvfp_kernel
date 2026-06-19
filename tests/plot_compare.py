import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K, N = 6144, 9216
M = []; tb = []; tf = []; tr = []
for line in open("/tmp/cmp_data.txt"):
    if not line.startswith("DATA,"):
        continue
    p = line.strip().split(",")
    M.append(int(p[1])); tb.append(float(p[2])); tf.append(float(p[4])); tr.append(float(p[6]))

def toks(ts):
    return [m / t / 1e3 for m, t in zip(M, ts)]

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# Left: throughput (Mtok/s)
ax[0].plot(M, toks(tb), "o-", color="tab:gray", label="bf16 (cuBLAS)")
ax[0].plot(M, toks(tf), "s-", color="tab:blue", label="single W4A4")
ax[0].plot(M, toks(tr), "^-", color="tab:red", label="residual W4A4 (2-level)")
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
ax[0].set_xticks(M); ax[0].set_xticklabels(M, rotation=45)
ax[0].set_xlabel("M (decode batch / tokens)"); ax[0].set_ylabel("throughput  (M tokens / s)")
ax[0].set_title("Throughput"); ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()

# Right: speedup vs bf16
sf = [b / f for b, f in zip(tb, tf)]
sr = [b / r for b, r in zip(tb, tr)]
ax[1].axhline(1.0, color="tab:gray", ls="--", label="bf16 baseline")
ax[1].plot(M, sf, "s-", color="tab:blue", label="single W4A4")
ax[1].plot(M, sr, "^-", color="tab:red", label="residual W4A4")
for x, y in zip(M, sf):
    ax[1].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color="tab:blue")
for x, y in zip(M, sr):
    ax[1].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7, color="tab:red")
ax[1].set_xscale("log", base=2); ax[1].set_xticks(M); ax[1].set_xticklabels(M, rotation=45)
ax[1].set_xlabel("M (decode batch / tokens)"); ax[1].set_ylabel("speedup vs bf16  (x)")
ax[1].set_title("Speedup over bf16  (higher = faster)"); ax[1].grid(True, alpha=0.3); ax[1].legend()

fig.suptitle("bf16 vs single-W4A4 vs residual-W4A4 (optimized)  -  B200, qkv_proj K=6144 N=9216, CUDA-graph\n"
             "quant kernel + GEMM (+ residual add); residual is sqrt(2)x more accurate than single W4A4", fontsize=11)
fig.tight_layout()
fig.savefig("/tmp/compare_bf16_w4a4.png", dpi=140, bbox_inches="tight")
print("saved")
