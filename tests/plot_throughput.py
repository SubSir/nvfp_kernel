import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K, N = 6144, 9216
M, tb, tf, tr = [], [], [], []
for line in open("/tmp/thr_data.txt"):
    if line.startswith("DATA,"):
        _, m, a, b, c = line.strip().split(",")
        M.append(int(m)); tb.append(float(a)); tf.append(float(b)); tr.append(float(c))

def toks(ts):   # Mtokens/s  (M useful tokens / latency_ms)
    return [m / t / 1e3 for m, t in zip(M, ts)]
def tflops(ts): # useful TFLOPS = 2*M*K*N / latency
    return [2.0 * m * K * N / (t * 1e-3) / 1e12 for m, t in zip(M, ts)]

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

for y, lab, c, mk in [(toks(tb), "bf16 (cuBLAS)", "tab:gray", "o"),
                      (toks(tf), "FP4 single-level", "tab:blue", "s"),
                      (toks(tr), "FP4 residual (2-level)", "tab:red", "^")]:
    ax[0].plot(M, y, marker=mk, label=lab, color=c)
ax[0].set_xscale("log", base=2); ax[0].set_xticks(M); ax[0].set_xticklabels(M, rotation=45)
ax[0].set_xlabel("M (decode batch / tokens)"); ax[0].set_ylabel("throughput  (M tokens / s)")
ax[0].set_title("Useful token throughput"); ax[0].grid(True, alpha=0.3); ax[0].legend()

for y, lab, c, mk in [(tflops(tb), "bf16 (cuBLAS)", "tab:gray", "o"),
                      (tflops(tf), "FP4 single-level", "tab:blue", "s"),
                      (tflops(tr), "FP4 residual (2-level)", "tab:red", "^")]:
    ax[1].plot(M, y, marker=mk, label=lab, color=c)
ax[1].axhline(7702, ls="--", color="green", alpha=0.6, label="FP4 peak, fp16-accum (7702)")
ax[1].axhline(4700, ls=":", color="purple", alpha=0.7, label="FP4 measured, fp32-accum (~4700)")
ax[1].set_xscale("log", base=2); ax[1].set_xticks(M); ax[1].set_xticklabels(M, rotation=45)
ax[1].set_xlabel("M (decode batch / tokens)"); ax[1].set_ylabel("useful throughput  (TFLOPS)")
ax[1].set_title("Effective compute throughput"); ax[1].grid(True, alpha=0.3); ax[1].legend(fontsize=8)

fig.suptitle(f"B200 NVFP4 GEMM throughput vs M  (MiniMax-M3 qkv_proj, K={K}, N={N}, CUDA-graph)", fontsize=12)
fig.tight_layout()
fig.savefig("/tmp/throughput_vs_M.png", dpi=140, bbox_inches="tight")
print("saved /tmp/throughput_vs_M.png")
