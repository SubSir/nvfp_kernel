import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K, N = 6144, 9216
M = []
mean = {"bf16": [], "fp4": [], "resid": []}
std = {"bf16": [], "fp4": [], "resid": []}
for line in open("/tmp/cmp_data.txt"):
    if not line.startswith("DATA,"):
        continue
    p = line.strip().split(",")
    M.append(int(p[1]))
    mean["bf16"].append(float(p[2])); std["bf16"].append(float(p[3]))
    mean["fp4"].append(float(p[4]));  std["fp4"].append(float(p[5]))
    mean["resid"].append(float(p[6])); std["resid"].append(float(p[7]))

styles = {"bf16": ("bf16 (cuBLAS)", "tab:gray", "o"),
          "fp4": ("single W4A4", "tab:blue", "s"),
          "resid": ("residual W4A4 (2-level)", "tab:red", "^")}

def toks(key):
    y = [m / t / 1e3 for m, t in zip(M, mean[key])]
    e = [yi * (s / t) for yi, t, s in zip(y, mean[key], std[key])]
    return y, e

def tflops(key):
    y = [2.0 * m * K * N / (t * 1e-3) / 1e12 for m, t in zip(M, mean[key])]
    e = [yi * (s / t) for yi, t, s in zip(y, mean[key], std[key])]
    return y, e

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

for key in ("bf16", "fp4", "resid"):
    lab, c, mk = styles[key]
    y, e = toks(key)
    ax[0].errorbar(M, y, yerr=e, marker=mk, color=c, label=lab, capsize=3)
ax[0].set_xscale("log", base=2); ax[0].set_xticks(M); ax[0].set_xticklabels(M, rotation=45)
ax[0].set_xlabel("M (decode batch / tokens)"); ax[0].set_ylabel("throughput  (M tokens / s)")
ax[0].set_title("Useful token throughput  (mean ± std)"); ax[0].grid(True, alpha=0.3); ax[0].legend()

for key in ("bf16", "fp4", "resid"):
    lab, c, mk = styles[key]
    y, e = tflops(key)
    ax[1].errorbar(M, y, yerr=e, marker=mk, color=c, label=lab, capsize=3)
ax[1].axhline(7702, ls="--", color="green", alpha=0.6, label="FP4 peak, fp16-accum (7702)")
ax[1].axhline(4500, ls=":", color="purple", alpha=0.7, label="FP4 ~fp32-accum ceiling (~4500)")
ax[1].set_xscale("log", base=2); ax[1].set_xticks(M); ax[1].set_xticklabels(M, rotation=45)
ax[1].set_xlabel("M (decode batch / tokens)"); ax[1].set_ylabel("useful throughput  (TFLOPS)")
ax[1].set_title("Effective compute throughput  (mean ± std)"); ax[1].grid(True, alpha=0.3); ax[1].legend(fontsize=8)

fig.suptitle("bf16 vs single-W4A4 vs residual-W4A4 (optimized)  -  B200, qkv_proj K=6144 N=9216, CUDA-graph", fontsize=12)
fig.tight_layout()
fig.savefig("/tmp/compare_throughput.png", dpi=140, bbox_inches="tight")
print("saved")
