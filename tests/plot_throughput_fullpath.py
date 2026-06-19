import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K, N = 6144, 9216
M = []
d = {"bf16": ([], []), "fp4": ([], []), "resid": ([], [])}
for line in open("/tmp/thr_data4.txt"):
    if not line.startswith("DATA,"):
        continue
    p = line.strip().split(",")
    m = int(p[1]); M.append(m)
    d["bf16"][0].append(float(p[2])); d["bf16"][1].append(float(p[3]))
    d["fp4"][0].append(float(p[4]));  d["fp4"][1].append(float(p[5]))
    d["resid"][0].append(float(p[6])); d["resid"][1].append(float(p[7]))

styles = {"bf16": ("bf16 (cuBLAS)", "tab:gray", "o"),
          "fp4": ("FP4 single-level", "tab:blue", "s"),
          "resid": ("FP4 residual (2-level)", "tab:red", "^")}

def toks(mean, std):           # M tokens / s -> Mtok/s, with propagated error
    y = [m / t / 1e3 for m, t in zip(M, mean)]
    e = [yi * (s / t) for yi, t, s in zip(y, mean, std)]
    return y, e

def tflops(mean, std):         # useful TFLOPS = 2*M*K*N / latency
    y = [2.0 * m * K * N / (t * 1e-3) / 1e12 for m, t in zip(M, mean)]
    e = [yi * (s / t) for yi, t, s in zip(y, mean, std)]
    return y, e

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

for key in ("bf16", "fp4", "resid"):
    lab, c, mk = styles[key]
    y, e = toks(*d[key])
    ax[0].errorbar(M, y, yerr=e, marker=mk, label=lab, color=c, capsize=3)
ax[0].set_xscale("log", base=2); ax[0].set_xticks(M); ax[0].set_xticklabels(M, rotation=45)
ax[0].set_xlabel("M (decode batch / tokens)"); ax[0].set_ylabel("throughput  (M tokens / s)")
ax[0].set_title("Useful token throughput  (mean ± std)"); ax[0].grid(True, alpha=0.3); ax[0].legend()

for key in ("bf16", "fp4", "resid"):
    lab, c, mk = styles[key]
    y, e = tflops(*d[key])
    ax[1].errorbar(M, y, yerr=e, marker=mk, label=lab, color=c, capsize=3)
ax[1].axhline(7702, ls="--", color="green", alpha=0.6, label="FP4 peak, fp16-accum (7702)")
ax[1].axhline(4500, ls=":", color="purple", alpha=0.7, label="FP4 ~fp32-accum ceiling (~4500)")
ax[1].set_xscale("log", base=2); ax[1].set_xticks(M); ax[1].set_xticklabels(M, rotation=45)
ax[1].set_xlabel("M (decode batch / tokens)"); ax[1].set_ylabel("useful throughput  (TFLOPS)")
ax[1].set_title("Effective compute throughput  (mean ± std)"); ax[1].grid(True, alpha=0.3); ax[1].legend(fontsize=8)

fig.suptitle(f"B200 NVFP4 GEMM throughput vs M  (MiniMax-M3 qkv_proj, K={K}, N={N}, full decode path: quant+GEMM(+add), mean of 60)", fontsize=12)
fig.tight_layout()
fig.savefig("/tmp/throughput_vs_M_fullpath.png", dpi=140, bbox_inches="tight")
print("saved")
