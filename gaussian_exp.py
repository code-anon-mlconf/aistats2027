import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from utils import kl, sample_hatD, pN_kl_lower, upper_bound


torch.manual_seed(0)
np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

x = torch.linspace(-6, 6, 1000, device=device, dtype=dtype)

KLs = [1, 4, 8, 12]
Ns = [4, 8, 16, 32, 64, 128, 256]
Ks = [1, 16, 64]

B_lower, M_lower, R_outer = 20_000, 20_000, 10_000


def gaussian(mu):
    p = torch.exp(-0.5 * (x - mu) ** 2)
    return p / p.sum()


pi0 = gaussian(0)


def make_target(target, ref=pi0, tol=1e-8):
    lo, hi = 0.0, 6.0
    for _ in range(100):
        mu = (lo + hi) / 2
        p = gaussian(mu)
        actual = kl(p, ref).item()
        if abs(actual - target) < tol:
            break
        if actual < target:
            lo = mu
        else:
            hi = mu
    assert abs(actual - target) < tol, f"Target KL {target} not reached: got {actual}"
    return mu, p


targets = {D: make_target(D) for D in KLs}
rows = []
results = {}

for D in KLs:
    mu, pib = targets[D]
    Dtrue = kl(pib, pi0).item()
    results[D] = {}

    for N in Ns:
        lower = pN_kl_lower(pi0, pib, N, B_lower, M_lower)
        upper = upper_bound(pi0, pib, N)

        results[D][N] = dict(lower=lower, upper=upper, K={})

        for K in Ks:
            # Fresh samples for every path, generated in batches of 100 paths.
            hat = torch.cat([
                sample_hatD(pi0, pib, N, min(100, R_outer - start) * K)
                for start in range(0, R_outer, 100)
            ])
            est = hat.reshape(R_outer, K).mean(1)
            q = torch.quantile(
                est, torch.tensor([.025, .975], device=device, dtype=dtype)
            )

            vals = (est.mean().item(), q[0].item(), q[1].item())
            results[D][N]["K"][K] = vals

            rows.append([
                D, Dtrue, mu, N, K, lower, upper,
                hat.mean().item(), *vals
            ])

df = pd.DataFrame(rows, columns=[
    "target_KL", "actual_KL", "mu", "N", "K",
    "pN_lower", "upper", "E_hat",
    "K_mean", "K_q025", "K_q975"
])
df.to_csv("kl_experiment.csv", index=False)


# ------------------------------------------------------------
# Plot: distributions + raw KL
# ------------------------------------------------------------

fig, ax = plt.subplots(2, 4, figsize=(20, 8), sharey="row")
pos = np.arange(len(Ns))
offsets = np.linspace(-.25, .25, len(Ks))

for c, D in enumerate(KLs):
    mu, pib = targets[D]
    Dtrue = kl(pib, pi0).item()

    # distributions
    ax[0, c].plot(x.cpu(), pi0.cpu(), "--", label=r"$\pi_0$")
    ax[0, c].plot(x.cpu(), pib.cpu(), label=r"$\pi_\beta$")
    ax[0, c].set_title(fr"$D_{{KL}}={Dtrue:.0f}$, $\mu={mu:.2f}$")
    ax[0, c].grid(alpha=.2)

    if c == 0:
        ax[0, c].legend()

    lower = np.array([results[D][N]["lower"] for N in Ns])
    upper = np.array([results[D][N]["upper"] for N in Ns])

    a = ax[1, c]

    a.plot(pos, lower, "s--", label=r"$D_{KL}(p_N||\pi_0)$")
    a.plot(pos, upper, "^-.", label="Upper bound")
    a.axhline(Dtrue, ls=":", label="Population KL")

    for off, K in zip(offsets, Ks):
        vals = np.array([results[D][N]["K"][K] for N in Ns])
        mean, lo, hi = vals.T
        points, = a.plot(pos + off, mean, "o", ms=4, label=f"K={K}")
        a.errorbar(
            pos + off, (lo + hi) / 2,
            yerr=(hi - lo) / 2,
            fmt="none", capsize=2, ecolor=points.get_color()
        )

    a.set_xticks(pos, Ns)
    a.set_xlabel("N")
    a.grid(axis="y", alpha=.25)

    if c == 0:
        a.set_ylabel("KL (nats)")


ax[1, 0].set_ylim(0, 6)

handles, labels = ax[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4)

fig.tight_layout(rect=[0, .08, 1, 1])
plt.savefig("kl_experiment.png", dpi=300, bbox_inches="tight")
plt.show()