import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.special import gammaln

from utils import kl, sample_hatD, upper_bound


# ============================================================
# Configuration
# ============================================================

L = 8

N_VALUES = np.arange(1, 16)

BETAS = np.array([
    4.0,
    8.0,
    12.0,
])

MC_REPS = 200_000
MC_CHUNK = 20_000

SEED = 12345

OUTPUT_NPZ = "softbon_alignment_three_beta_rows.npz"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DTYPE = torch.float64

print("Device:", DEVICE)


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Plot style
# ============================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})


# ============================================================
# Enumerate compositions
#
# c_1 + ... + c_L = N
# ============================================================

def compositions(n, k, prefix=()):
    if k == 1:
        yield prefix + (n,)
        return

    for c in range(n + 1):
        yield from compositions(
            n - c,
            k - 1,
            prefix + (c,),
        )


# ============================================================
# Limiting tilted target
#
# P_beta(y) proportional to Q(y) exp(beta r(y))
# ============================================================

def softbon_target(q, reward, beta):
    q = np.asarray(q, dtype=np.float64)
    reward = np.asarray(reward, dtype=np.float64)

    logits = np.log(q) + beta * reward
    logits -= np.max(logits)

    p_beta = np.exp(logits)
    p_beta /= p_beta.sum()

    return p_beta


# ============================================================
# Exact finite-N Soft-BoN output distribution
# ============================================================

def exact_softbon_output(
    q,
    reward,
    beta,
    N,
):

    q = np.asarray(q, dtype=np.float64)
    reward = np.asarray(reward, dtype=np.float64)

    q = q / q.sum()

    # State weight: w_j = exp(beta * reward_j).
    score_weights = np.exp(
        beta * reward
    )


    # ========================================================
    # Enumerate count vectors c: c_j >= 0 and sum_j c_j = N.
    # c_j is the number of times state j appears among N draws from q.
    # ========================================================

    counts = np.asarray(
        list(
            compositions(
                int(N),
                len(q),
            )
        ),
        dtype=np.int16,
    )


    # ========================================================
    # Exact multinomial probability
    # P(C=c) = [N! / prod_j c_j!] * prod_j q_j**c_j.
    # N! / prod_j c_j! counts the ordered draws with the same counts.
    # gammaln(n + 1) = log(Gamma(n + 1)) = log(n!).
    # log P(C=c) = log(N!) - sum_j log(c_j!) + sum_j c_j * log(q_j).
    # ========================================================

    log_prob = (
        gammaln(N + 1)
        - np.sum(
            gammaln(counts + 1),
            axis=1,
        )
        + counts @ np.log(q)
    )

    prob = np.exp(log_prob)
    prob /= prob.sum()  # Correct floating-point drift; sum_c P(C=c) = 1.


    # ========================================================
    # Soft-BoN denominator: Z(c) = sum_j c_j * w_j.
    # ========================================================

    Z = counts @ score_weights


    # ========================================================
    # Conditional output distribution: P(Y=j | C=c) = c_j * w_j / Z(c).
    # Each of the c_j copies of state j contributes weight w_j.
    # ========================================================

    conditional_output = (
        counts
        * score_weights[None, :]
        / Z[:, None]
    )


    # ========================================================
    # Exact marginal: p_N(j) = sum_c P(C=c) * P(Y=j | C=c).
    # Weight each count-vector row by its probability, then sum rows.
    # ========================================================

    pN = np.sum(
        prob[:, None]
        * conditional_output,
        axis=0,
    )

    pN /= pN.sum()  # Correct floating-point drift; sum_j p_N(j) = 1.


    # ========================================================
    # KL(p_N || q) = sum_j p_N(j) * log(p_N(j) / q_j).
    # ========================================================

    kl_pN_q = kl(
        torch.as_tensor(pN, dtype=DTYPE),
        torch.as_tensor(q, dtype=DTYPE),
    ).item()


    return {
        "pN": pN,
        "kl_pN_q": kl_pN_q,
        "num_count_vectors": len(counts),
    }


# ============================================================
# Monte Carlo E[Dhat_N]
# ============================================================

@torch.no_grad()
def monte_carlo_softbon_Dhat(q, p_beta, N, num_reps, chunk_size):
    samples = torch.cat([
        sample_hatD(q, p_beta, N, min(chunk_size, num_reps - start))
        for start in range(0, num_reps, chunk_size)
    ])
    mean_Dhat = samples.mean().item()
    std_Dhat = samples.std(unbiased=num_reps > 1).item()
    return {
        "mean_Dhat": mean_Dhat,
        "std_Dhat": std_Dhat,
        "se_Dhat": std_Dhat / math.sqrt(num_reps),
    }


# ============================================================
# Reference distribution
# ============================================================

Q = np.array([
    0.30,
    0.22,
    0.16,
    0.11,
    0.08,
    0.06,
    0.04,
    0.03,
])

Q /= Q.sum()
q_t = torch.as_tensor(Q, dtype=DTYPE, device=DEVICE)


# ============================================================
# Reward scenarios
# ============================================================

SCENARIOS = []


# ============================================================
# 1. Aligned
# ============================================================

r1 = np.array([
    1.00,
    0.85,
    0.70,
    0.55,
    0.40,
    0.25,
    0.10,
    0.00,
])

SCENARIOS.append({
    "name": "Aligned",
    "reward": r1,
})


# ============================================================
# 2. Mildly misaligned
# ============================================================

r2 = np.array([
    0.80,
    1.00,
    0.55,
    0.70,
    0.30,
    0.45,
    0.10,
    0.00,
])

SCENARIOS.append({
    "name": "Mildly misaligned",
    "reward": r2,
})


# ============================================================
# 3. Misaligned
# ============================================================

r3 = np.array([
    0.20,
    0.35,
    0.10,
    0.60,
    0.45,
    0.80,
    0.65,
    1.00,
])

SCENARIOS.append({
    "name": "Misaligned",
    "reward": r3,
})


# ============================================================
# 4. Anti-aligned
# ============================================================

r4 = np.array([
    0.00,
    0.10,
    0.20,
    0.30,
    0.45,
    0.60,
    0.80,
    1.00,
])

SCENARIOS.append({
    "name": "Anti-aligned",
    "reward": r4,
})


# ============================================================
# Dimensions
# ============================================================

NUM_SCENARIOS = len(
    SCENARIOS
)

NUM_BETAS = len(
    BETAS
)

NUM_N = len(
    N_VALUES
)


# ============================================================
# Allocate arrays
# ============================================================

REWARDS = np.zeros(
    (
        NUM_SCENARIOS,
        L,
    ),
    dtype=np.float64,
)


TARGET_P = np.zeros(
    (
        NUM_SCENARIOS,
        NUM_BETAS,
        L,
    ),
    dtype=np.float64,
)


TARGET_KL = np.zeros(
    (
        NUM_SCENARIOS,
        NUM_BETAS,
    ),
    dtype=np.float64,
)


PN_ALL = np.zeros(
    (
        NUM_SCENARIOS,
        NUM_BETAS,
        NUM_N,
        L,
    ),
    dtype=np.float64,
)


KL_PN_Q = np.zeros(
    (
        NUM_SCENARIOS,
        NUM_BETAS,
        NUM_N,
    ),
    dtype=np.float64,
)


EXPECTED_DHAT_MC = np.zeros(
    (
        NUM_SCENARIOS,
        NUM_BETAS,
        NUM_N,
    ),
    dtype=np.float64,
)


DHAT_STD_MC = np.zeros_like(
    EXPECTED_DHAT_MC
)


DHAT_SE_MC = np.zeros_like(
    EXPECTED_DHAT_MC
)


AMINIAN_BOUND = np.zeros_like(EXPECTED_DHAT_MC)


NUM_COUNT_VECTORS = np.zeros(
    NUM_N,
    dtype=np.int64,
)


# ============================================================
# Run experiment
# ============================================================

for scenario_idx, scenario in enumerate(
    SCENARIOS
):

    reward = scenario[
        "reward"
    ].copy()

    REWARDS[
        scenario_idx
    ] = reward


    print()
    print("=" * 100)
    print(
        f"SCENARIO {scenario_idx + 1}: "
        f"{scenario['name']}"
    )
    print("=" * 100)


    for beta_idx, beta in enumerate(
        BETAS
    ):

        print()
        print("-" * 100)
        print(
            f"beta = {beta:g}"
        )
        print("-" * 100)


        # ====================================================
        # Limiting target
        # ====================================================

        p_beta = softbon_target(
            Q,
            reward,
            float(beta),
        )
        p_beta_t = torch.as_tensor(p_beta, dtype=DTYPE, device=DEVICE)

        TARGET_P[
            scenario_idx,
            beta_idx,
            :
        ] = p_beta


        target_kl = kl(p_beta_t, q_t).item()

        TARGET_KL[
            scenario_idx,
            beta_idx
        ] = target_kl


        print(
            f"KL(P_beta || Q) = "
            f"{target_kl:.9f}"
        )


        # ====================================================
        # Loop over N
        # ====================================================

        for n_idx, N in enumerate(
            N_VALUES
        ):

            # ================================================
            # Exact finite-N output
            # ================================================

            exact_out = exact_softbon_output(
                Q,
                reward,
                float(beta),
                int(N),
            )


            PN_ALL[
                scenario_idx,
                beta_idx,
                n_idx,
                :
            ] = exact_out[
                "pN"
            ]


            KL_PN_Q[
                scenario_idx,
                beta_idx,
                n_idx
            ] = exact_out[
                "kl_pN_q"
            ]


            if (
                scenario_idx == 0
                and beta_idx == 0
            ):

                NUM_COUNT_VECTORS[
                    n_idx
                ] = exact_out[
                    "num_count_vectors"
                ]


            # ================================================
            # MC estimator
            # ================================================

            mc_out = monte_carlo_softbon_Dhat(
                q_t,
                p_beta_t,
                int(N),
                MC_REPS,
                MC_CHUNK,
            )
            AMINIAN_BOUND[scenario_idx, beta_idx, n_idx] = upper_bound(
                q_t, p_beta_t, int(N)
            )


            EXPECTED_DHAT_MC[
                scenario_idx,
                beta_idx,
                n_idx
            ] = mc_out[
                "mean_Dhat"
            ]


            DHAT_STD_MC[
                scenario_idx,
                beta_idx,
                n_idx
            ] = mc_out[
                "std_Dhat"
            ]


            DHAT_SE_MC[
                scenario_idx,
                beta_idx,
                n_idx
            ] = mc_out[
                "se_Dhat"
            ]


            print(
                f"N={N:2d}   "
                f"KL(P_N || Q)="
                f"{exact_out['kl_pN_q']: .8f}   "
                f"E[Dhat_N]="
                f"{mc_out['mean_Dhat']: .8f}   "
                f"SE="
                f"{mc_out['se_Dhat']:.2e}   "
                f"Aminian="
                f"{AMINIAN_BOUND[scenario_idx, beta_idx, n_idx]: .8f}"
            )


# ============================================================
# Save arrays
# ============================================================

SCENARIO_NAMES = np.array(
    [
        scenario["name"]
        for scenario in SCENARIOS
    ],
    dtype="U32",
)


np.savez_compressed(
    OUTPUT_NPZ,

    scenario_names=SCENARIO_NAMES,
    N_values=N_VALUES,
    betas=BETAS,

    L=np.array(L),

    mc_reps=np.array(
        MC_REPS
    ),

    mc_chunk=np.array(
        MC_CHUNK
    ),

    seed=np.array(
        SEED
    ),

    Q=Q,
    rewards=REWARDS,

    target_P=TARGET_P,
    target_kl=TARGET_KL,

    P_N=PN_ALL,
    kl_PN_Q=KL_PN_Q,

    expected_Dhat_mc=EXPECTED_DHAT_MC,
    Dhat_std_mc=DHAT_STD_MC,
    Dhat_se_mc=DHAT_SE_MC,

    aminian_bound=AMINIAN_BOUND,

    num_count_vectors=NUM_COUNT_VECTORS,
)


print()
print("=" * 100)
print("Saved:", OUTPUT_NPZ)
print("=" * 100)


# ============================================================
# Main figure
#
# 4 columns:
#
# Aligned
# Mildly misaligned
# Misaligned
# Anti-aligned
#
#
# Row 1:
# Q versus reward
#
# Row 2:
# beta = 1
#
# Row 3:
# beta = 2
#
# Row 4:
# beta = 4
# ============================================================

fig, axes = plt.subplots(
    4,
    4,
    figsize=(13.8, 10.2),
)


states = np.arange(
    1,
    L + 1
)

bar_width = 0.36


# ============================================================
# Top row
# ============================================================

for scenario_idx, scenario in enumerate(
    SCENARIOS
):

    ax = axes[
        0,
        scenario_idx
    ]

    reward = REWARDS[
        scenario_idx
    ]


    ax.bar(
        states - bar_width / 2,
        Q,
        width=bar_width,
        label="Q(y)",
    )


    ax.bar(
        states + bar_width / 2,
        reward,
        width=bar_width,
        label="r(x,y)",
    )


    ax.set_title(
        scenario["name"],
        pad=8,
    )


    ax.set_ylim(
        0.0,
        1.05,
    )


    ax.set_xticks(
        states
    )


    ax.set_xlabel(
        "y"
    )


    ax.grid(
        axis="y",
        alpha=0.12,
    )


    if scenario_idx == 0:

        ax.set_ylabel(
            "Mass / reward"
        )


# ============================================================
# Bottom three rows
# ============================================================

for beta_idx, beta in enumerate(
    BETAS
):

    row_idx = (
        beta_idx + 1
    )


    for scenario_idx, scenario in enumerate(
        SCENARIOS
    ):

        ax = axes[
            row_idx,
            scenario_idx
        ]


        # ====================================================
        # Exact KL(P_N || Q)
        # ====================================================

        ax.plot(
            N_VALUES,
            KL_PN_Q[
                scenario_idx,
                beta_idx,
                :
            ],
            marker="o",
            markersize=3.4,
            linewidth=1.6,
            label="KL(P_N || Q)",
        )


        # ====================================================
        # Monte Carlo E[Dhat_N]
        # ====================================================

        ax.plot(
            N_VALUES,
            EXPECTED_DHAT_MC[
                scenario_idx,
                beta_idx,
                :
            ],
            marker="s",
            markersize=3.2,
            linewidth=1.5,
            label="E[Dhat_N]",
        )


        # ====================================================
        # 95% MC confidence interval
        # ====================================================

        lower = (
            EXPECTED_DHAT_MC[
                scenario_idx,
                beta_idx,
                :
            ]
            - 1.96
            * DHAT_SE_MC[
                scenario_idx,
                beta_idx,
                :
            ]
        )


        upper = (
            EXPECTED_DHAT_MC[
                scenario_idx,
                beta_idx,
                :
            ]
            + 1.96
            * DHAT_SE_MC[
                scenario_idx,
                beta_idx,
                :
            ]
        )


        ax.fill_between(
            N_VALUES,
            lower,
            upper,
            alpha=0.10,
            linewidth=0,
        )


        # ====================================================
        # Limiting KL(P_beta || Q)
        # ====================================================

        ax.axhline(
            TARGET_KL[
                scenario_idx,
                beta_idx
            ],
            linestyle="--",
            linewidth=1.4,
            label="KL(P_beta || Q)",
        )


        # ====================================================
        # Aminian bound
        # ====================================================

        ax.plot(
            N_VALUES,
            AMINIAN_BOUND[
                scenario_idx,
                beta_idx,
                :
            ],
            linestyle=":",
            linewidth=1.9,
            label="Aminian bound",
        )


        # ====================================================
        # Axis
        # ====================================================

        ax.set_xlim(
            N_VALUES[0],
            N_VALUES[-1],
        )


        ax.set_xticks([
            1,
            5,
            10,
            15,
        ])


        ax.grid(
            axis="y",
            alpha=0.12,
        )


        # Only bottom row gets N label
        if row_idx == 3:

            ax.set_xlabel(
                "N"
            )


        # First column gets beta label
        if scenario_idx == 0:

            ax.set_ylabel(
                f"beta={beta:g}\nDivergence"
            )


# ============================================================
# Same y-range within each beta row
# ============================================================

for row_idx in range(
    1,
    4
):

    ymin = np.inf
    ymax = -np.inf


    for scenario_idx in range(
        NUM_SCENARIOS
    ):

        low, high = axes[
            row_idx,
            scenario_idx
        ].get_ylim()

        ymin = min(
            ymin,
            low,
        )

        ymax = max(
            ymax,
            high,
        )


    span = ymax - ymin

    if span <= 0:
        span = 1.0


    ymin = max(
        0.0,
        ymin - 0.02 * span,
    )

    ymax = (
        ymax + 0.03 * span
    )


    for scenario_idx in range(
        NUM_SCENARIOS
    ):

        axes[
            row_idx,
            scenario_idx
        ].set_ylim(
            ymin,
            ymax,
        )


# ============================================================
# Hide x tick labels on middle rows
# ============================================================

for row_idx in [
    1,
    2,
]:

    for scenario_idx in range(
        NUM_SCENARIOS
    ):

        axes[
            row_idx,
            scenario_idx
        ].tick_params(
            labelbottom=False
        )


# ============================================================
# Shared legend
# ============================================================

top_handles, top_labels = (
    axes[0, 0]
    .get_legend_handles_labels()
)


bottom_handles, bottom_labels = (
    axes[1, 0]
    .get_legend_handles_labels()
)


handles = (
    top_handles
    + bottom_handles
)

labels = (
    top_labels
    + bottom_labels
)


fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        0.995,
    ),
    ncol=6,
    frameon=False,
    columnspacing=1.4,
    handlelength=2.2,
)


# ============================================================
# Layout
# ============================================================

fig.subplots_adjust(
    left=0.07,
    right=0.99,
    bottom=0.065,
    top=0.92,
    wspace=0.23,
    hspace=0.33,
)


# ============================================================
# Save
# ============================================================

FIGURE_PDF = (
    "softbon_alignment_three_beta_rows.pdf"
)

FIGURE_PNG = (
    "softbon_alignment_three_beta_rows.png"
)


fig.savefig(
    FIGURE_PDF,
    bbox_inches="tight",
)


fig.savefig(
    FIGURE_PNG,
    dpi=300,
    bbox_inches="tight",
)


print()
print(
    "Saved figure:",
    FIGURE_PDF
)

print(
    "Saved figure:",
    FIGURE_PNG
)


plt.show()


# ============================================================
# Numeric summary at largest N
# ============================================================

print()
print("=" * 100)
print(
    f"SUMMARY AT N={N_VALUES[-1]}"
)
print("=" * 100)


for scenario_idx, scenario in enumerate(
    SCENARIOS
):

    print()
    print(
        scenario["name"]
    )

    print(
        "-" * 90
    )


    for beta_idx, beta in enumerate(
        BETAS
    ):

        actual = KL_PN_Q[
            scenario_idx,
            beta_idx,
            -1
        ]

        estimator = EXPECTED_DHAT_MC[
            scenario_idx,
            beta_idx,
            -1
        ]

        target = TARGET_KL[
            scenario_idx,
            beta_idx
        ]

        aminian = AMINIAN_BOUND[
            scenario_idx,
            beta_idx,
            -1
        ]


        if actual > 0:
            ratio = (
                aminian / actual
            )
        else:
            ratio = np.inf


        print(
            f"beta={beta:4.1f}   "
            f"KL(P_N || Q)={actual:.6f}   "
            f"E[Dhat_N]={estimator:.6f}   "
            f"KL(P_beta || Q)={target:.6f}   "
            f"Aminian={aminian:.6f}   "
            f"bound/actual={ratio:.2f}"
        )