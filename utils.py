import numpy as np
import torch


def kl(p, q):
    return torch.sum(p * torch.log(p / q))


@torch.no_grad()
def sample_hatD(pi0, pib, N, B=50_000):
    logw = torch.log(pib / pi0)
    idx = torch.multinomial(pi0, B * N, replacement=True).reshape(B, N)
    lam = torch.softmax(logw[idx], dim=1)
    return (lam * torch.log(N * lam)).sum(1)


@torch.no_grad()
def pN_kl_lower(pi0, pib, N, B=50_000, M=2_000, chunk=64):
    if N == 1:
        return 0.0  # One candidate is drawn from pi0, so p_1 = pi0.

    logw = torch.log(pib / pi0)
    w = torch.exp(logw - logw.max())
    total = 0.0

    for start in range(0, B, chunk):
        b = min(chunk, B - start)

        # Y ~ p_N
        idx = torch.multinomial(pi0, b * N, replacement=True).reshape(b, N)
        lam = torch.softmax(logw[idx], dim=1)
        j = torch.multinomial(lam, 1).squeeze(1)
        Y = idx[torch.arange(b, device=pi0.device), j]
        wY = w[Y]

        # estimate p_N(Y) / pi0(Y)
        acc = torch.zeros(b, device=pi0.device, dtype=pi0.dtype)
        mc = max(1, min(M, 4_000_000 // max(1, b * (N - 1))))

        for m in range(0, M, mc):
            mm = min(mc, M - m)
            aux = torch.multinomial(
                pi0, b * mm * (N - 1), replacement=True
            ).reshape(b, mm, N - 1)
            S = w[aux].sum(2)
            acc += (wY[:, None] / (wY[:, None] + S)).sum(1)

        total += torch.log(N * acc / M).sum().item()

    return total / B


def upper_bound(pi0, pib, N):
    logw = torch.log(pib / pi0)
    R = (logw.max() - logw.min()).item()
    return np.log(N / (1 + (N - 1) * np.exp(-R)))