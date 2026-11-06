"""Differentiable CU-bin probabilities for the single-head Beta policy."""
import torch


def beta_bin_probabilities(alpha, beta, cus_num):
    """Integrate each Beta over [j/J, (j+1)/J], retaining shape gradients.

    Uses the incomplete-beta continued fraction and symmetry in NIST DLMF
    8.17.4, 8.17.22--23: https://dlmf.nist.gov/8.17.v
    PyTorch 2.5 Beta.cdf is unimplemented. Double precision avoids cancellation
    when differencing CDFs; no sampling, soft binning, or SciPy detach is used.
    """
    a, b = alpha.double().unsqueeze(-1), beta.double().unsqueeze(-1)
    if cus_num == 1:
        return (a + b) * 0.0 + 1.0
    x = torch.arange(1, cus_num, dtype=a.dtype, device=a.device) / cus_num
    reflect = x > (a + 1.0) / (a + b + 2.0)
    a, b, x = (torch.where(reflect, b, a), torch.where(reflect, a, b),
               torch.where(reflect, 1.0 - x, x))

    def nonzero(value):
        return torch.where(value.abs() < 1e-30,
                           torch.where(value < 0, -1e-30, 1e-30), value)

    c = torch.ones_like(x)
    d = 1.0 / nonzero(1.0 - (a + b) * x / (a + 1.0))
    fraction = d
    for m in range(1, 201):
        even = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
        odd = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
        for coefficient in (even, odd):
            d = 1.0 / nonzero(1.0 + coefficient * d)
            c = nonzero(1.0 + coefficient / c)
            delta = c * d
            fraction = fraction * delta
        # A minimum depth also converges parameter derivatives when integer
        # shapes make the value's continued fraction terminate early.
        if m >= 32 and bool(torch.all((delta - 1.0).abs() < 1e-12)):
            break
    else:
        raise RuntimeError("Beta CU-bin continued fraction did not converge")
    log_scale = (a*x.log() + b*torch.log1p(-x)
                 + torch.lgamma(a+b) - torch.lgamma(a) - torch.lgamma(b))
    tail = log_scale.exp() * fraction / a
    cdf = torch.where(reflect, 1.0 - tail, tail)
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf,
                     torch.ones_like(cdf[..., :1])), dim=-1)
    probabilities = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0.0)
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


def expected_duplicate_matching(probabilities):
    """E[sum_j max(N_j-1, 0)] for independent UAV CU choices [B, U, J]."""
    selected = torch.zeros_like(probabilities[:, 0])
    expected = torch.zeros_like(selected[:, :1])
    for probability in probabilities.unbind(dim=1):
        expected = expected + (probability * selected).sum(dim=-1, keepdim=True)
        selected = selected + (1.0 - selected) * probability
    return expected
