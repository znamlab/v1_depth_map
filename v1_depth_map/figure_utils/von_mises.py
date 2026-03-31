"""Tools for fitting von Mises mixture models to **axial** orientation data.

Axial data (e.g. tuning angles in the range [-45°, 135°]) has periodicity π
rather than 2π: two angles that differ by 180° represent the *same* orientation.
The standard fix is to **double** the angles so they span the full circle,
fit the mixture, then **halve** the estimated means back.

Public API
----------
fit_axial_von_mises_mixture(thetas_rad, k, ...)
    Fit k von Mises components to axial data via EM.
axial_mixture_density(theta_range, pi_k, mu_k, kappa_k)
    Evaluate fitted density on a range of angles.
model_selection(thetas_rad, max_k, ...)
    Fit k = 1 … max_k and return BIC/AIC results.
plot_model_selection(thetas_rad, results, ...)
    Two-panel figure: BIC/AIC curves + histogram with best-fit overlay.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ive
import matplotlib.pyplot as plt

__all__ = [
    "fit_axial_von_mises_mixture",
    "axial_mixture_density",
    "model_selection",
    "plot_model_selection",
]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _vonmises_pdf(x: np.ndarray, mu: float, kappa: float) -> np.ndarray:
    """Evaluates the von Mises PDF at x.

    Uses `scipy.special.ive` (exponentially scaled I0) to avoid overflow
    for large kappa.

    Args:
        x: Circular data at which to evaluate the density.
        mu: Mean angle.
        kappa: Concentration parameter.

    Returns:
        np.ndarray: The probability density at x.
    """
    return np.exp(kappa * (np.cos(x - mu) - 1)) / (2 * np.pi * ive(0, kappa))


def _kappa_from_R(R: float) -> float:
    """Approximates kappa from the mean resultant length R ∈ [0, 1).

    Uses the approximation from Mardia & Jupp (2000).

    Args:
        R: Mean resultant length.

    Returns:
        float: The approximated concentration parameter kappa.
    """
    R = float(np.clip(R, 0.0, 0.9999))
    if R < 0.53:
        return 2 * R + R**3 + (5 / 6) * R**5
    elif R < 0.85:
        return -0.4 + 1.39 * R + 0.43 / (1 - R)
    else:
        val = R**3 - 4 * R**2 + 3 * R
        return 1 / val if val > 1e-10 else 100.0


# ---------------------------------------------------------------------------
# EM fitting
# ---------------------------------------------------------------------------


def _fit_von_mises_em(
    x: np.ndarray,
    k: int,
    max_iter: int = 1000,
    tol: float = 1e-6,
    n_restarts: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple:
    """Fits a k-component von Mises mixture to circular data via EM.

    Args:
        x: Circular data in radians, wrapped to [-π, π].
        k: Number of components.
        max_iter: Maximum EM iterations per restart. Defaults to 1000.
        tol: Convergence threshold on log-likelihood change. Defaults to 1e-6.
        n_restarts: Number of random restarts; the best solution is kept.
            Defaults to 20.
        rng: Random number generator for reproducibility.

    Returns:
        tuple: (pi_k, mu_k, kappa_k, ll, bic, aic) where:
            pi_k: Mixing weights (k,).
            mu_k: Means in radians (k,).
            kappa_k: Concentration parameters (k,).
            ll: Final log-likelihood.
            bic: Bayesian Information Criterion.
            aic: Akaike Information Criterion.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float)
    n = len(x)
    best_ll = -np.inf
    best_params = None

    for _ in range(n_restarts):
        # Initialise means by sampling data points
        idx = rng.choice(n, size=k, replace=False)
        pi_k = np.ones(k) / k
        mu_k = x[idx].copy()
        kappa_k = np.full(k, 2.0)

        ll = ll_old = -np.inf

        for _iter in range(max_iter):
            # ---- E-step ----
            # pdf_k: (n, k)
            pdf_k = np.column_stack(
                [_vonmises_pdf(x, mu_k[j], kappa_k[j]) for j in range(k)]
            )
            weighted = pdf_k * pi_k  # (n, k)
            row_sum = weighted.sum(axis=1)  # (n,)
            row_sum = np.maximum(row_sum, 1e-300)
            gamma = weighted / row_sum[:, None]  # (n, k)

            ll = float(np.sum(np.log(row_sum)))
            if abs(ll - ll_old) < tol:
                break
            ll_old = ll

            # ---- M-step ----
            N_k = gamma.sum(axis=0)  # (k,)
            pi_k = N_k / n

            for j in range(k):
                if N_k[j] < 1e-10:
                    # Reinitialise dead component
                    mu_k[j] = x[rng.integers(n)]
                    kappa_k[j] = 1.0
                    continue
                cs = np.dot(gamma[:, j], np.cos(x))
                sn = np.dot(gamma[:, j], np.sin(x))
                mu_k[j] = np.arctan2(sn, cs)
                kappa_k[j] = _kappa_from_R(np.hypot(cs, sn) / N_k[j])

        if ll > best_ll:
            best_ll = ll
            best_params = (pi_k.copy(), mu_k.copy(), kappa_k.copy(), ll)

    if best_params is None:
        raise RuntimeError(f"EM failed to find solution for k={k}")

    pi_k, mu_k, kappa_k, ll = best_params
    p = 3 * k - 1  # free parameters
    bic = -2 * ll + p * np.log(n)
    aic = -2 * ll + 2 * p
    return pi_k, mu_k, kappa_k, ll, bic, aic


# ---------------------------------------------------------------------------
# Axial-data wrappers
# ---------------------------------------------------------------------------


def fit_axial_von_mises_mixture(
    thetas_rad,
    k: int,
    *,
    max_iter: int = 1000,
    tol: float = 1e-6,
    n_restarts: int = 30,
    rng: np.random.Generator | None = None,
) -> tuple:
    """Fits a k-component von Mises mixture to axial orientation data.

    Because axial data has period π (not 2π), we first double the angles to
    map them onto the full circle, run the EM algorithm there, then convert
    the estimated means back to the original (half) space.

    Args:
        thetas_rad: Axial orientation angles in radians (e.g., in [-π/2, π/2]).
        k: Number of mixture components.
        max_iter: Maximum EM iterations per restart. Defaults to 1000.
        tol: Convergence threshold. Defaults to 1e-6.
        n_restarts: Number of random restarts. Defaults to 30.
        rng: Random number generator for reproducibility.

    Returns:
        tuple: (pi_k, mu_k, kappa_k, ll, bic, aic) where:
            pi_k: Mixing weights (k,), sorted by mean angle.
            mu_k: Component means in the original angle space [radians] (k,).
            kappa_k: Concentration parameters in the doubled-angle space (k,).
            ll: Log-likelihood in the doubled-angle space.
            bic: Bayesian Information Criterion.
            aic: Akaike Information Criterion.

    Notes:
        * The kappa_k values correspond to the doubled-angle representation.
          To convert to an effective concentration for the original angular scale,
          note that a von Mises with kappa on angle 2θ is equivalent to a
          "wrapped Cauchy" or a different distribution on θ; for plotting use
          axial_mixture_density() which handles the Jacobian correctly.
        * Because of the doubling, each orientation θ also appears at θ+π/2 on
          the doubled circle. axial_mixture_density() accounts for this
          automatically.
    """
    thetas_rad = np.asarray(thetas_rad, dtype=float)
    # Double and wrap to [-π, π]
    x2 = (2.0 * thetas_rad + np.pi) % (2 * np.pi) - np.pi

    pi_k, mu2, kappa_k, ll, bic, aic = _fit_von_mises_em(
        x2,
        k,
        max_iter=max_iter,
        tol=tol,
        n_restarts=n_restarts,
        rng=rng,
    )

    # Map means back to original space
    mu_k = mu2 / 2.0
    # Wrap to [-π/2, π/2]
    mu_k = (mu_k + np.pi / 2) % np.pi - np.pi / 2

    # Sort components by mean angle
    order = np.argsort(mu_k)
    return pi_k[order], mu_k[order], kappa_k[order], ll, bic, aic


def axial_mixture_density(
    theta_range,
    pi_k: np.ndarray,
    mu_k: np.ndarray,
    kappa_k: np.ndarray,
) -> np.ndarray:
    """Evaluates the fitted axial mixture density at arbitrary angles.

    Applies the change-of-variables Jacobian |d(2θ)/dθ| = 2 so that the
    returned values integrate to 1 over the axial range [-π/2, π/2].

    Args:
        theta_range: Angles in radians at which to evaluate the density.
        pi_k: Mixing weights.
        mu_k: Component means.
        kappa_k: Concentration parameters.

    Returns:
        np.ndarray: Probability density at each angle in theta_range.
    """
    theta_range = np.asarray(theta_range, dtype=float)
    x2 = (2.0 * theta_range + np.pi) % (2 * np.pi) - np.pi
    density = np.zeros_like(theta_range)
    for j in range(len(pi_k)):
        density += pi_k[j] * _vonmises_pdf(x2, 2.0 * mu_k[j], kappa_k[j])
    return density * 2.0  # Jacobian factor


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def model_selection(
    thetas_rad,
    max_k: int = 6,
    *,
    n_restarts: int = 30,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Fits k = 1 … max_k components and returns BIC / AIC for each.

    Args:
        thetas_rad: Axial orientation data in radians.
        max_k: Maximum number of components to try. Defaults to 6.
        n_restarts: EM restarts per k. Defaults to 30.
        seed: Random seed for reproducibility.
        verbose: If True, print a summary line per k. Defaults to True.

    Returns:
        dict: results[k] is a dict with keys:
            pi_k, mu_k, kappa_k, ll, bic, aic.

    Examples:
        >>> from v1_depth_map.figure_utils.von_mises import model_selection
        >>> import numpy as np
        >>> thetas = np.radians(ndf.g2d_theta_treadmill)
        >>> results = model_selection(thetas, max_k=5, seed=42)
    """
    rng = np.random.default_rng(seed)
    thetas_rad = np.asarray(thetas_rad, dtype=float)
    results = {}

    for k in range(1, max_k + 1):
        pi_k, mu_k, kappa_k, ll, bic, aic = fit_axial_von_mises_mixture(
            thetas_rad,
            k,
            n_restarts=n_restarts,
            rng=rng,
        )
        results[k] = dict(
            pi_k=pi_k, mu_k=mu_k, kappa_k=kappa_k, ll=ll, bic=bic, aic=aic
        )
        if verbose:
            means_deg = np.degrees(mu_k).round(1)
            print(
                f"k={k}  ll={ll:9.1f}  BIC={bic:9.1f}  AIC={aic:9.1f}  "
                f"means={means_deg} °  weights={pi_k.round(3)}"
            )

    best_bic = min(results, key=lambda k: results[k]["bic"])
    best_aic = min(results, key=lambda k: results[k]["aic"])
    if verbose:
        print(f"\nBest k by BIC: {best_bic}")
        print(f"Best k by AIC: {best_aic}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_model_selection(
    thetas_rad,
    results: dict,
    *,
    angle_range_deg: tuple = (-45, 135),
    bin_width_deg: float = 5.0,
    figsize: tuple = (14, 5),
    colors: tuple = ("steelblue", "darkorange"),
    ax_criteria: plt.Axes | None = None,
    ax_hist: plt.Axes | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Two-panel figure for axial von Mises model selection.

    Left panel shows BIC and AIC as a function of k; dotted vertical lines mark
    the respective minima. Right panel shows a histogram of the orientation data
    overlaid with the fitted mixture density for the best-BIC and best-AIC models.

    Args:
        thetas_rad: Raw orientation data in radians (used for histogram).
        results: Output of model_selection().
        angle_range_deg: x-axis limits of the histogram panel (degrees).
            Defaults to (-45, 135).
        bin_width_deg: Histogram bin width in degrees. Defaults to 5.0.
        figsize: Figure size (width, height) in inches. Defaults to (14, 5).
        colors: Colours for BIC and AIC lines/overlays. Defaults to
            ("steelblue", "darkorange").
        ax_criteria: Pre-existing axes for criteria plot. Defaults to None.
        ax_hist: Pre-existing axes for histogram plot. Defaults to None.

    Returns:
        tuple: (fig, axes, results) where:
            fig: The created or used Figure.
            axes: Tuple of Axes (ax_criteria, ax_hist).
            results: The results dictionary passed in.

    Examples:
        >>> from v1_depth_map.figure_utils.von_mises import model_selection, plot_model_selection
        >>> import numpy as np
        >>> thetas = np.radians(ndf.g2d_theta_treadmill)
        >>> results = model_selection(thetas, max_k=5, seed=42)
        >>> fig, axes = plot_model_selection(thetas, results)
    """
    thetas_rad = np.asarray(thetas_rad, dtype=float)
    ks = sorted(results.keys())
    best_bic = min(ks, key=lambda k: results[k]["bic"])
    best_aic = min(ks, key=lambda k: results[k]["aic"])

    # ---- Set up axes ----
    if ax_criteria is None or ax_hist is None:
        fig, (ax_criteria, ax_hist) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = ax_criteria.get_figure()

    col_bic, col_aic = colors

    # ---- Left panel: BIC and AIC ----
    bics = [results[k]["bic"] for k in ks]
    aics = [results[k]["aic"] for k in ks]

    ax_criteria.plot(ks, bics, "o-", color=col_bic, label="BIC")
    ax_criteria.plot(ks, aics, "s--", color=col_aic, label="AIC")
    ax_criteria.axvline(best_bic, color=col_bic, alpha=0.35, linestyle=":")
    ax_criteria.axvline(best_aic, color=col_aic, alpha=0.35, linestyle=":")
    ax_criteria.set_xlabel("Number of components k")
    ax_criteria.set_ylabel("Information criterion")
    ax_criteria.set_title("Model selection (lower = better)")
    ax_criteria.legend()
    ax_criteria.set_xticks(ks)

    # ---- Right panel: histogram + fitted density ----
    lo, hi = angle_range_deg
    bins = np.arange(lo, hi + bin_width_deg, bin_width_deg)
    ax_hist.hist(
        np.degrees(thetas_rad),
        bins=bins,
        density=True,
        color="lightgray",
        edgecolor="white",
        alpha=0.85,
        label="Data",
    )

    theta_rng = np.linspace(np.radians(lo), np.radians(hi), 600)

    for best_k, ls, col, label_prefix in [
        (best_bic, "-", col_bic, "BIC"),
        (best_aic, "--", col_aic, "AIC"),
    ]:
        r = results[best_k]
        dens_rad = axial_mixture_density(theta_rng, r["pi_k"], r["mu_k"], r["kappa_k"])
        # Convert density from per-radian to per-degree
        dens_deg = dens_rad / np.degrees(1)
        ax_hist.plot(
            np.degrees(theta_rng),
            dens_deg,
            linestyle=ls,
            color=col,
            linewidth=2,
            label=f"{label_prefix} best k={best_k}",
        )
        for mu_j in r["mu_k"]:
            ax_hist.axvline(np.degrees(mu_j), color=col, alpha=0.35, linestyle=":")

    ax_hist.set_xlabel("Orientation (degrees)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Axial von Mises mixture – best models")
    ax_hist.set_xlim(lo, hi)
    ax_hist.legend(fontsize=8)

    plt.tight_layout()
    return fig, (ax_criteria, ax_hist), results
