"""Spatial structure of the sphere RFs (flanks, asymmetry, Gabor fit) for comparison with grating tuning.

RF maps are the closed-loop RF coefficients (n_rois, k_folds, ndepths * n_ele * n_azi + bias), averaged
over folds, at the depth of the RF peak. Map coordinates: x = azimuth index (increasing azimuth), y =
elevation index (increasing elevation), angles counterclockwise from +x in degrees, pixels of `RES` deg.

Flank analysis (`flank_metrics`): circular moments of the negative part of the map in a disk around the
RF peak. The flank axis is the direction joining the flanks, i.e. the wave vector of the grating that the
centre-flank structure would match (flanks above and below the centre: axis 90 deg, horizontal bars).
A 2D Gabor fit (`fit_gabor`) gives the same orientation with SF and phase (even / odd symmetry).
"""

import numpy as np
from scipy.optimize import least_squares

NDEPTHS = 8
FRAME_SHAPE = (16, 24)  # (elevation, azimuth)
RES = 5  # deg per pixel
MAX_DIST = 4  # radius of the surround around the peak, pixels


def rf_maps(coef, folds=None, ndepths=NDEPTHS, frame_shape=FRAME_SHAPE):
    """Fold-mean RF volumes, (n_rois, ndepths, n_ele, n_azi).

    Args:
        coef (np.ndarray): (n_rois, k_folds, n_features) coefficients, bias last.
        folds (list, optional): folds to average. Defaults to all.
    """
    if folds is not None:
        coef = coef[:, folds]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(coef[:, :, :-1], axis=1)
    return mean.reshape(len(coef), ndepths, *frame_shape)


def peak_index(volumes):
    """(depth, ele, azi) index of the maximum of each RF volume; -1 for all-NaN volumes."""
    out = np.full((len(volumes), 3), -1, dtype=int)
    for i, v in enumerate(volumes):
        if np.isfinite(v).any():
            out[i] = np.unravel_index(np.nanargmax(v), v.shape)
    return out


def is_centred(peak, max_dist=MAX_DIST, frame_shape=FRAME_SHAPE):
    """Peaks at least `max_dist` pixels from every edge, so that the surround disk fits in the map."""
    iy, ix = peak[:, 1], peak[:, 2]
    return (
        (peak[:, 0] >= 0)
        & (iy >= max_dist)
        & (iy <= frame_shape[0] - 1 - max_dist)
        & (ix >= max_dist)
        & (ix <= frame_shape[1] - 1 - max_dist)
    )


def flank_metrics(img, iy, ix, noise, radius=MAX_DIST):
    """Flank axis, depth and asymmetry of the negative surround of one RF map around its peak.

    The surround is the disk of `radius` pixels around the peak, without the peak pixel. Its negative
    part N = max(-map, 0) / noise is the flank mass, in units of the ipsilateral RF noise (z). With phi
    the angle of each pixel seen from the peak:
    - axis = arg(sum N exp(2i phi)) / 2: the direction joining the flanks, i.e. the wave vector of the
      matching grating (flanks above and below the centre: 90 deg, horizontal bars);
    - anisotropy = |sum N exp(2i phi)| / sum N: 0 for a ring, 1 for flanks on a single line;
    - asym = |sum N exp(i phi)| / sum N: 0 for symmetric flanks, 1 for a single flank; asym_dir is the
      direction of the deeper side (deg, [0, 360));
    - flank_z: deepest surround value on each side of the line through the peak orthogonal to the axis
      (side 0 towards `axis`), in z;
    - oriented_z = |sum N exp(2i phi)| / n_pixels: strength of the oriented flank structure (weight of
      the axis in the population tests);
    - elong_axis, elongation: the same axial moment for the positive part, i.e. the long axis of the
      RF centre (deg, [0, 180)) and how elongated it is ([0, 1]).

    Returns:
        dict
    """
    ny, nx = img.shape
    y, x = np.mgrid[:ny, :nx]
    dy, dx = y - iy, x - ix
    r = np.hypot(dx, dy)
    sel = (r > 0) & (r <= radius) & np.isfinite(img)
    phi = np.arctan2(dy[sel], dx[sel])
    mass = np.clip(-img[sel], 0, None) / noise
    total = mass.sum()
    m2 = (mass * np.exp(2j * phi)).sum()
    m1 = (mass * np.exp(1j * phi)).sum()
    axis = np.angle(m2) / 2
    along = np.cos(phi - axis)  # > 0 on the side of `axis`
    # elongation of the positive centre: axial moment of the positive mass around the peak
    pos = np.clip(img[sel], 0, None) / noise
    e2 = (pos * np.exp(2j * phi)).sum()
    vals = img[sel] / noise
    flank_z = np.array(
        [
            max(-vals[along > 0].min(), 0) if (along > 0).any() else np.nan,
            max(-vals[along < 0].min(), 0) if (along < 0).any() else np.nan,
        ]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(
            centre_z=img[iy, ix] / noise,
            surround_z=total / sel.sum(),
            axis=np.rad2deg(axis) % 180,
            anisotropy=np.abs(m2) / total,
            asym=np.abs(m1) / total,
            asym_dir=np.rad2deg(np.angle(m1)) % 360,
            flank_z=flank_z,
            oriented_z=np.abs(m2) / sel.sum(),
            elong_axis=np.rad2deg(np.angle(e2) / 2) % 180,
            elongation=np.abs(e2) / pos.sum(),
        )


def _gabor(p, x, y):
    amp, x0, y0, sx, sy, theta, freq, phase, offset = p
    c, s = np.cos(theta), np.sin(theta)
    xr = (x - x0) * c + (y - y0) * s  # along the wave vector
    yr = -(x - x0) * s + (y - y0) * c
    env = np.exp(-0.5 * (xr / sx) ** 2 - 0.5 * (yr / sy) ** 2)
    return amp * env * np.cos(2 * np.pi * freq * xr + phase) + offset


def fit_gabor(img, iy, ix):
    """2D Gabor fit of one RF map, started at the peak from 4 orientations x 2 frequencies.

    Returns:
        dict: theta (wave vector, deg [0, 180)), sf (cpd), phase (rad, 0 = even with a positive centre),
        sigma_x / sigma_y (deg, across / along the bars), rsq.
    """
    ny, nx = img.shape
    y, x = np.mgrid[:ny, :nx]
    x, y, z = x.ravel(), y.ravel(), img.ravel()
    lo = [0, -0.5, -0.5, 0.5, 0.5, -np.inf, 0, -np.inf, -np.inf]
    hi = [np.inf, nx - 0.5, ny - 0.5, 12, 12, np.inf, 0.5, np.inf, np.inf]
    best = None
    for theta in np.deg2rad([0, 45, 90, 135]):
        for freq in [0.08, 0.2]:
            p0 = [max(img[iy, ix], 1e-6), ix, iy, 2, 2, theta, freq, 0, 0]
            try:
                r = least_squares(lambda p: _gabor(p, x, y) - z, p0, bounds=(lo, hi))
            except ValueError:
                continue
            if best is None or r.cost < best.cost:
                best = r
    if best is None:
        return dict(
            theta=np.nan,
            sf=np.nan,
            phase=np.nan,
            sigma_x=np.nan,
            sigma_y=np.nan,
            rsq=np.nan,
        )
    amp, _, _, sx, sy, theta, freq, phase, _ = best.x
    # canonical form: theta in [0, pi) (theta + pi with -phase is the same Gabor), phase in (-pi, pi]
    theta = theta % (2 * np.pi)
    if theta >= np.pi:
        theta, phase = theta - np.pi, -phase
    phase = np.angle(np.exp(1j * phase))
    ss_res = 2 * best.cost
    ss_tot = ((z - z.mean()) ** 2).sum()
    return dict(
        theta=np.rad2deg(theta),
        sf=freq / RES,
        phase=phase,
        sigma_x=sx * RES,
        sigma_y=sy * RES,
        rsq=1 - ss_res / ss_tot,
    )


def flow_direction(azi, ele):
    """Direction (deg, map coordinates) of the image motion of a static object during forward running.

    The focus of expansion is at azimuth 0, elevation 0; points move away from it along great circles.
    In (azimuth, elevation) degrees the displacement is proportional to
    (sin(azi) / cos(ele), sin(ele) cos(azi)).
    """
    a, e = np.deg2rad(azi), np.deg2rad(ele)
    return np.rad2deg(np.arctan2(np.sin(e) * np.cos(a), np.sin(a) / np.cos(e))) % 360


def axial_diff(a, b, period=180):
    """Signed difference a - b wrapped to [-period / 2, period / 2)."""
    return (np.asarray(a) - np.asarray(b) + period / 2) % period - period / 2
