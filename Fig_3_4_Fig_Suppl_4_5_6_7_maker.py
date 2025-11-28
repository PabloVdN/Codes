"""
Generates Figure 3, Figure 4, and Supplementary Figures S4–S7:
full processing pipeline for Stokes maps (S1, S2, S3), purity (ρ),
and LPB dispersion fits across three detunings.

Overview
--------
This script loads polarization-resolved PL images (.sif), builds energy–momentum
axes, applies wavelength→energy rescaling, subtracts smooth backgrounds, computes
Stokes maps and purity, performs a joint LPB fit from column-wise extrema,
maps PL minima to detuning via a quadratic relation, and renders all figures
with consistent axes and external colorbars.

End-to-end pipeline
-------------------
1) Load, rotate, and lightly denoise each .sif per detuning and analyzer channel.
2) Build (sinθ, E) meshes and convert to (k_parallel, E); k depends on E → 2D K_grid.
3) Apply a nm→eV Jacobian-like factor row-wise to preserve spectral density.
4) Zero out columns beyond the measured pixel window to avoid edge artifacts.
5) Per-channel background subtraction: column-wise model (linear poly + two skewed Lorentzians);
   then compute Stokes maps S1, S2, S3 via S = (A−B)/(A+B) with an intensity mask.
6) Compute intensity-weighted averages ⟨S1⟩, ⟨S2⟩, ⟨S3⟩ and the Stokes modulus ρ
   for H, σ⁺, σ⁻ pump configurations at each detuning.
7) Pick column-wise maxima from background-subtracted PL (H-channel sum) and run a
   joint least-squares fit of the coupled-oscillator LPB across the three panels
   to estimate shared photon curvature A and Rabi splitting Ω, and per-panel k-offsets and detunings.
8) Convert LPB minimum energy E_min_PL (at k = off) into reflectivity detuning δ (meV)
   via a quadratic mapping (coefficients taken from Fig_2 maker).
9) Render Figure 3 (LPB overlays on PL), Figure 4 (bar charts of ⟨S3⟩ and ⟨ρ⟩ vs δ),
   and Supplementary 3×3 grids (Stokes and purity) with consistent color scales
   and external vertical colorbars.

Cells guide
-----------
• Cell 1 - Imports & globals:
    Physical constants (h, c), sinθ limits, mask threshold, fixed exciton energy Ex,
    fit k-range, paths and per-detuning configuration, plus helpers for plotting,
    defect removal, nm→eV, background subtraction, Stokes calculation, grouped bars,
    zeroing columns, and LPB/UPB analytic models.

• Cell 2 - Main data extraction (per detuning):
    – Load all analyzer/polarization channels (H/V, D/A, σ⁺/σ⁻), rotate and denoise.
    – Build sinθ→column mapping using per-panel pixel windows; calibrate energy
      from matching .asc files (nm → eV).
    – Construct 2D grids Y_ev and K_parallel; apply Jacobian-like scaling of intensity.
    – Zero outside columns; perform column-wise background subtraction.
    – Compute S1/S2/S3 maps and store them in 3D matrices [energy, x, detuning].
    – Compute intensity-weighted averages ⟨S1⟩, ⟨S2⟩, ⟨S3⟩ and purity ⟨ρ⟩ for H, σ⁺, σ⁻.
    – Cache K/Y grids, k-limits, and a background-subtracted PL image for later LPB fitting.

• Cell 3 — Detuning extraction:
    – From PL images (three detunings), pick column-wise maxima → (k, E) seeds.
    – Build fit lists and run joint least_squares (method='lm') for LPB across panels.
    – Evaluate E_min_PL via LPB at k = off; map E_min_PL → δ (meV) using δ = a·E_min_PL² + b·E_min_PL + c
      (coefficients from Fig_2; update here if they change).
    – Plot background-subtracted PL (normalized) per detuning with consistent axes.
    – Overlay LPB from the joint fit and show the red curve with maxima used for fitting.
    – Add an external vertical colorbar (normalized intensity).

• Cell 4 - Figure 4 (grouped bar charts):
    – Two panels: ⟨S3⟩ vs δ and ⟨ρ⟩ vs δ (σ⁺, H, σ⁻), ordered by detuning labels.
    – Global legend centered above both subplots; consistent spines and grid.

• Cells 5 to 7 - Supplementary Figures S4–S6 (Stokes 3×3 at each detuning):
    – For detuning #1/#2/#3, render a 3×3 grid: rows (σ⁺, H, σ⁻) × cols (S1, S2, S3),
      fixed energy window and k-limits, seismic colormap (vmin=−0.5, vmax=0.5),
      external right-side colorbar labeled S₁,₂,₃.

• Cell 8 - Supplementary Figure S7 (purity 3×3 across detunings):
    – Purity ρ = √(S1² + S2² + S3²), 3×3 grid: rows (σ⁺, H, σ⁻) × cols (three detunings),
      custom white→red colormap (top half of 'seismic'), external colorbar labeled ρ.

• Cell 9 - Final Figure 3 (combined 2×3):
    – For each detuning column, stitch top-row S3 and bottom-row purity:
      left half from σ⁻ and right half from σ⁺, split at k ≈ 0 (optional dashed line).
    – Keep panel-specific K/Y grids and k-limits; add two vertical colorbars (S3, ρ),
      panel labels (a–f), optional text overlays and σ labels (top/bottom).

Key notes
---------
• nm→eV Jacobian-like rescaling preserves spectral density when changing variables.
• Background removal is column-wise (linear polynomial + two skewed Lorentzians),
  subtracting only the polynomial component to avoid distorting spectral features.
• Stokes maps use an intensity mask built from (A+B) with a configurable threshold
  to suppress low-SNR regions; NaN/Inf handling is explicit.
• Joint LPB fit uses Levenberg–Marquardt (no bounds) for parity with Fig_2 maker;
  uncertainties can be estimated from the Jacobian if needed.
• The three detunings are not calculated from the LPB fit but from the interpolation 
  in the last cell of Fig_2_maker.py, which yields δ = a·E_min_PL² + b·E_min_PL + c.
  This is done so to extract the detunings using white light reflectivity. 
  The LPB fit is to calculate where the E_min_PL is.
  
"""


# ============================== Imports =================================== #

# Standard libs
import os
import re
import inspect
import numpy as np
import matplotlib.pyplot as plt

# Matplotlib helpers
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable

# Image denoising (simple local mean blur) for hot-pixel suppression
import cv2

# Nonlinear modeling (background removal) and parameters container
from lmfit import Model, Parameters

# SIF reader (Andor camera)
import sif_parser

# Joint least-squares optimizer for the polariton dispersion fit
from scipy.optimize import least_squares


# ============================ Global parameters =========================== #
# Physical constants
h = 4.135667696e-15  # Planck constant [eV·s]
c = 299792458        # Speed of light [m/s]

# Original horizontal measurement axis is angular (encoded as sin(theta))
# This pair defines the calibrated sin(theta) range across the illuminated pixels.
sin1, sin2 = -0.75, 0.75

# Mask threshold: pixels where the positive part of (A+B) is below
# threshold_percent% of its maximum are set to zero in S-calculation.
threshold_percent = 15

# Parameters used when fitting the coupled oscillator (LPB/UPB) dispersion
Ex_fixed = 2.404  # Exciton energy [eV] (kept fixed)

# k_parallel range used for LPB least-squares fitting (for column maxima selection)
fit_rangE_min_PL, fit_range_max = -7.0, 7.0  # [µm^-1]


# ================================ Configs ================================= #
# Absolute path to the current script
current_path = os.path.dirname(os.path.abspath(__file__))

# Root folder where the polariton datasets live
base_folder = os.path.abspath(os.path.join(current_path, "..", "Measurements", "S3_polariton"))

# Three configurations (high/mid/low detuning) with their dynamic paths and pixel windows
configurations = [
    {
        "initial_path": os.path.join(base_folder, "polariton_hig_det_wavelength_525_pl_").replace("\\", "/"),
        "final_path": ".sif",
        "pixel1": 120, "pixel2": 875,
        "detuning_num": 1,
    },
    {
        "initial_path": os.path.join(base_folder, "polariton_mid_det_wavelength_525_pl_").replace("\\", "/"),
        "final_path": ".sif",
        "pixel1": 110, "pixel2": 890,
        "detuning_num": 2,
    },
    {
        "initial_path": os.path.join(base_folder, "polariton_low_det_wavelength_535_pl_").replace("\\", "/"),
        "final_path": ".sif",
        "pixel1": 110, "pixel2": 890,
        "detuning_num": 3,
    }
]


# =============================== Functions ================================ #
def plot_image(image, title, K_parallel, Y_ev, pixel1, pixel2):
    """
    Plot a 2D image on (K_parallel, Y_ev) with a colorbar and axis labels.

    The horizontal axis is limited to the measured pixel window [pixel1, pixel2].
    This is a generic helper for quick inspection of raw/processed intensity maps.

    Parameters
    ----------
    image : 2D np.ndarray
        Image to plot (same shape as K_parallel and Y_ev).
    title : str or None
        Title shown above the figure; if None, it attempts to infer the variable name.
    K_parallel, Y_ev : 2D np.ndarray
        Grids used as X and Y axes for pcolormesh.
    pixel1, pixel2 : int
        Horizontal pixel indices delimiting the measured window.
    """
    if title is None:
        # Best-effort inference of the variable name passed in (for quick-look only)
        frame = inspect.currentframe().f_back
        names = [name for name, val in frame.f_locals.items() if val is image]
        title = names[0] if names else ''
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(K_parallel, Y_ev, image, shading='auto', cmap='viridis')

    # Limit the horizontal axis to the measured pixel window
    k_min = K_parallel[0, pixel1]
    k_max = K_parallel[0, pixel2]
    ax.set_xlim(k_min, k_max)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Intensity', size=16)
    cbar.ax.tick_params(labelsize=16)

    # Axis labels/ticks
    ax.set_xlabel(r'$k_{\parallel}$ ($\mu m^{-1}$)', fontsize=18)
    ax.set_ylabel('Energy (eV)', fontsize=18)
    ax.set_title(title, fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()


def setup_axis(image, title, K_parallel, Y_ev, pixel1, pixel2):
    """
    Plot a Stokes-like map with fixed range [-1,1] and masked out-of-range values in green.
    Uses (K_parallel, Y_ev) axes and limits the horizontal range to the measured window.

    Note
    ----
    The colorbar label is set to 'Intensity' in this function. When using it for
    Stokes maps, consider adjusting the label externally to show S1/S2/S3 explicitly.
    """
    masked_image = np.ma.masked_outside(image, -1, 1)
    cmap = plt.get_cmap('seismic').copy()
    cmap.set_bad(color='green')
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(K_parallel, Y_ev, masked_image, shading='auto', cmap=cmap, vmin=-1, vmax=1)
    k_min = K_parallel[0, pixel1]
    k_max = K_parallel[0, pixel2]
    ax.set_xlim(k_min, k_max)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Intensity', size=16)  # label text kept as-is
    cbar.ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$k_{\parallel}$ ($\mu m^{-1}$)', fontsize=18)
    ax.set_ylabel('Energy (eV)', fontsize=18)
    ax.set_title(title, fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()


def S_calculator_and_mask(image_A, image_B):
    """
    Compute S = (A - B) / (A + B) and mask out pixels where (A+B) is weak.

    Rules implemented
    -----------------
    - Only the positive part of (A+B) contributes to the thresholding.
    - Non-finite values (NaN/Inf) in the ratio are replaced with 0.
    - Pixels outside the intensity mask are set to 0 in S.

    Parameters
    ----------
    image_A, image_B : 2D np.ndarray
        Intensity images corresponding to a given analyzer pair (e.g., H/V).

    Returns
    -------
    image_S : 2D np.ndarray
        Stokes-like map with masked regions set to 0.
    """
    diff_image = image_A - image_B
    sum_image = image_A + image_B

    # Positive part of the sum used for robust masking
    sum_pos = np.clip(sum_image, 0, None)
    threshold = (threshold_percent / 100.0) * np.nanmax(sum_pos)
    mask = sum_pos > threshold

    # Robust ratio with finite check
    with np.errstate(divide='ignore', invalid='ignore'):
        image_S = np.true_divide(diff_image, sum_image)
        image_S[~np.isfinite(image_S)] = 0.0

    # Apply intensity mask: outside becomes 0
    image_S[~mask] = 0.0
    return image_S


def remove_defects(image, threshold_factor=1.3):
    """
    Suppress local bright defects (hot pixels) by comparing each pixel to a local mean.

    Method
    ------
    - Compute a 5x5 box blur to estimate the local mean.
    - Replace pixels above threshold_factor * local_mean by the local mean.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image to be filtered (in-place safe: we copy internally).
    threshold_factor : float
        Scaling factor above which a pixel is considered a defect.

    Returns
    -------
    filtered_image : 2D np.ndarray
        Output image with hot pixels suppressed.
    """
    filtered_image = image.copy()
    kernel_size = 5
    mean_local = cv2.blur(image, (kernel_size, kernel_size))
    threshold = threshold_factor * mean_local
    mask_defects = image > threshold
    filtered_image[mask_defects] = mean_local[mask_defects]
    return filtered_image


def nm_to_ev(wavelength_nm):
    """
    Convert wavelength in nm to photon energy in eV using E = h·c / λ.

    Parameters
    ----------
    wavelength_nm : array-like or float
        Wavelength(s) in nanometers.

    Returns
    -------
    energy_ev : array-like or float
        Energy in electronvolts.
    """
    return h * c / (wavelength_nm * 1e-9)


def rotate_image(filepath):
    """
    Load a .sif file, squeeze singleton dimensions, and rotate 90° clockwise.

    After rotation:
    - axis 0 corresponds to energy (rows),
    - axis 1 corresponds to x/angle (columns).

    Parameters
    ----------
    filepath : str
        Full path to the .sif file.

    Returns
    -------
    image_rot : 2D np.ndarray
        Rotated image ready for further processing.
    """
    data, info = sif_parser.np_open(filepath)
    image = data.squeeze()
    return np.rot90(image, k=-1)


def remove_polynomial_background(image, Y_ev):
    """
    Column-wise background removal:
    Fit [linear polynomial + skewed Lorentzian + skewed Lorentzian] along energy,
    and subtract only the polynomial component. If a column is all zeros or a fit
    fails, the column is left unchanged.

    Notes
    -----
    - This routine is CPU-intensive (one nonlinear fit per column).
    - The second Lorentzian's center is fixed to 2.395 eV as a heuristic.
    - Failures are caught and the original data is preserved for that column.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image (energy by position).
    Y_ev : 2D np.ndarray
        Energy grid used to build the x-axis for fitting (only min/max are used).

    Returns
    -------
    image_no_bg : 2D np.ndarray
        Image with only the polynomial background removed column-wise.
    """
    def polynomial_1(x, c0, c1):
        return c0 + c1 * x

    def skewed_lorentzian(x, amp, center, gamma, alpha):
        return amp / (1 + (((x - center) / (gamma * (1 + alpha * np.sign(x - center)))) ** 2))

    model = (Model(polynomial_1, prefix='poly_')
             + Model(skewed_lorentzian, prefix='lorentz_')
             + Model(skewed_lorentzian, prefix='lorentz2_'))

    n_energies, n_positions = image.shape
    Y_ev = np.asarray(Y_ev)
    x_data = np.linspace(np.nanmin(Y_ev), np.nanmax(Y_ev), n_energies)
    image_no_bg = np.zeros_like(image)

    for j in range(n_positions):
        y_data = image[:, j]

        # Leave empty columns unchanged
        if np.allclose(y_data, 0) or np.nanmax(y_data) <= 0:
            image_no_bg[:, j] = y_data
            continue

        max_y = np.nanmax(y_data)
        max_y_index = np.nanargmax(y_data)
        x_max_y = x_data[max_y_index]

        params = Parameters()
        params.add('poly_c0', value=0.01 * max_y)
        params.add('poly_c1', value=0.01 * max_y)

        # Main Lorentzian (free center near maximum)
        params.add('lorentz_amp', value=max_y * 0.7, min=max_y * 0.2, max=max_y * 1.1)
        params.add('lorentz_center', value=x_max_y, min=x_max_y - 0.05, max=x_max_y + 0.05)
        params.add('lorentz_gamma', value=0.01, min=0.0, max=0.05)
        params.add('lorentz_alpha', value=0, vary=False)

        # Secondary Lorentzian (fixed center at 2.395 eV; heuristic shoulder)
        params.add('lorentz2_amp', value=max_y * 0.25, min=0, max=max_y * 0.5)
        params.add('lorentz2_center', value=2.395, vary=False)
        params.add('lorentz2_gamma', value=0.04, min=0.0, max=0.05)
        params.add('lorentz2_alpha', value=0, vary=False)

        try:
            result = model.fit(y_data, params, x=x_data, method='least_squares')
            bg = result.eval_components(x=x_data)['poly_']
            image_no_bg[:, j] = y_data - bg
        except Exception as e:
            print(f"Fit failed for column {j}: {e}")
            image_no_bg[:, j] = y_data

    return image_no_bg


def compute_Si_average_with_mask(Si_image, intensity_image_A, intensity_image_B, threshold_percent):
    """
    Weighted average of a Stokes-like quantity over pixels where (A+B)
    exceeds threshold_percent% of its maximum. Returns scalar average.

    Important
    ---------
    - The mask is built from (A+B) without clipping negatives here.
    - If the valid-weight sum is zero, the function returns 0.0.

    Parameters
    ----------
    Si_image : 2D np.ndarray
        Stokes-like map (e.g., S1, S2, or S3).
    intensity_image_A, intensity_image_B : 2D np.ndarray
        The two analyzer images used to build Si_image.
    threshold_percent : float
        Percentage of the global maximum used to define the mask.

    Returns
    -------
    average_S : float
        Intensity-weighted average within the masked region (0.0 if no support).
    """
    intensity_sum = intensity_image_A + intensity_image_B
    threshold = (threshold_percent / 100.0) * np.max(intensity_sum)
    mask = intensity_sum > threshold

    # Work on copies to avoid modifying inputs outside this function
    Si_image = Si_image.copy()
    intensity_sum = intensity_sum.copy()

    Si_image[~mask] = 0
    intensity_sum[~mask] = 0

    numerator = np.sum(Si_image * intensity_sum)
    denominator = np.sum(intensity_sum)

    if denominator == 0:
        return 0.0
    average_S = numerator / denominator
    return average_S


def plot_grouped_bars(x_pos, x_ticklabels, series_triplet, colors,
                      legend_labels=None, xlabel='', ylabel='', title='',
                      title_pad=30, legend_loc=('upper left', (0.02, 1.10), 3),
                      ax=None):
    """
    Draw grouped bars for three series (e.g., σ+, H, σ−) at each x position.

    Parameters
    ----------
    x_pos : 1D array
        X positions of the groups.
    x_ticklabels : list of str
        Labels shown under each group (e.g., detuning values).
    series_triplet : tuple of 1D arrays
        Three series to plot at each group (y_A, y_B, y_C).
    colors : list of str
        Colors for the bars.
    legend_labels : list of str or None
        Labels for the legend (length 3). If None, no legend is drawn.
    xlabel, ylabel, title : str
        Axis labels and main title.
    title_pad : float
        Padding for the title.
    legend_loc : tuple
        (loc_string, bbox_to_anchor, ncol) for the legend placement.
    ax : matplotlib Axes or None
        If provided, bars are drawn on this axis; otherwise, a new figure is created.

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
        Figure and axis used for plotting.
    """

    GROUP_WIDTH = 0.78
    BAR_WIDTH = GROUP_WIDTH / 3.0 * 0.95
    OFFSETS = (-BAR_WIDTH, 0.0, +BAR_WIDTH)

    y_A, y_B, y_C = series_triplet

    # Create figure/axis if not provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot bars with safe handling of legend_labels
    ax.bar(x_pos + OFFSETS[0], y_A, width=BAR_WIDTH, color=colors[0],
           linewidth=1.0, label=legend_labels[0] if legend_labels else None)
    ax.bar(x_pos + OFFSETS[1], y_B, width=BAR_WIDTH, color=colors[1],
           linewidth=1.0, label=legend_labels[1] if legend_labels else None)
    ax.bar(x_pos + OFFSETS[2], y_C, width=BAR_WIDTH, color=colors[2],
           linewidth=1.0, label=legend_labels[2] if legend_labels else None)

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, pad=title_pad)

    # X ticks
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_ticklabels, fontsize=FS_TICK)

    # Y ticks
    ax.tick_params(axis='y', labelsize=FS_TICK, length=0)
    ax.tick_params(axis='x', length=0)

    # Frame and grid: keep all spines visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    ax.grid(axis='y', linestyle='--', alpha=0.35)

    # Add legend only if labels are provided
    if legend_labels:
        loc, bta, ncol = legend_loc
        ax.legend(loc=loc, bbox_to_anchor=bta, ncol=ncol,
                  fontsize=FS_LEGEND, frameon=False)

    return fig, ax



def zero_outside_columns(M, p1, p2):
    """
    Zero out columns outside the measured pixel window [p1, p2] in matrix M.

    This protects downstream steps from edge artifacts coming from unilluminated
    or uncollected regions.

    Parameters
    ----------
    M : 2D np.ndarray
        Input matrix to be zeroed outside [p1, p2].
    p1, p2 : int
        Column index limits (inclusive) for the measured window.

    Returns
    -------
    M : 2D np.ndarray
        The same matrix with zeroed columns outside the valid window.
    """
    M[:, :p1] = 0.0
    M[:, p2+1:] = 0.0
    return M

# Residual builder for the joint fit across the three PL panels
def joint_residuals(x, k_list, E_list, w_list):
    """
    Concatenate residuals (LPB_model(k) - E_data) across the three panels.

    Parameters
    ----------
    x : array-like, shape (8,)
        Packed parameters: [A, Omega, off1, delta1, off2, delta2, off3, delta3]
    k_list : list of np.ndarray
        k vectors per panel (length = 3)
    E_list : list of np.ndarray
        Energy vectors per panel (length = 3)
    w_list : list of float
        Optional panel weights (length = 3)

    Returns
    -------
    residuals : 1D np.ndarray
        Concatenated residuals for least_squares.
    """
    A, Omega = float(x[0]), float(x[1])
    off = np.array([x[2], x[4], x[6]], dtype=float)
    delt = np.array([x[3], x[5], x[7]], dtype=float)

    res_all = []
    for j, (k, E, w) in enumerate(zip(k_list, E_list, w_list)):
        if k.size == 0:
            continue
        Em = LPB(k, off[j], Ex_fixed, A, delt[j], Omega)  # analytical LPB
        # Weighted residuals (use sqrt(w) like in Fig_2)
        res_j = (Em - E) * np.sqrt(max(w, 0.0))
        res_all.append(res_j)

    return np.concatenate(res_all) if res_all else np.array([], dtype=float)

# Normalize for robust peak-picking (visual-only) like in Fig_2
def _normalize(img):
    """Return img / max(img) for display; leaves img untouched if max is 0/NaN."""
    m = np.nanmax(img)
    return img / m if (m not in (None, 0) and np.isfinite(m)) else img



def LPB(x, off, Ex, A, delta, omega):
    """Lower polariton branch dispersion from the coupled oscillator model."""
    return 0.5 * (Ex + (A*(x - off)**2 + delta + Ex)
                  - np.sqrt(((A*(x - off)**2 + delta)**2 + omega**2)))


def UPB(x, off, Ex, A, delta, omega):
    """Upper polariton branch dispersion from the coupled oscillator model."""
    return 0.5 * (Ex + (A*(x - off)**2 + delta + Ex)
                  + np.sqrt(((A*(x - off)**2 + delta)**2 + omega**2)))

# %% 
# ========================================================================= #
# =================== Data extraction from .sif files ===================== #
# ========================================================================= #

# Track global energy limits across all detunings (used for consistent figure axes)
energy_min_global = float('inf')
energy_max_global = float('-inf')

# Per-detuning containers for axes and other intermediates
sin_min_all, sin_max_all = {}, {}
energy_min_all, energy_max_all = {}, {}
wavelength_central_all = {}
Y_ev_all, K_parallel_all = {}, {}
k_limits_all = {}
image_for_fit_all = {}

# Detuning identifiers
detuning_nums = [cfg["detuning_num"] for cfg in configurations]
n_det = len(detuning_nums)

# Averages <S1>, <S2>, <S3> per incoming polarization (H, σ⁺, σ⁻)
S1_H = np.full(n_det, np.nan, dtype=float); S1_R = np.full(n_det, np.nan, dtype=float); S1_L = np.full(n_det, np.nan, dtype=float)
S2_H = np.full(n_det, np.nan, dtype=float); S2_R = np.full(n_det, np.nan, dtype=float); S2_L = np.full(n_det, np.nan, dtype=float)
S3_H = np.full(n_det, np.nan, dtype=float); S3_R = np.full(n_det, np.nan, dtype=float); S3_L = np.full(n_det, np.nan, dtype=float)

# Stokes modulus <P> per incoming polarization (H, σ⁺, σ⁻)
P_H = np.full(n_det, np.nan, dtype=float)
P_R = np.full(n_det, np.nan, dtype=float)
P_L = np.full(n_det, np.nan, dtype=float)

# Preallocate 3D arrays to store Stokes maps [energy, x, detuning_index]
n_energy, n_x = 1024, 1024
n_det = len(configurations)
dtype_store = np.float32

imageH_S1_matrix = np.zeros((n_energy, n_x, n_det), dtype=dtype_store)
imageR_S1_matrix = np.zeros((n_energy, n_x, n_det), dtype=dtype_store)
imageL_S1_matrix = np.zeros((n_energy, n_x, n_det), dtype=dtype_store)

imageH_S2_matrix = np.zeros((n_energy, n_x, n_det), dtype=dtype_store)
imageR_S2_matrix = np.zeros((n_energy, n_x, n_det), dtype=dtype_store)
imageL_S2_matrix = np.zeros((n_energy, n_x, n_det), dtype=dtype_store)

imageH_S3_matrix = np.zeros((n_energy, n_x, n_det), dtype=dtype_store)
imageR_S3_matrix = np.zeros((n_energy, n_x, n_det), dtype=dtype_store)
imageL_S3_matrix = np.zeros((n_energy, n_x, n_det), dtype=dtype_store)

# ------------------------------------------------------------------------- #
# Main loop across detunings: load channels, build axes, rescale intensity,
# zero edges, subtract background, compute Stokes maps and averages.
# ------------------------------------------------------------------------- #
for config in configurations:  # Loop across the three detunings
    detuning_num = config["detuning_num"]
    initial_path = config["initial_path"]
    final_path = config["final_path"]
    pixel1, pixel2 = config["pixel1"], config["pixel2"]

    # Extract central wavelength from path name to locate the matching .asc calibration
    match = re.search(r'wavelength_(\d+)_', os.path.basename(initial_path))
    if match:
        wavelength_central = int(match.group(1))
        wavelength_central_all[config["detuning_num"]] = wavelength_central

    # Path to calibration spectra folder: up two levels, then Measurements/Calibration_wavelength
    current_path = os.path.dirname(os.path.abspath(__file__))
    calibration_folder = os.path.abspath(os.path.join(current_path, "..", "Measurements", "Calibration_wavelength"))

    # Build and open the corresponding .asc calibration file (first column = wavelength in nm)
    asc_filename = f"{wavelength_central}_spectrum.asc"
    asc_path = os.path.join(calibration_folder, asc_filename).replace("\\", "/")
    with open(asc_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # Simple parser assuming numeric first token per line (comma allowed as decimal separator)
        nm_values = np.array([float(line.split()[0].replace(",", ".")) for line in lines])

    # -------------------- Load each polarization/analyzer channel -------------------- #
    # Naming convention: compose filepaths for incoming polarization and analyzer basis.
    # H in, measure H/V
    filepathH_H = initial_path + "H_L4ext_0_sample_L4_0_L2_0_H" + final_path  # Pump in H, measured in H
    filepathH_V = initial_path + "H_L4ext_0_sample_L4_0_L2_45_H" + final_path  # Pump in H, measured in V
    imageH_H = remove_defects(rotate_image(filepathH_H))
    imageH_V = remove_defects(rotate_image(filepathH_V))

    # R in, measure H/V
    filepathR_H = initial_path + "H_L4ext_45_sample_L4_0_L2_0_H" + final_path  # Pump in R, measured in H
    filepathR_V = initial_path + "H_L4ext_45_sample_L4_0_L2_45_H" + final_path  # Pump in R, measured in V
    imageR_H = remove_defects(rotate_image(filepathR_H))
    imageR_V = remove_defects(rotate_image(filepathR_V))

    # L in, measure H/V
    filepathL_H = initial_path + "H_L4ext_-45_sample_L4_0_L2_0_H" + final_path  # Pump in L, measured in H
    filepathL_V = initial_path + "H_L4ext_-45_sample_L4_0_L2_45_H" + final_path  # Pump in L, measured in V
    imageL_H = remove_defects(rotate_image(filepathL_H))
    imageL_V = remove_defects(rotate_image(filepathL_V))

    # H in, measure D/A
    filepathH_D = initial_path + "H_L4ext_0_sample_L4_45_L2_22.5_H" + final_path  # Pump in H, measured in D
    filepathH_A = initial_path + "H_L4ext_0_sample_L4_45_L2_-22.5_H" + final_path  # Pump in H, measured in A
    imageH_D = remove_defects(rotate_image(filepathH_D))
    imageH_A = remove_defects(rotate_image(filepathH_A))

    # R in, measure D/A
    filepathR_D = initial_path + "H_L4ext_45_sample_L4_45_L2_22.5_H" + final_path  # Pump in R, measured in D
    filepathR_A = initial_path + "H_L4ext_45_sample_L4_45_L2_-22.5_H" + final_path  # Pump in R, measured in A
    imageR_D = remove_defects(rotate_image(filepathR_D))
    imageR_A = remove_defects(rotate_image(filepathR_A))

    # L in, measure D/A
    filepathL_D = initial_path + "H_L4ext_-45_sample_L4_45_L2_22.5_H" + final_path  # Pump in L, measured in D
    filepathL_A = initial_path + "H_L4ext_-45_sample_L4_45_L2_-22.5_H" + final_path  # Pump in L, measured in A
    imageL_D = remove_defects(rotate_image(filepathL_D))
    imageL_A = remove_defects(rotate_image(filepathL_A))

    # H in, measure σ⁺/σ⁻
    filepathH_R = initial_path + "H_L4ext_0_sample_L4_-45_L2_45_H" + final_path  # Pump in H, measured in σ⁺
    filepathH_L = initial_path + "H_L4ext_0_sample_L4_45_L2_45_H" + final_path   # Pump in H, measured in σ⁻
    imageH_R = remove_defects(rotate_image(filepathH_R))
    imageH_L = remove_defects(rotate_image(filepathH_L))

    # R in, measure σ⁺/σ⁻
    filepathR_R = initial_path + "H_L4ext_45_sample_L4_-45_L2_45_H" + final_path  # Pump in R, measured in σ⁺
    filepathR_L = initial_path + "H_L4ext_45_sample_L4_45_L2_45_H" + final_path   # Pump in R, measured in σ⁻
    imageR_R = remove_defects(rotate_image(filepathR_R))
    imageR_L = remove_defects(rotate_image(filepathR_L))

    # L in, measure σ⁺/σ⁻
    filepathL_R = initial_path + "H_L4ext_-45_sample_L4_-45_L2_45_H" + final_path  # Pump in L, measured in σ⁺
    filepathL_L = initial_path + "H_L4ext_-45_sample_L4_45_L2_45_H" + final_path   # Pump in L, measured in σ⁻
    imageL_R = remove_defects(rotate_image(filepathL_R))
    imageL_L = remove_defects(rotate_image(filepathL_L))

    # -------------------- Build horizontal mapping (pixels → sin(theta)) -------------------- #
    # All images are 1024 x 1024 after rotation; use H→H to get the shape
    pixels_total = imageH_H.shape[1]  # = 1024

    # Linear mapping in sin(theta) across the illuminated window [pixel1, pixel2]
    sin_per_pixel = (sin2 - sin1) / (pixel2 - pixel1)  # increment in sin(theta) per pixel
    sin_min = sin1 - (pixel1 * sin_per_pixel)         # sin(theta) at column 0
    sin_max = sin_min + (pixels_total * sin_per_pixel)  # sin(theta) at last column

    # -------------------- Convert spectral axis: nm -> eV -------------------- #
    ev_values = nm_to_ev(nm_values)

    # -------------------- Build meshes for plotting: (X_sin, Y_ev) and K_parallel -------------------- #
    # Note that k depends on energy; the K grid is non-linear in x for each energy row.
    x_edges = np.linspace(sin_min, sin_max, pixels_total + 1)    # horizontal bin edges
    x_values = (x_edges[:-1] + x_edges[1:]) / 2                  # horizontal bin centers
    X_sin, Y_ev = np.meshgrid(x_values, ev_values)               # axes grids

    # Store per-detuning energy grid
    Y_ev_all[config["detuning_num"]] = Y_ev

    energy_max = Y_ev.max()
    energy_min = Y_ev.min()
    energy_per_pixel = (energy_max - energy_min) / pixels_total  # not used downstream

    energy_min_global = min(energy_min_global, energy_min)
    energy_max_global = max(energy_max_global, energy_max)

    # k_parallel = (2π * E / (h c)) * sin(theta), in µm^-1 (convert m^-1 → µm^-1)
    K_parallel = (2 * np.pi * Y_ev / (h * c)) * X_sin * 1e-6

    # -------------------- Apply nm→eV Jacobian-like factor to all intensity images -------------------- #
    # This factor expresses a variable change from λ to E on the (Y_ev) grid and is applied uniformly.
    jacobian = (np.abs(-(((h * c)**2) * 1e15))) / (2 * np.pi * Y_ev**3)

    imageH_H_final = imageH_H * jacobian
    imageH_V_final = imageH_V * jacobian
    imageH_D_final = imageH_D * jacobian
    imageH_A_final = imageH_A * jacobian
    imageH_R_final = imageH_R * jacobian
    imageH_L_final = imageH_L * jacobian

    imageR_H_final = imageR_H * jacobian
    imageR_V_final = imageR_V * jacobian
    imageR_D_final = imageR_D * jacobian
    imageR_A_final = imageR_A * jacobian
    imageR_R_final = imageR_R * jacobian
    imageR_L_final = imageR_L * jacobian

    imageL_H_final = imageL_H * jacobian
    imageL_V_final = imageL_V * jacobian
    imageL_D_final = imageL_D * jacobian
    imageL_A_final = imageL_A * jacobian
    imageL_R_final = imageL_R * jacobian
    imageL_L_final = imageL_L * jacobian

    # -------------------- Zero out columns outside the measurement window -------------------- #
    imageH_H_final = zero_outside_columns(imageH_H_final, pixel1, pixel2)
    imageH_V_final = zero_outside_columns(imageH_V_final, pixel1, pixel2)
    imageH_D_final = zero_outside_columns(imageH_D_final, pixel1, pixel2)
    imageH_A_final = zero_outside_columns(imageH_A_final, pixel1, pixel2)
    imageH_R_final = zero_outside_columns(imageH_R_final, pixel1, pixel2)
    imageH_L_final = zero_outside_columns(imageH_L_final, pixel1, pixel2)

    imageR_H_final = zero_outside_columns(imageR_H_final, pixel1, pixel2)
    imageR_V_final = zero_outside_columns(imageR_V_final, pixel1, pixel2)
    imageR_D_final = zero_outside_columns(imageR_D_final, pixel1, pixel2)
    imageR_A_final = zero_outside_columns(imageR_A_final, pixel1, pixel2)
    imageR_R_final = zero_outside_columns(imageR_R_final, pixel1, pixel2)
    imageR_L_final = zero_outside_columns(imageR_L_final, pixel1, pixel2)

    imageL_H_final = zero_outside_columns(imageL_H_final, pixel1, pixel2)
    imageL_V_final = zero_outside_columns(imageL_V_final, pixel1, pixel2)
    imageL_D_final = zero_outside_columns(imageL_D_final, pixel1, pixel2)
    imageL_A_final = zero_outside_columns(imageL_A_final, pixel1, pixel2)
    imageL_R_final = zero_outside_columns(imageL_R_final, pixel1, pixel2)
    imageL_L_final = zero_outside_columns(imageL_L_final, pixel1, pixel2)

    # ========================================================================= #
    # Background subtraction (per channel), Stokes maps, and storing results
    # ========================================================================= #

    # 1) Define once the 1D energy axis for this detuning (rows of the images)
    energy_axis_ev = Y_ev_all[detuning_num][:, 0]

    # --- S1, incoming H ---
    plot_image(imageH_H_final, 'Raw. H pumping, measured in H', K_parallel, Y_ev, pixel1, pixel2)
    imageH_H_processed = remove_polynomial_background(imageH_H_final, energy_axis_ev)
    plot_image(imageH_H_processed, 'Processed. H pumping, measured in H', K_parallel, Y_ev, pixel1, pixel2)

    plot_image(imageH_V_final, 'Raw. H pumping, measured in V', K_parallel, Y_ev, pixel1, pixel2)
    imageH_V_processed = remove_polynomial_background(imageH_V_final, energy_axis_ev)
    plot_image(imageH_V_processed, 'Processed. H pumping, measured in V', K_parallel, Y_ev, pixel1, pixel2)

    imageH_S1 = S_calculator_and_mask(imageH_H_processed, imageH_V_processed)
    setup_axis(imageH_S1, 'S1 for incoming H polarization', K_parallel, Y_ev, pixel1, pixel2)
    imageH_S1_matrix[:, :, detuning_num-1] = imageH_S1

    # Use H→(H+V) background-subtracted sum to locate dispersion maxima for LPB fitting
    image_for_fit = np.clip(imageH_H_processed + imageH_V_processed, 0, None)
    image_for_fit_all[detuning_num] = image_for_fit
    K_parallel_all[detuning_num] = K_parallel
    k_limits_all[detuning_num] = (K_parallel[0, pixel1], K_parallel[0, pixel2])

    # --- S1, incoming R ---
    plot_image(imageR_H_final, r'Raw. $\sigma^{+}$ pumping, measured in H', K_parallel, Y_ev, pixel1, pixel2)
    imageR_H_processed = remove_polynomial_background(imageR_H_final, energy_axis_ev)
    plot_image(imageR_H_processed, r'Processed. $\sigma^{+}$ pumping, measured in H', K_parallel, Y_ev, pixel1, pixel2)

    plot_image(imageR_V_final, r'Raw. $\sigma^{+}$ pumping, measured in V', K_parallel, Y_ev, pixel1, pixel2)
    imageR_V_processed = remove_polynomial_background(imageR_V_final, energy_axis_ev)
    plot_image(imageR_V_processed, r'Processed. $\sigma^{+}$ pumping, measured in V', K_parallel, Y_ev, pixel1, pixel2)

    imageR_S1 = S_calculator_and_mask(imageR_H_processed, imageR_V_processed)
    setup_axis(imageR_S1, r'S1 for incoming $\sigma^{+}$ polarization', K_parallel, Y_ev, pixel1, pixel2)
    imageR_S1_matrix[:, :, detuning_num-1] = imageR_S1

    # --- S1, incoming L ---
    plot_image(imageL_H_final, r'Raw. $\sigma^{-}$ pumping, measured in H', K_parallel, Y_ev, pixel1, pixel2)
    imageL_H_processed = remove_polynomial_background(imageL_H_final, energy_axis_ev)
    plot_image(imageL_H_processed, r'Processed. $\sigma^{-}$ pumping, measured in H', K_parallel, Y_ev, pixel1, pixel2)

    plot_image(imageL_V_final, r'Raw. $\sigma^{-}$ pumping, measured in V', K_parallel, Y_ev, pixel1, pixel2)
    imageL_V_processed = remove_polynomial_background(imageL_V_final, energy_axis_ev)
    plot_image(imageL_V_processed, r'Processed. $\sigma^{-}$ pumping, measured in V', K_parallel, Y_ev, pixel1, pixel2)

    imageL_S1 = S_calculator_and_mask(imageL_H_processed, imageL_V_processed)
    setup_axis(imageL_S1, r'S1 for incoming $\sigma^{-}$ polarization', K_parallel, Y_ev, pixel1, pixel2)
    imageL_S1_matrix[:, :, detuning_num-1] = imageL_S1

    # --- S2, incoming H ---
    plot_image(imageH_D_final, 'Raw. H pumping, measured in D', K_parallel, Y_ev, pixel1, pixel2)
    imageH_D_processed = remove_polynomial_background(imageH_D_final, energy_axis_ev)
    plot_image(imageH_D_processed, 'Processed. H pumping, measured in D', K_parallel, Y_ev, pixel1, pixel2)

    plot_image(imageH_A_final, 'Raw. H pumping, measured in A', K_parallel, Y_ev, pixel1, pixel2)
    imageH_A_processed = remove_polynomial_background(imageH_A_final, energy_axis_ev)
    plot_image(imageH_A_processed, 'Processed. H pumping, measured in A', K_parallel, Y_ev, pixel1, pixel2)

    imageH_S2 = S_calculator_and_mask(imageH_D_processed, imageH_A_processed)
    setup_axis(imageH_S2, 'S2 for incoming H polarization', K_parallel, Y_ev, pixel1, pixel2)
    imageH_S2_matrix[:, :, detuning_num-1] = imageH_S2

    # --- S2, incoming R ---
    plot_image(imageR_D_final, r'Raw. $\sigma^{+}$ pumping, measured in D', K_parallel, Y_ev, pixel1, pixel2)
    imageR_D_processed = remove_polynomial_background(imageR_D_final, energy_axis_ev)
    plot_image(imageR_D_processed, r'Processed. $\sigma^{+}$ pumping, measured in D', K_parallel, Y_ev, pixel1, pixel2)

    plot_image(imageR_A_final, r'Raw. $\sigma^{+}$ pumping, measured in A', K_parallel, Y_ev, pixel1, pixel2)
    imageR_A_processed = remove_polynomial_background(imageR_A_final, energy_axis_ev)
    plot_image(imageR_A_processed, r'Processed. $\sigma^{+}$ pumping, measured in A', K_parallel, Y_ev, pixel1, pixel2)

    imageR_S2 = S_calculator_and_mask(imageR_D_processed, imageR_A_processed)
    setup_axis(imageR_S2, r'S2 for incoming $\sigma^{+}$ polarization', K_parallel, Y_ev, pixel1, pixel2)
    imageR_S2_matrix[:, :, detuning_num-1] = imageR_S2

    # --- S2, incoming L ---
    plot_image(imageL_D_final, r'Raw. $\sigma^{-}$ pumping, measured in D', K_parallel, Y_ev, pixel1, pixel2)
    imageL_D_processed = remove_polynomial_background(imageL_D_final, energy_axis_ev)
    plot_image(imageL_D_processed, r'Processed. $\sigma^{-}$ pumping, measured in D', K_parallel, Y_ev, pixel1, pixel2)

    plot_image(imageL_A_final, r'Raw. $\sigma^{-}$ pumping, measured in A', K_parallel, Y_ev, pixel1, pixel2)
    imageL_A_processed = remove_polynomial_background(imageL_A_final, energy_axis_ev)
    plot_image(imageL_A_processed, r'Processed. $\sigma^{-}$ pumping, measured in A', K_parallel, Y_ev, pixel1, pixel2)

    imageL_S2 = S_calculator_and_mask(imageL_D_processed, imageL_A_processed)
    setup_axis(imageL_S2, r'S2 for incoming $\sigma^{-}$ polarization', K_parallel, Y_ev, pixel1, pixel2)
    imageL_S2_matrix[:, :, detuning_num-1] = imageL_S2

    # --- S3, incoming H ---
    plot_image(imageH_R_final, r'Raw. H pumping, measured in $\sigma^{+}$', K_parallel, Y_ev, pixel1, pixel2)
    imageH_R_processed = remove_polynomial_background(imageH_R_final, energy_axis_ev)
    plot_image(imageH_R_processed, r'Processed. H pumping, measured in $\sigma^{+}$', K_parallel, Y_ev, pixel1, pixel2)

    plot_image(imageH_L_final, r'Raw. H pumping, measured in $\sigma^{-}$', K_parallel, Y_ev, pixel1, pixel2)
    imageH_L_processed = remove_polynomial_background(imageH_L_final, energy_axis_ev)
    plot_image(imageH_L_processed, r'Processed. H pumping, measured in $\sigma^{-}$', K_parallel, Y_ev, pixel1, pixel2)

    imageH_S3 = S_calculator_and_mask(imageH_R_processed, imageH_L_processed)
    setup_axis(imageH_S3, 'S3 for incoming H polarization', K_parallel, Y_ev, pixel1, pixel2)
    imageH_S3_matrix[:, :, detuning_num-1] = imageH_S3

    # --- S3, incoming σ⁺ ---
    plot_image(imageR_R_final, r'Raw. $\sigma^{+}$ pumping, measured in $\sigma^{+}$', K_parallel, Y_ev, pixel1, pixel2)
    imageR_R_processed = remove_polynomial_background(imageR_R_final, energy_axis_ev)
    plot_image(imageR_R_processed, r'Processed. $\sigma^{+}$ pumping, measured in $\sigma^{+}$', K_parallel, Y_ev, pixel1, pixel2)

    plot_image(imageR_L_final, r'Raw. $\sigma^{+}$ pumping, measured in $\sigma^{-}$', K_parallel, Y_ev, pixel1, pixel2)
    imageR_L_processed = remove_polynomial_background(imageR_L_final, energy_axis_ev)
    plot_image(imageR_L_processed, r'Processed. $\sigma^{+}$ pumping, measured in $\sigma^{-}$', K_parallel, Y_ev, pixel1, pixel2)

    imageR_S3 = S_calculator_and_mask(imageR_R_processed, imageR_L_processed)
    setup_axis(imageR_S3, r'S3 for incoming $\sigma^{+}$ polarization', K_parallel, Y_ev, pixel1, pixel2)
    imageR_S3_matrix[:, :, detuning_num-1] = imageR_S3

    # --- S3, incoming σ⁻ ---
    plot_image(imageL_R_final, r'Raw. $\sigma^{-}$ pumping, measured in $\sigma^{+}$', K_parallel, Y_ev, pixel1, pixel2)
    imageL_R_processed = remove_polynomial_background(imageL_R_final, energy_axis_ev)
    plot_image(imageL_R_processed, r'Processed. $\sigma^{-}$ pumping, measured in $\sigma^{+}$', K_parallel, Y_ev, pixel1, pixel2)

    plot_image(imageL_L_final, r'Raw. $\sigma^{-}$ pumping, measured in $\sigma^{-}$', K_parallel, Y_ev, pixel1, pixel2)
    imageL_L_processed = remove_polynomial_background(imageL_L_final, energy_axis_ev)
    plot_image(imageL_L_processed, r'Processed. $\sigma^{-}$ pumping, measured in $\sigma^{-}$', K_parallel, Y_ev, pixel1, pixel2)

    imageL_S3 = S_calculator_and_mask(imageL_R_processed, imageL_L_processed)
    setup_axis(imageL_S3, r'S3 for incoming $\sigma^{-}$ polarization', K_parallel, Y_ev, pixel1, pixel2)
    imageL_S3_matrix[:, :, detuning_num-1] = imageL_S3

    # --- Averages <S1>, <S2>, <S3> per incoming polarization (weighted by (A+B) mask) ---
    S1_R[detuning_num - 1] = compute_Si_average_with_mask(imageR_S1, imageR_H_processed, imageR_V_processed, threshold_percent=threshold_percent)
    S1_H[detuning_num - 1] = compute_Si_average_with_mask(imageH_S1, imageH_H_processed, imageH_V_processed, threshold_percent=threshold_percent)
    S1_L[detuning_num - 1] = compute_Si_average_with_mask(imageL_S1, imageL_H_processed, imageL_V_processed, threshold_percent=threshold_percent)

    S2_R[detuning_num - 1] = compute_Si_average_with_mask(imageR_S2, imageR_D_processed, imageR_A_processed, threshold_percent=threshold_percent)
    S2_H[detuning_num - 1] = compute_Si_average_with_mask(imageH_S2, imageH_D_processed, imageH_A_processed, threshold_percent=threshold_percent)
    S2_L[detuning_num - 1] = compute_Si_average_with_mask(imageL_S2, imageL_D_processed, imageL_A_processed, threshold_percent=threshold_percent)

    S3_R[detuning_num - 1] = compute_Si_average_with_mask(imageR_S3, imageR_R_processed, imageR_L_processed, threshold_percent=threshold_percent)
    S3_H[detuning_num - 1] = compute_Si_average_with_mask(imageH_S3, imageH_R_processed, imageH_L_processed, threshold_percent=threshold_percent)
    S3_L[detuning_num - 1] = compute_Si_average_with_mask(imageL_S3, imageL_R_processed, imageL_L_processed, threshold_percent=threshold_percent)

    # --- Average Stokes modulus <P> per incoming polarization ---
    masterR_S = np.sqrt(imageR_S1**2 + imageR_S2**2 + imageR_S3**2)
    masterH_S = np.sqrt(imageH_S1**2 + imageH_S2**2 + imageH_S3**2)
    masterL_S = np.sqrt(imageL_S1**2 + imageL_S2**2 + imageL_S3**2)

    P_R[detuning_num - 1] = compute_Si_average_with_mask(
        masterR_S,
        (imageR_H_processed + imageR_D_processed + imageR_R_processed)/3,
        (imageR_V_processed + imageR_A_processed + imageR_L_processed)/3,
        threshold_percent=threshold_percent
    )
    P_H[detuning_num - 1] = compute_Si_average_with_mask(
        masterH_S,
        (imageH_H_processed + imageH_D_processed + imageH_R_processed)/3,
        (imageH_V_processed + imageH_A_processed + imageH_L_processed)/3,
        threshold_percent=threshold_percent
    )
    P_L[detuning_num - 1] = compute_Si_average_with_mask(
        masterL_S,
        (imageL_H_processed + imageL_D_processed + imageL_R_processed)/3,
        (imageL_V_processed + imageL_A_processed + imageL_L_processed)/3,
        threshold_percent=threshold_percent
    )

    # Optional quick console check in compact format
    print(f"[Detuning {detuning_num}] R: <S1,S2,S3> = {S1_R[detuning_num - 1]:.4f}, {S2_R[detuning_num - 1]:.4f}, {S3_R[detuning_num - 1]:.4f} \n P = {P_R[detuning_num - 1]:.4f}")
    print(f"[Detuning {detuning_num}] H: <S1,S2,S3> = {S1_H[detuning_num - 1]:.4f}, {S2_H[detuning_num - 1]:.4f}, {S3_H[detuning_num - 1]:.4f} \n P = {P_H[detuning_num - 1]:.4f}")
    print(f"[Detuning {detuning_num}] L: <S1,S2,S3> = {S1_L[detuning_num - 1]:.4f}, {S2_L[detuning_num - 1]:.4f}, {S3_L[detuning_num - 1]:.4f} \n P = {P_L[detuning_num - 1]:.4f}")

# %% 
# ========================================================================= #
# STEP 3 — Extraction of detuning
# ========================================================================= #

# Extracting necessary data
detuning_order = [1, 2, 3]  # three PL panels
image_bgsub_list_norm = [_normalize(image_for_fit_all[d]) for d in detuning_order]
K_parallel_list = [K_parallel_all[d] for d in detuning_order]
Y_ev_list = [Y_ev_all[d] for d in detuning_order]
k_limits_list = [k_limits_all[d] for d in detuning_order]

# The detunings are not extracted from the fits, the fits are only used to find E_min_PL,
# which is only the minimum energy of the PL

k_fit_list, E_fit_list, x_plot_grid = [], [], []
for i in range(3):
    Kp, Yev, img = K_parallel_list[i], Y_ev_list[i], image_bgsub_list_norm[i]

    # Column-wise maxima → seed points (k, E) for the LPB dispersion
    max_row_idx = np.nanargmax(img, axis=0)
    cols_idx = np.arange(img.shape[1])
    x_data = Kp[max_row_idx, cols_idx]  # k∥ at maxima
    y_data = Yev[max_row_idx, cols_idx]  # E at maxima

    # Fit mask inside the global k-window; relax gracefully if too few points
    mask_fit = (x_data >= fit_rangE_min_PL) & (x_data <= fit_range_max) & np.isfinite(y_data)
    x_fit = x_data[mask_fit]
    y_fit = y_data[mask_fit]
    if x_fit.size < 5:  # fallback: use all finite maxima if masked selection is too small
        mfin = np.isfinite(y_data)
        x_fit = x_data[mfin]
        y_fit = y_data[mfin]

    k_fit_list.append(np.asarray(x_fit, dtype=float))
    E_fit_list.append(np.asarray(y_fit, dtype=float))
    x_plot_grid.append(x_data)  # dense grid used later for smooth overlays


# Collect the points and default weights
k_list = [np.asarray(k, dtype=float) for k in k_fit_list]
E_list = [np.asarray(E, dtype=float) for E in E_fit_list]
weights_default = [1.0, 1.0, 1.0]
w_list = [float(weights_default[min(len(weights_default)-1, i)]) for i in range(3)]

# Initial guess (similar to Fig_2_maker; adjust if your dataset differs)
# Packing: x = [A, Omega, off1, delta1, off2, delta2, off3, delta3]
init_guess = {
    "A": 0.004,       # eV·μm^2
    "omega": 0.200,   # eV
    "off": [-0.10, 0.10, 0.05],   # μm^-1
    "delta": [ 0.10, 0.00,-0.07], # eV
}
A0 = float(init_guess["A"])
Omega0 = float(init_guess["omega"])
off0 = np.array(init_guess["off"], dtype=float)
delta0 = np.array(init_guess["delta"], dtype=float)
x0 = np.array([A0, Omega0, off0[0], delta0[0], off0[1], delta0[1], off0[2], delta0[2]], dtype=float)

# Note on bounds:
# We deliberately use method='lm' (Levenberg–Marquardt) for parity with Fig_2, which does not support bounds.
# If bounds are needed, switch to method='trf' and provide 'bounds=(lb, ub)'.
ls = least_squares(joint_residuals, x0=x0, args=(k_list, E_list, w_list), method='lm')
x_hat = ls.x
A_hat, Omega_hat = float(x_hat[0]), float(x_hat[1])
off_hat = [float(x_hat[2]), float(x_hat[4]), float(x_hat[6])]
delt_hat = [float(x_hat[3]), float(x_hat[5]), float(x_hat[7])]

# Detuning extraction
# a, b, and c are parameters obtained from the last cell in Fig_2_maker. 
# They define a polynomial transformation that converts the minimum energy value from PL 
# into the corresponding detuning in reflectivity. 
# The transformation is delta = *E_min_PL**2 + b*E_min_PL + c where E_min_PL is the minimum energy of the PL
a, b, c = 1.296303e+04, -5.741813e+04, 6.347917e+04
deltas_meV = []
for i in range(3):
    # LPB minimum occurs at k = off_hat[i]; evaluate LPB at that point
    Emin_PL = LPB(off_hat[i], off_hat[i], Ex_fixed, A_hat, delt_hat[i], Omega_hat)
    # Apply quadratic relation to compute delta in meV
    delta_meV = a * Emin_PL**2 + b * Emin_PL + c
    # Round to 1e-2 eV (i.e., 10 meV precision)
    delta_meV_rounded = round(delta_meV, -1)
    deltas_meV.append(delta_meV_rounded)

# -------------------------- Figure 3 (dispersion overlay) -------------------------- #
# Overlay LPB/UPB + bare photon on background-subtracted PL images.
# We keep parameter source from DE → LS; visuals match previous outputs.
fig3, axes3 = plt.subplots(1, 3, figsize=(30, 11), sharey=True)
plt.subplots_adjust(wspace=0.12)
RIGHT_PAD = 0.94
im_for_cbar = None

for i in range(3):
    ax = axes3[i]
    Kp = K_parallel_list[i]
    Yev = Y_ev_list[i]
    img = image_bgsub_list_norm[i]
    k_min, k_max = k_limits_list[i]

    # Background-subtracted PL image (normalized)
    im = ax.pcolormesh(Kp, Yev, img, shading='auto', cmap='viridis')
    im_for_cbar = im

    # Set consistent energy and k-limits
    ax.set_ylim(energy_min_global, energy_max_global)
    ax.set_xlim(k_min, k_max)

    # Overlays from the joint LS solution
    off_i, del_i= off_hat[i], delt_hat[i]
    x_dense = np.asarray(x_plot_grid[i], dtype=float)
    lpb = LPB(x_dense, off_i, Ex_fixed, A_hat, del_i, Omega_hat)

    ax.plot(x_dense, lpb, '.', color='cyan', markersize=6, label='LPB')

    # Points used for fitting (in red)
    ax.plot(k_list[i], E_list[i], '-', color='red', linewidth=1.2, alpha=0.9, label='Column maxima')

    # Axis formatting (paper style)
    ax.set_xlabel(r'$k_{\parallel}$ $( \mu m ^{-1})$', fontsize=40)
    if i == 0:
        ax.set_ylabel('Energy (eV)', fontsize=40)
        ax.tick_params(axis='y', which='major', labelsize=35, left=True, labelleft=True)
    else:
        ax.tick_params(axis='y', which='major', labelsize=35, left=False, labelleft=False)
    ax.tick_params(axis='x', which='major', labelsize=35, bottom=True, labelbottom=True)
    ax.set_title(rf'$\delta \sim {deltas_meV[i]:.0f} \mathrm{{meV}}$', fontsize=45, pad=16)

# External colorbar
plt.tight_layout(rect=[0.0, 0.0, RIGHT_PAD, 1.0])
cax = fig3.add_axes([0.96, 0.15, 0.015, 0.70])  # [left, bottom, width, height]
cb = fig3.colorbar(im_for_cbar, cax=cax)
cb.set_label('Normalized intensity (a. u.)', fontsize=40)
cb.ax.tick_params(labelsize=35)
plt.show()

# %% 
# =============================== Figure 4 ================================= #
# Style (font sizes)
FS_TITLE = 20
FS_LABEL = 20
FS_TICK = 18
FS_LEGEND = 16

# Order bars by detuning value (descending)
order = np.argsort(deltas_meV)[::-1]
xticklabels = [rf'{deltas_meV[i]:.0f}' for i in order]
pos = np.arange(len(order))

# For consistency, rewrite xticklabels from the raw list (kept as in original code)
xticklabels = [
    rf' ${val:.0f}$'
    for val in deltas_meV
]

# Colors/labels for the three pump configurations (σ+, H, σ−)
COLORS = ['#D62728', '#B0B0B0', '#1F77B4']  # red, grey, blue
LABELS = [r'$\sigma^{+}$', 'H', r'$\sigma^{-}$']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) S3 vs detuning
plot_grouped_bars(x_pos=pos, x_ticklabels=xticklabels,
                  series_triplet=(S3_R[order], S3_H[order], S3_L[order]),
                  colors=COLORS, legend_labels=None,  # Disable local legend
                  xlabel=r'$\delta$ (meV)', ylabel=r'$\langle S_{3}\rangle$',
                  title='', ax=axes[0])

# (b) Purity vs detuning
plot_grouped_bars(x_pos=pos, x_ticklabels=xticklabels,
                  series_triplet=(P_R[order], P_H[order], P_L[order]),
                  colors=COLORS, legend_labels=None,  # Disable local legend
                  xlabel=r'$\delta$ (meV)', ylabel=r'$\langle \rho \rangle$',
                  title='', ax=axes[1])

# Add panel labels (a) and (b) in top-left corners
axes[0].text(-0.22, 0.99, '(a)', transform=axes[0].transAxes,
             ha='left', va='top', fontsize=25, fontweight='bold')
axes[1].text(-0.22, 0.99, '(b)', transform=axes[1].transAxes,
             ha='left', va='top', fontsize=25, fontweight='bold')

# === NEW: Ensure full frame (all spines visible) ===
for ax in axes:
    # Make all spines visible and set a consistent thickness
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

# === Add a single global legend centered above both subplots ===

fig.legend(LABELS, loc='upper center', ncol=3,
           fontsize=FS_LEGEND + 2,
           frameon=False,
           bbox_to_anchor=(0.55, 1.12),
           handlelength=2.5,  # Make color boxes wider
           handleheight=1.5)  # Make color boxes taller


# Increase horizontal space between subplots
plt.subplots_adjust(wspace=0.001)  # A bit of space between both subplots

# Adjust layout
plt.tight_layout()

# --- Rasterize just images (pcolormesh) ---
for ax in axes:
    for artist in ax.collections:
        artist.set_rasterized(True) # The rest is not renderized!

# Create the foler "Figures" if it doesn't exist
figures_folder = os.path.join(current_path, "Figures")
os.makedirs(figures_folder, exist_ok=True)
# Whole path to save the figure
save_path = os.path.join(figures_folder, "Fig.Suppl.4.pdf")
# Actually save it
fig.savefig(save_path, format='pdf', bbox_inches='tight', transparent=False, metadata=None, dpi=300)

plt.show()
# %% 
# ============================== Figure S. 4 ================================== #
# =========== Stokes 3x3 — First detuning ===========
detuning_num = 1  # 1→ high, 2→ middle, 3→ low detuning
idx = detuning_num - 1  # 0-based index into * _matrix[:, :, idx]

# Retrieve precomputed S1, S2, S3 for each incoming polarization (σ+, H, σ−)
# (They were filled during the main loop above.)
imageH_S1 = imageH_S1_matrix[:, :, idx]
imageH_S2 = imageH_S2_matrix[:, :, idx]
imageH_S3 = imageH_S3_matrix[:, :, idx]

imageR_S1 = imageR_S1_matrix[:, :, idx]
imageR_S2 = imageR_S2_matrix[:, :, idx]
imageR_S3 = imageR_S3_matrix[:, :, idx]

imageL_S1 = imageL_S1_matrix[:, :, idx]
imageL_S2 = imageL_S2_matrix[:, :, idx]
imageL_S3 = imageL_S3_matrix[:, :, idx]

# Colormap for Stokes maps: 'seismic' in range [-0.5, 0.5], 'bad' (masked) in green
CMAP_ST = plt.get_cmap('seismic').copy()
CMAP_ST.set_bad('green')
NORM_ST = Normalize(vmin=-0.5, vmax=0.5)

# 3x3 grid: rows = (σ+, H, σ−), cols = (S1, S2, S3)
nrows, ncols = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 30))
stokes_grid = [
    [imageR_S1, imageR_S2, imageR_S3],  # row 0: σ+
    [imageH_S1, imageH_S2, imageH_S3],  # row 1: H
    [imageL_S1, imageL_S2, imageL_S3],  # row 2: σ−
]

# k-axis limits and energy window (consistent with previous purity figure)
k_min = K_parallel_all[detuning_num][0, pixel1]
k_max = K_parallel_all[detuning_num][0, pixel2]
ymin_ev, ymax_ev = 2.2177, 2.4504

# Use the energy grid corresponding to THIS detuning
Y_ev_plot = Y_ev_all[detuning_num]
K_plot = K_parallel_all[detuning_num]

# Draw the 9 panels
for r in range(nrows):
    for c in range(ncols):
        im = axes[r, c].pcolormesh(
            K_plot, Y_ev_plot, stokes_grid[r][c],
            shading='auto', cmap=CMAP_ST, norm=NORM_ST
        )
        axes[r, c].set_xlim(k_min, k_max)
        axes[r, c].set_ylim(ymin_ev, ymax_ev)

# Axis formatting: show energy ticks only on left column; k-axis only on bottom row
for r in range(nrows):
    for c in range(ncols):
        ax = axes[r, c]
        ax.tick_params(axis='both', which='major', labelsize=27)
        if c == 0:
            ax.set_ylabel('Energy (eV)', fontsize=45)
            ax.tick_params(axis='y', which='major', labelsize=40, left=True, labelleft=True)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        if r == nrows - 1:
            ax.set_xlabel(r'$k_{\parallel}$ $( \mu m^{-1})$', fontsize=45)
            ax.tick_params(axis='x', which='major', labelsize=40, bottom=True, labelbottom=True)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Column headers only on the top row
col_titles = [r'$S_{1}$', r'$S_{2}$', r'$S_{3}$']
for i in range(ncols):
    axes[0, i].set_title(col_titles[i], fontsize=50, fontweight='normal', pad=20)

# Ensure no titles in middle/bottom rows
for i in range(ncols):
    axes[1, i].set_title('')
    axes[2, i].set_title('')

# Leave right margin for a vertical colorbar; also left pad to place σ+/H/σ− labels
LEFT_FIG_PAD = 0.10
plt.tight_layout(rect=[LEFT_FIG_PAD, 0.0, 0.94, 0.965])

# Add σ+, H, σ− labels on the far left, vertically centered per row
row_labels = [r'$\boldsymbol{\sigma}^{+}$', r'$\mathbf{H}$', r'$\boldsymbol{\sigma}^{-}$']
for r, label in enumerate(row_labels):
    bbox = axes[r, 0].get_position()
    y_text = 0.5 * (bbox.y0 + bbox.y1)
    x_text = LEFT_FIG_PAD - 0.015
    fig.text(x_text, y_text, label, ha='right', va='center', fontsize=50, fontweight='bold', color='black')

# Supertitle with detuning value (rounded to integer meV)
delta_meV = int(np.rint(deltas_meV[idx]))
fig.suptitle(r'$ \delta \sim {} \,\mathrm{{meV}}$'.format(delta_meV),
             fontsize=60, fontweight='normal',
             x=0.50 + LEFT_FIG_PAD/2, y=1)

# Right-side colorbar (seismic, range -0.5 to 0.5)
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.75])
sm = ScalarMappable(norm=NORM_ST, cmap=CMAP_ST)
sm.set_array([])  # compatibility
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=40)
cbar.set_ticks([-0.5, 0.0, 0.5])
cbar.set_label(r'$S_{1,2,3}$', fontweight='bold', fontsize=60)


# --- Rasterize just images (pcolormesh) ---
for ax in axes.flatten():  # CORRECT
    for artist in ax.collections:
        artist.set_rasterized(True)  # Only rasterize images


# Create the foler "Figures" if it doesn't exist
figures_folder = os.path.join(current_path, "Figures")
os.makedirs(figures_folder, exist_ok=True)
# Whole path to save the figure
save_path = os.path.join(figures_folder, "Fig.Suppl.4.pdf")
# Actually save it
fig.savefig(save_path, format='pdf', bbox_inches='tight', transparent=False, metadata=None, dpi=300)


plt.show()

# %% 
# ============================== Figure S. 5 ================================== #
# =========== Stokes 3x3 — Second detuning ===========
detuning_num = 2  # 1→ high, 2→ middle, 3→ low detuning
idx = detuning_num - 1  # 0-based index into * _matrix[:, :, idx]

# Retrieve precomputed S1, S2, S3 for each incoming polarization (σ+, H, σ−)
# (They were filled during the main loop above.)
imageH_S1 = imageH_S1_matrix[:, :, idx]
imageH_S2 = imageH_S2_matrix[:, :, idx]
imageH_S3 = imageH_S3_matrix[:, :, idx]

imageR_S1 = imageR_S1_matrix[:, :, idx]
imageR_S2 = imageR_S2_matrix[:, :, idx]
imageR_S3 = imageR_S3_matrix[:, :, idx]

imageL_S1 = imageL_S1_matrix[:, :, idx]
imageL_S2 = imageL_S2_matrix[:, :, idx]
imageL_S3 = imageL_S3_matrix[:, :, idx]

# Colormap for Stokes maps: 'seismic' in range [-0.5, 0.5], 'bad' (masked) in green
CMAP_ST = plt.get_cmap('seismic').copy()
CMAP_ST.set_bad('green')
NORM_ST = Normalize(vmin=-0.5, vmax=0.5)

# 3x3 grid: rows = (σ+, H, σ−), cols = (S1, S2, S3)
nrows, ncols = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 30))
stokes_grid = [
    [imageR_S1, imageR_S2, imageR_S3],  # row 0: σ+
    [imageH_S1, imageH_S2, imageH_S3],  # row 1: H
    [imageL_S1, imageL_S2, imageL_S3],  # row 2: σ−
]

# k-axis limits and energy window (consistent with previous purity figure)
k_min = K_parallel_all[detuning_num][0, pixel1]
k_max = K_parallel_all[detuning_num][0, pixel2]
ymin_ev, ymax_ev = 2.2177, 2.4504

# Use the energy grid corresponding to THIS detuning
Y_ev_plot = Y_ev_all[detuning_num]
K_plot = K_parallel_all[detuning_num]

# Draw the 9 panels
for r in range(nrows):
    for c in range(ncols):
        im = axes[r, c].pcolormesh(
            K_plot, Y_ev_plot, stokes_grid[r][c],
            shading='auto', cmap=CMAP_ST, norm=NORM_ST
        )
        axes[r, c].set_xlim(k_min, k_max)
        axes[r, c].set_ylim(ymin_ev, ymax_ev)

# Axis formatting: show energy ticks only on left column; k-axis only on bottom row
for r in range(nrows):
    for c in range(ncols):
        ax = axes[r, c]
        ax.tick_params(axis='both', which='major', labelsize=27)
        if c == 0:
            ax.set_ylabel('Energy (eV)', fontsize=45)
            ax.tick_params(axis='y', which='major', labelsize=40, left=True, labelleft=True)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        if r == nrows - 1:
            ax.set_xlabel(r'$k_{\parallel}$ $( \mu m^{-1})$', fontsize=45)
            ax.tick_params(axis='x', which='major', labelsize=40, bottom=True, labelbottom=True)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Column headers only on the top row
col_titles = [r'$S_{1}$', r'$S_{2}$', r'$S_{3}$']
for i in range(ncols):
    axes[0, i].set_title(col_titles[i], fontsize=50, fontweight='normal', pad=20)

# Ensure no titles in middle/bottom rows
for i in range(ncols):
    axes[1, i].set_title('')
    axes[2, i].set_title('')

# Leave right margin for a vertical colorbar; also left pad to place σ+/H/σ− labels
LEFT_FIG_PAD = 0.10
plt.tight_layout(rect=[LEFT_FIG_PAD, 0.0, 0.94, 0.965])

# Add σ+, H, σ− labels on the far left, vertically centered per row
row_labels = [r'$\boldsymbol{\sigma}^{+}$', r'$\mathbf{H}$', r'$\boldsymbol{\sigma}^{-}$']
for r, label in enumerate(row_labels):
    bbox = axes[r, 0].get_position()
    y_text = 0.5 * (bbox.y0 + bbox.y1)
    x_text = LEFT_FIG_PAD - 0.015
    fig.text(x_text, y_text, label, ha='right', va='center', fontsize=50, fontweight='bold', color='black')

# Supertitle with detuning value (rounded to integer meV)
delta_meV = int(np.rint(deltas_meV[idx]))
fig.suptitle(r'$ \delta \sim {} \,\mathrm{{meV}}$'.format(delta_meV),
             fontsize=60, fontweight='normal',
             x=0.50 + LEFT_FIG_PAD/2, y=1)

# Right-side colorbar (seismic, range -0.5 to 0.5)
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.75])
sm = ScalarMappable(norm=NORM_ST, cmap=CMAP_ST)
sm.set_array([])  # compatibility
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=40)
cbar.set_ticks([-0.5, 0.0, 0.5])
cbar.set_label(r'$S_{1,2,3}$', fontweight='bold', fontsize=60)

# --- Rasterize just images (pcolormesh) ---
for ax in axes.flatten():  
    for artist in ax.collections:
        artist.set_rasterized(True)  # Only rasterize images

# Create the foler "Figures" if it doesn't exist
figures_folder = os.path.join(current_path, "Figures")
os.makedirs(figures_folder, exist_ok=True)
# Whole path to save the figure
save_path = os.path.join(figures_folder, "Fig.Suppl.5.pdf")
# Actually save it
fig.savefig(save_path, format='pdf', bbox_inches='tight', transparent=False, metadata=None, dpi=300)


plt.show()
# %% 
# ============================== Figure S. 6 ================================== #
# =========== Stokes 3x3 — Third detuning ===========
detuning_num = 3  # 1→ high, 2→ middle, 3→ low detuning
idx = detuning_num - 1  # 0-based index into * _matrix[:, :, idx]

# Retrieve precomputed S1, S2, S3 for each incoming polarization (σ+, H, σ−)
# (They were filled during the main loop above.)
imageH_S1 = imageH_S1_matrix[:, :, idx]
imageH_S2 = imageH_S2_matrix[:, :, idx]
imageH_S3 = imageH_S3_matrix[:, :, idx]

imageR_S1 = imageR_S1_matrix[:, :, idx]
imageR_S2 = imageR_S2_matrix[:, :, idx]
imageR_S3 = imageR_S3_matrix[:, :, idx]

imageL_S1 = imageL_S1_matrix[:, :, idx]
imageL_S2 = imageL_S2_matrix[:, :, idx]
imageL_S3 = imageL_S3_matrix[:, :, idx]

# Colormap for Stokes maps: 'seismic' in range [-0.5, 0.5], 'bad' (masked) in green
CMAP_ST = plt.get_cmap('seismic').copy()
CMAP_ST.set_bad('green')
NORM_ST = Normalize(vmin=-0.5, vmax=0.5)

# 3x3 grid: rows = (σ+, H, σ−), cols = (S1, S2, S3)
nrows, ncols = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 30))
stokes_grid = [
    [imageR_S1, imageR_S2, imageR_S3],  # row 0: σ+
    [imageH_S1, imageH_S2, imageH_S3],  # row 1: H
    [imageL_S1, imageL_S2, imageL_S3],  # row 2: σ−
]

# k-axis limits and energy window (consistent with previous purity figure)
k_min = K_parallel_all[detuning_num][0, pixel1]
k_max = K_parallel_all[detuning_num][0, pixel2]
ymin_ev, ymax_ev = 2.2177, 2.4504

# Use the energy grid corresponding to THIS detuning
Y_ev_plot = Y_ev_all[detuning_num]
K_plot = K_parallel_all[detuning_num]

# Draw the 9 panels
for r in range(nrows):
    for c in range(ncols):
        im = axes[r, c].pcolormesh(
            K_plot, Y_ev_plot, stokes_grid[r][c],
            shading='auto', cmap=CMAP_ST, norm=NORM_ST
        )
        axes[r, c].set_xlim(k_min, k_max)
        axes[r, c].set_ylim(ymin_ev, ymax_ev)

# Axis formatting: show energy ticks only on left column; k-axis only on bottom row
for r in range(nrows):
    for c in range(ncols):
        ax = axes[r, c]
        ax.tick_params(axis='both', which='major', labelsize=27)
        if c == 0:
            ax.set_ylabel('Energy (eV)', fontsize=45)
            ax.tick_params(axis='y', which='major', labelsize=40, left=True, labelleft=True)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        if r == nrows - 1:
            ax.set_xlabel(r'$k_{\parallel}$ $( \mu m^{-1})$', fontsize=45)
            ax.tick_params(axis='x', which='major', labelsize=40, bottom=True, labelbottom=True)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Column headers only on the top row
col_titles = [r'$S_{1}$', r'$S_{2}$', r'$S_{3}$']
for i in range(ncols):
    axes[0, i].set_title(col_titles[i], fontsize=50, fontweight='normal', pad=20)

# Ensure no titles in middle/bottom rows
for i in range(ncols):
    axes[1, i].set_title('')
    axes[2, i].set_title('')

# Leave right margin for a vertical colorbar; also left pad to place σ+/H/σ− labels
LEFT_FIG_PAD = 0.10
plt.tight_layout(rect=[LEFT_FIG_PAD, 0.0, 0.94, 0.965])

# Add σ+, H, σ− labels on the far left, vertically centered per row
row_labels = [r'$\boldsymbol{\sigma}^{+}$', r'$\mathbf{H}$', r'$\boldsymbol{\sigma}^{-}$']
for r, label in enumerate(row_labels):
    bbox = axes[r, 0].get_position()
    y_text = 0.5 * (bbox.y0 + bbox.y1)
    x_text = LEFT_FIG_PAD - 0.015
    fig.text(x_text, y_text, label, ha='right', va='center', fontsize=50, fontweight='bold', color='black')

# Supertitle with detuning value (rounded to integer meV)
delta_meV = int(np.rint(deltas_meV[idx]))
fig.suptitle(r'$ \delta \sim {} \,\mathrm{{meV}}$'.format(delta_meV),
             fontsize=60, fontweight='normal',
             x=0.50 + LEFT_FIG_PAD/2, y=1)

# Right-side colorbar (seismic, range -0.5 to 0.5)
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.75])
sm = ScalarMappable(norm=NORM_ST, cmap=CMAP_ST)
sm.set_array([])  # compatibility
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=40)
cbar.set_ticks([-0.5, 0.0, 0.5])
cbar.set_label(r'$S_{1,2,3}$', fontweight='bold', fontsize=60)

# --- Rasterize just images (pcolormesh) ---
for ax in axes.flatten():
    for artist in ax.collections:
        artist.set_rasterized(True)  # Only rasterize images

# Create the foler "Figures" if it doesn't exist
figures_folder = os.path.join(current_path, "Figures")
os.makedirs(figures_folder, exist_ok=True)
# Whole path to save the figure
save_path = os.path.join(figures_folder, "Fig.Suppl.6.pdf")
# Actually save it
fig.savefig(save_path, format='pdf', bbox_inches='tight', transparent=False, metadata=None, dpi=300)


plt.show()

# %%
# ============================== Figure S. 7 ================================== #
# =========== 3x3 Purities ===========
# Build a purity colormap from the top half (white→red) of 'seismic'; keep masked in green
campa = plt.get_cmap('seismic').copy()
cmapa = campa
campa.set_bad(color='green')

try:
    CMAP_PUR = ListedColormap(campa(np.linspace(0.5, 1.0, 256)))
    CMAP_PUR.set_bad(campa._rgba_bad)
except Exception:
    _seis = plt.get_cmap('seismic')
    CMAP_PUR = ListedColormap(_seis(np.linspace(0.5, 1.0, 256)))
    CMAP_PUR.set_bad('green')

NORM_PUR = Normalize(vmin=0.0, vmax=1.0)

# Figure layout: rows = pump polarization (σ+, H, σ−), columns = detuning panels
nrows, ncols = 3, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 30))

# Fixed energy window
ymin_ev, ymax_ev = 2.2177, 2.4504

# Loop over detuning panels
for i in range(len(detuning_nums)):
    detuning_idx = detuning_nums[i] - 1

    # PER-PANEL K/Y GRIDS AND k-LIMITS
    K_plot = K_parallel_all[detuning_nums[i]]
    Y_plot = Y_ev_all[detuning_nums[i]]
    k_min = K_plot[0, pixel1]
    k_max = K_plot[0, pixel2]

    # Compute purity for each polarization: ρ = sqrt(S1² + S2² + S3²)
    imageH_total = np.sqrt(imageH_S1_matrix[:, :, detuning_idx]**2 +
                           imageH_S2_matrix[:, :, detuning_idx]**2 +
                           imageH_S3_matrix[:, :, detuning_idx]**2)

    imageR_total = np.sqrt(imageR_S1_matrix[:, :, detuning_idx]**2 +
                           imageR_S2_matrix[:, :, detuning_idx]**2 +
                           imageR_S3_matrix[:, :, detuning_idx]**2)

    imageL_total = np.sqrt(imageL_S1_matrix[:, :, detuning_idx]**2 +
                           imageL_S2_matrix[:, :, detuning_idx]**2 +
                           imageL_S3_matrix[:, :, detuning_idx]**2)

    pur_list = [imageR_total, imageH_total, imageL_total]  # row order: σ+, H, σ−

    # Plot each row for this detuning with correct K/Y and k-limits
    for r in range(nrows):
        im = axes[r, i].pcolormesh(
            K_plot, Y_plot, np.clip(pur_list[r], 0.0, 1.0),
            shading='auto', cmap=CMAP_PUR, norm=NORM_PUR
        )
        axes[r, i].set_xlim(k_min, k_max)
        axes[r, i].set_ylim(ymin_ev, ymax_ev)

# Shared axis formatting
for r in range(nrows):
    for c in range(ncols):
        ax = axes[r, c]
        ax.tick_params(axis='both', which='major', labelsize=27)
        if c == 0:
            ax.set_ylabel('Energy (eV)', fontsize=45)
            ax.tick_params(axis='y', which='major', labelsize=40, left=True, labelleft=True)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        if r == nrows - 1:
            ax.set_xlabel(r'$k_{\parallel}$ $( \mu m^{-1})$', fontsize=45)
            ax.tick_params(axis='x', which='major', labelsize=40, bottom=True, labelbottom=True)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Column titles include detuning (rounded)
for i in range(ncols):
    axes[0, i].set_title(
        rf'$\delta \sim {int(deltas_meV[i])} ~\mathrm{{meV}}$',
        fontsize=45, fontweight='normal', pad=20
    )

# Layout and row labels (σ+, H, σ−)
LEFT_FIG_PAD = 0.10
plt.tight_layout(rect=[LEFT_FIG_PAD, 0.0, 0.94, 0.965])

row_labels = [r'$\boldsymbol{\sigma}^{+}$', r'$\mathbf{H}$', r'$\boldsymbol{\sigma}^{-}$']
for r, label in enumerate(row_labels):
    bbox = axes[r, 0].get_position()
    y_text = 0.5 * (bbox.y0 + bbox.y1)
    x_text = LEFT_FIG_PAD - 0.015
    fig.text(x_text, y_text, label, ha='right', va='center', fontsize=50, fontweight='bold', color='black')

# Remove supertitle and add colorbar with label ρ
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.75])
sm = ScalarMappable(norm=NORM_PUR, cmap=CMAP_PUR)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=40)
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
cbar.set_label(r'$\rho$', fontweight='bold', fontsize=60)

# --- Rasterize just images (pcolormesh) ---
for ax in axes.flatten(): 
    for artist in ax.collections:
        artist.set_rasterized(True)  # Only rasterize images

# Create the foler "Figures" if it doesn't exist
figures_folder = os.path.join(current_path, "Figures")
os.makedirs(figures_folder, exist_ok=True)
# Whole path to save the figure
save_path = os.path.join(figures_folder, "Fig.Suppl.7.pdf")
# Actually save it
fig.savefig(save_path, format='pdf', bbox_inches='tight', transparent=False, metadata=None, dpi=300)


plt.show()

# %% 
# =============================== Final Fig. 3 ============================= #
# =========== Final 2x3 figure (combined left/right panels) ===========
# NOTE: This block assembles S3 (top) and purity (bottom) per detuning,
#       stitching the left half from σ− and the right half from σ+.

S3_CBAR_MIN, S3_CBAR_MAX = -1.0, 1.0
PUR_CBAR_MIN, PUR_CBAR_MAX = 0.0, 1.0

DRAW_SPLIT_AT_K0 = True  # draw a vertical dashed line at k = 0
SPLIT_OFFSET_PX = 0      # optional column offset for the split

TEXT_TOP = ''            # optional text overlay in the top row
TEXT_BOTTOM = ''         # optional text overlay in the bottom row
TEXT_FONTSIZE = 50
TEXT_COLOR = 'black'
TEXT_WEIGHT = 'bold'
TEXT_TOP_POS = (0.50, 0.80)
TEXT_BOTTOM_POS = (0.50, 0.80)

SHOW_SIGMA_LABELS = True
SIGMA_LEFT_LABEL = r'$\sigma^{-}$'
SIGMA_RIGHT_LABEL = r'$\sigma^{+}$'
SIGMA_LABEL_FONTSIZE = 50
SIGMA_LEFT_POS = (0.25, 0.95)
SIGMA_RIGHT_POS = (0.75, 0.95)

# Purity colormap: take the top half (white→red) of 'seismic'; keep 'bad' as green.
CMAP_PUR = ListedColormap(campa(np.linspace(0.5, 1.0, 256)))
try:
    CMAP_PUR.set_bad(campa._rgba_bad)
except Exception:
    CMAP_PUR.set_bad('green')

# Normalizations for S3 and purity
NORM_S3 = Normalize(vmin=S3_CBAR_MIN, vmax=S3_CBAR_MAX)
NORM_PUR = Normalize(vmin=PUR_CBAR_MIN, vmax=PUR_CBAR_MAX)

ncols = len(detuning_nums)
fig, axes = plt.subplots(2, ncols, figsize=(10 * ncols, 22))
im_top_for_cbar = None
im_bottom_for_cbar = None

for i in range(ncols):
    det_num = detuning_nums[i] - 1

    # Use K/Y grids of the current panel (detuning)
    K_plot = K_parallel_all[detuning_nums[i]]
    Y_plot = Y_ev_all[detuning_nums[i]]

    # Column index closest to k = 0 in THIS panel (used to stitch left σ− / right σ+)
    j0 = int(np.argmin(np.abs(K_plot[0, :] - 0.0)))
    mid_idx = int(np.clip(j0 + SPLIT_OFFSET_PX, 1, K_plot.shape[1] - 1))

    # k-limits of THIS panel (measured pixel window)
    k_min = K_plot[0, pixel1]
    k_max = K_plot[0, pixel2]

    # Pull S3 (R,L) and S1,S2 for both σ+ and σ−, then compute total purity
    imageR_S3 = imageR_S3_matrix[:, :, det_num]
    imageL_S3 = imageL_S3_matrix[:, :, det_num]

    imageR_S1 = imageR_S1_matrix[:, :, det_num]
    imageR_S2 = imageR_S2_matrix[:, :, det_num]
    imageR_tot = np.sqrt(imageR_S1**2 + imageR_S2**2 + imageR_S3**2)

    imageL_S1 = imageL_S1_matrix[:, :, det_num]
    imageL_S2 = imageL_S2_matrix[:, :, det_num]
    imageL_tot = np.sqrt(imageL_S1**2 + imageL_S2**2 + imageL_S3**2)

    # Combine halves: left half from σ−, right half from σ+
    s3_combined = np.zeros_like(imageR_S3)
    s3_combined[:, :mid_idx] = imageL_S3[:, :mid_idx]
    s3_combined[:, mid_idx:] = imageR_S3[:, mid_idx:]

    pur_combined = np.zeros_like(imageR_tot)
    pur_combined[:, :mid_idx] = imageL_tot[:, :mid_idx]
    pur_combined[:, mid_idx:] = imageR_tot[:, mid_idx:]
    pur_combined = np.clip(pur_combined, PUR_CBAR_MIN, PUR_CBAR_MAX)

    ax_top = axes[0, i]
    ax_bot = axes[1, i]

    # Optional text overlays (top row)
    ax_top.text(TEXT_TOP_POS[0], TEXT_TOP_POS[1], TEXT_TOP, transform=ax_top.transAxes,
                ha='center', va='center', fontsize=TEXT_FONTSIZE, color=TEXT_COLOR, fontweight=TEXT_WEIGHT)

    # σ labels at the top
    if SHOW_SIGMA_LABELS:
        ax_top.text(SIGMA_LEFT_POS[0], SIGMA_LEFT_POS[1], SIGMA_LEFT_LABEL, transform=ax_top.transAxes,
                    ha='left', va='top', fontsize=SIGMA_LABEL_FONTSIZE, color='black')
        ax_top.text(SIGMA_RIGHT_POS[0], SIGMA_RIGHT_POS[1], SIGMA_RIGHT_LABEL, transform=ax_top.transAxes,
                    ha='right', va='top', fontsize=SIGMA_LABEL_FONTSIZE, color='black')

    if DRAW_SPLIT_AT_K0:
        ax_top.axvline(0, color='black', linestyle='--', linewidth=1.75)
        
    # Panel labels a,b,c,d,e,f for top and bottom rows
    panel_labels_top = ['(a)', '(b)', '(c)']   # adjust according to ncols
    panel_labels_bottom = ['(d)', '(e)', '(f)']  # adjust according to ncols
    label_pos = (0.03, 0.13)
    label_fontsize = 60 
    ax_top.text(label_pos[0], label_pos[1], panel_labels_top[i], transform=ax_top.transAxes,
                ha='left', va='top', fontsize=label_fontsize, fontweight='bold', color='black')
    ax_bot.text(label_pos[0], label_pos[1], panel_labels_bottom[i], transform=ax_bot.transAxes,                
                ha='left', va='top', fontsize=label_fontsize, fontweight='bold', color='black')
        
    # Top row: S3 combined (panel-specific K/Y)
    im_top = ax_top.pcolormesh(K_plot, Y_plot, s3_combined,
                               shading='auto', cmap=campa, norm=NORM_S3)
    im_top_for_cbar = im_top
    ax_top.set_xlim(k_min, k_max)
    ax_top.set_ylim(ymin_ev, ymax_ev)  # keep energy range identical

    # Bottom row: purity combined (panel-specific K/Y)
    im_bot = ax_bot.pcolormesh(K_plot, Y_plot, pur_combined,
                               shading='auto', cmap=CMAP_PUR, norm=NORM_PUR)
    im_bottom_for_cbar = im_bot
    ax_bot.set_xlim(k_min, k_max)
    ax_bot.set_ylim(ymin_ev, ymax_ev)

    if i == 0:
        ax_top.set_ylabel('Energy (eV)', fontsize=50)
        ax_top.tick_params(axis='y', which='major', labelsize=40, left=True, labelleft=True)

        ax_bot.set_ylabel('Energy (eV)', fontsize=50)
        ax_bot.tick_params(axis='y', which='major', labelsize=40, left=True, labelleft=True)
    else:
        ax_top.set_ylabel('')
        ax_top.tick_params(axis='y', which='both', left=False, labelleft=False)

        ax_bot.set_ylabel('')
        ax_bot.tick_params(axis='y', which='both', left=False, labelleft=False)

    ax_top.set_xlabel('')
    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    ax_bot.set_xlabel(r'$k_{\parallel}$ $( \mu m^{-1})$', fontsize=50)
    ax_bot.tick_params(axis='x', which='major', labelsize=40, bottom=True, labelbottom=True)

    ax_bot.text(TEXT_BOTTOM_POS[0], TEXT_BOTTOM_POS[1], TEXT_BOTTOM, transform=ax_bot.transAxes,
                ha='center', va='center', fontsize=TEXT_FONTSIZE, color=TEXT_COLOR, fontweight=TEXT_WEIGHT)

    if SHOW_SIGMA_LABELS:
        ax_bot.text(SIGMA_LEFT_POS[0], SIGMA_LEFT_POS[1], SIGMA_LEFT_LABEL, transform=ax_bot.transAxes,
                    ha='left', va='top', fontsize=SIGMA_LABEL_FONTSIZE, color='black')
        ax_bot.text(SIGMA_RIGHT_POS[0], SIGMA_RIGHT_POS[1], SIGMA_RIGHT_LABEL, transform=ax_bot.transAxes,
                    ha='right', va='top', fontsize=SIGMA_LABEL_FONTSIZE, color='black')

    if DRAW_SPLIT_AT_K0:
        ax_bot.axvline(0, color='black', linestyle='--', linewidth=1.75)

# Column headers (detuning) on the first row
detuning_titles = [
    rf'$\delta \sim {int(deltas_meV[i])} \ \mathrm{{meV}}$'
    for i in range(ncols)
]
for i in range(ncols):
    axes[0, i].set_title(detuning_titles[i], fontsize=45, fontweight='normal', pad=20)

plt.tight_layout(rect=[0.0, 0.0, 0.94, 1.0])

# Two vertical colorbars (top: S3, bottom: purity)
cax_top = fig.add_axes([0.95, 0.59, 0.02, 0.35])
cb_top = fig.colorbar(im_top_for_cbar, cax=cax_top)
cb_top.set_label(r'$S_{3}$', fontweight='bold', fontsize=60)  # top colorbar label
cb_top.ax.tick_params(labelsize=35)

cax_bot = fig.add_axes([0.95, 0.10, 0.02, 0.35])
cb_bot = fig.colorbar(im_bottom_for_cbar, cax=cax_bot)
cb_bot.set_label(r'$\rho$', fontweight='bold', fontsize=60)  # bottom colorbar label
cb_bot.ax.tick_params(labelsize=35)

# --- Rasterize just images (pcolormesh) ---
for ax in axes.flatten(): 
    for artist in ax.collections:
        artist.set_rasterized(True)  # Only rasterize images

# Create the foler "Figures" if it doesn't exist
figures_folder = os.path.join(current_path, "Figures")
os.makedirs(figures_folder, exist_ok=True)
# Whole path to save the figure
save_path = os.path.join(figures_folder, "Fig.3.pdf")
# Actually save it
fig.savefig(save_path, format='pdf', bbox_inches='tight', transparent=False, metadata=None, dpi=300)

plt.show()

