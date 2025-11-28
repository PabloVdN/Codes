
"""
Generates Figure 2: full processing and visualization pipeline for PL (top row)
and white-light reflectivity (bottom row).

Overview
--------
This script loads Andor .sif frames, removes detector defects, calibrates the spectral
axis from wavelength to energy, builds 2D energy–momentum meshes (Y_ev, K_parallel),
applies the nm→eV Jacobian to preserve spectral density, and renders a raw 2×3 grid
(PL on top, reflectivity on bottom). It then corrects white-light reflectivity using
a measured lamp profile, removes a smooth k-background from PL, extracts column-wise
extrema (maxima/minima), performs joint fits of the coupled-oscillator dispersion,
and finally combines PL+reflectivity into a single k′-centered figure with analytical
overlays. The last cell builds a quadratic mapping between PL LPB minima and
reflectivity detuning for use in downstream scripts.
 
End-to-end pipeline
-------------------
1) Load each .sif (once), clean hot pixels/outliers, and rotate to (rows = energy, columns = angle/pixels).
2) Calibrate the spectral axis from wavelength[nm] (matching .asc file) to energy[eV].
3) Build 2D meshes: X_sin (sinθ per column) and Y_ev (energy per row); compute K_parallel(E, sinθ) in μm⁻¹. 
4) Apply the nm→eV Jacobian to obtain image_ev, preserving intensity density under variable change. 
5) Plot the raw 2×3 grid (PL on top, white-light reflectivity on bottom), set k-limits per panel and a shared energy window; cache all intermediate arrays. 
6) Extract the white-light background from an empty reflectivity .sif, average over an x-window to get the lamp spectral profile, smooth and cubic-interpolate it.
7) Process PL (remove smooth k-background, pick column maxima) and reflectivity (divide by lamp profile, pick column minima); normalize for display and build fit masks in k.
8) Run joint least-squares fits (PL and reflectivity independently) for LPB/UPB/photon mode; overlay analytical branches, report δ, Ω, A and uncertainties; print results.
9) Combine PL (right side) and reflectivity (left side) per detuning into a 1×3 figure in k′=k−k₀; remap reflectivity onto the PL grid, blend at the seam, add overlays and labels.

Cells guide
-----------
• Cell 1 - Header & helpers:
    Physical constants (h, c) and a fixed exciton energy extracted from white light absorption
    measurements as cited in the article Ex = 2.404 eV; wavelength→energy conversion (nm_to_ev); 
    hot-pixel suppression (local mean); k-background removal via row-averaged O(1) polynomial plus 
    robust skewed Lorentzians; analytical LPB/UPB and photon mode; residuals for joint fitting;
    file lists, pixel windows, and k-fit ranges. 

• Cell 2 — Raw 2×3 figure + precomputation:
    – Load and rotate each .sif; defect removal.
    – Extract central wavelength from filename, read matching .asc, convert nm→eV.
    – Build sinθ per column from pixel pairs, then X_sin and Y_ev meshes; compute K_parallel(E, sinθ).
    – Apply the nm→eV Jacobian to reweight intensities (image_ev).
    – Plot the 2×3 grid, set per-panel k-limits and a shared energy window.
    – Cache image_ev, K_parallel, Y_ev, k-limits, and energy ranges for later reuse.

• Cell 3 — Reflectivity background (lamp profile):
    – Load an empty reflectivity .sif, rotate and clean; read its .asc and convert nm→eV.
    – Map pixel columns to sample position (μm); render energy vs position.
    – Average a chosen x-window (mean or Gaussian-weighted), mildly smooth, and
      build a cubic interpolation of the lamp spectral profile for downstream correction.
      
• Cell 4 — Processed 2×3 (corrections + extrema + joint fit):
    – Top row (PL): subtract a smooth k-background using the row-averaged O(1) polynomial;
      normalize; pick column-wise maxima; mask to a predefined k-window. 
    – Joint least-squares fits (per triplet): pack parameters [A, Ω, off₁, δ₁, off₂, δ₂, off₃, δ₃],
      solve with method='lm' (no bounds), estimate uncertainties from (JᵀJ)⁻¹, overlay LPB/UPB/photon,
      and title panels with δ±σ. Console prints for A, Ω, δ, off.

• Cell 5 — Final combination (PL + reflectivity in k′), create Fig. 2:
    – Use fitted offsets to recenter each side at k′=0; normalize left/right independently.
    – Remap reflectivity left half onto PL k′-grid (row-wise linear interpolation), blend the seam.
    – Draw LPB/UPB/photon using reflectivity-fit parameters; set limits and labels; add
      “Refl.” and “PL” badges.

• Cell 6 — δ_reflectivity vs PL LPB minimum energy:
    – Collect PL fit parameters and compute E_min (via LPB(k=off)) for each detuning.
    – Collect reflectivity δ values; fit a quadratic δ(meV) from resistivity vs E_min_PL(eV); plot the curve
      and print coefficients that are used in Fig_3_4_Fig_Suppl_4_5_6_7 maker, as each detunign there
      is calculated from the minimum energy of the PL measurement.

Key notes
---------
• The nm→eV Jacobian ensures the change of variables preserves spectral density.
• The k-background removal for PL uses an O(1) polynomial derived from a robust
  row-averaged energy profile within a k-window, subtracting only the smooth part.
• Joint fits use Levenberg–Marquardt ('lm') for parity with reflectivity; uncertainties
  computed from the residual Jacobian’s covariance approximation.
• The final 1×3 combination aligns panels in k′ and blends reflectivity into the PL grid
  to visualize both processes coherently across detunings.
• In the last cell we calculate δ_refl = a·E_min_PL² + b·E_min_PL + c.
  This is done so to extract the detunings using white light reflectivity for Fig_3_4_Fig_Suppl_4_5_6_7_maker.py 
  as those figures do not contain any reflectivity measurement, making the interpolation necessary.
  LPB is fitting the PL measurents (top 2x3 of cell 4) to accurately find E_min_PL.
"""


import os
import re
import sif_parser
import matplotlib.pyplot as plt
import cv2
import numpy as np
from lmfit import Model, Parameters
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

# === Parameters and helper functions ===
# Physical constants
h = 4.135667696e-15  # Planck's constant [eV·s]
c = 299792458        # Speed of light [m/s]

# Fixed exciton energy used by the analytical model
Ex_fixed = 2.404

# Wavelength (nm) → Energy (eV)
def nm_to_ev(wavelength_nm):
    """Convert wavelength in nm to photon energy in eV."""
    return h * c / (wavelength_nm * 1e-9)

# Replace outlier/hot pixels by the local mean (simple defect filtering)
def remove_defects(image, threshold_factor=1.3):
    """Mitigate detector defects by replacing high outliers with the local mean."""
    filtewhite_image = image.copy()
    kernel_size = 5
    mean_local = cv2.blur(image, (kernel_size, kernel_size))
    threshold = threshold_factor * mean_local
    mask_defects = image > threshold
    filtewhite_image[mask_defects] = mean_local[mask_defects]
    return filtewhite_image

# Subtracts an O(1) polynomial background along k using an averaged window around k≈0
def remove_polynomial_background_only_k(image_rotated, Y_ev, K_parallel,
                                        k_window, min_cols):
    """
    Remove a smooth k‑background by fitting (and subtracting) an O(1) polynomial
    to a 1D energy profile built row-wise by averaging columns whose
    k_parallel(E_i, x_j) falls inside the k-window for each energy row.

    Parameters
    ----------
    image_rotated : (Ny, Nx) ndarray
        Intensity image with rows = energy samples and columns = angular/pixel axis
        (same orientation as Y_ev and K_parallel).
    Y_ev : (Ny, Nx) ndarray
        2D energy mesh (eV). Typically built with `np.meshgrid(E_axis, x_values)`.
        Only the first column is used as the energy axis for fitting (E_axis = Y_ev[:, 0]).
    K_parallel : (Ny, Nx) ndarray
        2D k_parallel(E, x) in μm^−1 (same shape as image_rotated).
    k_window : tuple(float, float)
        (k_min, k_max) μm^−1 window used to select, for each energy row,
        which columns to average. Default in callers: (−1.0, 1.0).
    min_cols : int
        Minimum number of columns to average at each energy. If fewer columns
        lie inside the k‑window, a nearest‑to‑k=0 fallback is used to reach
        at least `min_cols` columns.

    Returns
    -------
    image_without_bg : (Ny, Nx) ndarray
        Image after subtracting the fitted O(1) background from every column.
    """
    # -- Helper models: O(1) polynomial + (optional) skewed Lorentzians (kept for robustness on profile)
    def O_1(x, c0, c1):
        return c0 + c1 * x

    def skewed_lorentzian(x, amp, center, gamma, alpha):
        return amp / (1.0 + ((x - center) / (gamma * (1.0 + alpha * np.sign(x - center))))**2)

    modelo = (
        Model(O_1, prefix='poly_')
        + Model(skewed_lorentzian, prefix='lorentz_')
        + Model(skewed_lorentzian, prefix='lorentz2_')
    )

    Ny, Nx = image_rotated.shape

    # Energy axis taken from the 2D mesh (use first column)
    E_axis = np.asarray(Y_ev[:, 0], dtype=float)  # (Ny,)
    k_lo, k_hi = k_window
    y_profile = np.empty(Ny, dtype=float)

    # -- Build the energy profile by row‑wise averaging inside the k‑window
    for i in range(Ny):
        k_row = K_parallel[i, :]  # (Nx,)
        mask = (k_row >= k_lo) & (k_row <= k_hi)
        if np.count_nonzero(mask) >= max(1, min_cols):
            # Simple mean over selected columns
            vals = image_rotated[i, mask]
            y_profile[i] = np.nanmean(vals) if vals.size else np.nan
        else:
            # Fallback: take the closest `min_cols` columns to k=0 for this row
            order = np.argsort(np.abs(k_row))[:max(1, min_cols)]
            vals = image_rotated[i, order]
            y_profile[i] = np.nanmean(vals) if vals.size else np.nan

    # Replace any remaining NaNs by local interpolation (robustness)
    if np.any(~np.isfinite(y_profile)):
        finite = np.isfinite(y_profile)
        if finite.any():
            y_profile[~finite] = np.interp(
                np.flatnonzero(~finite),
                np.flatnonzero(finite),
                y_profile[finite]
            )
        else:
            # Degenerate case: no finite data; return original image
            return image_rotated.copy()

    # -- Robust initial guesses from the averaged profile
    max_y = float(np.nanmax(y_profile)) if np.isfinite(np.nanmax(y_profile)) else 1.0
    i_max = int(np.nanargmax(y_profile)) if np.isfinite(np.nanmax(y_profile)) else Ny // 2
    E_max = float(E_axis[i_max])

    params = Parameters()
    # O(1) polynomial (background)
    params.add('poly_c0', value=0.01 * max_y)
    params.add('poly_c1', value=0.00 * max_y)
    # First Lorentzian (LPB‑like)
    params.add('lorentz_amp', value=max_y * 0.7, min=max_y * 0.2, max=max_y * 1.1)
    params.add('lorentz_center', value=E_max, min=E_max - 0.05, max=E_max + 0.05)
    params.add('lorentz_gamma', value=0.01, min=0.0, max=0.05)
    params.add('lorentz_alpha', value=0, vary=False)
    # Second Lorentzian (exciton‑like)
    params.add('lorentz2_amp', value=max_y * 0.25, min=0.0, max=max_y * 0.5)
    params.add('lorentz2_center', value=2.395, vary=False)
    params.add('lorentz2_gamma', value=0.04, min=0.0, max=0.05)
    params.add('lorentz2_alpha', value=0, vary=False)

    # -- Fit and extract the O(1) background
    try:
        result = modelo.fit(y_profile, params, x=E_axis, method='least_squares')
        bg = result.eval_components(x=E_axis)['poly_']  # (Ny,)
    except Exception as e:
        print(f"[WARN] k‑background fit failed for window {k_window}: {e}")
        bg = np.zeros_like(E_axis, dtype=float)

    # -- Subtract the same smooth background (as a function of E) from all columns
    image_without_bg = image_rotated - bg[:, np.newaxis]
    return image_without_bg
def joint_residuals(x, k_list, E_list, w_list):
    """
    Compute stacked residuals for joint fitting across 3 panels.

    Parameters
    ----------
    x : array-like, shape (8,)
        Model parameters: [A, Omega, off1, delta1, off2, delta2, off3, delta3]
    k_list : list of np.ndarray
        List of k-vectors for each panel (length 3).
    E_list : list of np.ndarray
        List of energy values for each panel (length 3).
    w_list : list of float
        Weight for each panel (length 3).

    Returns
    -------
    residuals : np.ndarray
        Stacked residuals vector (1D).
    """
    A, Omega = x[0], x[1]
    off = np.array([x[2], x[4], x[6]], dtype=float)
    delt = np.array([x[3], x[5], x[7]], dtype=float)
    res = []
    for j, (k, E, w) in enumerate(zip(k_list, E_list, w_list)):
        Em = LPB(k, off[j], Ex_fixed, A, delt[j], Omega)
        rj = (Em - E) * np.sqrt(max(w, 0.0))  # weighted residuals
        res.append(rj)
    return np.concatenate(res) if res else np.array([], dtype=float)

# Convert the x‑window (μm) to column indices in order to extract the spectral profile
def x_to_col(x_um):
    # inverse mapping: j = pixel1_bg + (x - pos1)/pos_per_pixel
    j = pixel1_bg + (x_um - pos1) / pos_per_pixel
    return int(np.clip(np.rint(j), 0, Nx - 1))


# Lower Polariton Branch (analytical expression)
def LPB(x, off, Ex, A, delta, omega):
    return (Ex + A * (x - off)**2 + delta + Ex - np.sqrt((A * (x - off)**2 + delta)**2 + omega**2)) * 0.5

# Upper Polariton Branch (analytical expression)
def UPB(x, off, Ex, A, delta, omega):
    return (Ex + A * (x - off)**2 + delta + Ex + np.sqrt((A * (x - off)**2 + delta)**2 + omega**2)) * 0.5

def photon_mode(k, off, delta, A):
    return Ex_fixed + delta + A * (k - off)**2

# === Measurement file configuration ===
# Path relative to this script → ../../Measurements/PL_reflec
current_path = os.path.dirname(os.path.abspath(__file__))
target_folder = os.path.abspath(os.path.join(current_path, "..", "Measurements", "PL_reflec"))

# SIF files: first three = PL, last three = white‑light reflectivity
filenames = [
    "polariton_hig_det_wavelength_535_pl.sif",
    "polariton_mid_det_wavelength_530_pl.sif",
    "polariton_low_det_wavelength_535_pl.sif",
    "polariton_hig_det_wavelength_535_wl.sif",
    "polariton_mid_det_wavelength_530_wl.sif",
    "polariton_low_det_wavelength_535_wl.sif"
]

# Build absolute paths
filepaths = [os.path.join(target_folder, name).replace("\\", "/") for name in filenames]

# Early warning if any file is missing
for filepath in filepaths:
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")

# Angular range (objective NA = 0.75). Measurements recorded as E vs sin(theta).
sin1, sin2 = -0.75, 0.75
pos1, pos2 = -((1024/2)*13.3)/50, ((1024/2*13.3))/50  # 1024 pixels; 13.3 µm per pixel on the CCD; 50× objective magnification

# Per-file pixel pairs used for angular calibration: we will show the content between these pixels (X-axis range)
pixel_pairs = [
    (100, 900),  # File 1
    (90, 920),   # File 2
    (70, 910),   # File 3
    (90, 920),   # File 4
    (100, 910),  # File 5
    (70, 920),   # File 6
]

# k-range (μm^−1) used to mask points for LPB fits in each panel
fit_ranges = [
    (-5, 5),     # File 1
    (-6.5, 6.5), # File 2
    (-8, 8),     # File 3
    (-6, 6),     # File 4
    (-6.5, 6.5), # File 5
    (-8, 8),     # File 6
]

# %% 2x3 raw figure + precomputation for later reuse
# Caches for reuse (avoid I/O and recomputation)
images_ev_list = []  # Energy-domain images (Jacobian applied), one per file
K_parallels = []     # k_parallel mesh (Ny x Nx), depends on energy → 2D
Y_evs = []           # Energy mesh (Ny x Nx)
k_limits = []        # Per-panel (k_min, k_max) for consistent display
energy_mins = []     # Per-panel energy min
energy_maxs = []     # Per-panel energy max

# Create 2x3 figure for PL and reflectivity
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

# Loop through files: load, calibrate axes, apply Jacobian, and plot
for idx, filepath in enumerate(filepaths):
    # --- Load SIF and basic preprocessing
    data, info = sif_parser.np_open(filepath)
    image = data.squeeze()
    image = remove_defects(image)        # Replace hot pixels by local mean
    image_rotated = np.rot90(image, k=-1)  # Rotate: rows = Energy, cols = angle/pixels

    # --- Angular calibration: compute sin(theta) per column
    pixel1, pixel2 = pixel_pairs[idx]    # Reference pixel positions
    pixels_total = image.shape[0]        # Total rows before rotation
    sin_per_pixel = (sin2 - sin1) / (pixel2 - pixel1)
    sin_min = sin1 - (pixel1 * sin_per_pixel)
    sin_max = sin_min + (pixels_total * sin_per_pixel)

    # --- Extract central wavelength from filename to find matching .asc
    match = re.search(r'wavelength_(\d+)_', os.path.basename(filepath))
    if not match:
        raise ValueError("Could not extract 'wavelength_central' from filename.")
    wavelength_central = int(match.group(1))

    # --- Load spectral calibration (.asc): pixel → nm
    current_path = os.path.dirname(os.path.abspath(__file__))
    calibration_folder = os.path.abspath(
        os.path.join(current_path, "..", "Measurements", "Calibration_wavelength")
    )
    asc_filename = f"{wavelength_central}_spectrum.asc"
    asc_path = os.path.join(calibration_folder, asc_filename).replace("\\", "/")
    if not os.path.exists(asc_path):
        raise FileNotFoundError(f"Calibration file not found: {asc_path}")
    with open(asc_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # First column in .asc is wavelength in nm (decimal separator may be a comma)
    nm_values = np.array([float(line.split()[0].replace(",", ".")) for line in lines])

    # Convert nm → eV (energy axis per row)
    ev_values = nm_to_ev(nm_values)

    # Cache per-measurement energy ranges
    energy_min = float(ev_values.min())
    energy_max = float(ev_values.max())
    energy_mins.append(energy_min)
    energy_maxs.append(energy_max)

    # --- Build centered sin(theta) axis per column
    x_edges = np.linspace(sin_min, sin_max, image_rotated.shape[1] + 1)
    x_values = (x_edges[:-1] + x_edges[1:]) / 2

    # --- 2D meshes: X_sin (sinθ), Y_ev (energy)
    # Y_ev varies along rows; X_sin varies along columns.
    X_sin, Y_ev = np.meshgrid(x_values, ev_values)

    # --- k_parallel(E, sinθ) in μm^−1: k = (2π / hc) * E * sinθ
    # Note: k depends on E → K_parallel is a 2D matrix
    K_parallel = (2 * np.pi * Y_ev / (h * c)) * X_sin * 1e-6

    # --- Apply Jacobian for nm→eV scaling
    # The λ→E change introduces a sign (decreasing λ increases E); keep positive weights via abs().
    # Jacobian for intensity reweighting from wavelength to energy domain
    jacobian = (np.abs(-((h * c) ** 2) * 1e15) / (2 * np.pi * (Y_ev ** 3)))
    image_ev = image_rotated * jacobian

    # --- Plot (simple 2x3 layout)
    ax = axes[idx]
    im = ax.pcolormesh(K_parallel, Y_ev, image_ev, shading='auto', cmap='viridis')

    # Determine displayed k-range for this panel based on reference columns
    k_min = K_parallel[0, pixel1]
    k_max = K_parallel[0, pixel2]
    k_limits.append((float(k_min), float(k_max)))
    ax.set_xlim(k_min, k_max)
    ax.set_xlabel(r'$k_{\parallel}$ $( \mu m ^{-1})$', fontsize=18)

    # Y-label only on the leftmost subplots
    if idx % 3 == 0:
        ax.set_ylabel('Energy (eV)', fontsize=18)
    else:
        ax.set_ylabel('')

    # Colorbar per subplot (kept consistent with original structure)
    if idx % 3 == 2:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Intensity', fontsize=16)
        cbar.ax.tick_params(labelsize=16)
    else:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=16)

    # Styling
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    ax.tick_params(axis='both', which='major', labelsize=16, width=2.5, length=8)

    # Cache for later reuse
    images_ev_list.append(image_ev.copy())   # Ny x Nx
    K_parallels.append(K_parallel.copy())    # Ny x Nx
    Y_evs.append(Y_ev.copy())                # Ny x Nx

# Shared energy window across all panels
global_energy_min = max(energy_mins)
global_energy_max = min(energy_maxs)
for ax in axes:
    ax.set_ylim(global_energy_min, global_energy_max)

# Titles & layout
fig.subplots_adjust(hspace=0.545, wspace=0.4)
fig.text(0.51, 0.91, "Emission at non-resonant driving", ha='center', fontsize=24)
fig.text(0.51, 0.445, "White light reflectivity", ha='center', fontsize=24)
plt.show()

# %% Extract background from an empty reflectivity measurement
# Folder: ../../Measurements/Calibration_white_light
target_folder = os.path.abspath(os.path.join(current_path, "..", "Measurements", "Calibration_white_light"))

# Reflectivity background file (used to extract the white‑light spectral profile)
filename = "reflectivity_background_wavelength_535_wl.sif"
filepath = os.path.join(target_folder, filename).replace("\\", "/")

# Extract central wavelength from filename
match = re.search(r'wavelength_(\d+)_', os.path.basename(filepath))
if match:
    wavelength_central = int(match.group(1))

# Matching energy calibration folder and file
calibration_folder = os.path.abspath(os.path.join(current_path, "..", "Measurements", "Calibration_wavelength"))
asc_filename = f"{wavelength_central}_spectrum.asc"
asc_path = os.path.join(calibration_folder, asc_filename).replace("\\", "/")

# Read .asc calibration (wavelength column)
if not os.path.exists(asc_path):
    raise FileNotFoundError(f"Calibration file not found: {asc_path}")
with open(asc_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
# First column is wavelength in nm; comma decimal separators may appear
nm_values = np.array([float(line.split()[0].replace(",", ".")) for line in lines])

# Load background SIF and pre-process
data, info = sif_parser.np_open(filepath)
image = data.squeeze()

# Remove hot pixels/defects and rotate to (Energy rows, Pixel columns)
image_filtered = remove_defects(image)
image_rotated = np.rot90(image_filtered, k=-1)

# Convert wavelength (nm) to energy (eV)
ev_values = nm_to_ev(nm_values)

# 2D mesh of pixel centers (X_pix) vs energy (Y_ev)
X_pix, Y_ev = np.meshgrid(x_values, ev_values)

# Apply Jacobian for λ → E transformation to preserve spectral density
# The negative sign from inversion is removed by abs()
jacobian = (np.abs(-(h*c)*1e9) / (Y_ev**2))
image_final = image_rotated * jacobian

# === FIGURE 1 === Energy vs x (μm) in the reflectivity background measurement
fig, ax = plt.subplots(figsize=(10, 6))

# Use the pixel‑pair calibration corresponding to the 535 nm reflectivity file
# (change index 3 if you prefer to reference a different reflectivity dataset)
pixel1_bg, pixel2_bg = pixel_pairs[3]  # e.g., (90, 920) for wl@535 nm
Nx = image_rotated.shape[1]

# Linear pixel → position mapping enforcing pixel1→pos1 and pixel2→pos2
pos_per_pixel = (pos2 - pos1) / (pixel2_bg - pixel1_bg)  # μm per CCD pixel (in the sample plane)
j_cols = np.arange(Nx)  # column indices [0..Nx-1]
x_values_um = pos1 + (j_cols - pixel1_bg) * pos_per_pixel  # position (μm) for each image column

# 2D grids: X_pos (μm) vs energy (eV)
X_pos, Y_ev = np.meshgrid(x_values_um, ev_values)

# Render the image as (x, E)
im = ax.pcolormesh(X_pos, Y_ev, image_final, shading='auto', cmap='viridis')
plt.colorbar(im, label='Intensity')

# x-axis limits in μm (between pos1 and pos2, as requested)
ax.set_xlim(pos1, pos2)

# Labels / styling
ax.set_ylabel('Energy (eV)', fontsize=14)
plt.xlabel(r'x (μm)', fontsize=18)  # <- exactly as requested
ax.set_title('Energy vs position', fontsize=16)
ax.tick_params(axis='both', labelsize=12)

# ------- Averaging window in x: center −75 μm with half‑width 30 μm -------
x_center_um = -75.0
half_width_um = 30.0
x_start_um = x_center_um - half_width_um
x_end_um = x_center_um + half_width_um

col_center = x_to_col(x_center_um)
col_start = x_to_col(x_start_um)
col_end = x_to_col(x_end_um)

# Safety: swap if round‑off makes the order inverted
if col_end < col_start:
    col_start, col_end = col_end, col_start

# Visual guides for the averaging window (drawn in x‑units, not pixels)
ax.axvline(x_start_um, color='white', linestyle='--', linewidth=1.2)
ax.axvline(x_end_um, color='white', linestyle='--', linewidth=1.2)
ax.axvline(x_center_um, color='white', linestyle=':', linewidth=1.2)
plt.show()

# --- Extract the averaged spectral profile over the window [col_start : col_end] ---
# 'image_final' has shape (Ny_energy, Nx_columns); columns were mapped to x (μm) above.
window = image_final[:, col_start:col_end + 1]

# Choose averaging mode: simple mean (default) or Gaussian‑weighted mean
use_weighted = False  # set to True to enable Gaussian‑weighted averaging
if not use_weighted:
    # Simple arithmetic mean across selected columns
    line_profile_avg = np.mean(window, axis=1)
    avg_label = "Mean profile"
else:
    # Gaussian‑weighted mean giving more weight to the central column
    cols = np.arange(col_start, col_end + 1)
    # Standard deviation as a fraction of the half window; adjust the factor if needed
    sigma_cols = 0.4 * max(1, (col_end - col_start) / 2)
    # Convert the target center (x_center_um) to a fractional column index for the weights
    j_center = pixel1_bg + (x_center_um - pos1) / pos_per_pixel
    weights = np.exp(-0.5 * ((cols - j_center) / sigma_cols) ** 2)
    weights = weights / weights.sum()
    # Weighted sum along columns (broadcast weights)
    line_profile_avg = (window * weights).sum(axis=1)
    avg_label = f"Gaussian-weighted profile (x={x_center_um}±{half_width_um} μm)"

# --- Smoothing and interpolation of the averaged spectral profile ---
# Apply a mild 1D Gaussian smoothing in the energy-index domain
sigma_smooth = 4  # adjust if needed; 0 disables smoothing
line_profile_smooth = gaussian_filter1d(line_profile_avg, sigma=sigma_smooth) if sigma_smooth > 0 else line_profile_avg

# Build a cubic interpolant of the smoothed profile vs energy (E is strictly monotonic)
interp_func = interp1d(ev_values, line_profile_smooth, kind='cubic')

# Create a denser energy grid to draw a smooth curve (min 600 points for clarity)
n_dense = max(len(image_final), 600)
energies_interp = np.linspace(ev_values.min(), ev_values.max(), n_dense)
line_profile_interp = interp_func(energies_interp)

# === FIGURE 2 === Averaged spectral profile (original vs cubic interpolation)
fig_profile, ax_profile = plt.subplots(figsize=(10, 6))
ax_profile.plot(line_profile_avg, ev_values, 'o', ms=3, label=avg_label)
ax_profile.plot(line_profile_interp, energies_interp, '-r', lw=1.8, label='Interpolation')
ax_profile.set_ylabel('Energy (eV)', fontsize=14)
ax_profile.set_xlabel('Intensity (a. u.)', fontsize=14)
ax_profile.set_title(f'Averaged profile around x = {x_center_um} ± {half_width_um} μm', fontsize=16)
ax_profile.tick_params(axis='both', labelsize=12)
ax_profile.grid(True, alpha=0.3)
ax_profile.legend()
plt.show()

# %% Correct reflectivity with background profile and remove k‑background from PL
# Outputs kept for next steps (fits/overlays/export)
corrected_images = []   # Processed: PL (k‑background removed) / Reflectivity (WL‑corrected)
display_images = []     # Normalized for visualization
extrema_per_panel = []  # Per‑panel dictionaries with extrema and ready‑to‑fit arrays
k_lines = []            # k per column (1D)
x_data_per_panel = []   # k‑axis used to evaluate analytical curves (length = Nx)
profile_vectors = []    # Spectral profile used in bottom row (None for top row)
joint_fit = {}          # Fitted parameters per panel

# --------- Figure 2x3 (processed images with extrema overlays) ---------
fig_corr, axes_corr = plt.subplots(2, 3, figsize=(20, 12))
axes_corr = axes_corr.flatten()

for idx, (img_ev, K_parallel, Y_ev) in enumerate(zip(images_ev_list, K_parallels, Y_evs)):
    ax = axes_corr[idx]

    # Common axes
    k_min, k_max = k_limits[idx]
    ax.set_xlim(k_min, k_max)
    ax.set_ylim(global_energy_min, global_energy_max)
    ax.set_xlabel(r'$k_{\parallel}$ $( \mu m ^{-1})$', fontsize=18)
    if idx % 3 == 0:
        ax.set_ylabel('Energy (eV)', fontsize=18)

    # Ticks & spines styling
    ax.tick_params(axis='both', which='major', labelsize=16, width=2.5, length=8)
    for sp in ax.spines.values():
        sp.set_linewidth(2.5)

    # Helpers for later curves
    k_line = K_parallel[-1, :]
    k_lines.append(k_line)
    x_data = np.linspace(float(k_line[0]), float(k_line[-1]), K_parallel.shape[1])
    x_data_per_panel.append(x_data)

    # ================= TOP ROW (PL): background removal + maxima =================
    if idx < 3:
        # Remove smooth k‑background while preserving spectral structure
        image_proc = remove_polynomial_background_only_k(
            img_ev,
            Y_ev,
            K_parallel,
            k_window=(-1.0, 1.0),
            min_cols=5
        )

        # Normalize for display
        den = np.nanmax(image_proc)
        image_disp = (image_proc / den) if (np.isfinite(den) and den > 0) else image_proc

        # Column‑wise extrema: maxima for PL
        row_idx = np.argmax(image_proc, axis=0)
        E_cols = Y_ev[:, 0][row_idx]
        k_cols = K_parallel[row_idx, np.arange(K_parallel.shape[1])]

        # Fit mask within predefined k‑window (some datasets lack robust extrema outside these ranges)
        fit_min, fit_max = fit_ranges[idx]
        mask_fit = (k_cols >= fit_min) & (k_cols <= fit_max)
        k_fit = k_cols[mask_fit]
        E_fit = E_cols[mask_fit]

        # Plot image + red maxima
        im = ax.pcolormesh(K_parallel, Y_ev, image_disp, shading='auto', cmap='viridis')
        ax.plot(k_cols[mask_fit], E_cols[mask_fit], '-', color='red', markersize=5, label='PL maxima')

        # Store
        corrected_images.append(image_proc)
        display_images.append(image_disp)
        profile_vectors.append(None)
        extrema_per_panel.append({
            'type': 'PL',
            'row_idx': row_idx,
            'E_cols': E_cols,
            'k_cols': k_cols,
            'mask_fit': mask_fit,
            'k_fit': k_fit,
            'E_fit': E_fit,
        })

        if idx == 2:
            cbar = fig_corr.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label('Normalized intensity (a. u.)', fontsize=16)

    # ============== BOTTOM ROW (Reflectivity): WL‑profile correction + minima ==============
    else:
        # Build the spectral profile vector aligned to this panel's energy axis (E‑axis = Y_ev[:,0])
        # Interpolate the measured profile (energies_interp, line_profile_interp) onto the panel E‑axis
        E_axis = Y_ev[:, 0]
        prof_vec = np.interp(E_axis, energies_interp, line_profile_interp)  # (Ny,)

        # Ensure strictly positive values before division
        prof_vec = np.array(prof_vec, dtype=float)
        if not np.any(prof_vec > 0):
            raise ValueError("White‑light spectral profile has no positive values.")
        small = np.nanmin(prof_vec[prof_vec > 0])
        prof_vec[prof_vec <= 0] = small
        profile_matrix = prof_vec[:, np.newaxis]  # broadcast along columns

        # Correct and normalize for display
        with np.errstate(divide='ignore', invalid='ignore'):
            image_corr = np.where(profile_matrix > 0, img_ev / profile_matrix, 0.0)
        den = np.nanmax(image_corr)
        image_disp = (image_corr / den) if (np.isfinite(den) and den > 0) else image_corr

        # Column‑wise extrema: minima for reflectivity
        row_idx = np.argmin(image_corr, axis=0)
        E_cols = Y_ev[:, 0][row_idx]
        k_cols = K_parallel[row_idx, np.arange(K_parallel.shape[1])]

        # Fit mask within predefined k‑window (consistent with PL)
        fit_min, fit_max = fit_ranges[idx]
        mask_fit = (k_cols >= fit_min) & (k_cols <= fit_max)
        k_fit = k_cols[mask_fit]
        E_fit = E_cols[mask_fit]

        # Plot image + red minima
        im = ax.pcolormesh(K_parallel, Y_ev, image_disp, shading='auto', cmap='viridis')
        cbar.ax.tick_params(labelsize=16)
        ax.plot(k_cols[mask_fit], E_cols[mask_fit], '-', color='red', markersize=5, label='Reflect minima')

        # Store
        corrected_images.append(image_corr)
        display_images.append(image_disp)
        profile_vectors.append(prof_vec)
        extrema_per_panel.append({
            'type': 'Reflect',
            'row_idx': row_idx,
            'E_cols': E_cols,
            'k_cols': k_cols,
            'mask_fit': mask_fit,
            'k_fit': k_fit,
            'E_fit': E_fit,
        })

        if idx == 5:
            cbar = fig_corr.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label('Normalized intensity (a. u.)', fontsize=16)

# Titles & layout
fig_corr.subplots_adjust(hspace=0.575, wspace=0.3)
fig_corr.text(0.51, 0.93, "Emission at non-resonant driving", ha='center', fontsize=24)
fig_corr.text(0.51, 0.465, "White light reflectivity", ha='center', fontsize=24)

# --- Joint fitting with SciPy least_squares ---
# This version performs a global fit across triplets (PL or Reflect) with NO bounds.
# You provide initial guesses for A, omega, off (3 panels), and delta (3 panels).
# The output dictionary 'joint_fit' keeps the same schema used by later steps.

# ===== 1) User initial guesses =====
# You can pass scalars or 3‑element sequences for 'off' and 'delta'.
init_guess = {
    "A": 0.004,               # eV·μm^2 (photon-mode curvature)
    "omega": 0.200,           # eV (Rabi splitting)
    "off": [-0.11, 0.13, 0.058],   # μm^−1 (k offset per panel)
    "delta": [0.100, 0.005, -0.070],  # eV (detuning per panel)
}

# ===== 2) Build the initial vector x0 in the expected packing =====
# Packing order must match downstream code: x = [A, Omega, off1, delta1, off2, delta2, off3, delta3]
off0 = init_guess.get("off")
delta0 = init_guess.get("delta")
A0 = init_guess["A"]
Omega0 = init_guess["omega"]
x0 = np.array([A0, Omega0, off0[0], delta0[0], off0[1], delta0[1], off0[2], delta0[2]], dtype=float)

# ===== 3) Define residuals (same physics as before; no bounds now) =====


# Choose which triplets to fit jointly. PL and reflectivity are fitted independently (different processes).
# The detunings are those coming from the fit of the reflectivity.
# The fit is only done in the PL to find out where the minimun energy (E_min) for PL is for each detuning in the reflecitity.
# In the last cell of this code, a O2 polynomial is built by interpolation to get the detuning
# calculated using reflectivity given a E_min of the PL. This will be used in code Fig_3_4_Fig_Suppl_4_5_6_7_maker.py to find
# the correct detuning coming from PL measurements.
triplets = {
    "PL": [0, 1, 2],
    "Reflect": [3, 4, 5],
}

# Optional: per‑curve weights to balance datasets with very different number of points
weights_default = [1.0, 1.0, 1.0]

# ===== 4) Run the joint fit per triplet (PL and Reflect) with method='lm' (no bounds) =====
for label, idxs in triplets.items():
    # -- Collect the k/E points and weights for the 3 panels --
    k_list, E_list, w_list = [], [], []
    for ii in idxs:
        k_i = np.asarray(extrema_per_panel[ii]['k_fit'], dtype=float)
        E_i = np.asarray(extrema_per_panel[ii]['E_fit'], dtype=float)
        m = np.isfinite(k_i) & np.isfinite(E_i)
        k_list.append(k_i[m]); E_list.append(E_i[m])
        # Use weights_default[j] where j=0,1,2 (protect against length mismatches)
        w_list.append(float(weights_default[min(len(weights_default)-1, len(w_list))]))

    # -- Solve unconstrained nonlinear least‑squares --
    # method='lm' (Levenberg–Marquardt).
    ls = least_squares(joint_residuals, x0=x0, args=(k_list, E_list, w_list), method='lm')
    x_hat = ls.x
    A_hat, Omega_hat = x_hat[0], x_hat[1]
    off_hat = [x_hat[2], x_hat[4], x_hat[6]]
    delt_hat = [x_hat[3], x_hat[5], x_hat[7]]

    # --- Uncertainties from Jacobian (approximate covariance) ---
    J = ls.jac  # Jacobian of residuals
    m = ls.fun.size  # number of residuals
    n = x_hat.size   # number of parameters
    JTJ = J.T @ J
    JTJ_inv = np.linalg.inv(JTJ)
    dof = max(m - n, 1)  # degrees of freedom
    s_sq = 2 * ls.cost / dof
    cov = JTJ_inv * s_sq
    perr = np.sqrt(np.clip(np.diag(cov), 0, np.inf))  # standard errors

    A_err_hat = float(perr[0])
    Omega_err_hat = float(perr[1])
    off_err_hat = [float(perr[2]),
                   float(perr[4]),
                   float(perr[6])]
    delt_err_hat = [float(perr[3]),
                    float(perr[5]),
                    float(perr[7])]

    # -- Persist per‑panel results with uncertainties --
    for j_local, i_panel in enumerate(idxs):
        joint_fit[i_panel] = {
            "A": float(A_hat),
            "omega": float(Omega_hat),
            "off": float(off_hat[j_local]),
            "delta": float(delt_hat[j_local]),
            "Ex": float(Ex_fixed),
            # Errors:
            "A_err": float(A_err_hat),
            "omega_err": float(Omega_err_hat),
            "off_err": float(off_err_hat[j_local]),
            "delta_err": float(delt_err_hat[j_local]),
        }

    # --- Overlays: plot analytical branches on top of images ---
    for j_local, i_panel in enumerate(idxs):
        ax = axes_corr[i_panel]
        k_plot = np.asarray(x_data_per_panel[i_panel], dtype=float)
        off_j, del_j = off_hat[j_local], delt_hat[j_local]
        # Compute LPB for both rows (PL and Reflectivity)
        lpb = LPB(k_plot, off_j, Ex_fixed, A_hat, del_j, Omega_hat)
        if label == "Reflect":  # Bottom row: keep full overlay
            upb = UPB(k_plot, off_j, Ex_fixed, A_hat, del_j, Omega_hat)
            photon = Ex_fixed + del_j + A_hat * (k_plot - off_j)**2
            step = 50
            ax.plot(k_plot[::step], lpb[::step], '.', color='cyan', markersize=5, label='LPB')
            ax.plot(k_plot[::step], upb[::step], '.', color='white', markersize=5, label='UPB')
            ax.plot(k_plot, photon, '--', color='white', linewidth=2.0, label='Photonic mode')
            ax.axhline(Ex_fixed, linestyle='--', color='cyan', linewidth=2.0, label=r'$E_x$')
            delta_err_j = delt_err_hat[j_local]
            # Title with detuning and uncertainty
            ax.set_title(
                rf"$\delta = {del_j*1e3:.0f}\pm{(delta_err_j*1e3 if np.isfinite(delta_err_j) else np.nan):.0f}\,\mathrm{{meV}}$",
                fontsize=16, pad=8
            )
        else:  # Top row (PL): minimal overlay
            step = 50
            ax.plot(k_plot[::step], lpb[::step], '.', color='cyan', markersize=5, label='LPB')

# -- Console output with ± errors for the reflectivity fits --
print("Reflectivity joint fit result:")
print(f" A = {A_hat*1e3:.6f} ± {A_err_hat*1e3:.6f} meV·μm^2")
print(f" Omega = {Omega_hat*1e3:.1f} ± {Omega_err_hat*1e3:.1f} meV")
print(f" delta (meV) = {[round(d*1e3,1) for d in delt_hat]} ± {[round(e*1e3,1) for e in delt_err_hat]}")
print(f" off (μm^-1) = {[round(o,3) for o in off_hat]} ± {[round(e,3) for e in off_err_hat]}")

# %% FINAL COMBINATION.
# Goal: for each detuning, find the k position of the LPB minimum and redefine it as the new k'=0.
fig_mix, axes_mix = plt.subplots(1, 3, figsize=(30, 10))
axes_mix = np.atleast_1d(axes_mix)
im_for_cbar = None

# Keep same aspect behavior
for ax in axes_mix:
    ax.set_aspect('auto', adjustable='box')

blend_cols = 1  # subtle blend at the seam

for idx in range(3):
    ax = axes_mix[idx]

    # --- Inputs (right = PL idx, left = Reflect idx+3) ---
    image_pl = corrected_images[idx]
    Y_pl = Y_evs[idx]
    K_pl = K_parallels[idx]

    image_ref = corrected_images[idx + 3]
    Y_ref = Y_evs[idx + 3]
    K_ref = K_parallels[idx + 3]

    Ny, Nx = image_pl.shape

    # Get the offsets from the LPB fit (separately for PL and reflectivity)
    k0_pl  = float(joint_fit[idx]['off'])     # Offset for the PL
    k0_ref = float(joint_fit[idx+3]['off'])   # Offset for the reflectivity

    # Shift both k-grids to k' = k − k0 (each with its own k0)
    Kp_pl  = K_pl  - k0_pl
    Kp_ref = K_ref - k0_ref

    # Nearest column to k' = 0 on each grid
    kline_pl = np.asarray(Kp_pl[-1, :], dtype=float)
    kline_ref = np.asarray(Kp_ref[-1, :], dtype=float)

    # Find the column index relative to those k values
    s_pl = int(np.nanargmin(np.abs(kline_pl - 0.0)))
    s_ref = int(np.nanargmin(np.abs(kline_ref - 0.0)))

    # Visual‑only normalization per side
    left_slice  = image_ref[:, :max(0, s_ref)]
    right_slice = image_pl[:, min(Nx, s_pl):]

    # Normalize sides independently
    left_max  = np.nanmax(left_slice)
    right_max = np.nanmax(right_slice)
    left_norm  = image_ref / left_max
    right_norm = image_pl / right_max

    # Our target grid is the PL grid. Adapt the reflectivity grid to the PL grid (left side).
    Kp_target = Kp_pl
    Y_target  = Y_pl

    # Prepare the left side: remap reflectivity left half into the PL k‑grid
    Ny_dst, Nx_dst = Kp_target[:, :s_pl].shape  # shape of the left side according to PL
    left_on_pl_kp = np.zeros((Ny, s_pl), dtype=float)  # container for remapped reflectivity
    kdst = Kp_target[-1, :s_pl]                    # target k grid (left of cut) for PL
    ksrc = Kp_ref[-1, :max(1, s_ref)]              # source k grid (left of cut) for reflectivity

    # Row-wise linear interpolation of the reflectivity left side onto PL k-grid
    for r in range(Ny):
        f = interp1d(ksrc, left_norm[r, :max(1, s_ref)],
                     kind='linear', bounds_error=False, fill_value='extrapolate')
        left_on_pl_kp[r, :] = f(kdst)

    right_on_pl_kp = right_norm[:, s_pl:]

    # Assemble + small blend at the seam
    combined = np.zeros_like(image_pl, dtype=float)
    combined[:, :s_pl] = left_on_pl_kp
    combined[:, s_pl:] = right_on_pl_kp

    if blend_cols > 0 and 0 < s_pl < Nx:
        for b in range(blend_cols):
            jL = s_pl - 1 - b
            jR = s_pl + b
            if jL >= 0:
                w = (b + 1) / (blend_cols + 1)
                col_R = right_norm[:, min(Nx-1, s_pl + b)]
                combined[:, jL] = (1 - w) * combined[:, jL] + w * col_R
            if jR < Nx:
                w = (b + 1) / (blend_cols + 1)
                col_L = left_on_pl_kp[:, max(0, s_pl - 1 - b)] if left_on_pl_kp.shape[1] > 0 else 0.0
                combined[:, jR] = (1 - w) * combined[:, jR] + w * col_L

    # --- Overlays (use reflectivity fit for analytical curves, as in original) ---
    fit_refl = joint_fit[idx + 3] if (isinstance(joint_fit, dict) and (idx + 3) in joint_fit) \
        else (joint_fit[str(idx + 3)] if (isinstance(joint_fit, dict) and str(idx + 3) in joint_fit) else {})
    Ex_fit    = float(fit_refl.get('Ex', Ex_fixed))
    off_fit   = float(fit_refl.get('off', 0.0))
    off_err   = float(fit_refl.get('off_err', 0.0))
    A_fit     = float(fit_refl.get('A', 0.0))
    A_err     = float(fit_refl.get('A_err', 0.0))
    delta_fit = float(fit_refl.get('delta', 0.0))
    delta_err = float(fit_refl.get('delta_err', 0.0))
    omega_fit = float(fit_refl.get('omega', 0.0))
    omega_err = float(fit_refl.get('omega_err', 0.0))

    # 1D k‑grid used for curves (use reflectivity panel x‑data, like the original)
    x_data_k  = x_data_per_panel[idx + 3]
    x_data_kp = x_data_k - off_fit  # for plotting against k' = k − off

    lpb = LPB(x_data_k,  off_fit, Ex_fit, A_fit, delta_fit, omega_fit)
    upb = UPB(x_data_k,  off_fit, Ex_fit, A_fit, delta_fit, omega_fit)
    photonic = A_fit * (x_data_k - off_fit)**2 + delta_fit + Ex_fixed

    # --- DRAW (same style) ---
    im = ax.pcolormesh(Kp_target, Y_target, combined, shading='auto', cmap='viridis')
    im_for_cbar = im
    step = 50
    ax.plot(x_data_kp[::step], lpb[::step], '.', color='cyan',  markersize=10, label='LPB')
    ax.plot(x_data_kp[::step], upb[::step], '.', color='white', markersize=10, label='UPB')
    ax.plot(x_data_kp,       photonic, '--', color='white', linewidth=4.0, label='Photon mode')
    ax.axhline(Ex_fixed, linestyle='--', color='cyan', linewidth=4.0, label=r'$E_x$')
    ax.set_title(rf"$\delta = {delta_fit*1000:.0f} \pm {delta_err*1000:.0f}\,\mathrm{{meV}}$", fontsize=40, pad=16)

    # Same limits (by pixel columns) but in k'
    # --- Compute new x‑limits using old pixel cuts, adjusted for offsets ---
    # Pixel cuts from previous images
    pixel1_pl,  pixel2_pl  = pixel_pairs[idx]     # PL panel
    pixel1_refl, pixel2_refl = pixel_pairs[idx+3] # Reflectivity panel

    # Offsets from the fits (or fallback to 0 if missing)
    off_refl = float(joint_fit[idx+3].get('off', 0.0)) if (idx+3) in joint_fit else 0.0
    off_pl   = float(joint_fit[idx].get('off', 0.0))   if idx in joint_fit   else 0.0

    # Convert pixel cuts to k' coordinates
    k_left  = K_parallels[idx+3][0, pixel1_refl] - off_refl  # left bound (reflectivity side)
    k_right = K_parallels[idx][0, pixel2_pl]     - off_pl    # right bound (PL side)

    # Apply limits
    ax.set_xlim(min(k_left, k_right), max(k_left, k_right))
    ax.set_ylim(global_energy_min, global_energy_max)

    # Keep the same label text to match "exactly the same" output
    ax.set_xlabel(r"$k_{\parallel}$ $( \mu m ^{-1})$", fontsize=40)
    if idx == 0:
        ax.set_ylabel("Energy (eV)", fontsize=40)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis='y', which='both', labelleft=(idx == 0))
    ax.tick_params(axis='both', which='major', labelsize=40)

    ax.text(0.10, 0.05, "Refl.", transform=ax.transAxes,
            fontsize=35, color='white', ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    ax.text(0.90, 0.05, "PL", transform=ax.transAxes,
            fontsize=35, color='white', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

# --- Global colorbar: identical placement to the first 1x3 ---
cbar_ax = fig_mix.add_axes([1, 0.15, 0.02, 0.75])
cbar = fig_mix.colorbar(im_for_cbar, cax=cbar_ax)
cbar.set_label('Normalized intensity (a. u.)', fontsize=40)
cbar.ax.tick_params(labelsize=40)


# Añadir etiquetas (a), (b), (c) en la esquina superior izquierda de cada subplot
labels = ['(a)', '(b)', '(c)']
for idx, ax in enumerate(axes_mix):
    ax.text(0.0, 1.1, labels[idx], transform=ax.transAxes,
            ha='left', va='top', fontsize=45, fontweight='bold')


# --- Same spacing between subplots as in the first 1x3 ---
fig_mix.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.12, wspace=0.025)


# --- Rasterize just images (pcolormesh) ---
for ax in axes_mix:
    for artist in ax.collections:
        artist.set_rasterized(True) # The rest is not renderized!

# Create the foler "Figures" if it doesn't exist
figures_folder = os.path.join(current_path, "Figures")
os.makedirs(figures_folder, exist_ok=True)
# Whole path to save the figure
save_path = os.path.join(figures_folder, "Fig.2.pdf")
# Actually save it
fig_mix.savefig(save_path, format='pdf', bbox_inches='tight', transparent=False, metadata=None, dpi=300)


plt.show()

# %% Reflectivity detuning vs PL LPB minimum energy using LPB() at k=off

# A O2 polynomial is built by interpolation to get the detuning calculated using 
# reflectivity given a E_min of the PL. This will be used in code Fig_3_4_Fig_Suppl_4_5_6_7_maker.py
# to find the correct detuning coming from PL measurements.

# ---- Indices: PL (top row) and Reflectivity (bottom row) ----
idxs_pl = [0, 1, 2]
idxs_refl = [3, 4, 5]

# ---- Collect PL fit parameters (to compute Emin via LPB(off,...)) ----
off_pl, Ex_pl, A_pl, d_pl, Om_pl = [], [], [], [], []
for i in idxs_pl:
    fit = joint_fit[i]  # assumes integer keys
    off_pl.append(float(fit.get("off", np.nan)))
    Ex_pl.append(float(fit.get("Ex", Ex_fixed)))
    A_pl.append(float(fit.get("A", np.nan)))
    d_pl.append(float(fit.get("delta", np.nan)))
    Om_pl.append(float(fit.get("omega", np.nan)))
off_pl = np.array(off_pl, dtype=float)
Ex_pl = np.array(Ex_pl, dtype=float)
A_pl  = np.array(A_pl,  dtype=float)
d_pl  = np.array(d_pl,  dtype=float)
Om_pl = np.array(Om_pl, dtype=float)

# ---- Collect Reflectivity detunings (Y values) ----
d_refl = []
for i in idxs_refl:
    fit = joint_fit[i]
    d_refl.append(float(fit.get("delta", np.nan)))
d_refl = np.array(d_refl, dtype=float)

# ---- Safety: need 3 finite PL and 3 finite Reflect entries ----
mask_pl = np.isfinite(off_pl) & np.isfinite(Ex_pl) & np.isfinite(A_pl) & np.isfinite(d_pl) & np.isfinite(Om_pl)
mask_re = np.isfinite(d_refl)
if mask_pl.sum() < 3 or mask_re.sum() < 3:
    raise RuntimeError("Not enough valid points (need 3 PL and 3 Reflect).")

# ---- Compute Emin_PL using LPB at k=off for each PL dataset ----
# Emin_i = LPB(off_i, off_i, Ex_i, A_i, delta_i, omega_i)
Emin_PL_eV = np.array([
    LPB(off_pl[i], off_pl[i], Ex_pl[i], A_pl[i], d_pl[i], Om_pl[i])
    for i in range(len(idxs_pl))
], dtype=float)

# ---- Axes: X = Emin from PL (eV), Y = detuning from Reflectivity (meV) ----
delta_reflect_meV = 1e3 * d_refl

# ---- Final finiteness check on plotted pairs ----
mask_all = np.isfinite(Emin_PL_eV) & np.isfinite(delta_reflect_meV)
if mask_all.sum() < 3:
    raise RuntimeError("Not enough finite (Emin_PL, δ_reflect) pairs to plot.")

# ---- Polynomial interpolation: δ_reflect = f(Emin_PL) ----
# With 3 points, a quadratic (deg=2) passes exactly through them.
coeffs = np.polyfit(Emin_PL_eV[mask_all], delta_reflect_meV[mask_all], deg=2)
poly = np.poly1d(coeffs)

# ---- Smooth curve for plotting ----
x_plot = np.linspace(Emin_PL_eV[mask_all].min(), Emin_PL_eV[mask_all].max(), 200)
y_plot = poly(x_plot)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(Emin_PL_eV, delta_reflect_meV, 'o', color='tab:blue',
        label='(PL $E_{\\min}$ , reflectivity δ)')
ax.plot(x_plot, y_plot, '-', color='tab:orange', label='Quadratic interpolation')
ax.set_xlabel('LPB $E_{\\min}$ from PL (eV)', fontsize=12)
ax.set_ylabel('Reflectivity $\\delta$ (meV)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# ---- Optional: print polynomial ----
print(" ")
print("Quadratic fit δ_reflect(meV) = a·Emin_PL^2 + b·Emin_PL + c (Emin_PL in eV)")
print(f"a = {coeffs[0]:.6e} b = {coeffs[1]:.6e} c = {coeffs[2]:.6e}")