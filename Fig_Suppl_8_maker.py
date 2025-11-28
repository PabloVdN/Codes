"""
Generates Supplementary Figure 8: processing pipeline for exciton-polarized PL,
Stokes-like maps (S1, S2, S3), and integrated spectra under different pump polarizations.

Overview
--------
This script loads .sif frames for multiple analyzer bases, removes detector defects,
calibrates the spectral axis from wavelength to energy, applies nm→eV Jacobian scaling,
subtracts smooth backgrounds column-wise, computes Stokes-like parameters with intensity masking,
renders processed images and Stokes maps, and integrates over a selected x-range to produce
energy-resolved spectra for σ⁺ and σ⁻ components under R and L pumping.

End-to-end pipeline
-------------------
1) Load each .sif file once, rotate to (rows = energy, columns = x/pixels), and remove hot pixels.
2) Calibrate the spectral axis using the matching .asc file (nm → eV).
3) Compute horizontal mapping (pixels → real-space positions in μm).
4) Apply nm→eV Jacobian to preserve intensity density under variable change.
5) Remove smooth background column-wise (fit: 1st-order polynomial + skewed Lorentzian).
6) Compute Stokes-like parameters S1, S2, S3 = (A − B)/(A + B) with intensity masking.
7) Plot processed intensity maps and Stokes maps for each pump/analyzer configuration.
8) Integrate intensity over a chosen x-range to obtain energy-resolved spectra for σ⁺ and σ⁻.
9) Render final figure with two panels (incoming R and L), overlay curves, add labels, and save PNG.

Cells guide
-----------
• Cell 1 - Header & helpers:
    Physical constants (h, c), nm→eV conversion, defect removal, polynomial background subtraction,
    Stokes calculator with masking, plotting utilities for raw/processed images and Stokes maps.

• Cell 2 - Data loading and calibration:
    – Read .asc calibration file, parse nm values, convert to eV.
    – Load all analyzer channels (H/V, D/A, σ⁺/σ⁻) for pump polarizations H, σ⁺, σ⁻.
    – Rotate images to (energy rows, x columns) and apply defect filtering.
    – Compute pixel→position mapping (μm) using CCD pixel size and objective magnification.
    – Build energy axis and apply nm→eV Jacobian to all intensity images.
    – For each analyzer pair, subtract smooth background column-wise (polynomial + Lorentzian).
    – Compute S1, S2, S3 maps with intensity masking; plot maps with fixed color range [-1, 1].

• Cell 3 - Integration and final figure:
    – Define x-range (μm), convert to pixel indices, integrate intensity over columns.
    – Build energy array and plot integrated spectra for σ⁺ and σ⁻  eission under σ⁺ and σ⁻ pumping.

Key notes
---------
• nm→eV Jacobian ensures correct intensity scaling when changing variables.
• Background removal subtracts only the polynomial component to preserve spectral features.
• Stokes maps use intensity masks to suppress low-SNR regions; NaN/Inf values are set to zero.
• Integration window is adjustable; spectra are normalized by column summation.
"""


import os
import re
import sif_parser
import inspect
import matplotlib.pyplot as plt
import numpy as np
import cv2
from lmfit import Model, Parameters

# Constants to convert nm -> eV
h = 4.135667696e-15  # Planck constant in eV·s
c = 299792458        # Speed of light in m/s

threshold_percent = 15  # Masking: if (A - B) is below this % of max(A + B), set 0 in S_calculator_and_mask

# Horizontal axis window (in pixels and converted spatial units later)
pixel1, pixel2 = 0, 1024  # left and right
# 1024 pixels; 13.3 µm per pixel on the CCD; 50× objective magnification
pos1, pos2 = -((1024/2)*13.3)/50, ((1024/2*13.3))/50

def plot_image(image, title):
    """
    Display a 2D image with physical axes.
    If no title is provided, attempt to infer the variable name.
    """
    if title is None:
        # Try to infer the variable name passed as argument
        frame = inspect.currentframe().f_back
        names = [name for name, val in frame.f_locals.items() if val is image]
        title = names[0] if names else ''

    # Larger figure
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        image,
        aspect='auto',
        extent=[pos_min, pos_max, energy_min, energy_max],
        # cmap='seismic'
    )
    cbar = plt.colorbar(im, label='Intensity')
    cbar.set_label('Intensity', size=16)
    cbar.ax.tick_params(labelsize=16)

    # Limit x-axis to the desired angular (real-space-projected) range
    ax.set_xlim(pos1, pos2)

    # Labels and title
    plt.ylabel('Energy (eV)', fontsize=18)
    plt.xlabel(r'x (μm)', fontsize=18)
    plt.title(title, fontsize=20)

    # Evenly spaced x ticks
    xticks = np.linspace(pos1, pos2, num=7)
    plt.xticks(xticks)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

def setup_axis(image, title):
    """
    Display a Stokes-parameter-like image with a fixed color range [-1, 1].
    Values outside [-1, 1] are masked and colored in green to highlight out-of-range values.
    """
    # Mask outside [-1, 1]
    masked_image = np.ma.masked_outside(image, -1, 1)

    # Colormap with a 'bad' (masked) color
    cmap = plt.get_cmap('seismic').copy()
    # Without this, values like -8 would be colored like -1 and +8 like +1.
    cmap.set_bad(color='green')

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        masked_image,
        aspect='auto',
        extent=[pos_min, pos_max, energy_min, energy_max],
        cmap=cmap,
        vmin=-1, vmax=1
    )
    cbar = plt.colorbar(im, ax=ax, label='Intensity')

    ax.set_ylabel('Energy (eV)', fontsize=18)
    ax.set_xlabel(r'x (μm)', fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.set_xlim(pos1, pos2)

    xticks = np.linspace(pos1, pos2, num=7)
    plt.xticks(xticks)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar.set_label('Intensity', fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.show()

def S_calculator_and_mask(image_A, image_B):
    """
    Compute a Stokes-like ratio (A - B) / (A + B), apply intensity-based masking, and return the result.
    Mask keeps pixels where (A + B) is greater than threshold_percent% of max(A + B).
    """
    # Sum and difference
    diff_image = image_A - image_B
    sum_image = image_A + image_B

    # Threshold on the sum
    threshold = (threshold_percent / 100.0) * np.max(sum_image)

    # Valid pixels: sum above threshold (absolute not needed here because sum_image is used directly)
    mask = np.abs(sum_image) > threshold

    # Compute S, guarding divisions
    with np.errstate(divide='ignore', invalid='ignore'):
        image_S = np.true_divide(diff_image, sum_image)
        image_S[~np.isfinite(image_S)] = 0  # Replace NaN/Inf with 0

    # Apply mask
    image_S[~mask] = 0
    return image_S

def remove_defects(image, threshold_factor=1.3):
    """
    Suppress hot-pixel-like defects by replacing outliers with the local mean.
    Pixels with intensity > threshold_factor * local_mean are replaced by local_mean.
    """
    filtered_image = image.copy()
    kernel_size = 5  # Neighborhood size

    # Local mean via box blur
    mean_local = cv2.blur(image, (kernel_size, kernel_size))

    # Defect mask and replacement
    threshold = threshold_factor * mean_local
    mask_defects = image > threshold
    filtered_image[mask_defects] = mean_local[mask_defects]

    return filtered_image

def nm_to_ev(wavelength_nm):
    """Convert wavelength (nm) to photon energy (eV)."""
    return h * c / (wavelength_nm * 1e-9)

# Read .sif, squeeze, and rotate
def rotate_image(filepath):
    """
    Load .sif data, squeeze extra dimensions, and rotate 90° clockwise
    (k = -1 in np.rot90).
    """
    data, info = sif_parser.np_open(filepath)
    image = data.squeeze()
    return np.rot90(image, k=-1)

def remove_polynomial_background_exciton(image_rotated, energy_min, energy_max):
    """
    Remove a smooth background (1st-order polynomial) while fitting a skewed Lorentzian.
    The polynomial component is evaluated and subtracted column-wise.
    """
    def polinomio_1(x, c0, c1):
        return c0 + c1 * x

    def skewed_lorentzian(x, amp, center, gamma, alpha):
        return amp / (1 + (((x - center)/(gamma*(1 + alpha*np.sign(x - center))))**2))

    # Build model: linear background + one skewed Lorentzian
    model = Model(polinomio_1, prefix='poly_') + Model(skewed_lorentzian, prefix='lorentz_')

    # Axes
    n_energias, n_posiciones = image_rotated.shape
    x_data = np.linspace(energy_min, energy_max, n_energias)
    imagen_pos_fondo = np.zeros_like(image_rotated)

    for j in range(n_posiciones):
        y_data = image_rotated[:, j]

        # Use the maximum as an initial guess for amplitude and center
        max_y = np.max(y_data)
        max_y_index = np.argmax(y_data)
        x_max_y = x_data[max_y_index]

        params = Parameters()
        params.add('poly_c0', value=0.01*max_y)
        params.add('poly_c1', value=0.01*max_y)

        # Lorentzian (LPB) parameters and bounds
        params.add('lorentz_amp', value=max_y*0.5, min=0, max=max_y*1.1)
        params.add('lorentz_center', value=x_max_y, min=x_max_y - 0.05, max=x_max_y + 0.05)
        params.add('lorentz_gamma', value=0.01, min=0, max=0.1)
        params.add('lorentz_alpha', value=0, vary=False)

        try:
            resultado = model.fit(y_data, params, x=x_data, method='least_squares')
            fondo = resultado.eval_components(x=x_data)['poly_']
            imagen_pos_fondo[:, j] = y_data - fondo
        except Exception as e:
            print(f"Fit failed at column {j}: {e}")
            imagen_pos_fondo[:, j] = y_data  # Keep original if fit fails

    return imagen_pos_fondo

def compute_Si_average_with_mask(Si_image, intensity_image_A, intensity_image_B, threshold_percent):
    """
    Weighted average of a Stokes image (S1, S2, or S3) considering only pixels
    where (A + B) exceeds threshold_percent% of its maximum.
    Returns the average (scalar). Error estimation is omitted here.
    """
    # Intensity sum and threshold
    intensity_sum = intensity_image_A + intensity_image_B
    threshold = (threshold_percent / 100.0) * np.max(intensity_sum)

    # Valid pixels
    mask = intensity_sum > threshold

    # Zero out invalid pixels (weights and values)
    Si_image[~mask] = 0
    intensity_sum[~mask] = 0

    # Weighted average
    numerator = np.sum(Si_image * intensity_sum)
    denominator = np.sum(intensity_sum)
    if denominator == 0:
        return 0.0, 0.0  # Avoid division by zero

    average_S = numerator / denominator
    return average_S

#%%
# Parameters and data paths
# We send H/σ⁺/σ⁻ and measure in H, V, D, A, σ⁺, σ⁻

# Base path for measurements
current_path = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(current_path, "..", "Measurements", "S3_exciton"))
initial_path = base_path + "/exciton_wavelength_525_pl_"
final_path = ".sif"

# Extract central wavelength from filename prefix
match = re.search(r'wavelength_(\d+)_', os.path.basename(initial_path))
if match:
    wavelength_central = int(match.group(1))

# Path to calibration spectrum (two levels up, then Measurements/Calibration_wavelength)
current_path = os.path.dirname(os.path.abspath(__file__))
calibration_folder = os.path.abspath(os.path.join(current_path, "..", "Measurements", "Calibration_wavelength"))

# .asc calibration filename
asc_filename = f"{wavelength_central}_spectrum.asc"

# Full path as a normalized string
ruta_asc = os.path.join(calibration_folder, asc_filename).replace("\\", "/")

# Read the .asc file (first column only)
if not os.path.exists(ruta_asc):
    raise FileNotFoundError(f"Calibration file not found: {ruta_asc}")

with open(ruta_asc, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Parse wavelength values (nm), replace commas with dots
nm_values = np.array([float(line.split()[0].replace(",", ".")) for line in lines])

# ---------------- Load each polarization channel, rotate, and filter defects ----------------
# H in, measure H/V
filepathH_H = initial_path + "H_L4ext_0_sample_L4_0_L2_0_H" + final_path   # Pump in H, measured in H
filepathH_V = initial_path + "H_L4ext_0_sample_L4_0_L2_45_H" + final_path  # Pump in H, measured in V
imageH_H = remove_defects(rotate_image(filepathH_H))
imageH_V = remove_defects(rotate_image(filepathH_V))

# R in, measure H/V
filepathR_H = initial_path + "H_L4ext_45_sample_L4_0_L2_0_H" + final_path  # Pump in R, measured in H
filepathR_V = initial_path + "H_L4ext_45_sample_L4_0_L2_45_H" + final_path # Pump in R, measured in V
imageR_H = remove_defects(rotate_image(filepathR_H))
imageR_V = remove_defects(rotate_image(filepathR_V))

# L in, measure H/V
filepathL_H = initial_path + "H_L4ext_-45_sample_L4_0_L2_0_H" + final_path # Pump in L, measured in H
filepathL_V = initial_path + "H_L4ext_-45_sample_L4_0_L2_45_H" + final_path# Pump in L, measured in V
imageL_H = remove_defects(rotate_image(filepathL_H))
imageL_V = remove_defects(rotate_image(filepathL_V))

# H in, measure D/A
filepathH_D = initial_path + "H_L4ext_0_sample_L4_45_L2_22.5_H" + final_path  # Pump in H, measured in D
filepathH_A = initial_path + "H_L4ext_0_sample_L4_45_L2_-22.5_H" + final_path # Pump in H, measured in A
imageH_D = remove_defects(rotate_image(filepathH_D))
imageH_A = remove_defects(rotate_image(filepathH_A))

# R in, measure D/A
filepathR_D = initial_path + "H_L4ext_45_sample_L4_45_L2_22.5_H" + final_path  # Pump in R, measured in D
filepathR_A = initial_path + "H_L4ext_45_sample_L4_45_L2_-22.5_H" + final_path # Pump in R, measured in A
imageR_D = remove_defects(rotate_image(filepathR_D))
imageR_A = remove_defects(rotate_image(filepathR_A))

# L in, measure D/A
filepathL_D = initial_path + "H_L4ext_-45_sample_L4_45_L2_22.5_H" + final_path  # Pump in L, measured in D
filepathL_A = initial_path + "H_L4ext_-45_sample_L4_45_L2_-22.5_H" + final_path # Pump in L, measured in A
imageL_D = remove_defects(rotate_image(filepathL_D))
imageL_A = remove_defects(rotate_image(filepathL_A))

# H in, measure σ⁺/σ⁻
filepathH_R = initial_path + "H_L4ext_0_sample_L4_-45_L2_45_H" + final_path   # Pump in H, measured in σ⁺
filepathH_L = initial_path + "H_L4ext_0_sample_L4_45_L2_45_H" + final_path    # Pump in H, measured in σ⁻
imageH_R = remove_defects(rotate_image(filepathH_R))
imageH_L = remove_defects(rotate_image(filepathH_L))

# R in, measure σ⁺/σ⁻
filepathR_R = initial_path + "H_L4ext_45_sample_L4_-45_L2_45_H" + final_path  # Pump in σ⁺, measured in σ⁺
filepathR_L = initial_path + "H_L4ext_45_sample_L4_45_L2_45_H" + final_path   # Pump in σ⁺, measured in σ⁻
imageR_R = remove_defects(rotate_image(filepathR_R))
imageR_L = remove_defects(rotate_image(filepathR_L))

# L in, measure σ⁺/σ⁻
filepathL_R = initial_path + "H_L4ext_-45_sample_L4_-45_L2_45_H" + final_path # Pump in σ⁻, measured in σ⁺
filepathL_L = initial_path + "H_L4ext_-45_sample_L4_45_L2_45_H" + final_path  # Pump in σ⁻, measured in σ⁻
imageL_R = remove_defects(rotate_image(filepathL_R))
imageL_L = remove_defects(rotate_image(filepathL_L))

# Compute pixel-to-angle mapping (used for horizontal axis scaling)
pixels_total = imageH_H.shape[1]
pos_per_pixel = (pos2 - pos1) / (pixel2 - pixel1)
pos_min = pos1 - (pixel1 * pos_per_pixel)
pos_max = pos_min + (pixels_total * pos_per_pixel)

# Convert wavelength axis to energy (eV) and compute bin edges
ev_values = nm_to_ev(nm_values)
ev_edges = np.zeros(len(ev_values) + 1)  # N pixels -> N+1 edges
# Midpoints between consecutive values to define edges
ev_edges[1:-1] = (ev_values[:-1] + ev_values[1:]) / 2
# First and last edges extrapolated from spacing
ev_edges[0] = ev_values[0] - (ev_values[1] - ev_values[0]) / 2
ev_edges[-1] = ev_values[-1] + (ev_values[-1] - ev_values[-2]) / 2

energy_max = ev_values.max()
energy_min = ev_values.min()
energy_per_pixel = (energy_max - energy_min) / pixels_total

# Build pixel-axis (columns) and energy axis (rows)
x_edges = np.linspace(pixel1, pixel2, imageH_H.shape[1] + 1)
x_values = (x_edges[:-1] + x_edges[1:]) / 2

# 2D mesh of pixel centers (X_pix) vs energy (Y_ev)
X_pix, Y_ev = np.meshgrid(x_values, ev_values)

# Jacobian for λ -> E transformation.
# The negative sign comes from the inverse relation between λ and E;
# we use abs() because we only need the magnitude for intensity correction.
jacobian = (np.abs(-((h*c)*1e9)) / (Y_ev**2))

# Apply Jacobian to all images
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

 # --- S1, incoming H ---
plot_image(imageH_H_final, 'Raw. H pumping, measured in H')
imageH_H_processed = remove_polynomial_background_exciton(imageH_H_final, energy_min, energy_max)
plot_image(imageH_H_processed, 'Processed. H pumping, measured in H')

plot_image(imageH_V_final, 'Raw. H pumping, measured in V')
imageH_V_processed = remove_polynomial_background_exciton(imageH_V_final, energy_min, energy_max)
plot_image(imageH_V_processed, 'Processed. H pumping, measured in V')

imageH_S1 = S_calculator_and_mask(imageH_H_processed, imageH_V_processed)
setup_axis(imageH_S1, 'S1 for incoming H polarization')

 # --- S1, incoming R ---
plot_image(imageR_H_final, r'Raw. $\sigma^{+}$ pumping, measured in H')
imageR_H_processed = remove_polynomial_background_exciton(imageR_H_final, energy_min, energy_max)
plot_image(imageR_H_processed, r'Processed. $\sigma^{+}$ pumping, measured in H')

plot_image(imageR_V_final, r'Raw. $\sigma^{+}$ pumping, measured in V')
imageR_V_processed = remove_polynomial_background_exciton(imageR_V_final, energy_min, energy_max)
plot_image(imageR_V_processed, r'Processed. $\sigma^{+}$ pumping, measured in V')

imageR_S1 = S_calculator_and_mask(imageR_H_processed, imageR_V_processed)
setup_axis(imageR_S1, r'S1 for incoming $\sigma^{+}$ polarization')

 # --- S1, incoming L ---
plot_image(imageL_H_final, r'Raw. $\sigma^{-}$ pumping, measured in H')
imageL_H_processed = remove_polynomial_background_exciton(imageL_H_final, energy_min, energy_max)
plot_image(imageL_H_processed, r'Processed. $\sigma^{-}$ pumping, measured in H')

plot_image(imageL_V_final, r'Raw. $\sigma^{-}$ pumping, measured in V')
imageL_V_processed = remove_polynomial_background_exciton(imageL_V_final, energy_min, energy_max)
plot_image(imageL_V_processed, r'Processed. $\sigma^{-}$ pumping, measured in V')

imageL_S1 = S_calculator_and_mask(imageL_H_processed, imageL_V_processed)
setup_axis(imageL_S1, r'S1 for incoming $\sigma^{-}$ polarization')

 # --- S2, incoming H ---
plot_image(imageH_D_final, 'Raw. H pumping, measured in D')
imageH_D_processed = remove_polynomial_background_exciton(imageH_D_final, energy_min, energy_max)
plot_image(imageH_D_processed, 'Processed. H pumping, measured in D')

plot_image(imageH_A_final, 'Raw. H pumping, measured in A')
imageH_A_processed = remove_polynomial_background_exciton(imageH_A_final, energy_min, energy_max)
plot_image(imageH_A_processed, 'Processed. H pumping, measured in A')

imageH_S2 = S_calculator_and_mask(imageH_D_processed, imageH_A_processed)
setup_axis(imageH_S2, 'S2 for incoming H polarization')

 # --- S2, incoming R ---
plot_image(imageR_D_final, r'Raw. $\sigma^{+}$ pumping, measured in D')
imageR_D_processed = remove_polynomial_background_exciton(imageR_D_final, energy_min, energy_max)
plot_image(imageR_D_processed, r'Processed. $\sigma^{+}$ pumping, measured in D')

plot_image(imageR_A_final, r'Raw. $\sigma^{+}$ pumping, measured in A')
imageR_A_processed = remove_polynomial_background_exciton(imageR_A_final, energy_min, energy_max)
plot_image(imageR_A_processed, r'Processed. $\sigma^{+}$ pumping, measured in A')

imageR_S2 = S_calculator_and_mask(imageR_D_processed, imageR_A_processed)
setup_axis(imageR_S2, r'S2 for incoming $\sigma^{+}$ polarization')

 # --- S2, incoming L ---
plot_image(imageL_D_final, r'Raw. $\sigma^{-}$ pumping, measured in D')
imageL_D_processed = remove_polynomial_background_exciton(imageL_D_final, energy_min, energy_max)
plot_image(imageL_D_processed, r'Processed. $\sigma^{-}$ pumping, measured in D')

plot_image(imageL_A_final, r'Raw. $\sigma^{-}$ pumping, measured in A')
imageL_A_processed = remove_polynomial_background_exciton(imageL_A_final, energy_min, energy_max)
plot_image(imageL_A_processed, r'Processed. $\sigma^{-}$ pumping, measured in A')

imageL_S2 = S_calculator_and_mask(imageL_D_processed, imageL_A_processed)
setup_axis(imageL_S2, r'S2 for incoming $\sigma^{-}$ polarization')

 # --- S3, incoming H ---
plot_image(imageH_R_final, r'Raw. H pumping, measured in $\sigma^{+}$')
imageH_R_processed = remove_polynomial_background_exciton(imageH_R_final, energy_min, energy_max)
plot_image(imageH_R_processed, r'Processed. H pumping, measured in $\sigma^{+}$')

plot_image(imageH_L_final, r'Raw. H pumping, measured in $\sigma^{-}$')
imageH_L_processed = remove_polynomial_background_exciton(imageH_L_final, energy_min, energy_max)
plot_image(imageH_L_processed, r'Processed. H pumping, measured in $\sigma^{-}$')

imageH_S3 = S_calculator_and_mask(imageH_R_processed, imageH_L_processed)
setup_axis(imageH_S3, 'S3 for incoming H polarization')

 # --- S3, incoming σ⁺ ---
plot_image(imageR_R_final, r'Raw. $\sigma^{+}$ pumping, measured in $\sigma^{+}$')
imageR_R_processed = remove_polynomial_background_exciton(imageR_R_final, energy_min, energy_max)
plot_image(imageR_R_processed, r'Processed. $\sigma^{+}$ pumping, measured in $\sigma^{+}$')

plot_image(imageR_L_final, r'Raw. $\sigma^{+}$ pumping, measured in $\sigma^{-}$')
imageR_L_processed = remove_polynomial_background_exciton(imageR_L_final, energy_min, energy_max)
plot_image(imageR_L_processed, r'Processed. $\sigma^{+}$ pumping, measured in $\sigma^{-}$')

imageR_S3 = S_calculator_and_mask(imageR_R_processed, imageR_L_processed)
setup_axis(imageR_S3, r'S3 for incoming $\sigma^{+}$ polarization')

 # --- S3, incoming σ⁻ ---
plot_image(imageL_R_final, r'Raw. $\sigma^{-}$ pumping, measured in $\sigma^{+}$')
imageL_R_processed = remove_polynomial_background_exciton(imageL_R_final, energy_min, energy_max)
plot_image(imageL_R_processed, r'Processed. $\sigma^{-}$ pumping, measured in $\sigma^{+}$')

plot_image(imageL_L_final, r'Raw. $\sigma^{-}$ pumping, measured in $\sigma^{-}$')
imageL_L_processed = remove_polynomial_background_exciton(imageL_L_final, energy_min, energy_max)
plot_image(imageL_L_processed, r'Processed. $\sigma^{-}$ pumping, measured in $\sigma^{-}$')

imageL_S3 = S_calculator_and_mask(imageL_R_processed, imageL_L_processed)
setup_axis(imageL_S3, r'S3 for incoming $\sigma^{-}$ polarization')

#%% Figure 10: integrate over a chosen x-range
x_min = -5
x_max = 15
# Convert x-range to pixel indices
x_min_integration = int((x_min - pos_min) / pos_per_pixel)
x_max_integration = int((x_max - pos_min) / pos_per_pixel)

# Integrate columns (sum over x) for each required dataset
integrated_R_R = np.sum(imageR_R_processed[:, x_min_integration:x_max_integration], axis=1)
integrated_R_L = np.sum(imageR_L_processed[:, x_min_integration:x_max_integration], axis=1)
integrated_L_R = np.sum(imageL_R_processed[:, x_min_integration:x_max_integration], axis=1)
integrated_L_L = np.sum(imageL_L_processed[:, x_min_integration:x_max_integration], axis=1)

# Energy array for plotting (flip to match image orientation)
energy_values = np.flip(np.linspace(energy_min, energy_max, len(integrated_L_R)))

# Two subplots (R and L incoming polarizations)
fig, axs = plt.subplots(1, 2, figsize=(20, 8))

# --- Incoming R (σ+ and σ-) ---
axs[0].plot(energy_values, integrated_R_R, label=r'$I_{\sigma^+}$')
axs[0].plot(energy_values, integrated_R_L, label=r'$I_{\sigma^-}$')
axs[0].set_xlabel('Energy (eV)', fontsize=30)
axs[0].set_ylabel('Intensity (a. u.)', fontsize=30)
axs[0].grid(True)
axs[0].tick_params(axis='both', which='major', labelsize=25)
axs[0].yaxis.offsetText.set_fontsize(25)
axs[0].legend(loc='lower center', fontsize=40, frameon=False)
axs[0].text(
    0.98, 0.98, r'$\sigma^+$', transform=axs[0].transAxes,
    fontsize=35, verticalalignment='top', horizontalalignment='right',
    bbox=dict(facecolor='white', edgecolor='none', pad=2.0)
)

# --- Incoming L (σ+ and σ-) ---
axs[1].plot(energy_values, integrated_L_R, label=r'$I_{\sigma^+}$')
axs[1].plot(energy_values, integrated_L_L, label=r'$I_{\sigma^-}$')
axs[1].set_xlabel('Energy (eV)', fontsize=30)
axs[1].set_ylabel('Intensity (a. u.)', fontsize=30)
axs[1].grid(True)
axs[1].tick_params(axis='both', which='major', labelsize=25)
axs[1].yaxis.offsetText.set_fontsize(25)
axs[1].legend(loc='lower center', fontsize=40, frameon=False)
axs[1].text(
    0.98, 0.98, r'$\sigma^-$', transform=axs[1].transAxes,
    fontsize=35, verticalalignment='top', horizontalalignment='right',
    bbox=dict(facecolor='white', edgecolor='none', pad=2.0)
)

# Add panel labels (a) and (b) in top-left corners
axs[0].text(-0.19, 0.99, '(a)', transform=axs[0].transAxes,
             ha='left', va='top', fontsize=30, fontweight='bold')
axs[1].text(-0.19, 0.99, '(b)', transform=axs[1].transAxes,
             ha='left', va='top', fontsize=30, fontweight='bold')
# Increase horizontal space between subplots
plt.subplots_adjust(wspace=7.5) # A bit of space between both subplots

plt.tight_layout()
# --- Rasterize just images (pcolormesh) ---
for ax in axs:
    for artist in ax.collections:
        artist.set_rasterized(True) # The rest is not renderized!

# Create the foler "Figures" if it doesn't exist
figures_folder = os.path.join(current_path, "Figures")
os.makedirs(figures_folder, exist_ok=True)
# Whole path to save the figure
save_path = os.path.join(figures_folder, "Fig.Suppl.8.pdf")
# Actually save it
fig.savefig(save_path, format='pdf', bbox_inches='tight', transparent=False, metadata=None, dpi=300)


plt.show()



