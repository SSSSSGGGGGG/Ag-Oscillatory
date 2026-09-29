# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:25:05 2026

@modified: sgao
"""
"""Quantify radial propagation relative to the confirmed r = 15.5 µm illumination circle."""
from pathlib import Path
import math
import os
from openpyxl import load_workbook
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, detrend, hilbert, sosfiltfilt

# ============================================================
# 1. WORKING DIRECTORY / INPUT PARAMETERS
# ============================================================

ROOT = Path(
    r"C:/Users/sgao/OneDrive - ICIQ/Documents/SG/2025/PIV/Yufen/Videos/No.2"
)
os.chdir(ROOT)

# Code 2 file-naming parameters used by the existing XLSX reader.
window_size = 32
search_size = 12
displacement_threshold_h = 10
n_values = range(1, 376, 1)

OUT = (
    ROOT
    / "outputs"
    / "wave_analysis"
    / "small_circle_propagation"
)
OUT.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename, dpi=300):
    """Save a matplotlib figure robustly on Windows/OneDrive paths."""
    filename = str(filename)
    primary_path = OUT / filename
    try:
        primary_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(primary_path), dpi=dpi, bbox_inches="tight")
        return primary_path
    except OSError as exc:
        # Fall back to a local file if OneDrive/path handling blocks the primary save.
        fallback_dir = ROOT / "outputs" / "wave_analysis" / "small_circle_propagation_local"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / filename
        print(f"WARNING: Could not save figure to {primary_path!s}: {exc}")
        print(f"         Saving figure instead to {fallback_path!s}")
        fig.savefig(str(fallback_path), dpi=dpi, bbox_inches="tight")
        return fallback_path

FPS = 25.0
PX_PER_UM = 4.8563
ILLUMINATION_DIAMETER_UM = 31.0
BOUNDARY_UM = ILLUMINATION_DIAMETER_UM / 2.0
REFERENCE_MIN_UM = 3.0
PHASE_MAX_RADIUS_UM = 55.0
BAND_HZ = (0.4, 2.6)
WINDOW_S = 5.0

# All criteria before a phase speed is reported.
MIN_PHASE_R2 = 0.70
MIN_PHASE_SPAN_RAD = 0.5
MIN_PEAK_PROMINENCE = 0.5

# Minimum number of radial points required for phase fitting.
MIN_VALID_RADIAL_POINTS = 5


# ============================================================
# 2. READ RADIAL VELOCITY XLSX FILES
# ============================================================

profiles_radius = []
profiles_vr = []
profiles_vtheta = []
profiles_vtotal = []
times = []

print("=" * 70)
print("READING RADIAL VELOCITY FILES")
print("=" * 70)

for n in n_values:

    file_path = (
        ROOT
        / f"radial_velocity_{n} "
        f"w_{window_size} "
        f"s_{search_size} "
        f"dth_{displacement_threshold_h}.xlsx"
    )

    print(f"Reading: {file_path.name}")

    if not file_path.exists():
        print("  File not found - skipping")
        continue

    wb = load_workbook(file_path, data_only=True)
    ws = wb.active

    headers = [cell.value for cell in ws[1]]

    required = [
        "time_s",
        "radius_um",
        "mean_Vr_um_s",
        "mean_Vtheta_um_s",
    ]

    for name in required:
        if name not in headers:
            raise ValueError(
                f"'{name}' not found in {file_path.name}"
            )

    time_col = headers.index("time_s")
    radius_col = headers.index("radius_um")
    vr_col = headers.index("mean_Vr_um_s")
    vtheta_col = headers.index("mean_Vtheta_um_s")

    file_time = []
    file_radius = []
    file_vr = []
    file_vtheta = []

    for row_number, row in enumerate(
        ws.iter_rows(min_row=2, values_only=True),
        start=2
    ):

        if row[time_col] is None or row[radius_col] is None:
            continue

        try:
            t = float(row[time_col])
            r = float(row[radius_col])
        except (ValueError, TypeError):
            continue

        vr = (
            0.0
            if row[vr_col] is None
            else float(row[vr_col])
        )

        vtheta = (
            0.0
            if row[vtheta_col] is None
            else float(row[vtheta_col])
        )

        if not (
            np.isfinite(t)
            and np.isfinite(r)
            and np.isfinite(vr)
            and np.isfinite(vtheta)
        ):
            continue

        file_time.append(t)
        file_radius.append(r)
        file_vr.append(vr)
        file_vtheta.append(vtheta)

    if len(file_time) == 0:
        print("  No valid data - skipping")
        continue

    file_time = np.asarray(file_time, dtype=float)
    file_radius = np.asarray(file_radius, dtype=float)
    file_vr = np.asarray(file_vr, dtype=float)
    file_vtheta = np.asarray(file_vtheta, dtype=float)

    # Definition of total velocity.
    file_vtotal = np.sqrt(
        file_vr**2 + file_vtheta**2
    )

    unique_times = np.unique(file_time)
    file_time_value = unique_times[0]

    order = np.argsort(file_radius)

    sorted_radius = file_radius[order]
    sorted_vr = file_vr[order]
    sorted_vtheta = file_vtheta[order]
    sorted_vtotal = file_vtotal[order]

    times.append(file_time_value)

    profiles_radius.append(sorted_radius)
    profiles_vr.append(sorted_vr)
    profiles_vtheta.append(sorted_vtheta)
    profiles_vtotal.append(sorted_vtotal)


# ============================================================
# 3. CHECK AND SORT DATA
# ============================================================

if len(profiles_vr) == 0:
    raise ValueError(
        "No valid radial velocity data found."
    )

times = np.asarray(times, dtype=float)

time_order = np.argsort(times)

times = times[time_order]

profiles_radius = [
    profiles_radius[i]
    for i in time_order
]

profiles_vr = [
    profiles_vr[i]
    for i in time_order
]

profiles_vtheta = [
    profiles_vtheta[i]
    for i in time_order
]

profiles_vtotal = [
    profiles_vtotal[i]
    for i in time_order
]

print()
print(f"Loaded {len(times)} time points")
print(
    f"Time range: "
    f"{times[0]:.2f} - {times[-1]:.2f} s"
)


# ============================================================
# 4. CREATE COMMON RADIAL GRID
# ============================================================

common_radius_min = max(
    np.min(r)
    for r in profiles_radius
)

common_radius_max = min(
    np.max(r)
    for r in profiles_radius
)

if common_radius_max <= common_radius_min:
    raise ValueError(
        "The radial profiles do not have an overlapping radial range."
    )

n_common_radius = len(profiles_radius[0])

radius_values = np.linspace(
    common_radius_min,
    common_radius_max,
    n_common_radius,
)

print(
    f"Radial grid: {len(radius_values)} bins "
    f"from {radius_values[0]:.2f} to "
    f"{radius_values[-1]:.2f} µm"
)


# ============================================================
# 5. INTERPOLATE ALL VELOCITY PROFILES
# ============================================================

print()
print("Interpolating velocity profiles...")

interpolated_vr = []
interpolated_vtheta = []
interpolated_vtotal = []

for i in range(len(times)):

    r = np.asarray(
        profiles_radius[i],
        dtype=float,
    )

    vr = np.asarray(
        profiles_vr[i],
        dtype=float,
    )

    vtheta = np.asarray(
        profiles_vtheta[i],
        dtype=float,
    )

    vtotal = np.asarray(
        profiles_vtotal[i],
        dtype=float,
    )

    unique_r, unique_indices = np.unique(
        r,
        return_index=True,
    )

    vr = vr[unique_indices]
    vtheta = vtheta[unique_indices]
    vtotal = vtotal[unique_indices]

    r = unique_r

    if len(r) < 2:
        raise ValueError(
            f"Profile {i} has fewer than "
            f"2 unique radial points."
        )

    vr_interp = np.interp(
        radius_values,
        r,
        vr,
    )

    vtheta_interp = np.interp(
        radius_values,
        r,
        vtheta,
    )

    vtotal_interp = np.interp(
        radius_values,
        r,
        vtotal,
    )

    interpolated_vr.append(vr_interp)
    interpolated_vtheta.append(vtheta_interp)
    interpolated_vtotal.append(vtotal_interp)

    if i % 25 == 0:
        print(
            f"  Interpolated profile "
            f"{i + 1}/{len(times)}"
        )

profiles_vr = interpolated_vr
profiles_vtheta = interpolated_vtheta
profiles_vtotal = interpolated_vtotal


radii = np.asarray(radius_values, dtype=float)

# Primary propagation variable: radial velocity Vr.
velocity_vr = np.asarray(profiles_vr, dtype=float)

# Parallel secondary analysis variable: total velocity magnitude Vtotal.
velocity_vtotal = np.asarray(profiles_vtotal, dtype=float)

# Keep the original variable name as an explicit alias for the primary Vr analysis.
velocity = velocity_vr

print()
print("Data shapes:")
print(f"  times: {times.shape}")
print(f"  radii: {radii.shape}")
print(f"  radial velocity matrix Vr: {velocity_vr.shape}")
print(f"  total velocity matrix Vtotal: {velocity_vtotal.shape}")

if np.any(~np.isfinite(velocity_vr)):
    raise RuntimeError(
        "Unexpected missing/non-finite radial-velocity values "
        "after XLSX loading and interpolation."
    )

if np.any(~np.isfinite(velocity_vtotal)):
    raise RuntimeError(
        "Unexpected missing/non-finite total-velocity values "
        "after XLSX loading and interpolation."
    )


# ============================================================
# 6. PROCESS Vr AND Vtotal IN PARALLEL
# ============================================================

# Vtotal is derived ONCE from the original interpolated PIV components above.
# From this point, Vr and Vtotal are analyzed as parallel branches.
# Both branches use the same band-pass filter for propagation analysis.

sos = butter(
    3,
    BAND_HZ,
    btype="bandpass",
    fs=FPS,
    output="sos",
)

filtered_vr = sosfiltfilt(
    sos,
    detrend(velocity_vr, axis=0),
    axis=0,
)

# Vtotal follows the same filtering procedure independently.
filtered_vtotal = sosfiltfilt(
    sos,
    detrend(velocity_vtotal, axis=0),
    axis=0,
)


filtered = filtered_vr

inside = (
    (radii >= REFERENCE_MIN_UM)
    & (radii <= BOUNDARY_UM)
)

if np.sum(inside) == 0:
    raise RuntimeError(
        "No radial bins are available inside the defined "
        "illumination/reference region."
    )

# Vr reference: derived from the FILTERED Vr branch only.
reference_vr = np.median(
    filtered_vr[:, inside],
    axis=1,
)
reference_vr -= np.mean(reference_vr)

# Vtotal reference: derived from the FILTERED Vtotal branch.
reference_vtotal = np.median(
    filtered_vtotal[:, inside],
    axis=1,
)
reference_vtotal -= np.mean(reference_vtotal)


reference = reference_vr

def fit_phase_window(signal_matrix, reference_signal, start, n_samples, maximum_radius_um, forced_frequency_hz=None):
    """Fit phase versus radius in one window without assuming a full-video frequency."""
    stop = start + n_samples
    local_time = np.arange(n_samples) / FPS
    taper = np.hanning(n_samples)
    ref = reference_signal[start:stop] - np.mean(reference_signal[start:stop])
    frequencies = np.fft.rfftfreq(n_samples, 1 / FPS)
    ref_fft = np.fft.rfft(ref * taper)
    allowed = (frequencies >= BAND_HZ[0]) & (frequencies <= BAND_HZ[1])
    allowed_indices = np.flatnonzero(allowed)
    if forced_frequency_hz is None:
        peak_index = allowed_indices[np.argmax(np.abs(ref_fft[allowed]) ** 2)]
        frequency = float(frequencies[peak_index])
    else:
        frequency = float(forced_frequency_hz)
        if not (BAND_HZ[0] <= frequency <= BAND_HZ[1]):
            raise ValueError(
                f"Forced frequency {frequency:.6g} Hz is outside BAND_HZ={BAND_HZ}."
            )
        peak_index = int(np.argmin(np.abs(frequencies - frequency)))

    peak_power = float(abs(ref_fft[peak_index]) ** 2)
    peak_prominence = peak_power / max(float(np.median(abs(ref_fft[allowed]) ** 2)), 1e-12)

    local = signal_matrix[start:stop] - np.mean(signal_matrix[start:stop], axis=0, keepdims=True)
    kernel = taper * np.exp(-2j * np.pi * frequency * local_time)
    coefficient = np.sum(local * kernel[:, None], axis=0)
    reference_coefficient = np.sum(ref * kernel)

    use = (radii >= BOUNDARY_UM) & (radii <= maximum_radius_um)
    radial_position = radii[use]
    phase = np.unwrap(np.angle(coefficient[use] * np.conj(reference_coefficient)))
    phase -= phase[0]
    amplitude = np.abs(coefficient[use])
    weights = amplitude / max(np.max(amplitude), 1e-12)

    distance = radial_position - radial_position[0]
    design = np.column_stack([distance, np.ones(len(distance))])
    sqrt_weight = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(design * sqrt_weight[:, None], phase * sqrt_weight, rcond=None)
    predicted_phase = design @ beta
    weighted_mean = np.average(phase, weights=np.maximum(weights, 1e-12))
    residual_ss = float(np.sum(weights * (phase - predicted_phase) ** 2))
    total_ss = float(np.sum(weights * (phase - weighted_mean) ** 2))
    phase_r2 = 1.0 - residual_ss / max(total_ss, 1e-12)
    phase_slope = float(beta[0])
    fitted_phase_span = float(abs(phase_slope) * (distance[-1] - distance[0]))
    signed_phase_speed = float(-2 * np.pi * frequency / phase_slope) if abs(phase_slope) > 1e-12 else math.inf
    delay = -phase / (2 * np.pi * frequency)
    fitted_delay = -predicted_phase / (2 * np.pi * frequency)
    delay_slope = float(-phase_slope / (2 * np.pi * frequency))

    # Negative delays are NOT a quality criterion.
    # They remain part of the calculation/results. They are only excluded
    # from the plotted raw-delay points below.
    endpoint_delay = float(delay[-1])
    effective_endpoint_speed = (
        float(distance[-1] / endpoint_delay)
        if endpoint_delay > 0
        else math.nan
    )

    accepted = bool(
        phase_slope < 0
        and phase_r2 >= MIN_PHASE_R2
        and fitted_phase_span >= MIN_PHASE_SPAN_RAD
        and peak_prominence >= MIN_PEAK_PROMINENCE
    )
    failed = []
    if phase_slope >= 0:
        failed.append("not_outward")
    if phase_r2 < MIN_PHASE_R2:
        failed.append("phase_r2")
    if fitted_phase_span < MIN_PHASE_SPAN_RAD:
        failed.append("phase_span")
    if peak_prominence < MIN_PEAK_PROMINENCE:
        failed.append("spectral_prominence")

    row = {
        "start_s": float(times[start]),
        "end_s": float(times[stop - 1]),
        "center_s": float(times[start + n_samples // 2]),
        "window_s": float(n_samples / FPS),
        "phase_fit_maximum_radius_um": float(maximum_radius_um),
        "dominant_frequency_hz": frequency,
        "peak_to_median_spectral_power": peak_prominence,
        "phase_slope_rad_per_um": phase_slope,
        "phase_vs_radius_weighted_r2": phase_r2,
        "fitted_phase_span_rad": fitted_phase_span,
        "delay_slope_s_per_um": delay_slope,
        "signed_phase_speed_um_s": signed_phase_speed,
        "phase_delay_to_outermost_fitted_radius_s": endpoint_delay,
        "effective_boundary_to_outer_radius_speed_um_s": effective_endpoint_speed,
        "accepted_outward_propagation": accepted,
        "failed_quality_criteria": ";".join(failed),
    }
    profile = pd.DataFrame({
        "window_start_s": float(times[start]),
        "window_end_s": float(times[stop - 1]),
        "window_center_s": float(times[start + n_samples // 2]),
        "dominant_frequency_hz": frequency,
        "radius_um": radial_position,
        "distance_beyond_illumination_boundary_um": radial_position - BOUNDARY_UM,
        "phase_relative_to_boundary_rad": phase,
        # Keep ALL raw delays in the numerical output.
        "phase_derived_delay_s": delay,
        "fitted_delay_s": fitted_delay,
        "spectral_amplitude_au": amplitude,
        "accepted_outward_propagation": accepted,
        # Plotting mask only. This does NOT affect acceptance or fitting.
        "plot_raw_delay": delay >= 0,
    })
    return row, profile


# ============================================================
# Vtotal PHASE ANALYSIS
# ============================================================

def fit_phase_window_vtotal(
    signal_matrix,
    reference_signal,
    start,
    n_samples,
    maximum_radius_um,
    frequency_hz,
):
    """
    Apply the same phase-vs-radius analysis to Vtotal.

    IMPORTANT:
    Vtotal is a scalar velocity magnitude, so this is a
    phase-propagation/coherence analysis of the total-speed signal.
    It is NOT interpreted as radial propagation of particles.
    """
    return fit_phase_window(
        signal_matrix,
        reference_signal,
        start,
        n_samples,
        maximum_radius_um,
        forced_frequency_hz=frequency_hz,
    )


# ============================================================
# 7. Vr AND Vtotal AMPLITUDE / PHASE-LOCKING PROFILES
# ============================================================


rms_amplitude_vr = np.sqrt(np.mean(filtered_vr ** 2, axis=0))
analytic_vr = hilbert(filtered_vr, axis=0)
reference_analytic_vr = hilbert(reference_vr)
active_time_vr = (
    np.abs(reference_analytic_vr)
    >= np.percentile(np.abs(reference_analytic_vr), 40)
)
phase_difference_vr = np.angle(
    analytic_vr * np.conj(reference_analytic_vr[:, None])
)
phase_vector_vr = np.mean(
    np.exp(1j * phase_difference_vr[active_time_vr]),
    axis=0,
)
phase_locking_vr = np.abs(phase_vector_vr)
mean_phase_vr = np.unwrap(np.angle(phase_vector_vr))
mean_phase_vr -= np.median(mean_phase_vr[inside])

# Parallel Vtotal analysis uses the FILTERED total-velocity magnitude,

rms_amplitude_vtotal = np.sqrt(np.mean(filtered_vtotal ** 2, axis=0))
analytic_vtotal = hilbert(filtered_vtotal, axis=0)
reference_analytic_vtotal = hilbert(reference_vtotal)
active_time_vtotal = (
    np.abs(reference_analytic_vtotal)
    >= np.percentile(np.abs(reference_analytic_vtotal), 40)
)
phase_difference_vtotal = np.angle(
    analytic_vtotal * np.conj(reference_analytic_vtotal[:, None])
)
phase_vector_vtotal = np.mean(
    np.exp(1j * phase_difference_vtotal[active_time_vtotal]),
    axis=0,
)
phase_locking_vtotal = np.abs(phase_vector_vtotal)
mean_phase_vtotal = np.unwrap(np.angle(phase_vector_vtotal))
mean_phase_vtotal -= np.median(mean_phase_vtotal[inside])

# Raw, unfiltered RMS comparison. Because Vtotal is calculated from the same
# original Vr and Vtheta samples, RMS(Vtotal) should be >= RMS(Vr) pointwise
# after averaging over time (up to numerical/interpolation effects).
raw_rms_vr = np.sqrt(np.mean(velocity_vr ** 2, axis=0))
raw_rms_vtotal = np.sqrt(np.mean(velocity_vtotal ** 2, axis=0))

raw_rms_profiles = pd.DataFrame({
    "radius_um": radii,
    "Vr_raw_rms_um_s": raw_rms_vr,
    "Vtotal_raw_rms_um_s": raw_rms_vtotal,
})
raw_rms_profiles.to_csv(
    OUT / "raw_unfiltered_Vr_vs_Vtotal_RMS.csv",
    index=False,
)

radial_profiles = pd.DataFrame({
    "radius_um": radii,
    "radius_px": radii * PX_PER_UM,
    "inside_illuminated_region": radii <= BOUNDARY_UM,

    # Vr
    "Vr_rms_um_s": rms_amplitude_vr,
    "Vr_phase_locking_value_0_to_1": phase_locking_vr,
    "Vr_mean_phase_vs_inside_rad": mean_phase_vr,

    # Vtotal
    "Vtotal_raw_rms_um_s": raw_rms_vtotal,
    "Vtotal_phase_locking_value_0_to_1": phase_locking_vtotal,
    "Vtotal_mean_phase_vs_inside_rad": mean_phase_vtotal,
})

radial_profiles.to_csv(
    OUT / "Vr_and_Vtotal_amplitude_coherence_profiles.csv",
    index=False,
)

# -----------------------------------------------------------------------------
# Raw / unfiltered RMS comparison
# -----------------------------------------------------------------------------
# These RMS values are calculated directly from the interpolated raw velocity
# matrices, before detrending, bandpass filtering, or Hilbert analysis.
# Therefore, pointwise |Vr| <= Vtotal is preserved in the RMS comparison.
raw_rms_vr = np.sqrt(np.mean(velocity_vr ** 2, axis=0))
raw_rms_vtotal = np.sqrt(np.mean(velocity_vtotal ** 2, axis=0))

raw_rms_profiles = pd.DataFrame({
    "radius_um": radii,
    "radius_px": radii * PX_PER_UM,
    "inside_illuminated_region": radii <= BOUNDARY_UM,
    "Vr_raw_rms_um_s": raw_rms_vr,
    "Vtotal_raw_rms_um_s": raw_rms_vtotal,
})
raw_rms_profiles.to_csv(
    OUT / "raw_unfiltered_Vr_vs_Vtotal_RMS.csv",
    index=False,
)

# Figure 11: raw / unfiltered RMS comparison
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(radii, raw_rms_vr, lw=2, label="Raw Vr RMS")
ax.plot(radii, raw_rms_vtotal, lw=2, label="Raw Vtotal RMS")
ax.axvspan(0, BOUNDARY_UM, alpha=0.15, label="Illuminated region")
ax.axvline(BOUNDARY_UM, lw=1.5, ls="--", label="Illumination boundary")
ax.set_xlabel("Radius (µm)")
ax.set_ylabel("Unfiltered RMS velocity (µm s$^{-1}$)")
ax.set_title("Raw / unfiltered Vr and Vtotal RMS")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_figure(fig, "Figure_11_raw_unfiltered_Vr_vs_Vtotal_RMS.png", dpi=300)
plt.close(fig)


rms_amplitude = rms_amplitude_vr
phase_locking = phase_locking_vr
mean_phase = mean_phase_vr


# Independent, non-overlapping five-second windows provide the primary summary.
n_window = int(round(WINDOW_S * FPS))
nonoverlap_rows = []
nonoverlap_profiles = []
for start in range(0, len(times) - n_window + 1, n_window):
    row, profile = fit_phase_window(filtered, reference, start, n_window, PHASE_MAX_RADIUS_UM)
    nonoverlap_rows.append(row)
    nonoverlap_profiles.append(profile)
nonoverlap = pd.DataFrame(nonoverlap_rows)
phase_delay_profiles = pd.concat(nonoverlap_profiles, ignore_index=True)
nonoverlap.to_csv(OUT / "propagation_windows_nonoverlapping.csv", index=False)
phase_delay_profiles.to_csv(OUT / "phase_delay_profiles_nonoverlapping.csv", index=False)


# Overlapping windows reveal when propagation is present, but are not treated as independent replicates.
sliding_rows = []
for start in range(0, len(times) - n_window + 1, int(FPS)):
    row, _ = fit_phase_window(filtered, reference, start, n_window, PHASE_MAX_RADIUS_UM)
    sliding_rows.append(row)
sliding = pd.DataFrame(sliding_rows)
sliding.to_csv(OUT / "propagation_windows_sliding.csv", index=False)


# Zone summaries. Ranges are temporal ranges across four non-overlapping blocks, not confidence intervals.
zones = [
    ("inside illuminated region", REFERENCE_MIN_UM, BOUNDARY_UM),
    ("outside: 15.5–30 µm", BOUNDARY_UM, 30.0),
    ("outside: 30–45 µm", 30.0, 45.0),
    ("outside: 45–60 µm", 45.0, 60.0),
    ("outside: 60–74 µm", 60.0, float(np.max(radii) + 1e-6)),
]
zone_rows = []
inside_median_amplitude = float(np.median(rms_amplitude[inside]))
for name, low, high in zones:
    use = (radii >= low) & (radii < high)
    block_amplitudes = []
    for start in range(0, len(times) - n_window + 1, n_window):
        block_rms = np.sqrt(np.mean(filtered[start:start + n_window, use] ** 2, axis=0))
        block_amplitudes.append(float(np.median(block_rms)))
    overall_amp = float(np.median(rms_amplitude[use]))
    zone_rows.append({
        "zone": name,
        "radius_min_um": low,
        "radius_max_um": high,
        "number_of_radial_bins": int(np.sum(use)),
        "median_rms_radial_velocity_um_s": overall_amp,
        "amplitude_relative_to_inside": overall_amp / inside_median_amplitude,
        "median_across_5s_blocks_um_s": float(np.median(block_amplitudes)),
        "minimum_across_5s_blocks_um_s": float(np.min(block_amplitudes)),
        "maximum_across_5s_blocks_um_s": float(np.max(block_amplitudes)),
        "median_phase_locking_value": float(np.median(phase_locking[use])),
    })
zones_table = pd.DataFrame(zone_rows)
zones_table.to_csv(OUT / "inside_outside_zone_summary.csv", index=False)


# Sensitivity to window duration and the outer radius used for the phase fit.
sensitivity_rows = []
for window_s in (4.0, 5.0, 6.0):
    samples = int(round(window_s * FPS))
    for maximum_radius in (45.0, 50.0, 55.0, 60.0):
        results = []
        for start in range(0, len(times) - samples + 1, int(FPS)):
            row, _ = fit_phase_window(filtered, reference, start, samples, maximum_radius)
            if row["accepted_outward_propagation"]:
                results.append(row["signed_phase_speed_um_s"])
        if results:
            q25, median, q75 = np.percentile(results, [25, 50, 75])
            minimum, maximum = np.min(results), np.max(results)
        else:
            q25 = median = q75 = minimum = maximum = math.nan
        sensitivity_rows.append({
            "window_s": window_s,
            "phase_fit_maximum_radius_um": maximum_radius,
            "accepted_sliding_windows": len(results),
            "median_phase_speed_um_s": median,
            "q25_phase_speed_um_s": q25,
            "q75_phase_speed_um_s": q75,
            "minimum_phase_speed_um_s": minimum,
            "maximum_phase_speed_um_s": maximum,
        })
sensitivity = pd.DataFrame(sensitivity_rows)
sensitivity.to_csv(OUT / "phase_speed_parameter_sensitivity.csv", index=False)


# Algorithmic validation using synthetic radial phase waves with a known speed.
rng = np.random.default_rng(20260831)
synthetic_rows = []
known_speed = 100.0
synthetic_n = int(WINDOW_S * FPS)
synthetic_time = np.arange(synthetic_n) / FPS
distance_from_boundary = np.maximum(radii - BOUNDARY_UM, 0)
for frequency in (0.8, 1.2, 1.4):
    for replicate in range(25):
        phase = 2 * np.pi * frequency * synthetic_time[:, None] - (
            2 * np.pi * frequency * distance_from_boundary[None, :] / known_speed
        )
        attenuation = np.exp(-distance_from_boundary / 120.0)
        synthetic = attenuation[None, :] * np.sin(phase) + 0.35 * rng.normal(size=phase.shape)
        synthetic_reference = np.median(synthetic[:, inside], axis=1)
        # The fitter uses the global time array only for labels; start=0 is valid here.
        row, _ = fit_phase_window(synthetic, synthetic_reference, 0, synthetic_n, PHASE_MAX_RADIUS_UM)
        synthetic_rows.append({
            "frequency_hz": frequency,
            "replicate": replicate + 1,
            "known_phase_speed_um_s": known_speed,
            "recovered_phase_speed_um_s": row["signed_phase_speed_um_s"],
            "phase_vs_radius_weighted_r2": row["phase_vs_radius_weighted_r2"],
            "accepted_by_quality_screen": row["accepted_outward_propagation"],
        })
synthetic_validation = pd.DataFrame(synthetic_rows)
synthetic_validation.to_csv(OUT / "synthetic_phase_speed_validation.csv", index=False)
synthetic_accepted = synthetic_validation[synthetic_validation["accepted_by_quality_screen"]]


# ============================================================
# 8. PARALLEL Vtotal WINDOWED PHASE ANALYSIS
# ============================================================

# The existing non-overlapping/sliding propagation analysis remains Vr.
# Vtotal receives the same windowing and quality-screen logic in parallel,
# but its results are explicitly labeled as Vtotal phase analysis.

vtotal_nonoverlap_rows = []
vtotal_nonoverlap_profiles = []

for start in range(
    0,
    len(times) - n_window + 1,
    n_window,
):
    # Force Vtotal to use exactly the frequency selected independently for Vr.
    vr_row = nonoverlap[nonoverlap["center_s"] == float(times[start + n_window // 2])].iloc[0]
    vr_frequency = float(vr_row["dominant_frequency_hz"])
    row, profile = fit_phase_window_vtotal(
        filtered_vtotal,
        reference_vtotal,
        start,
        n_window,
        PHASE_MAX_RADIUS_UM,
        frequency_hz=vr_frequency,
    )
    row["Vr_selected_frequency_hz"] = vr_frequency
    row["Vtotal_frequency_source"] = "Vr"
    vtotal_nonoverlap_rows.append(row)
    vtotal_nonoverlap_profiles.append(profile)

vtotal_nonoverlap = pd.DataFrame(vtotal_nonoverlap_rows)

if vtotal_nonoverlap_profiles:
    vtotal_phase_delay_profiles = pd.concat(
        vtotal_nonoverlap_profiles,
        ignore_index=True,
    )
else:
    vtotal_phase_delay_profiles = pd.DataFrame()

vtotal_nonoverlap.to_csv(
    OUT / "Vtotal_propagation_windows_nonoverlapping.csv",
    index=False,
)
vtotal_phase_delay_profiles.to_csv(
    OUT / "Vtotal_phase_delay_profiles_nonoverlapping.csv",
    index=False,
)

vtotal_sliding_rows = []

for start in range(
    0,
    len(times) - n_window + 1,
    int(FPS),
):
    # For sliding Vtotal windows, use the Vr-selected frequency from the
    # corresponding sliding Vr window.
    vr_row = sliding.loc[sliding["start_s"] == float(times[start])].iloc[0]
    vr_frequency = float(vr_row["dominant_frequency_hz"])
    row, _ = fit_phase_window_vtotal(
        filtered_vtotal,
        reference_vtotal,
        start,
        n_window,
        PHASE_MAX_RADIUS_UM,
        frequency_hz=vr_frequency,
    )
    row["Vr_selected_frequency_hz"] = vr_frequency
    row["Vtotal_frequency_source"] = "Vr"
    vtotal_sliding_rows.append(row)

vtotal_sliding = pd.DataFrame(vtotal_sliding_rows)

vtotal_sliding.to_csv(
    OUT / "Vtotal_propagation_windows_sliding.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# Figure 10: Vtotal phase-derived delay
# Same style and plotting logic as Figure 4.
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

colors = ["#1b9e77", "#377eb8", "#999999", "#e41a1c"]

for index, row in vtotal_nonoverlap.iterrows():
    profile = vtotal_phase_delay_profiles[
        vtotal_phase_delay_profiles["window_center_s"] == row["center_s"]
    ]

    label_time = (
        f"{row['start_s']:.0f}–"
        f"{row['end_s'] + 1/FPS:.0f} s"
    )

    if row["accepted_outward_propagation"]:
        label = (
            f"{label_time}: "
            f"{row['signed_phase_speed_um_s']:.0f} µm s⁻¹"
        )

        plot_profile = profile[profile["plot_raw_delay"]]

        ax.plot(
            plot_profile["radius_um"],
            plot_profile["phase_derived_delay_s"],
            color=colors[index % len(colors)],
            alpha=0.45,
            lw=1,
        )

        ax.plot(
            profile["radius_um"],
            profile["fitted_delay_s"],
            color=colors[index % len(colors)],
            lw=2.2,
            label=label,
        )

ax.axvline(
    BOUNDARY_UM,
    color="#ff4fb3",
    lw=1.6,
    ls=(0, (2, 2)),
)
ax.axhline(
    0,
    color="#555555",
    lw=0.8,
)
ax.set(
    title="Vtotal phase-derived delay (negative raw delays omitted from plot)",
    xlabel="Radius (µm)",
    ylabel="Delay relative to first external annulus (s)",
)
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.2)
fig.tight_layout()
save_figure(fig, "Figure_10_Vtotal_phase_derived_delay.png", dpi=300)
plt.close(fig)

# Figure 1: radial-velocity kymograph
# ------------------------------------------------------------
# Display-only smoothing of the filtered Vr signal.
velocity_display = gaussian_filter(
    filtered_vr,
    sigma=(0.75, 1.0),
)
vmax = float(
    np.percentile(
        np.abs(velocity_display[:, radii <= min(60.0, np.max(radii))]),
        98.5,
    )
)
vmax = max(vmax, 1e-12)
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(
    velocity_display.T,
    origin="lower",
    aspect="auto",
    extent=[times[0], times[-1], radii[0], radii[-1]],
    cmap="PuOr_r",
    vmin=-vmax,
    vmax=vmax,
    interpolation="bilinear",
)
ax.axhline(BOUNDARY_UM, color="#ff4fb3", lw=2, ls=(0, (2, 2)))
ax.set(
    title="Radial-velocity kymograph",
    xlabel="Time (s)",
    ylabel="Radius (µm)",
)
fig.colorbar(im, ax=ax, label="Radial velocity (µm s$^{-1}$)")
fig.tight_layout()
save_figure(fig, "Figure_1_radial_velocity_kymograph.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------
# Figure 2: oscillatory motion amplitude
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(radii, rms_amplitude, color="#1f78b4", lw=2)
ax.axvspan(
    0,
    BOUNDARY_UM,
    color="#ff4fb3",
    alpha=0.13,
    label="Illuminated region",
)
ax.axvline(
    BOUNDARY_UM,
    color="#ff4fb3",
    lw=1.6,
    ls=(0, (2, 2)),
)
ax.set(
    title="Oscillatory motion amplitude",
    xlabel="Radius (µm)",
    ylabel="RMS radial velocity (µm s$^{-1}$)",
)
ax.legend(frameon=False)
ax.grid(alpha=0.2)
fig.tight_layout()
save_figure(fig, "Figure_2_oscillatory_motion_amplitude.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------
# Figure 3: phase locking
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(radii, phase_locking, color="#33a02c", lw=2)
ax.axvspan(
    0,
    BOUNDARY_UM,
    color="#ff4fb3",
    alpha=0.13,
)
ax.axvline(
    BOUNDARY_UM,
    color="#ff4fb3",
    lw=1.6,
    ls=(0, (2, 2)),
)
ax.axhline(
    0.5,
    color="#666666",
    lw=1,
    ls="--",
)
ax.set(
    title="Phase locking to the illuminated region",
    xlabel="Radius (µm)",
    ylabel="Phase-locking value",
)
ax.set_ylim(0, 1.03)
ax.grid(alpha=0.2)
fig.tight_layout()
save_figure(fig, "Figure_3_phase_locking.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------
# Figure 4: phase-derived delay
#
# IMPORTANT:
# Negative raw delays are removed ONLY from this plot.
# They are NOT used as an acceptance/rejection criterion.
# The fitted delay remains unchanged.
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

colors = ["#1b9e77", "#377eb8", "#999999", "#e41a1c"]

for index, row in nonoverlap.iterrows():
    profile = phase_delay_profiles[
        phase_delay_profiles["window_center_s"] == row["center_s"]
    ]

    label_time = (
        f"{row['start_s']:.0f}–"
        f"{row['end_s'] + 1/FPS:.0f} s"
    )

    if row["accepted_outward_propagation"]:
        label = (
            f"{label_time}: "
            f"{row['signed_phase_speed_um_s']:.0f} µm s⁻¹"
        )

        # Only the negative raw-delay points are omitted from plotting.
        plot_profile = profile[profile["plot_raw_delay"]]

        ax.plot(
            plot_profile["radius_um"],
            plot_profile["phase_derived_delay_s"],
            color=colors[index % len(colors)],
            alpha=0.45,
            lw=1,
        )

        # Fitted delay is plotted completely; it is not refit after
        # removing negative raw-delay points.
        ax.plot(
            profile["radius_um"],
            profile["fitted_delay_s"],
            color=colors[index % len(colors)],
            lw=2.2,
            label=label,
        )

ax.axvline(
    BOUNDARY_UM,
    color="#ff4fb3",
    lw=1.6,
    ls=(0, (2, 2)),
)
ax.axhline(
    0,
    color="#555555",
    lw=0.8,
)
ax.set(
    title="Phase-derived delay (negative raw delays omitted from plot)",
    xlabel="Radius (µm)",
    ylabel="Delay relative to first external annulus (s)",
)
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.2)
fig.tight_layout()
save_figure(fig, "Figure_4_phase_derived_delay.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------
# Figure 5: phase speed from accepted sliding windows
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
if len(sliding):
    accepted_sliding = sliding[
        sliding["accepted_outward_propagation"]
    ]
    if len(accepted_sliding):
        ax.plot(
            accepted_sliding["center_s"],
            accepted_sliding["signed_phase_speed_um_s"],
            marker="o",
            lw=1.5,
        )

ax.set(
    title="Phase-propagation speed over time",
    xlabel="Window center time (s)",
    ylabel="Phase speed (µm s$^{-1}$)",
)
ax.grid(alpha=0.2)
fig.tight_layout()
save_figure(fig, "Figure_5_phase_speed_over_time.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------
# Figure 6: parameter sensitivity
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
for maximum_radius in sorted(
    sensitivity["phase_fit_maximum_radius_um"].unique()
):
    subset = sensitivity[
        sensitivity["phase_fit_maximum_radius_um"]
        == maximum_radius
    ]
    ax.plot(
        subset["window_s"],
        subset["median_phase_speed_um_s"],
        marker="o",
        lw=1.5,
        label=f"{maximum_radius:.0f} µm",
    )

ax.set(
    title="Phase-speed sensitivity to analysis parameters",
    xlabel="Window duration (s)",
    ylabel="Median phase speed (µm s$^{-1}$)",
)
ax.legend(frameon=False, title="Maximum fit radius")
ax.grid(alpha=0.2)
fig.tight_layout()
save_figure(fig, "Figure_6_phase_speed_sensitivity.png", dpi=300)
plt.close(fig)


# ============================================================
# FIGURE 12: RAW / UNFILTERED Vtotal Kymograph
# ============================================================

# Vtotal was calculated once from the original PIV components:
#     Vtotal = sqrt(Vr^2 + Vtheta^2)
# No band-pass filtering is applied to Vtotal here.
# The saved CSV contains the original interpolated values. A small Gaussian
# smoothing is used ONLY for display of the image.
vtotal_raw_kymograph_source = pd.DataFrame({
    "time_s": np.repeat(times, len(radii)),
    "radius_um": np.tile(radii, len(times)),
    "raw_total_velocity_um_s": velocity_vtotal.reshape(-1),
})
vtotal_raw_kymograph_source.to_csv(
    OUT / "total_velocity_kymograph_raw_plotting_data.csv",
    index=False,
)

vtotal_raw_display = gaussian_filter(
    velocity_vtotal,
    sigma=(0.75, 1.0),
)

vtotal_raw_vmax = float(
    np.percentile(
        vtotal_raw_display[:, radii <= min(60.0, np.max(radii))],
        98.5,
    )
)

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(
    vtotal_raw_display.T,
    origin="lower",
    aspect="auto",
    extent=[times[0], times[-1], radii[0], radii[-1]],
    cmap="viridis",
    vmin=0,
    vmax=vtotal_raw_vmax,
    interpolation="bilinear",
)
# # ax.axhline(BOUNDARY_UM, color="#ff4fb3", lw=2, ls=(0, (2, 2)))
# ax.set(
#     title="Unfiltered total-velocity kymograph",
#     xlabel="Time (s)",
#     ylabel="Radius (µm)",
# )
# fig.colorbar(im, ax=ax, label="Total velocity (µm s$^{-1}$)")
ax.axis("off")
fig.tight_layout()
save_figure(fig, "Figure_12_total_velocity_kymograph_raw_no_axis.png", dpi=300)
plt.close(fig)


# ============================================================
# FIGURES 7–9: PARALLEL Vtotal ANALYSIS
# ============================================================

# Figure 7: Vtotal oscillatory amplitude.
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(
    radii,
    raw_rms_vtotal,
    color="#756bb1",
    lw=2,
)
ax.axvspan(
    0,
    BOUNDARY_UM,
    color="#ff4fb3",
    alpha=0.13,
    label="Illuminated region",
)
ax.axvline(
    BOUNDARY_UM,
    color="#ff4fb3",
    lw=1.6,
    ls=(0, (2, 2)),
)
ax.set(
    title="Unfiltered total-velocity RMS amplitude",
    xlabel="Radius (µm)",
    ylabel="Raw RMS Vtotal (µm s$^{-1}$)",
)
ax.legend(frameon=False)
ax.grid(alpha=0.2)
fig.tight_layout()
save_figure(fig, "Figure_7_Vtotal_oscillatory_amplitude.png", dpi=300)
plt.close(fig)

# Figure 8: Vtotal phase locking.
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(
    radii,
    phase_locking_vtotal,
    color="#756bb1",
    lw=2,
)
ax.axvspan(
    0,
    BOUNDARY_UM,
    color="#ff4fb3",
    alpha=0.13,
)
ax.axvline(
    BOUNDARY_UM,
    color="#ff4fb3",
    lw=1.6,
    ls=(0, (2, 2)),
)
ax.axhline(
    0.5,
    color="#666666",
    lw=1,
    ls="--",
)
ax.set(
    title="Vtotal phase locking to the illuminated region",
    xlabel="Radius (µm)",
    ylabel="Vtotal phase-locking value",
)
ax.set_ylim(0, 1.03)
ax.grid(alpha=0.2)
fig.tight_layout()
save_figure(fig, "Figure_8_Vtotal_phase_locking.png", dpi=300)
plt.close(fig)

# Figure 9: direct comparison of Vr and Vtotal RMS amplitudes.
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(
    radii,
    raw_rms_vr,
    lw=2,
    label="Vr (raw)",
)
ax.plot(
    radii,
    raw_rms_vtotal,
    lw=2,
    label="Vtotal (raw)",
)
ax.axvline(
    BOUNDARY_UM,
    color="#ff4fb3",
    lw=1.6,
    ls=(0, (2, 2)),
)
ax.set(
    title="Unfiltered radial vs total velocity RMS",
    xlabel="Radius (µm)",
    ylabel="Raw RMS velocity (µm s$^{-1}$)",
)
ax.legend(frameon=False)
ax.grid(alpha=0.2)
fig.tight_layout()
save_figure(fig, "Figure_9_raw_Vr_vs_Vtotal_RMS.png", dpi=300)
plt.close(fig)

accepted_nonoverlap = nonoverlap[nonoverlap["accepted_outward_propagation"]]
accepted_speeds = accepted_nonoverlap["signed_phase_speed_um_s"].to_numpy()
endpoint_delays = accepted_nonoverlap["phase_delay_to_outermost_fitted_radius_s"].to_numpy()
effective_endpoint_speeds = accepted_nonoverlap["effective_boundary_to_outer_radius_speed_um_s"].to_numpy()
accepted_vtotal = vtotal_nonoverlap[
    vtotal_nonoverlap["accepted_outward_propagation"]
]
accepted_vtotal_speeds = (
    accepted_vtotal["signed_phase_speed_um_s"].to_numpy()
)

summary = {
    "primary_propagation_variable": "Vr",
    "parallel_total_velocity_analysis": True,
    "total_velocity_is_filtered": False,
    "video": "small circle.avi",
    "duration_s": float(times[-1]),
    "illumination_diameter_um": ILLUMINATION_DIAMETER_UM,
    "illumination_boundary_radius_um": BOUNDARY_UM,
    "radial_reference_um": [REFERENCE_MIN_UM, BOUNDARY_UM],
    "oscillation_band_hz": list(BAND_HZ),
    "independent_5s_windows": int(len(nonoverlap)),
    "windows_passing_propagation_screen": int(len(accepted_nonoverlap)),
    "accepted_phase_speeds_um_s": accepted_speeds.tolist(),
    "median_phase_speed_um_s": float(np.median(accepted_speeds)),
    "minimum_phase_speed_um_s": float(np.min(accepted_speeds)),
    "maximum_phase_speed_um_s": float(np.max(accepted_speeds)),
    "phase_delay_from_boundary_adjacent_annulus_to_54_77um_s": endpoint_delays.tolist(),
    "median_phase_delay_to_54_77um_s": float(np.median(endpoint_delays)),
    "effective_boundary_to_54_77um_speeds_um_s": effective_endpoint_speeds.tolist(),
    "median_effective_boundary_to_54_77um_speed_um_s": float(np.median(effective_endpoint_speeds)),

    # Parallel RAW Vtotal phase-analysis results.
    "vtotal_windows_passing_propagation_screen": int(len(accepted_vtotal)),
    "vtotal_phase_speeds_um_s": accepted_vtotal_speeds.tolist(),
    "vtotal_median_phase_speed_um_s": (
        float(np.median(accepted_vtotal_speeds))
        if len(accepted_vtotal_speeds)
        else math.nan
    ),

    "synthetic_known_speed_um_s": known_speed,
    "synthetic_accepted_fraction": float(len(synthetic_accepted) / len(synthetic_validation)),
    "synthetic_median_recovered_speed_um_s": float(np.median(synthetic_accepted["recovered_phase_speed_um_s"])),
    "synthetic_median_absolute_error_um_s": float(np.median(np.abs(synthetic_accepted["recovered_phase_speed_um_s"] - known_speed))),
}

