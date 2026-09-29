# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:25:05 2026

@author: sgao
"""

"""
texture filter
PIV analysis + independent original/filtered PIV saving
+ texture analysis + signed radial velocity Vr + tangential velocity Vtheta + magnitude images
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from openpyxl import Workbook
import os
import matplotlib
matplotlib.use("Agg")
# ============================================================
# OPEN FOLDER
# ============================================================
os.chdir("C:/Users/sgao/OneDrive - ICIQ/Documents/SG/2025/PIV/Yufen/Videos/No.2")

# ============================================================
# CREATE PARULA COLORMAP
# ============================================================
def create_parula_colormap():
    parula_data = [
        [0.2081,0.1663,0.5292],[0.2116,0.1898,0.5777],
        [0.2123,0.2138,0.6270],[0.2081,0.2386,0.6771],
        [0.1959,0.2645,0.7279],[0.1707,0.2919,0.7792],
        [0.1253,0.3242,0.8303],[0.0591,0.3598,0.8683],
        [0.0117,0.3875,0.8820],[0.0060,0.4086,0.8828],
        [0.0165,0.4266,0.8786],[0.0329,0.4430,0.8720],
        [0.0498,0.4586,0.8641],[0.0629,0.4737,0.8554],
        [0.0723,0.4887,0.8467],[0.0779,0.5040,0.8384],
        [0.0793,0.5200,0.8312],[0.0749,0.5375,0.8263],
        [0.0641,0.5570,0.8240],[0.0488,0.5772,0.8228],
        [0.0343,0.5966,0.8199],[0.0265,0.6137,0.8135],
        [0.0239,0.6287,0.8038],[0.0231,0.6418,0.7913],
        [0.0228,0.6535,0.7768],[0.0267,0.6642,0.7608],
        [0.0384,0.6743,0.7436],[0.0590,0.6838,0.7254],
        [0.0843,0.6928,0.7062],[0.1133,0.7015,0.6859],
        [0.1453,0.7098,0.6646],[0.1801,0.7177,0.6424],
        [0.2178,0.7250,0.6193],[0.2586,0.7317,0.5954],
        [0.3022,0.7376,0.5712],[0.3482,0.7424,0.5473],
        [0.3953,0.7459,0.5244],[0.4420,0.7481,0.5033],
        [0.4871,0.7491,0.4840],[0.5300,0.7491,0.4665],
        [0.5709,0.7485,0.4494],[0.6099,0.7473,0.4327],
        [0.6473,0.7456,0.4165],[0.6834,0.7435,0.4007],
        [0.7184,0.7411,0.3854],[0.7525,0.7384,0.3706],
        [0.7858,0.7356,0.3565],[0.8185,0.7327,0.3430],
        [0.8507,0.7299,0.3302],[0.8824,0.7274,0.3181],
        [0.9139,0.7258,0.3067],[0.9450,0.7261,0.2963],
        [0.9739,0.7314,0.2875],[0.9938,0.7455,0.2809],
        [0.9990,0.7653,0.2771],[0.9955,0.7861,0.2770],
        [0.9880,0.8066,0.2805],[0.9789,0.8271,0.2875],
        [0.9697,0.8481,0.2975],[0.9626,0.8705,0.3100],
        [0.9589,0.8949,0.3246],[0.9598,0.9218,0.3396],
        [0.9661,0.9514,0.3538],[0.9763,0.9831,0.3665]
    ]
    return LinearSegmentedColormap.from_list("parula", parula_data)

parula_cmap = create_parula_colormap()

# ============================================================
# PIV FUNCTION
# ============================================================
def piv_corrected(frame_a, frame_b, window_size=64, overlap=0.75, search_size=8):
    height, width = frame_a.shape
    step = int(window_size * (1 - overlap))

    y_coords = np.arange(0, height - window_size, step)
    x_coords = np.arange(0, width - window_size, step)

    vectors = np.zeros((len(y_coords),len(x_coords),2))
    correlation_scores = np.zeros((len(y_coords),len(x_coords)))
    window_std = np.zeros((len(y_coords),len(x_coords)))

    window_sums_a = np.zeros((len(y_coords),len(x_coords)))
    window_sums_sq_a = np.zeros((len(y_coords),len(x_coords)))

    for i,y in enumerate(y_coords):
        for j,x in enumerate(x_coords):
            window_a = frame_a[y:y+window_size,x:x+window_size]
            window_sums_a[i,j] = np.sum(window_a)
            window_sums_sq_a[i,j] = np.sum(window_a**2)
            window_std[i,j] = np.std(window_a)

    dy_range = np.arange(-search_size,search_size+1)
    dx_range = np.arange(-search_size,search_size+1)

    for i,y in enumerate(y_coords):
        for j,x in enumerate(x_coords):
            window_a = frame_a[y:y+window_size,x:x+window_size]
            best_corr = -1
            best_dx,best_dy = 0,0

            sum_a = window_sums_a[i,j]
            sum_sq_a = window_sums_sq_a[i,j]
            n_pixels = window_size*window_size

            for dy in dy_range:
                y_shifted = y+dy
                if 0 <= y_shifted < height-window_size:
                    for dx in dx_range:
                        x_shifted = x+dx
                        if 0 <= x_shifted < width-window_size:
                            window_b = frame_b[
                                y_shifted:y_shifted+window_size,
                                x_shifted:x_shifted+window_size
                            ]

                            sum_b = np.sum(window_b)
                            sum_sq_b = np.sum(window_b**2)
                            sum_ab = np.sum(window_a*window_b)

                            numerator = sum_ab-(sum_a*sum_b)/n_pixels
                            denominator = np.sqrt(
                                (sum_sq_a-sum_a**2/n_pixels)*
                                (sum_sq_b-sum_b**2/n_pixels)
                            )

                            if denominator > 1e-10:
                                corr_val = numerator/denominator
                                if corr_val > best_corr:
                                    best_corr = corr_val
                                    best_dx = dx
                                    best_dy = dy

            vectors[i,j] = [best_dx,best_dy]
            correlation_scores[i,j] = best_corr

    return vectors,x_coords,y_coords,correlation_scores,window_std

# ============================================================
# FILTER LARGE DISPLACEMENTS
# ============================================================
def filter_large_displacements(vectors, threshold_high=5):
    displacement_magnitude = np.sqrt(
        vectors[:,:,0]**2+vectors[:,:,1]**2
    )

    large_displacement_mask = displacement_magnitude > threshold_high

    vectors_filtered = vectors.copy()
    vectors_filtered[large_displacement_mask] = np.nan

    print(
        f"Filtered {np.sum(large_displacement_mask)} "
        f"vectors with displacement > {threshold_high} pixels"
    )

    return vectors_filtered,large_displacement_mask,displacement_magnitude

# ============================================================
# PARAMETERS
# ============================================================
window_size = 32
overlap = 0.5
search_size = 12
FPS = 25
size_px = 0.2059
sigma = 0.9

start = 1
end = 376 # 15s
interval = 1

displacement_threshold_h = 10
sigma1 = 7

# ============================================================
# TEXTURE PARAMETER
# ============================================================
texture_threshold = 0.1

# Set initially to 0.0 so no texture filtering is applied.
# After checking the histogram, change this value.

# ============================================================
# RADIAL PARAMETERS
# ============================================================
max_radius_px = 600
radial_bin_width_px = 5.0
radial_bin_width = radial_bin_width_px*size_px

# ============================================================
# MAIN PIV LOOP
# ============================================================
for idx in range(start,end):

    n1,n2 = idx,idx+interval

    print("\n"+"="*60)
    print(f"Processing frames {n1} → {n2}")
    print("="*60)

    # ========================================================
    # LOAD IMAGES
    # ========================================================
    fr1 = cv2.imread(f"Crop frame {n1}.png")
    fr2 = cv2.imread(f"Crop frame {n2}.png")

    if fr1 is None or fr2 is None:
        print(f"Skipping frames {n1} → {n2} (files not found)")
        continue

    # ========================================================
    # IMAGE DIMENSIONS AND CENTER
    # ========================================================
    h_orig,w_orig = fr1.shape[:2]

    center_x_px = w_orig/2
    center_y_px = h_orig/2

    # ========================================================
    # GRAYSCALE
    # ========================================================
    fr1_g = cv2.cvtColor(fr1,cv2.COLOR_BGR2GRAY)
    fr2_g = cv2.cvtColor(fr2,cv2.COLOR_BGR2GRAY)

    fr1_norm = fr1_g.astype(np.float32)
    fr2_norm = fr2_g.astype(np.float32)

    global_mean = (np.mean(fr1_norm)+np.mean(fr2_norm))/2
    global_std = (np.std(fr1_norm)+np.std(fr2_norm))/2

    if global_std == 0:
        global_std = 1.0

    fr1_norm = (fr1_norm-global_mean)/global_std
    fr2_norm = (fr2_norm-global_mean)/global_std

    # ========================================================
    # RUN PIV
    # ========================================================
    (
        vectors_original,
        x_coords,
        y_coords,
        correlation,
        window_std
    ) = piv_corrected(
        fr1_norm,
        fr2_norm,
        window_size=window_size,
        overlap=overlap,
        search_size=search_size
    )

    print(f"Grid: {len(y_coords)} × {len(x_coords)} vectors")

    # ========================================================
    # TEXTURE HISTOGRAM
    # ========================================================
    valid_std = window_std[np.isfinite(window_std)]

    plt.figure(figsize=(8,5))
    plt.hist(
        valid_std,
        bins=50,
        edgecolor="black"
    )

    if texture_threshold > 0:
        plt.axvline(
            texture_threshold,
            linestyle="--",
            linewidth=2,
            label=f"Threshold = {texture_threshold:.3f}"
        )
        plt.legend()

    plt.xlabel("Interrogation-window standard deviation")
    plt.ylabel("Number of PIV windows")
    plt.title(f"Texture distribution: frame {n1} → {n2}")
    plt.tight_layout()

    # plt.savefig(
    #     f"Texture_histogram_{n1} "
    #     f"w_{window_size} s_{search_size} "
    #     f"dth_{displacement_threshold_h}.png",
    #     dpi=200,
    #     bbox_inches="tight"
    # )

    plt.close()

    # ========================================================
    # TEXTURE FILTER
    # ========================================================
    texture_mask = window_std < texture_threshold

    print(
        f"Texture threshold = {texture_threshold:.3f}"
    )
    print(
        f"Low-texture windows = {np.sum(texture_mask)} "
        f"({100*np.mean(texture_mask):.2f}%)"
    )

    # ========================================================
    # FILTER LARGE DISPLACEMENTS
    # ========================================================
    (
        vectors_filtered,
        large_disp_mask,
        displacement_magnitude
    ) = filter_large_displacements(
        vectors_original,
        threshold_high=displacement_threshold_h
    )

    # ========================================================
    # COMBINED FILTER
    # ========================================================
    invalid_mask = (
        large_disp_mask |
        texture_mask
    )

    vectors_filtered[invalid_mask] = 0#np.nan

    print(
        f"Total rejected vectors = "
        f"{np.sum(invalid_mask)} "
        f"({100*np.mean(invalid_mask):.2f}%)"
    )

    # ========================================================
    # TIME
    # ========================================================
    dt = interval/FPS
    time_s = n1/FPS

    # ========================================================
    # CONVERT TO µm
    # ========================================================
    original_vectors_um = vectors_original*size_px
    filtered_vectors_um = vectors_filtered*size_px

    max_radius = (
        min(max_radius_px,min(h_orig//2,w_orig//2))
        *size_px
    )

    # ========================================================
    # CONVERT TO VELOCITY
    # ========================================================
    original_velocity = original_vectors_um/dt
    filtered_velocity = filtered_vectors_um/dt

    # ========================================================
    # PIV GRID
    # ========================================================
    X,Y = np.meshgrid(
        x_coords+window_size//2,
        y_coords+window_size//2
    )

    X_um = X*size_px
    Y_um = Y*size_px

    # ========================================================
    # ORIGINAL VELOCITY
    # ========================================================
    U_original = original_velocity[:,:,0]
    V_original = original_velocity[:,:,1]

    original_speed = np.sqrt(
        U_original**2+V_original**2
    )

    # ========================================================
    # FILTERED VELOCITY
    # ========================================================
    U_filtered = filtered_velocity[:,:,0]
    V_filtered = filtered_velocity[:,:,1]

    filtered_speed = np.sqrt(
        U_filtered**2+V_filtered**2
    )

    # ========================================================
    # RADIAL DISTANCE
    # ========================================================
    center_x_um = center_x_px*size_px
    center_y_um = center_y_px*size_px

    dx_um = X_um-center_x_um
    dy_um = Y_um-center_y_um

    R_um = np.sqrt(
        dx_um**2+dy_um**2
    )

    # ========================================================
    # RADIAL UNIT VECTOR
    # ========================================================
    er_x = np.divide(
        dx_um,
        R_um,
        out=np.zeros_like(R_um),
        where=R_um > 0
    )

    er_y = np.divide(
        dy_um,
        R_um,
        out=np.zeros_like(R_um),
        where=R_um > 0
    )

    # ========================================================
    # TANGENTIAL UNIT VECTOR
    # ========================================================
    e_theta_x = -er_y
    e_theta_y = er_x

    # ========================================================
    # SIGNED RADIAL VELOCITY Vr
    # ========================================================
    radial_velocity = (
        U_filtered*er_x+
        V_filtered*er_y
    )

    # ========================================================
    # SIGNED TANGENTIAL VELOCITY Vtheta
    # ========================================================
    tangential_velocity = (
        U_filtered*e_theta_x +
        V_filtered*e_theta_y
    )

    # ========================================================
    # DATA STORAGE FOR THIS FRAME
    # ========================================================
    piv_original_data = []
    piv_filtered_data = []
    radial_data = []

    # ========================================================
    # ORIGINAL PIV DATA
    # ========================================================
    for i in range(len(y_coords)):
        for j in range(len(x_coords)):
            piv_original_data.append([
                n1,
                time_s,
                X[i,j],
                Y[i,j],
                X_um[i,j],
                Y_um[i,j],
                R_um[i,j],
                U_original[i,j],
                V_original[i,j],
                original_speed[i,j],
                displacement_magnitude[i,j],
                correlation[i,j],
                window_std[i,j]
            ])

    # ========================================================
    # FILTERED PIV DATA
    # ========================================================
    for i in range(len(y_coords)):
        for j in range(len(x_coords)):
            if invalid_mask[i,j]:
                continue

            piv_filtered_data.append([
                n1,
                time_s,
                X[i,j],
                Y[i,j],
                X_um[i,j],
                Y_um[i,j],
                R_um[i,j],
                U_filtered[i,j],
                V_filtered[i,j],
                filtered_speed[i,j],
                radial_velocity[i,j],
                tangential_velocity[i,j],
                displacement_magnitude[i,j],
                correlation[i,j],
                window_std[i,j]
            ])

    # ========================================================
    # RADIAL Vr AND Vtheta
    # ========================================================
    radial_edges = np.arange(
        0,
        max_radius+radial_bin_width,
        radial_bin_width
    )

    for bin_start,bin_end in zip(
        radial_edges[:-1],
        radial_edges[1:]
    ):

        radial_mask = (
            (R_um >= bin_start) &
            (R_um < bin_end) &
            (~invalid_mask) &
            np.isfinite(radial_velocity) &
            np.isfinite(tangential_velocity)
        )

        vr = radial_velocity[radial_mask]
        vtheta = tangential_velocity[radial_mask]

        if len(vr) == 0:
            mean_vr = np.nan
            std_vr = np.nan
            mean_vtheta = np.nan
            std_vtheta = np.nan
            n_vectors = 0
        else:
            mean_vr = np.mean(vr)
            std_vr = np.std(vr)
            mean_vtheta = np.mean(vtheta)
            std_vtheta = np.std(vtheta)
            n_vectors = len(vr)

        radius_center = (bin_start+bin_end)/2

        radial_data.append([
            n1,
            time_s,
            radius_center,
            mean_vr,
            std_vr,
            mean_vtheta,
            std_vtheta,
            n_vectors
        ])

    # ========================================================
    # FILE NAME
    # ========================================================
    base_name = (
        f"{n1} w_{window_size} "
        f"s_{search_size} dth_{displacement_threshold_h}"
    )

    # # ========================================================
    # # SAVE ORIGINAL PIV
    # # ========================================================
    # wb = Workbook()
    # ws = wb.active
    # ws.title = "PIV_original"

    # ws.append([
    #     "frame",
    #     "time_s",
    #     "x_px",
    #     "y_px",
    #     "x_um",
    #     "y_um",
    #     "radius_um",
    #     "vx_um_s",
    #     "vy_um_s",
    #     "speed_um_s",
    #     "displacement_px",
    #     "correlation",
    #     "window_std"
    # ])

    # for row in piv_original_data:
    #     ws.append(row)

    # wb.save(
    #     f"PIV_original_{base_name}.xlsx"
    # )

    # ========================================================
    # SAVE FILTERED PIV
    # ========================================================
    wb = Workbook()
    ws = wb.active
    ws.title = "PIV_filtered"

    ws.append([
        "frame",
        "time_s",
        "x_px",
        "y_px",
        "x_um",
        "y_um",
        "radius_um",
        "vx_um_s",
        "vy_um_s",
        "speed_um_s",
        "Vr_um_s",
        "Vtheta_um_s",
        "displacement_px",
        "correlation",
        "window_std"
    ])

    for row in piv_filtered_data:
        ws.append(row)

    wb.save(
        f"PIV_filtered_{base_name}.xlsx"
    )

    # ========================================================
    # SAVE RADIAL Vr AND Vtheta
    # ========================================================
    wb = Workbook()
    ws = wb.active
    ws.title = "radial_velocity"

    ws.append([
        "frame",
        "time_s",
        "radius_um",
        "mean_Vr_um_s",
        "std_Vr_um_s",
        "mean_Vtheta_um_s",
        "std_Vtheta_um_s",
        "n_vectors"
    ])

    for row in radial_data:
        ws.append(row)

    wb.save(
        f"radial_velocity_{base_name}.xlsx"
    )

    # ========================================================
    # SPEED STATISTICS
    # ========================================================
    valid_speeds = filtered_speed[
        np.isfinite(filtered_speed)
    ]

    valid_vr = radial_velocity[
        np.isfinite(radial_velocity)
    ]

    valid_vtheta = tangential_velocity[
        np.isfinite(tangential_velocity)
    ]

    if len(valid_speeds) > 0:
        print(
            f"Filtered mean speed = "
            f"{np.mean(valid_speeds):.3f} µm/s"
        )
        print(
            f"Filtered max speed = "
            f"{np.max(valid_speeds):.3f} µm/s"
        )

    if len(valid_vr) > 0:
        print(
            f"Mean Vr = "
            f"{np.mean(valid_vr):.3f} µm/s"
        )
        print(
            f"Max outward Vr = "
            f"{np.max(valid_vr):.3f} µm/s"
        )
        print(
            f"Max inward Vr = "
            f"{np.min(valid_vr):.3f} µm/s"
        )

    if len(valid_vtheta) > 0:
        print(
            f"Mean Vtheta = "
            f"{np.mean(valid_vtheta):.3f} µm/s"
        )
        print(
            f"Max Vtheta = "
            f"{np.max(valid_vtheta):.3f} µm/s"
        )
        print(
            f"Min Vtheta = "
            f"{np.min(valid_vtheta):.3f} µm/s"
        )

    # ========================================================
    # NaN-AWARE SPEED SMOOTHING
    # ========================================================
    valid_speed = np.isfinite(filtered_speed)
    speed_data = np.nan_to_num(
        filtered_speed,
        nan=0.0
    )

    speed_num = gaussian_filter(
        speed_data,
        sigma=sigma
    )

    speed_den = gaussian_filter(
        valid_speed.astype(float),
        sigma=sigma
    )

    magnitude_smooth = speed_num/np.maximum(
        speed_den,
        1e-12
    )

    # ========================================================
    # NaN-AWARE Vr SMOOTHING
    # ========================================================
    valid_vr_mask = np.isfinite(radial_velocity)
    vr_data = np.nan_to_num(
        radial_velocity,
        nan=0.0
    )

    vr_num = gaussian_filter(
        vr_data,
        sigma=sigma
    )

    vr_den = gaussian_filter(
        valid_vr_mask.astype(float),
        sigma=sigma
    )

    radial_velocity_smooth = vr_num/np.maximum(
        vr_den,
        1e-12
    )

    # ========================================================
    # VECTOR FIELD PLOT
    # ========================================================
    skip = 1

    X_sub = X[::skip,::skip]
    Y_sub = Y[::skip,::skip]
    U_sub = U_filtered[::skip,::skip]
    V_sub = V_filtered[::skip,::skip]
    magnitude_sub = filtered_speed[::skip,::skip]

    plt.figure(figsize=(10,8),dpi=150)
    plt.imshow(
        fr1_g,
        cmap="gray",
        alpha=0.4
    )

    Q = plt.quiver(
        X_sub,
        Y_sub,
        U_sub,
        V_sub,
        magnitude_sub,
        cmap="jet",
        angles="xy",
        scale_units="xy",
        scale=0.6,
        width=0.004,
        headwidth=3,
        headlength=4,
        alpha=1
    )

    plt.colorbar(
        Q,
        label="Speed (µm/s)"
    )

    plt.clim(0,20)
    plt.tight_layout()
    plt.axis("off")

    plt.savefig(
        f"P Vector_Field_{n1} "
        f"w_{window_size} s_{search_size} "
        f"dth_{displacement_threshold_h}  tt{texture_threshold} "
        f"in_{interval}.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # # ========================================================
    # # MAGNITUDE PLOT
    # # ========================================================
    # plt.figure(figsize=(8,6))

    # plt.imshow(
    #     magnitude_smooth,
    #     cmap=parula_cmap,
    #     vmin=0,
    #     vmax=17.5,
    #     interpolation="bilinear"
    # )

    # plt.colorbar(
    #     label="Speed (µm/s)"
    # )

    # plt.tight_layout()
    # plt.axis("off")

    # plt.savefig(
    #     f"P full_magnitude_{n1} "
    #     f"w_{window_size} s_{search_size} "
    #     f"dth_{displacement_threshold_h}  tt{texture_threshold}.png",
    #     dpi=150,
    #     bbox_inches="tight"
    # )

    # plt.close()

    # # ========================================================
    # # RADIAL Vr PLOT
    # # ========================================================
    # finite_vr = radial_velocity_smooth[
    #     np.isfinite(radial_velocity_smooth)
    # ]

    # if len(finite_vr) > 0:
    #     vr_limit = np.percentile(
    #         np.abs(finite_vr),
    #         98
    #     )
    #     vr_limit = max(vr_limit,1e-12)
    # else:
    #     vr_limit = 1

    # plt.figure(figsize=(8,6))

    # plt.imshow(
    #     radial_velocity_smooth,
    #     cmap="RdBu_r",
    #     vmin=-vr_limit,
    #     vmax=vr_limit,
    #     interpolation="bilinear"
    # )

    # plt.colorbar(
    #     label="Radial velocity $V_r$ (µm/s)"
    # )

    # plt.tight_layout()
    # plt.axis("off")

    # plt.savefig(
    #     f"P radial_velocity_{n1} "
    #     f"w_{window_size} s_{search_size} "
    #     f"dth_{displacement_threshold_h} tt{texture_threshold}.png",
    #     dpi=150,
    #     bbox_inches="tight"
    # )

    # plt.close()

    # ========================================================
    # EXPANDED MAGNITUDE PLOT
    # ========================================================
    h,w=filtered_speed.shape
    target_size=1024
    
    x_old=np.linspace(0,w-1,w)
    y_old=np.linspace(0,h-1,h)
    x_new=np.linspace(0,w-1,target_size)
    y_new=np.linspace(0,h-1,target_size)
    
    valid=np.isfinite(filtered_speed)
    speed0=np.nan_to_num(filtered_speed,nan=0.0)
    
    from scipy.interpolate import RegularGridInterpolator
    
    interp_speed=RegularGridInterpolator((y_old,x_old),speed0,bounds_error=False,fill_value=0)
    interp_valid=RegularGridInterpolator((y_old,x_old),valid.astype(float),bounds_error=False,fill_value=0)
    
    yy,xx=np.meshgrid(y_new,x_new,indexing="ij")
    points=np.column_stack((yy.ravel(),xx.ravel()))
    
    im_expanded=interp_speed(points).reshape(target_size,target_size)
    valid_expanded=interp_valid(points).reshape(target_size,target_size)
    
    smooth_num=gaussian_filter(im_expanded*valid_expanded,sigma=sigma1)
    smooth_den=gaussian_filter(valid_expanded,sigma=sigma1)
    
    im_smooth=smooth_num/np.maximum(smooth_den,1e-12)
    
    plt.imsave(
        f"Expanded magnitude_{n1} w_{window_size} s_{search_size} dth_{displacement_threshold_h} tt{texture_threshold}.png",
        im_smooth,cmap=parula_cmap,vmin=0,vmax=20
    )
    # ========================================================
    # SUMMARY
    # ========================================================
    print("\nSaved:")
    print(f"  PIV_original_{base_name}.xlsx")
    print(f"  PIV_filtered_{base_name}.xlsx")
    print(f"  radial_velocity_{base_name}.xlsx")
    print(f"  Texture_histogram_{n1} ...png")
    print(f"  P Vector_Field_{n1} ...png")
    print(f"  P full_magnitude_{n1} ...png")
    print(f"  P radial_velocity_{n1} ...png")
    print(f"  Expanded magnitude_{n1} ...png")

print("\n"+"="*60)
print("ALL PROCESSING COMPLETE")
print("="*60)
