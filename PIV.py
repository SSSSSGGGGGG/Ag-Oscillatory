# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 01:16:12 2025
@author: SG
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import os

"""Open folder"""
os.chdir("C:/Users/sgao/OneDrive - ICIQ/Documents/SG/2025/PIV/Yufen/Videos/No.2")

"""Create Parula colormap"""
def create_parula_colormap():
    
    parula_data = [
        [0.2081, 0.1663, 0.5292], [0.2116, 0.1898, 0.5777], [0.2123, 0.2138, 0.6270],
        [0.2081, 0.2386, 0.6771], [0.1959, 0.2645, 0.7279], [0.1707, 0.2919, 0.7792],
        [0.1253, 0.3242, 0.8303], [0.0591, 0.3598, 0.8683], [0.0117, 0.3875, 0.8820],
        [0.0060, 0.4086, 0.8828], [0.0165, 0.4266, 0.8786], [0.0329, 0.4430, 0.8720],
        [0.0498, 0.4586, 0.8641], [0.0629, 0.4737, 0.8554], [0.0723, 0.4887, 0.8467],
        [0.0779, 0.5040, 0.8384], [0.0793, 0.5200, 0.8312], [0.0749, 0.5375, 0.8263],
        [0.0641, 0.5570, 0.8240], [0.0488, 0.5772, 0.8228], [0.0343, 0.5966, 0.8199],
        [0.0265, 0.6137, 0.8135], [0.0239, 0.6287, 0.8038], [0.0231, 0.6418, 0.7913],
        [0.0228, 0.6535, 0.7768], [0.0267, 0.6642, 0.7607], [0.0384, 0.6743, 0.7436],
        [0.0590, 0.6838, 0.7254], [0.0843, 0.6928, 0.7062], [0.1133, 0.7015, 0.6859],
        [0.1453, 0.7098, 0.6646], [0.1801, 0.7177, 0.6424], [0.2178, 0.7250, 0.6193],
        [0.2586, 0.7317, 0.5954], [0.3022, 0.7376, 0.5712], [0.3482, 0.7424, 0.5473],
        [0.3953, 0.7459, 0.5244], [0.4420, 0.7481, 0.5033], [0.4871, 0.7491, 0.4840],
        [0.5300, 0.7491, 0.4665], [0.5709, 0.7485, 0.4494], [0.6099, 0.7473, 0.4327],
        [0.6473, 0.7456, 0.4165], [0.6834, 0.7435, 0.4007], [0.7184, 0.7411, 0.3854],
        [0.7525, 0.7384, 0.3706], [0.7858, 0.7356, 0.3565], [0.8185, 0.7327, 0.3430],
        [0.8507, 0.7299, 0.3302], [0.8824, 0.7274, 0.3181], [0.9139, 0.7258, 0.3067],
        [0.9450, 0.7261, 0.2963], [0.9739, 0.7314, 0.2875], [0.9938, 0.7455, 0.2809],
        [0.9990, 0.7653, 0.2771], [0.9955, 0.7861, 0.2770], [0.9880, 0.8066, 0.2805],
        [0.9789, 0.8271, 0.2875], [0.9697, 0.8481, 0.2975], [0.9626, 0.8705, 0.3100],
        [0.9589, 0.8949, 0.3246], [0.9598, 0.9218, 0.3396], [0.9661, 0.9514, 0.3538],
        [0.9763, 0.9831, 0.3665]
    ]
    return LinearSegmentedColormap.from_list('parula', parula_data)

parula_cmap = create_parula_colormap()

"""Create PIV function"""
def piv_corrected(frame_a, frame_b, window_size=64, overlap=0.75, search_size=8):
    
    height, width = frame_a.shape
    
    # Calculate step size
    step = int(window_size * (1 - overlap))
    
    # Create grid coordinates
    y_coords = np.arange(0, height - window_size, step)
    x_coords = np.arange(0, width - window_size, step)
    
    vectors = np.zeros((len(y_coords), len(x_coords), 2))
    correlation_scores = np.zeros((len(y_coords), len(x_coords)))
    
    # Pre-compute all window sums for frame_a for efficiency
    window_sums_a = np.zeros((len(y_coords), len(x_coords)))
    window_sums_sq_a = np.zeros((len(y_coords), len(x_coords)))
    
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            window_a = frame_a[y:y+window_size, x:x+window_size]
            window_sums_a[i, j] = np.sum(window_a)
            window_sums_sq_a[i, j] = np.sum(window_a**2)
    
    # OPTIMIZED: Use list comprehension and pre-allocated search ranges
    dy_range = np.arange(-search_size, search_size + 1)
    dx_range = np.arange(-search_size, search_size + 1)
    
    for i, y in enumerate(y_coords):
        # Pre-extract row from frame_b for efficiency
        row_start = max(0, y - search_size)
        row_end = min(height - window_size, y + search_size + window_size)
        frame_b_row = frame_b[row_start:row_end, :]
        
        for j, x in enumerate(x_coords):
            # Extract window A once
            window_a = frame_a[y:y+window_size, x:x+window_size]
            
            best_corr = -1
            best_dx, best_dy = 0, 0
            
            # Pre-compute sum_a and sum_sq_a for this window
            sum_a = window_sums_a[i, j]
            sum_sq_a = window_sums_sq_a[i, j]
            
            # Optimized search loop
            for dy in dy_range:
                y_shifted = y + dy
                if 0 <= y_shifted < height - window_size:
                    # Pre-extract column from frame_b_row
                    col_start = max(0, x - search_size)
                    col_end = min(width - window_size, x + search_size + window_size)
                    
                    for dx in dx_range:
                        x_shifted = x + dx
                        if 0 <= x_shifted < width - window_size:
                            # Extract shifted window
                            window_b_shifted = frame_b[y_shifted:y_shifted+window_size, 
                                                       x_shifted:x_shifted+window_size]
                            
                            # Fast correlation computation
                            sum_b = np.sum(window_b_shifted)
                            sum_sq_b = np.sum(window_b_shifted**2)
                            sum_ab = np.sum(window_a * window_b_shifted)
                            
                            # Fast normalized cross-correlation
                            numerator = sum_ab - (sum_a * sum_b) / (window_size * window_size)
                            denom = np.sqrt((sum_sq_a - sum_a**2/(window_size*window_size)) * 
                                          (sum_sq_b - sum_b**2/(window_size*window_size)))
                            
                            if denom > 1e-10:  # Avoid division by zero
                                corr_val = numerator / denom
                                
                                if corr_val > best_corr:
                                    best_corr = corr_val
                                    best_dx, best_dy = dx, dy
            
            vectors[i, j] = [best_dx, best_dy]
            correlation_scores[i, j] = best_corr
    
    return vectors, x_coords, y_coords, correlation_scores

"""Filter out noise- large displacement"""
def filter_large_displacements(vectors, threshold_low = 0 , threshold_high=5):
    """
    Filter out vectors with displacement magnitude larger than threshold
    Sets filtered vectors to [0, 0]
    """
    vectors_filtered = vectors.copy()
    displacement_magnitude = np.sqrt(vectors[:, :, 0]**2 + vectors[:, :, 1]**2)
    # small_displacement_mask = displacement_magnitude < threshold_low
    large_displacement_mask = displacement_magnitude > threshold_high
    
    # Combine masks - filter out both too small AND too large displacements
    outlier_mask =  large_displacement_mask  #small_displacement_mask |
    
    vectors_filtered[outlier_mask] = [0, 0]
        
    
    print(f"Filtered {np.sum(large_displacement_mask)} vectors with displacement > {threshold_high} pixels & < {threshold_low}")
    return vectors_filtered, large_displacement_mask, displacement_magnitude

# Pre-define parameters for consistency
window_size = 32
overlap = 0.5
search_size =12
FPS = 25
size_px = 0.2059
sigma = 0.5
block_size = 18
# number of frame
start =128
end = 129
interval = 1

for idx in range(start, end):
    n1, n2 = idx, idx + interval
    
    # Load images
    print(f"\n{'='*60}")  
    print(f"Processing frames {n1} → {n2}")
    print(f"{'='*60}")
    
    # Consecutive frames
    fr1 = cv2.imread(f"Crop frame {n1}.png") 
    fr2 = cv2.imread(f"Crop frame {n2}.png")
    
    # Get original dimensions
    h_orig, w_orig = fr1.shape[:2]  # Note: OpenCV uses (height, width)
    c_h, c_w = h_orig // 2, w_orig // 2
    
    if fr1 is None or fr2 is None:
        print(f"Skipping frames {n1} → {n2} (files not found)")
        continue
    
    def BGR_gray(fr):
        return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    
    fr1_g, fr2_g = BGR_gray(fr1), BGR_gray(fr2)
    
    fr1_norm = fr1_g.astype(np.float32)
    fr2_norm = fr2_g.astype(np.float32)
    
    global_mean = (np.mean(fr1_norm) + np.mean(fr2_norm)) / 2
    global_std = (np.std(fr1_norm) + np.std(fr2_norm)) / 2
    
    fr1_norm = (fr1_norm - global_mean) / global_std
    fr2_norm = (fr2_norm - global_mean) / global_std
    
    # ============ RUN PIV ============
    vectors, x_coords, y_coords, correlation = piv_corrected(
        fr1_norm, fr2_norm, 
        window_size=window_size,
        overlap=overlap,
        search_size=search_size
    )
    
    print(f"Grid: {len(y_coords)}×{len(x_coords)} vectors")
    
    # ============ FILTER NOISES ============
    displacement_threshold_l = 0
    displacement_threshold_h = 4  # pixels
    vectors_filtered, large_disp_mask, original_dis = filter_large_displacements(vectors, displacement_threshold_l, displacement_threshold_h)
    
    t = 1/FPS
    vectors_um = vectors_filtered * size_px
    velocities = vectors_um / t
    
    X, Y = np.meshgrid(x_coords + window_size//2, y_coords + window_size//2)
    U = velocities[:, :, 0]
    V = velocities[:, :, 1]
    magnitude = np.sqrt(U**2 + V**2)  # Original magnitude for vector scaling
    
    # Smooth
    U_filtered = gaussian_filter(U, sigma=sigma)
    V_filtered = gaussian_filter(V, sigma=sigma)
    magnitude_smooth = gaussian_filter(magnitude, sigma=sigma)  # Using raw magnitude, not normalized

    print(f"Speed stats - mean={magnitude.mean():.3f}, max={magnitude.max():.3f}")
        
    # ============ VISUALIZATION ============
    # Prepare data for vector plots
    skip = 1
    X_sub = X[::skip, ::skip]
    Y_sub = Y[::skip, ::skip]
    U_sub = U_filtered[::skip, ::skip]
    V_sub = V_filtered[::skip, ::skip]
    magnitude_sub = magnitude[::skip, ::skip]  # Raw magnitude
    
    # 1. Vector field plot
    plt.figure(figsize=(10, 8), dpi=150)
    plt.imshow(fr1_g, cmap='gray', alpha=0.4)
    
    vmax = np.percentile(magnitude_sub, 100) if np.max(magnitude_sub) > 0 else 1
    
    Q = plt.quiver(X_sub, Y_sub, U_sub, V_sub, 
                   magnitude_sub, cmap="jet", 
                   angles='xy', scale_units='xy', scale=0.6, width=0.004, headwidth=3, headlength=4, alpha=1)
    
    plt.colorbar(Q, label='Speed (µm/s)')  
    cclip=17.5
    plt.clim(0, vmax=cclip)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(f'P Vector_Field_{n1} w_{window_size} s_{search_size} dth_{displacement_threshold_h} in_{interval}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Magnitude plot (raw, not normalized)
    plt.figure(figsize=(8, 6))
    plt.imshow(magnitude_smooth, cmap=parula_cmap, 
               vmin=0, vmax=np.percentile(magnitude_smooth, 100))
    plt.colorbar(label='Speed (µm/s)') 
    plt.tight_layout()
    plt.axis("off")
    plt.imsave(f'P full_magnitude_{n1} w_{window_size} s_{search_size} dth_{displacement_threshold_h}.png', magnitude_smooth)
    plt.close()

print("\nProcessing complete!")  