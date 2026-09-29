# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:25:05 2026

@modified: sgao
"""
from __future__ import annotations; import argparse
import json; import math
from pathlib import Path; import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import numpy as np
import pandas as pd; from scipy.ndimage import gaussian_filter1d
from scipy.signal import detrend, periodogram

# ============================================================
# BASIC FUNCTIONS
# ============================================================
def weighted_mean(values, weights, mask):
    """Weighted average over valid pixels."""
    selected_values = values[mask]
    selected_weights = weights[mask]
    weight_sum = np.sum(selected_weights)
    if weight_sum <= 0:
        return math.nan
    return float( np.sum(selected_values * selected_weights) / weight_sum )
def peak_frequency(signal, fps, fmin, fmax):
    """
    Find the strongest frequency peak using a periodogram.
    Returns
    -------
    peak_frequency : float
        Frequency of strongest peak in the requested range.
    uncertainty : float
        Simple spectral-resolution estimate.
    frequencies : ndarray
        Full periodogram frequency axis.
    power : ndarray
        Periodogram power.
    """
    y = np.asarray(signal, dtype=float)
    if len(y) < 4:
        return math.nan, math.nan, np.array([]), np.array([])
    if not np.all(np.isfinite(y)):
        y = pd.Series(y).interpolate( limit_direction="both" ).to_numpy()
    y = detrend(y)
    frequencies, power = periodogram( y, fs=fps, window="hann" )
    keep = ( (frequencies >= fmin) & ( frequencies <= min(fmax, 0.49 * fps) ) )
    if not np.any(keep):
        return ( math.nan, math.nan, frequencies, power )
    f_selected = frequencies[keep]
    p_selected = power[keep]
    peak_index = int( np.argmax(p_selected) )
    peak = float( f_selected[peak_index] )
    # --------------------------------------------------------
    # Estimate spectral uncertainty.
    # --------------------------------------------------------
    peak_power = p_selected[peak_index]
    half_power = peak_power / 2.0
    above_half = np.where( p_selected >= half_power )[0]
    if len(above_half) > 1:
        width = float( f_selected[above_half[-1]] - f_selected[above_half[0]] )
    else:
        width = 0.0
    df = fps / len(y)
    uncertainty = max( df / 2.0, width / 2.0 )
    return ( peak, uncertainty, frequencies, power )
def frequency_is_reliable( frequency, uncertainty, duration ):
    """
    Conservative frequency-reliability test.
    Requirements:
        - positive finite frequency
        - at least 4 cycles
        - relative uncertainty <= 25%
    """
    if not np.isfinite(frequency):
        return False
    if frequency <= 0:
        return False
    if not np.isfinite(uncertainty):
        return False
    if frequency * duration < 4:
        return False
    if uncertainty / frequency > 0.25:
        return False
    return True
def wavelength_from_filename(filename):
    """
    Determine wavelength group from the first part of the filename.
    Examples:
        UV_1.mp4       -> UV
        blue_2.mp4     -> BLUE
        green_sample.mp4 -> GREEN
    """
    stem = Path(filename).stem.lower()
    return stem.split("_")[0]
# ============================================================
# ANALYZE ONE VIDEO
# ============================================================
def analyze_video( video: Path, output: Path, circle, settings ):
    """Analyze one video using optical flow."""
    print(f"Opening video: {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError( f"Cannot open video: {video}" )
    fps = float( cap.get(cv2.CAP_PROP_FPS) )
    n_total = int( cap.get(cv2.CAP_PROP_FRAME_COUNT) )
    if fps <= 0:
        cap.release()
        raise ValueError( f"Invalid FPS for {video}: {fps}" )
    if n_total < 2:
        cap.release()
        raise ValueError( f"Video must contain at least 2 frames: {video}" )
    # ========================================================
    # ANALYSIS DURATION
    # ========================================================
    maximum_duration = settings.get( "maximum_duration_s" )
    if maximum_duration is not None:
        maximum_duration = float( maximum_duration )
        n_analyze = min( n_total, max( 2, int( round( maximum_duration * fps ) ) ) )
    else:
        n_analyze = n_total
    # ========================================================
    # FIRST FRAME
    # ========================================================
    ok, first_full = cap.read()
    if not ok:
        cap.release()
        raise ValueError( f"Cannot read first frame from {video}" )
    # ========================================================
    # SETTINGS
    # ========================================================
    scale = float( settings.get( "resize_scale", 0.5 ) )
    if scale <= 0:
        cap.release()
        raise ValueError( "resize_scale must be greater than zero." )
    field_every = int( settings.get( "velocity_field_every_n_frames", 30 ) )
    arrow_step = int( settings.get( "velocity_field_arrow_step", 20 ) )
    arrow_scale = float( settings.get( "velocity_arrow_scale", 5.0 ) )
    if arrow_step <= 0:
        arrow_step = 20
    # ========================================================
    # CIRCLE
    # ========================================================
    cx0, cy0 = map( float, circle["center_px"] )
    radius0 = float( circle["radius_px"] )
    if radius0 <= 0:
        cap.release()
        raise ValueError( "Circle radius must be greater than zero." )
    padding = float( settings.get( "crop_padding_px", 0 ) )
    # ========================================================
    # CROP
    # ========================================================
    if padding > 0:
        crop_x0 = max( 0, int( np.floor( cx0 - radius0 - padding ) ) )
        crop_y0 = max( 0, int( np.floor( cy0 - radius0 - padding ) ) )
        crop_x1 = min( first_full.shape[1], int( np.ceil( cx0 + radius0 + padding ) ) )
        crop_y1 = min( first_full.shape[0], int( np.ceil( cy0 + radius0 + padding ) ) )
    else:
        crop_x0 = 0
        crop_y0 = 0
        crop_x1 = first_full.shape[1]
        crop_y1 = first_full.shape[0]
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        cap.release()
        raise ValueError( f"Invalid crop for {video}" )
    # ========================================================
    # PHYSICAL CALIBRATION
    # ========================================================
    physical_diameter = circle.get( "physical_diameter" )
    pixels_per_unit = circle.get( "pixels_per_unit" )
    unit = circle.get( "physical_unit", "um" )
    if pixels_per_unit is not None:
        pixels_per_unit = float( pixels_per_unit )
        if pixels_per_unit <= 0:
            cap.release()
            raise ValueError( "pixels_per_unit must be greater than zero." )
        length_scale = ( 1.0 / pixels_per_unit )
        length_unit = unit
    elif physical_diameter is not None:
        physical_diameter = float( physical_diameter )
        if physical_diameter <= 0:
            cap.release()
            raise ValueError( "physical_diameter must be greater than zero." )
        length_scale = ( physical_diameter / (2.0 * radius0) )
        length_unit = unit
    else:
        length_scale = 1.0
        length_unit = "px"
    # ========================================================
    # FRAME PREPROCESSING
    # ========================================================
    def prep(frame):
        cropped = frame[ crop_y0:crop_y1, crop_x0:crop_x1 ]
        small = cv2.resize( cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA )
        gray = cv2.cvtColor( small, cv2.COLOR_BGR2GRAY )
        blurred = cv2.GaussianBlur( gray, (5, 5), 0 )
        return blurred
    # ========================================================
    # FIRST PROCESSED FRAME
    # ========================================================
    prev = prep(first_full)
    h, w = prev.shape
    # ========================================================
    # CIRCLE IN RESIZED IMAGE
    # ========================================================
    cx = ( cx0 - crop_x0 ) * scale
    cy = ( cy0 - crop_y0 ) * scale
    radius = radius0 * scale
    # ========================================================
    # COORDINATE SYSTEM
    # ========================================================
    yy, xx = np.mgrid[ :h, :w ]
    dx = xx - cx
    dy = yy - cy
    rr = np.hypot( dx, dy )
    # Analyze inside circle.
    #
    # Exclude the very center because radial direction
    # is undefined there.
    mask = ( (rr < radius) & (rr > 0.05 * radius) )
    if not np.any(mask):
        cap.release()
        raise ValueError( f"No pixels inside analysis circle for {video}" )
    # ========================================================
    # RADIAL UNIT VECTOR
    # ========================================================
    erx = ( dx / np.maximum(rr, 1.0) )
    ery = ( dy / np.maximum(rr, 1.0) )
    # ========================================================
    # TANGENTIAL UNIT VECTOR
    # ========================================================
    etx = -ery
    ety = erx
    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================
    stem = video.stem.lower()
    run_out = ( output / stem )
    run_out.mkdir( parents=True, exist_ok=True )
    field_out = ( run_out / "velocity_fields" )
    field_out.mkdir( parents=True, exist_ok=True )
    # ========================================================
    # DATA CONTAINER
    # ========================================================
    rows = []
    # ========================================================
    # FRAME LOOP
    # ========================================================
    #
    # frame_no = 1 corresponds to:
    #
    #     frame 1 -> frame 2
    #
    # The velocity field is assigned to the first frame
    # of the pair.
    #
    frame_no = 1
    # This stores the ORIGINAL image corresponding to
    # "prev".
    #
    # It is important because the velocity field describes
    # motion from prev -> cur.
    prev_full = first_full.copy()
    while frame_no < n_analyze:
        ok, full = cap.read()
        if not ok:
            print( f"Stopped early at frame {frame_no + 1}" )
            break
        cur = prep(full)
        # ====================================================
        # OPTICAL FLOW
        #
        # KEEPING THE ORIGINAL PARAMETERS
        # ====================================================
        flow = cv2.calcOpticalFlowFarneback( prev, cur, None, 0.5, 3, 25, 3, 7, 1.5, 0 )
        # ====================================================
        # IMAGE TEXTURE
        # ====================================================
        gx = cv2.Sobel( prev, cv2.CV_32F, 1, 0, ksize=3 )
        gy = cv2.Sobel( prev, cv2.CV_32F, 0, 1, ksize=3 )
        texture = np.hypot( gx, gy )
        texture_inside = texture[mask]
        texture_threshold = np.percentile( texture_inside, 45 )
        valid = ( mask & ( texture >= texture_threshold ) )
        # ====================================================
        # VELOCITY IN ORIGINAL PIXELS / SECOND
        # ====================================================
        vx_px_s = ( flow[..., 0] * fps / scale )
        vy_px_s = ( flow[..., 1] * fps / scale )
        # ====================================================
        # PHYSICAL VELOCITY
        # ====================================================
        vx = ( vx_px_s * length_scale )
        vy = ( vy_px_s * length_scale )
        # ====================================================
        # RADIAL VELOCITY
        # ====================================================
        vr = ( vx * erx + vy * ery )
        # ====================================================
        # TANGENTIAL VELOCITY
        # ====================================================
        vt = ( vx * etx + vy * ety )
        # ====================================================
        # TOTAL VELOCITY
        # ====================================================
        v_total = np.hypot( vx, vy )
        # ====================================================
        # ANGLE 1
        #
        # Direction relative to image x-axis.
        #
        # 0 deg   = +x
        # 90 deg  = +y
        # -90 deg = -y
        # ====================================================
        angle_xy_deg = np.degrees( np.arctan2( vy, vx ) )
        # ====================================================
        # ANGLE 2
        #
        # Direction relative to local radial direction.
        #
        # 0 deg   = purely outward
        # 180/-180 = purely inward
        # +90 deg = tangential direction
        # ====================================================
        angle_radial_deg = np.degrees( np.arctan2( vt, vr ) )
        # ====================================================
        # TEXTURE WEIGHTS
        # ====================================================
        texture_max = np.percentile( texture_inside, 98 )
        weights = ( np.clip( texture, 0, texture_max ) + 1e-6 )
        # ====================================================
        # FRAME-LEVEL MEAN RADIAL VELOCITY
        # ====================================================
        mean_radial = weighted_mean( vr, weights, valid )
        # ====================================================
        # FRAME-LEVEL MEAN TANGENTIAL VELOCITY
        # ====================================================
        mean_tangential = weighted_mean( vt, weights, valid )
        # ====================================================
        # FRAME-LEVEL RMS TOTAL VELOCITY
        # ====================================================
        rms_total = weighted_mean( v_total ** 2, weights, valid )
        rms_total = ( math.sqrt(rms_total) if np.isfinite(rms_total) and rms_total >= 0 else math.nan )
        # ====================================================
        # FRAME-LEVEL RMS RADIAL VELOCITY
        # ====================================================
        rms_radial = weighted_mean( vr ** 2, weights, valid )
        rms_radial = ( math.sqrt(rms_radial) if np.isfinite(rms_radial) and rms_radial >= 0 else math.nan )
        # ====================================================
        # FRAME-LEVEL Cr
        #
        # Cr = RMS(vr) / RMS(v_total)
        # ====================================================
        if ( np.isfinite(rms_radial) and np.isfinite(rms_total) and rms_total > 0 ):
            Cr_frame = ( rms_radial / rms_total )
        else:
            Cr_frame = math.nan
        # ====================================================
        # TIME
        #
        # frame 1 -> frame 2 has time = 0 s
        # frame 2 -> frame 3 has time = 1/fps
        # etc.
        # ====================================================
        time_s = ( frame_no - 1 ) / fps
        # ====================================================
        # SAVE FRAME SUMMARY
        # ====================================================
        rows.append({ "frame": frame_no, "time_s": time_s, f"mean_radial_velocity_{length_unit}_s": mean_radial, f"mean_tangential_velocity_{length_unit}_s": mean_tangential, f"rms_radial_velocity_{length_unit}_s": rms_radial, f"rms_total_velocity_{length_unit}_s": rms_total, "Cr_radial_fraction": Cr_frame, "valid_texture_fraction": float( np.mean( valid[mask] ) ) })
        # ====================================================
        # SAVE SELECTED VELOCITY FIELD
        # ====================================================
        if ( field_every > 0 and frame_no % field_every == 0 ):
            # ------------------------------------------------
            # Numerical arrays
            #
            # vx, vy, v_total, vr, vt are saved in the
            # calibrated physical unit per second.
            #
            # If no calibration was supplied, the unit is
            # pixels/second.
            # ------------------------------------------------
            field_filename = ( field_out / f"frame_{frame_no:06d}.npz" )
            np.savez_compressed( field_filename, vx=vx, vy=vy, v_total=v_total, angle_xy_deg= angle_xy_deg, vr=vr, vt=vt, angle_radial_deg= angle_radial_deg, valid=valid,
                # Raw optical-flow velocities
                # in original pixels/second.
                vx_px_s=vx_px_s, vy_px_s=vy_px_s,
                # Metadata
                fps=fps, resize_scale=scale, crop_x0=crop_x0, crop_y0=crop_y0, center_x_px=cx0, center_y_px=cy0, radius_px=radius0, length_scale= length_scale, length_unit= length_unit, source_frame= frame_no, next_frame= frame_no + 1 )
            # ------------------------------------------------
            # VELOCITY OVERLAY
            #
            # IMPORTANT:
            #
            # The arrows describe motion from
            #
            #     frame_no -> frame_no + 1
            #
            # Therefore the arrows are drawn on frame_no,
            # the FIRST frame of the optical-flow pair.
            # ------------------------------------------------
            display = prev_full.copy()
            # ------------------------------------------------
            # Draw excitation circle
            # ------------------------------------------------
            cv2.circle( display, ( int(round(cx0)), int(round(cy0)) ), int(round(radius0)), (0, 0, 255), 2 )
            # ------------------------------------------------
            # Draw center
            # ------------------------------------------------
            cv2.drawMarker( display, ( int(round(cx0)), int(round(cy0)) ), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2 )
            # ------------------------------------------------
            # Draw velocity arrows
            # ------------------------------------------------
            for y in range( 0, h, arrow_step ):
                for x in range( 0, w, arrow_step ):
                    if not valid[y, x]:
                        continue
                    # Keep arrows inside circle.
                    if rr[y, x] >= radius:
                        continue
                    # ----------------------------------------
                    # Position in original image
                    # ----------------------------------------
                    x_orig = ( x / scale + crop_x0 )
                    y_orig = ( y / scale + crop_y0 )
                    # ----------------------------------------
                    # Convert velocity from px/s to
                    # px/frame for visualization.
                    #
                    # This does NOT change the saved
                    # numerical velocity.
                    # ----------------------------------------
                    ux = ( vx_px_s[y, x] / fps )
                    uy = ( vy_px_s[y, x] / fps )
                    # ----------------------------------------
                    # Visualization amplification.
                    #
                    # Arrow length is NOT quantitatively
                    # equal to displacement.
                    # ----------------------------------------
                    x2 = ( x_orig + ux * arrow_scale )
                    y2 = ( y_orig + uy * arrow_scale )
                    cv2.arrowedLine( display, ( int(round(x_orig)), int(round(y_orig)) ), ( int(round(x2)), int(round(y2)) ), (0, 255, 0), 1, tipLength=0.25 )
            # ------------------------------------------------
            # Frame information
            # ------------------------------------------------
            cv2.putText( display, ( f"Flow: frame {frame_no}" f" -> {frame_no + 1}   " f"t = {time_s:.2f} s" ), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA )
            cv2.putText( display, ( f"Arrow scale: {arrow_scale:g}x" ), (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA )
            # ------------------------------------------------
            # Save overlay
            # ------------------------------------------------
            overlay_filename = ( field_out / f"frame_{frame_no:06d}.png" )
            cv2.imwrite( str(overlay_filename), display )
        # ====================================================
        # MOVE TO NEXT FRAME
        # ====================================================
        prev = cur
        prev_full = full.copy()
        frame_no += 1
    cap.release()
    # ========================================================
    # DATAFRAME
    # ========================================================
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError( f"No optical-flow measurements were produced for {video}" )
    # ========================================================
    # COLUMN NAMES
    # ========================================================
    radial_col = ( f"mean_radial_velocity_" f"{length_unit}_s" )
    tangential_col = ( f"mean_tangential_velocity_" f"{length_unit}_s" )
    rms_radial_col = ( f"rms_radial_velocity_" f"{length_unit}_s" )
    rms_total_col = ( f"rms_total_velocity_" f"{length_unit}_s" )
    # ========================================================
    # OVERALL RMS RADIAL VELOCITY
    # ========================================================
    overall_rms_radial = float( np.sqrt( np.mean( df[rms_radial_col] ** 2 ) ) )
    # ========================================================
    # OVERALL RMS TOTAL VELOCITY
    # ========================================================
    overall_rms_total = float( np.sqrt( np.mean( df[rms_total_col] ** 2 ) ) )
    # ========================================================
    # Cr
    #
    # Cr = RMS(vr) / RMS(v_total)
    #
    # This is best interpreted as the fraction of measured
    # motion that is radial.
    # ========================================================
    if overall_rms_total > 0:
        Cr = ( overall_rms_radial / overall_rms_total )
    else:
        Cr = math.nan
    # ========================================================
    # COHERENT RADIAL RESPONSE
    #
    # Smooth mean radial velocity and calculate its standard
    # deviation.
    # ========================================================
    smooth_sigma = max( fps * 0.2, 1.0 )
    smooth_radial = ( gaussian_filter1d( df[radial_col].to_numpy(), smooth_sigma ) )
    coherent_radial_rms = float( np.std( smooth_radial ) )
    # ========================================================
    # TOTAL-VELOCITY VARIATION
    #
    # Total speed is always positive, so this should not be
    # called a signed "coherent velocity".
    # ========================================================
    smooth_total = ( gaussian_filter1d( df[rms_total_col].to_numpy(), smooth_sigma ) )
    total_velocity_variation_rms = float( np.std( smooth_total ) )
    # ========================================================
    # FREQUENCY SETTINGS
    # ========================================================
    fmin = float( settings.get( "minimum_frequency_hz", 0.05 ) )
    fmax = float( settings.get( "maximum_frequency_hz", 4.0 ) )
    # ========================================================
    # RADIAL VELOCITY FREQUENCY
    # ========================================================
    ( f_radial, u_radial, fr, pr ) = peak_frequency( df[radial_col], fps, fmin, fmax )
    # Duration between first and last velocity measurement.
    if len(df) > 1:
        analyzed_duration = ( df["time_s"].iloc[-1] - df["time_s"].iloc[0] )
    else:
        analyzed_duration = 0.0
    radial_frequency_detected = ( frequency_is_reliable( f_radial, u_radial, analyzed_duration ) )
    # ========================================================
    # TOTAL VELOCITY FREQUENCY
    # ========================================================
    ( f_total, u_total, ft, pt ) = peak_frequency( df[rms_total_col], fps, fmin, fmax )
    total_frequency_detected = ( frequency_is_reliable( f_total, u_total, analyzed_duration ) )
    # ========================================================
    # SAVE FRAME-LEVEL CSV
    # ========================================================
    df.to_csv( run_out / "collective_motion.csv", index=False )
    # ========================================================
    # REGION CHECK PLOT
    # ========================================================
    fig, ax = plt.subplots( figsize=(7, 7) )
    ax.imshow( cv2.cvtColor( first_full, cv2.COLOR_BGR2RGB ) )
    circle_patch = plt.Circle( (cx0, cy0), radius0, fill=False, color="red", lw=2 )
    ax.add_patch( circle_patch )
    ax.plot( cx0, cy0, "+", color="red", ms=14, mew=2 )
    ax.set( title=( f"{video.stem}: " "analyzed excitation region" ), xlim=( 0, first_full.shape[1] ), ylim=( first_full.shape[0], 0 ) )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig( run_out / "region_check.png", dpi=160 )
    plt.close(fig)
    # ========================================================
    # MOTION SIGNALS
    # ========================================================
    fig, axs = plt.subplots( 4, 1, figsize=(11, 10), sharex=True )
    # --------------------------------------------------------
    # Radial velocity
    # --------------------------------------------------------
    axs[0].plot( df["time_s"], df[radial_col], lw=0.8 )
    axs[0].axhline( 0, color="k", lw=0.6 )
    axs[0].set_ylabel( f"Mean radial velocity\n" f"({length_unit}/s)" )
    # --------------------------------------------------------
    # Tangential velocity
    # --------------------------------------------------------
    axs[1].plot( df["time_s"], df[tangential_col], lw=0.8 )
    axs[1].axhline( 0, color="k", lw=0.6 )
    axs[1].set_ylabel( f"Tangential velocity\n" f"({length_unit}/s)" )
    # --------------------------------------------------------
    # Total velocity
    # --------------------------------------------------------
    axs[2].plot( df["time_s"], df[rms_total_col], lw=0.8 )
    axs[2].set_ylabel( f"RMS total velocity\n" f"({length_unit}/s)" )
    # --------------------------------------------------------
    # Cr
    # --------------------------------------------------------
    axs[3].plot( df["time_s"], df["Cr_radial_fraction"], lw=0.8 )
    axs[3].set_ylabel( "Cr" )
    axs[3].set_xlabel( "Time (s)" )
    for ax in axs:
        ax.grid( alpha=0.25 )
    fig.suptitle( "Collective particle motion — " f"{video.stem}" )
    fig.tight_layout()
    fig.savefig( run_out / "motion_signals.png", dpi=180 )
    plt.close(fig)
    # ========================================================
    # FREQUENCY SPECTRA
    # ========================================================
    fig, axs = plt.subplots( 1, 2, figsize=(11, 4) )
    # --------------------------------------------------------
    # Radial spectrum
    # --------------------------------------------------------
    axs[0].plot( fr, pr )
    if radial_frequency_detected:
        axs[0].axvline( f_radial, color="r", ls="--" )
        axs[0].text( 0.98, 0.92, ( f"{f_radial:.3f} Hz" f"\n± {u_radial:.3f} Hz" ), transform=axs[0].transAxes, ha="right", va="top" )
    else:
        axs[0].text( 0.98, 0.92, "No reliable periodicity detected", transform=axs[0].transAxes, ha="right", va="top", color="darkred" )
    axs[0].set( title="Radial velocity", xlabel="Frequency (Hz)", ylabel="Power", xlim=(0, fmax) )
    # --------------------------------------------------------
    # Total velocity spectrum
    # --------------------------------------------------------
    axs[1].plot( ft, pt )
    if total_frequency_detected:
        axs[1].axvline( f_total, color="r", ls="--" )
        axs[1].text( 0.98, 0.92, ( f"{f_total:.3f} Hz" f"\n± {u_total:.3f} Hz" ), transform=axs[1].transAxes, ha="right", va="top" )
    else:
        axs[1].text( 0.98, 0.92, "No reliable periodicity detected", transform=axs[1].transAxes, ha="right", va="top", color="darkred" )
    axs[1].set( title="Total velocity", xlabel="Frequency (Hz)", ylabel="Power", xlim=(0, fmax) )
    for ax in axs:
        ax.grid( alpha=0.25 )
    fig.suptitle( "Oscillation spectra — " f"{video.stem}" )
    fig.tight_layout()
    fig.savefig( run_out / "spectra.png", dpi=180 )
    plt.close(fig)
    # ========================================================
    # RETURN SUMMARY
    # ========================================================
    return { "video": video.name, "wavelength": wavelength_from_filename( video.name ), "source_frame_count": n_total, "analyzed_frame_count": n_analyze, "number_of_velocity_measurements": len(df), "source_duration_s": (n_total - 1) / fps, "duration_s": analyzed_duration, "fps": fps, "analyzed_center_px": [cx0, cy0], "analyzed_radius_px": radius0, "distance_unit": length_unit,
        # ----------------------------------------------------
        # Radial motion
        # ----------------------------------------------------
        "overall_rms_radial_velocity": overall_rms_radial, "coherent_radial_response_rms": coherent_radial_rms, "radial_frequency_detected": radial_frequency_detected, "radial_frequency_hz": ( f_radial if radial_frequency_detected else None ), "radial_frequency_uncertainty_hz": ( u_radial if radial_frequency_detected else None ), "radial_candidate_peak_hz": ( f_radial if np.isfinite(f_radial) else None ),
        # ----------------------------------------------------
        # Total motion
        # ----------------------------------------------------
        "overall_rms_total_velocity": overall_rms_total, "total_velocity_variation_rms": total_velocity_variation_rms, "total_frequency_detected": total_frequency_detected, "total_frequency_hz": ( f_total if total_frequency_detected else None ), "total_frequency_uncertainty_hz": ( u_total if total_frequency_detected else None ), "total_candidate_peak_hz": ( f_total if np.isfinite(f_total) else None ),
        # ----------------------------------------------------
        # Radial fraction
        # ----------------------------------------------------
        "Cr": Cr }
# ============================================================
# COMPARISON BETWEEN VIDEOS / WAVELENGTHS
# ============================================================
def build_comparison( summaries, output ):
    """Build comparison plots across videos and wavelengths."""
    if not summaries:
        return
    # ========================================================
    # READ ALL VIDEO CSV FILES
    # ========================================================
    data_by_video = {}
    for item in summaries:
        stem = Path( item["video"] ).stem.lower()
        csv_path = ( output / stem / "collective_motion.csv" )
        if not csv_path.exists():
            continue
        data_by_video[stem] = pd.read_csv( csv_path )
    if not data_by_video:
        return
    unit = summaries[0]["distance_unit"]
    # ========================================================
    # PLOT 1
    #
    # Mean radial velocity vs time
    # ========================================================
    fig, ax = plt.subplots( figsize=(11, 5) )
    for item in summaries:
        stem = Path( item["video"] ).stem.lower()
        if stem not in data_by_video:
            continue
        data = data_by_video[stem]
        radial_col = data.filter( like="mean_radial_velocity" ).columns[0]
        ax.plot( data["time_s"], data[radial_col], lw=0.8, label=stem.upper(), alpha=0.65 )
    ax.axhline( 0, color="k", lw=0.6 )
    ax.set( xlabel="Time (s)", ylabel=( f"Mean radial velocity " f"({unit}/s)" ), title=( "Radial velocity response " "across wavelengths" ) )
    ax.legend( ncol=3, fontsize=8 )
    ax.grid( alpha=0.25 )
    fig.tight_layout()
    fig.savefig( output / "radial_velocity_comparison.png", dpi=180 )
    plt.close(fig)
    # ========================================================
    # PLOT 2
    #
    # RMS total velocity vs time
    # ========================================================
    fig, ax = plt.subplots( figsize=(11, 5) )
    for item in summaries:
        stem = Path( item["video"] ).stem.lower()
        if stem not in data_by_video:
            continue
        data = data_by_video[stem]
        total_col = data.filter( like="rms_total_velocity" ).columns[0]
        ax.plot( data["time_s"], data[total_col], lw=0.8, label=stem.upper(), alpha=0.65 )
    ax.set( xlabel="Time (s)", ylabel=( f"RMS total velocity " f"({unit}/s)" ), title=( "Total velocity response " "across wavelengths" ) )
    ax.legend( ncol=3, fontsize=8 )
    ax.grid( alpha=0.25 )
    fig.tight_layout()
    fig.savefig( output / "total_velocity_comparison.png", dpi=180 )
    plt.close(fig)
    # ========================================================
    # VIDEO-LEVEL COMPARISON TABLE
    # ========================================================
    comparison_rows = []
    for item in summaries:
        comparison_rows.append({ "video": item["video"], "wavelength": item["wavelength"], "RMS_radial_velocity": item[ "overall_rms_radial_velocity" ], "RMS_total_velocity": item[ "overall_rms_total_velocity" ], "Cr": item["Cr"], "coherent_radial_response": item[ "coherent_radial_response_rms" ], "total_velocity_variation": item[ "total_velocity_variation_rms" ], "radial_frequency_hz": item[ "radial_frequency_hz" ], "total_frequency_hz": item[ "total_frequency_hz" ] })
    comparison_df = pd.DataFrame( comparison_rows )
    comparison_df.to_csv( output / "comparison.csv", index=False )
    # ========================================================
    # PLOT 3
    #
    # RMS radial vs RMS total for each video
    # ========================================================
    names = [ Path( item["video"] ).stem.upper() for item in summaries ]
    radial_rms = [ item[ "overall_rms_radial_velocity" ] for item in summaries ]
    total_rms = [ item[ "overall_rms_total_velocity" ] for item in summaries ]
    x = np.arange( len(names) )
    width = 0.36
    fig, ax = plt.subplots( figsize=(11, 5) )
    ax.bar( x - width / 2, radial_rms, width, label="RMS radial velocity" )
    ax.bar( x + width / 2, total_rms, width, label="RMS total velocity" )
    ax.set( xticks=x, xticklabels=names, ylabel=f"Velocity ({unit}/s)", title=( "Radial motion versus total motion" ) )
    ax.legend()
    ax.grid( axis="y", alpha=0.25 )
    fig.tight_layout()
    fig.savefig( output / "radial_vs_total_comparison.png", dpi=180 )
    plt.close(fig)
    # ========================================================
    # PLOT 4
    #
    # Cr for each video
    # ========================================================
    Cr_values = [ item["Cr"] for item in summaries ]
    fig, ax = plt.subplots( figsize=(11, 5) )
    ax.bar( x, Cr_values )
    ax.set( xticks=x, xticklabels=names, ylabel="Cr", title=( "Radial fraction of total particle motion" ) )
    ax.grid( axis="y", alpha=0.25 )
    fig.tight_layout()
    fig.savefig( output / "Cr_comparison.png", dpi=180 )
    plt.close(fig)
    # ========================================================
    # WAVELENGTH-LEVEL SUMMARY
    #
    # If there are multiple videos with the same prefix,
    # calculate mean and standard deviation.
    # ========================================================
    wavelength_df = ( comparison_df .groupby("wavelength") .agg( RMS_radial_mean=( "RMS_radial_velocity", "mean" ), RMS_radial_std=( "RMS_radial_velocity", "std" ), RMS_total_mean=( "RMS_total_velocity", "mean" ), RMS_total_std=( "RMS_total_velocity", "std" ), Cr_mean=( "Cr", "mean" ), Cr_std=( "Cr", "std" ) ) .reset_index() )
    wavelength_df = wavelength_df.fillna(0)
    wavelength_df.to_csv( output / "wavelength_comparison.csv", index=False )
    # ========================================================
    # PLOT 5
    #
    # Wavelength RMS comparison
    # ========================================================
    wavelengths = ( wavelength_df["wavelength"] .str.upper() .tolist() )
    wx = np.arange( len(wavelengths) )
    fig, ax = plt.subplots( figsize=(9, 5) )
    ax.bar( wx - width / 2, wavelength_df[ "RMS_radial_mean" ], width, yerr=wavelength_df[ "RMS_radial_std" ], capsize=4, label="RMS radial velocity" )
    ax.bar( wx + width / 2, wavelength_df[ "RMS_total_mean" ], width, yerr=wavelength_df[ "RMS_total_std" ], capsize=4, label="RMS total velocity" )
    ax.set( xticks=wx, xticklabels=wavelengths, ylabel=f"Velocity ({unit}/s)", title=( "Particle motion by wavelength" ) )
    ax.legend()
    ax.grid( axis="y", alpha=0.25 )
    fig.tight_layout()
    fig.savefig( output / "wavelength_rms_comparison.png", dpi=180 )
    plt.close(fig)
    # ========================================================
    # PLOT 6
    #
    # Wavelength Cr comparison
    # ========================================================
    fig, ax = plt.subplots( figsize=(9, 5) )
    ax.bar( wx, wavelength_df["Cr_mean"], yerr=wavelength_df["Cr_std"], capsize=4 )
    ax.set( xticks=wx, xticklabels=wavelengths, ylabel="Cr", title=( "Radial fraction by wavelength" ) )
    ax.grid( axis="y", alpha=0.25 )
    fig.tight_layout()
    fig.savefig( output / "wavelength_Cr_comparison.png", dpi=180 )
    plt.close(fig)
    # ========================================================
    # MARKDOWN REPORT
    # ========================================================
    analysis_window = max( item["duration_s"] for item in summaries )
    lines = [ "# UV, blue, and green collective-motion analysis", "", ( f"Maximum analyzed window: " f"{analysis_window:g} seconds." ), "", ( "The analysis uses optical flow inside the " "supplied circular excitation region." ), "", ( "PCA was not used." ), "", ( "The radial and total velocity signals were " "analyzed independently." ), "", ( "| Video | Wavelength | RMS radial | " "RMS total | Cr | Radial frequency | " "Total frequency |" ), ( "|---|---|---:|---:|---:|---:|---:|" ) ]
    for item in summaries:
        radial_frequency = ( f"{item['radial_frequency_hz']:.3f} Hz" if item[ "radial_frequency_detected" ] else "Not detected" )
        total_frequency = ( f"{item['total_frequency_hz']:.3f} Hz" if item[ "total_frequency_detected" ] else "Not detected" )
        lines.append( f"| {Path(item['video']).stem} " f"| {item['wavelength'].upper()} " f"| {item['overall_rms_radial_velocity']:.4f} " f"| {item['overall_rms_total_velocity']:.4f} " f"| {item['Cr']:.4f} " f"| {radial_frequency} " f"| {total_frequency} |" )
    lines += [ "", "## Interpretation", "", ( "RMS radial velocity measures the magnitude " "of radial particle motion." ), "", ( "RMS total velocity measures the overall " "magnitude of particle motion regardless " "of direction." ), "", ( "Cr is defined as RMS radial velocity divided " "by RMS total velocity. It describes the " "fraction of measured motion that is radial." ), "", ( "The radial velocity is signed, so inward and " "outward motion have opposite signs." ), "", ( "Total velocity is non-negative. Therefore, " "its strongest spectral peak can sometimes " "occur at a harmonic of the radial-motion " "frequency." ), "", ( "The reported frequency uncertainty is a " "spectral-resolution estimate rather than " "a formal confidence interval." ), "", ( "Velocity-field arrows are amplified for " "visualization. The numerical velocity values " "are stored separately in the NPZ files." ) ]
    ( output / "analysis_report.md" ).write_text( "\n".join(lines), encoding="utf-8" )
# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser( description=( "Analyze collective particle motion " "inside a circular excitation region." ) )
    parser.add_argument( "--config", default="field_config.json", help=( "Path to the JSON configuration file." ) )
    args = parser.parse_args()
    # ========================================================
    # CONFIGURATION
    # ========================================================
    config_path = Path( args.config ).resolve()
    if not config_path.exists():
        raise FileNotFoundError( f"Config file not found: {config_path}" )
    cfg = json.loads( config_path.read_text( encoding="utf-8" ) )
    # ========================================================
    # ROOT DIRECTORY
    #
    # The root is now taken from the config file if supplied.
    #
    # Otherwise the directory containing the config file
    # is used.
    # ========================================================
    if "root_dir" in cfg:
        root = Path( cfg["root_dir"] ).expanduser()
        if not root.is_absolute():
            root = ( config_path.parent / root )
    else:
        root = config_path.parent
    root = root.resolve()
    print( f"Root directory: {root}" )
    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================
    output_dir = cfg.get( "output_dir", "outputs/particle_field" )
    output = ( root / output_dir )
    output.mkdir( parents=True, exist_ok=True )
    # ========================================================
    # REQUIRED CONFIGURATION
    # ========================================================
    if "videos" not in cfg:
        raise KeyError( "Config file must contain 'videos'." )
    if "excitation_circle" not in cfg:
        raise KeyError( "Config file must contain " "'excitation_circle'." )
    analysis_settings = cfg.get( "analysis", {} )
    # ========================================================
    # ANALYZE VIDEOS
    # ========================================================
    summaries = []
    for name in cfg["videos"]:
        video_path = ( root / name )
        if not video_path.exists():
            print( f"WARNING: video not found: " f"{video_path}" )
            continue
        print()
        print("=" * 70)
        print( f"Analyzing: {name}" )
        print("=" * 70)
        try:
            result = analyze_video( video_path, output, cfg["excitation_circle"], analysis_settings )
            summaries.append( result )
        except Exception as exc:
            print( f"ERROR while analyzing " f"{name}: {exc}" )
            raise
    if not summaries:
        raise RuntimeError( "No videos were successfully analyzed." )
    # ========================================================
    # SUMMARY TABLE
    # ========================================================
    summary_df = pd.DataFrame( summaries )
    summary_df.to_csv( output / "summary.csv", index=False )
    # ========================================================
    # SUMMARY JSON
    # ========================================================
    ( output / "summary.json" ).write_text( json.dumps( summaries, indent=2, default=lambda x: float(x) if isinstance( x, np.floating ) else x ), encoding="utf-8" )
    # ========================================================
    # COMPARISON
    # ========================================================
    build_comparison( summaries, output )
    # ========================================================
    # PRINT FINAL TABLE
    # ========================================================
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    display_columns = [ "video", "wavelength", "overall_rms_radial_velocity", "overall_rms_total_velocity", "Cr", "radial_frequency_hz", "total_frequency_hz" ]
    available_columns = [ column for column in display_columns if column in summary_df.columns ]
    print( summary_df[ available_columns ].to_string( index=False ) )
    print()
    print( f"Results saved to:\n{output}" )
# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()
