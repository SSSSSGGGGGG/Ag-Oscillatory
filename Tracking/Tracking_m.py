# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 17:30:17 2025

@author: gaosh

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from matplotlib import colors as mcolors
from PIL import Image, ImageDraw, ImageFont

def parula_colormap():
    """
    Create a Parula colormap similar to MATLAB's Parula.
    This is an approximation based on known Parula color values.
    """
    # Parula colormap data (approximation from MATLAB)
    parula_data = [
        [0.2422, 0.1504, 0.6603],
        [0.2504, 0.1650, 0.7076],
        [0.2578, 0.1818, 0.7511],
        [0.2647, 0.1978, 0.7952],
        [0.2707, 0.2147, 0.8364],
        [0.2751, 0.2342, 0.8710],
        [0.2783, 0.2559, 0.8991],
        [0.2803, 0.2782, 0.9221],
        [0.2813, 0.3006, 0.9414],
        [0.2810, 0.3228, 0.9579],
        [0.2795, 0.3447, 0.9717],
        [0.2760, 0.3667, 0.9829],
        [0.2699, 0.3892, 0.9906],
        [0.2602, 0.4123, 0.9952],
        [0.2440, 0.4358, 0.9988],
        [0.2209, 0.4603, 0.9973],
        [0.1963, 0.4847, 0.9892],
        [0.1834, 0.5074, 0.9798],
        [0.1786, 0.5289, 0.9682],
        [0.1764, 0.5499, 0.9520],
        [0.1687, 0.5703, 0.9359],
        [0.1540, 0.5902, 0.9218],
        [0.1460, 0.6091, 0.9079],
        [0.1380, 0.6276, 0.8973],
        [0.1248, 0.6459, 0.8883],
        [0.1113, 0.6635, 0.8763],
        [0.0952, 0.6798, 0.8598],
        [0.0689, 0.6948, 0.8394],
        [0.0297, 0.7082, 0.8163],
        [0.0036, 0.7203, 0.7917],
        [0.0067, 0.7312, 0.7660],
        [0.0433, 0.7411, 0.7394],
        [0.0964, 0.7500, 0.7120],
        [0.1408, 0.7584, 0.6842],
        [0.1717, 0.7670, 0.6554],
        [0.1938, 0.7758, 0.6251],
        [0.2161, 0.7843, 0.5923],
        [0.2470, 0.7918, 0.5567],
        [0.2906, 0.7973, 0.5188],
        [0.3406, 0.8008, 0.4789],
        [0.3909, 0.8029, 0.4394],
        [0.4456, 0.8024, 0.4103],
        [0.5044, 0.7993, 0.3906],
        [0.5616, 0.7942, 0.3746],
        [0.6174, 0.7876, 0.3595],
        [0.6720, 0.7793, 0.3446],
        [0.7242, 0.7698, 0.3293],
        [0.7738, 0.7598, 0.3160],
        [0.8203, 0.7498, 0.3057],
        [0.8634, 0.7406, 0.3013],
        [0.9035, 0.7330, 0.3046],
        [0.9393, 0.7288, 0.3194],
        [0.9728, 0.7298, 0.3344],
        [0.9956, 0.7434, 0.3302],
        [0.9970, 0.7659, 0.3002],
        [0.9952, 0.7893, 0.2695],
        [0.9892, 0.8136, 0.2389],
        [0.9786, 0.8386, 0.2099],
        [0.9676, 0.8639, 0.1860],
        [0.9610, 0.8890, 0.1696],
        [0.9597, 0.9135, 0.1645],
        [0.9628, 0.9373, 0.1726],
        [0.9691, 0.9606, 0.1929],
        [0.9769, 0.9839, 0.2207]
    ]
    
    # Create colormap
    parula_cmap = mcolors.LinearSegmentedColormap.from_list('parula', parula_data)
    return parula_cmap

def Tracking(file_path, ID, saved_combined_0, scale = 10, speedup = 1, objective = 63, speedmin = 5, speedmax = 95, progress_callback=None):
    # At the beginning
    if progress_callback:
        progress_callback(0.0)
    filename = os.path.basename(file_path)
    print(f"filename {filename}")
    folder_path = os.path.dirname(file_path)
    os.chdir(folder_path)
    
    output_name = f"Tracking_{ID}_{filename}"
    excelName = f"Location and speeds of particles_{ID}_{filename}"
    # Use parula colormap instead of jet
    colormap_name = "parula"  # Changed from "jet"
    
    # You can also use the CB module if it has parula, but we'll create our own
    # colormap_name = CB.LinearSegmentedColormap()  # Comment this out if CB doesn't have parula
    
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    bar_h = int(height * 0.4)
    bar_w = int(width * 0.02)
    
    # Microscope calibration
    PITCH_VALUES = {5: 2.5945, 10: 1.2972, 20: 0.6486, 40: 0.3243, 63: 0.2059, 100: 0.1297}
    pitch = PITCH_VALUES.get(objective)
    if pitch is None:
        print("The input objective is not correct!")
        
    
    """----------------------------------------- Excel -----------------------------------------"""    
    wb = Workbook()
    ws = wb.active
    
    ws["A1"] = "fps"
    ws["A2"] = f"={fps}"
    
    ws["A4"] = "Objective"
    ws["A5"] = f"={objective}"
    
    ws["A7"] = "pitch (um/px)"
    ws["A8"] = f"={pitch}"
    
    """----------------------------------------- Object detection -----------------------------------------"""
    # Extract all frames
    frames = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frames.append(img)
    cap.release()
    
    def particle_detection(frame_in): 
        edges = cv2.Canny(frame_in, threshold1=80, threshold2=150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        positions = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            positions.append((cx, cy))
        return positions
    
    # Calculate the distance between two points.
    def calculate_distance(point1, point2):
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    # Initialize combined dictionary for saving particles position in each frame.
    combined = {}
    
    combined = saved_combined_0
    
    if progress_callback:
        progress_callback(0.3)
    
    """----------------------------------------- Tracking -----------------------------------------"""
    # Track through all the remaining frames
    N = len(frames)
    max_allowed_distance = 50  # Maximum reasonable movement between frames
    
    for frame_idx in range(1, N):
         
        # Detect particles in current frame
        current_detections = particle_detection(frames[frame_idx])
        
        particle_order = {}
        used_detections = set()  # Track which current detections we've already assigned
        
        # For each existing particle, find its best match in current frame
        for particle_id in list(combined.keys()):
            # Get the last known position of this particle
            last_frame = max(combined[particle_id].keys())
            last_position = combined[particle_id][last_frame]
            
            best_match = None
            best_distance = float('inf')
            best_match_index = None
            
            if current_detections:
                # Find the closest current detection to this particle
                for idx, current_pos in enumerate(current_detections):
                    if idx in used_detections:
                        continue  # Skip already used detections
                        
                    distance = calculate_distance(last_position, current_pos)
                    
                    # Update best match if this is closer and within allowed distance
                    if distance < best_distance and distance < max_allowed_distance:
                        best_distance = distance
                        best_match = current_pos
                        best_match_index = idx
            
            # Only assign if we found a valid match
            if best_match is not None:
                
                particle_order[particle_id] = best_match
                used_detections.add(best_match_index)  # Mark this detection as used
            else:
                pass  
            
                # Don't assign any position - particle disappears
        
        # Update positions for particles that were successfully matched
        for particle_id, new_position in particle_order.items():
            combined[particle_id][frame_idx] = new_position
            
    """----------------------------------------- Speed and colorbar -----------------------------------------"""
    
    speeds = {}  # Dictionary to store all speeds for each particle
    
    for particle_id, positions_dict in combined.items():
       
        # Get sorted frame numbers
        frame_numbers = sorted(positions_dict.keys())
        number_frames = len(frame_numbers)
        
        # Initialize empty list for this particle's speeds
        speeds[particle_id] = {}
        
        # For the first frame, speed is 0
        first_frame = frame_numbers[0]
        speeds[particle_id][first_frame] = 0.0
        
        # Calculate speeds between consecutive frames for this particle
        for i in range(number_frames - 1):
            current_frame = frame_numbers[i]
            next_frame = frame_numbers[i + 1]
            
            # Get positions
            current_pos = positions_dict[current_frame]
            next_pos = positions_dict[next_frame]
            
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]
            speed = np.sqrt(dx**2 + dy**2) * fps * pitch
            
            # Store speed for the transition from current_frame to next_frame
            # We assign the speed to the next_frame since it represents the motion TO that frame
            speeds[particle_id][next_frame] = speed
            
    """----------------------------------------- Excel -----------------------------------------"""    
    if progress_callback:
        progress_callback(0.5)
        
    start_col = 3  # C column
    col_spacing = 4  # Skip 3 columns between value groups
    
    for il_idx, particle_id in enumerate(ID):
        
        # The column for frame
        col_num_1 = start_col -1 + (il_idx * col_spacing)
        col_letter_1 = get_column_letter(col_num_1)
        ws[f"{col_letter_1}1"] = "ID"
        ws[f"{col_letter_1}2"] = "Frame"
        
        # Main column for ID
        col_num = start_col + (il_idx * col_spacing)
        col_letter = get_column_letter(col_num)
        ws[f"{col_letter}1"] = f"{particle_id}"
        ws[f"{col_letter}2"] = "x"
        
        # Second column (x+1) for "y"
        col_num2 = start_col + 1 + (il_idx * col_spacing)
        col_letter2 = get_column_letter(col_num2)
        ws[f"{col_letter2}2"] = "y"
        
        # Third column (x+2) for "Speed"
        col_num3 = start_col + 2 + (il_idx * col_spacing)
        col_letter3 = get_column_letter(col_num3)
        ws[f"{col_letter3}2"] = "Speed"
        
        # Check if this particle ID exists in combined dictionary
        if particle_id in combined:
            # Get all frames for this particle and sort them
            frames_exl = sorted(combined[particle_id].keys())
            
            for row_idx, frame_num in enumerate(frames_exl):
                
                # Fill frame number
                ws[f"{col_letter_1}{row_idx + 3}"] = f"{frame_num}"
                # Start from row 3 for the data
                ws[f"{col_letter}{row_idx + 3}"] = f"{combined[particle_id][frame_num][0]}"  # x position
                ws[f"{col_letter2}{row_idx + 3}"] = f"{combined[particle_id][frame_num][1]}"  # y position
                
                # Add speed data if available
                if particle_id in speeds and frame_num in speeds[particle_id]:
                    ws[f"{col_letter3}{row_idx + 3}"] = f"{speeds[particle_id][frame_num]:.2f}"
                else:
                    ws[f"{col_letter3}{row_idx + 3}"] = "0.00"  # Default speed for first frame
    
    wb.save(f"{excelName}.xlsx")# The colorbar should be designed for all the particles, so the maximum speed should be 
    # the maximum for all the particles.
    def speedbar(speeds):
        
        if len(speeds) == 0:
            raise RuntimeError("No frames found / no speeds computed.")
        
        all_speeds = []
        for particle_speeds in speeds.values():
            all_speeds.extend(particle_speeds.values())
        
        # Use percentile-based range to ignore outliers
        all_speeds_array = np.array(all_speeds)
        vmax = speedmax #np.percentile(all_speeds_array, speedmax)  # 95th percentile to ignore extreme highs
        vmin = speedmin #np.percentile(all_speeds_array, speedmin)   # 5th percentile to ignore extreme lows
        
        # Fallback to min/max if percentiles don't work
        if vmax <= vmin:
            vmax = np.max(all_speeds_array)
            vmin = np.min(all_speeds_array)
        
        # Setup color and speed mapping 
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
        # Use parula colormap - check if it's available or create it
        if colormap_name == "parula":
            cmap = parula_colormap()
        else:
            cmap = plt.get_cmap(colormap_name)
        
        # Precompute color (BGR) for EACH SPEED of EACH PARTICLE
        colors_bgr = {}
        for particle_id, speed_dict in speeds.items():
            colors_bgr[particle_id] = {}
            for frame_num, speed_value in speed_dict.items():
                rgb = np.array(cmap(norm(speed_value))[:3]) * 255
                colors_bgr[particle_id][frame_num] = tuple(int(v) for v in rgb[::-1])  # reverse RGB→BGR 
            
        # Create color image 
        bar = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
        for y in range(bar_h):
            t = 1 - y / (bar_h - 1)
            rgb = np.array(cmap(t)[:3]) * 255
            bar[y, :, :] = rgb  # convert RGB→BGR
        
        bar_bgr = bar
        
        return colors_bgr, bar_bgr, vmax, vmin
    
    # Position of colorbar
    margin = bar_w
    x0 = width - 2*bar_w - margin
    y0 = height//2 - bar_h//2
    
    """----------------------------------------- Draw track trajectory into video -----------------------------------------"""
    
    cap = cv2.VideoCapture(filename)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_name, fourcc, fps * speedup, (width, height))
    
    frame_idx = 0
    num_frames = len(frames)
    
    # Define PIL font
    # Define PIL font with adaptive sizing
    # Base font sizes for different text elements
    base_font_size = min(17, int(width / 40))  # Scale with resolution
    small_font_size = max(10, int(width / 100))  # Smaller for labels
    
    try:
        FONT_REGULAR = ImageFont.truetype("arial.ttf", base_font_size)
        FONT_SMALL = ImageFont.truetype("arial.ttf", small_font_size)
        FONT_LARGE = ImageFont.truetype("arial.ttf", int(base_font_size * 1.2))
    except:
        # Fallback to default font if arial is not available
        FONT_REGULAR = ImageFont.load_default()
        FONT_SMALL = ImageFont.load_default()
        FONT_LARGE = ImageFont.load_default()
    
    TEXT_COLOR = (0, 0, 0)  # Black color
    
    def draw_rotated_text(pil_image, text, position, angle=90, color=(0, 0, 0), font=FONT_LARGE):
        """Draw rotated text using PIL"""
        draw = ImageDraw.Draw(pil_image)
        
        # Create a temporary image for the text
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        temp_img = Image.new('RGBA', (text_width + 10, text_height + 10), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        temp_draw.text((5, 5), text, fill=color + (255,), font=font)  # Add alpha channel
        
        # Rotate the temporary image
        rotated_img = temp_img.rotate(angle, expand=True, resample=Image.BICUBIC)
        
        # Paste rotated text onto main image
        x, y = position
        # Adjust position to center the rotated text
        rotated_width, rotated_height = rotated_img.size
        paste_x = x - rotated_width // 2
        paste_y = y - rotated_height // 2
        
        pil_image.paste(rotated_img, (paste_x, paste_y), rotated_img)
    
    # CREATE STATIC COLORBAR AND TEXT OVERLAY ONCE
    def create_colorbar_overlay():
        """Create a static image with colorbar and rotated text that can be overlaid on frames"""
        # Create a blank image with alpha channel
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Get speed data for colorbar
        speedpart = {k: speeds[k] for k in ID if k in speeds}
        colors_bgr, speedbar_results, speed_max, speed_min = speedbar(speedpart)
        
        # Convert speedbar to PIL image
        speedbar_pil = Image.fromarray(speedbar_results)
        
        # Paste colorbar onto overlay
        overlay.paste(speedbar_pil, (x0, y0))
        label_spacing = base_font_size + 5
        # Draw text labels
        draw.text((x0 - 10, y0 - int(label_spacing * 1.7)), f"{speed_max:.2f}", fill=TEXT_COLOR + (255,), font=FONT_LARGE)
        draw.text((x0 - 10, y0 + bar_h + 7), f"{speed_min:.2f}", fill=TEXT_COLOR + (255,), font=FONT_LARGE)
        
        # Draw rotated text
        rotated_text_x = x0 + bar_w + 8
        rotated_text_y = y0 + bar_h // 2
        # draw_rotated_text(overlay, "Speed (μm/s)", (rotated_text_x, rotated_text_y), 
        #                  angle=90, color=TEXT_COLOR, font=FONT_LARGE)
        
        return overlay, colors_bgr, speed_max, speed_min
    
    # Create the static overlay once
    static_overlay, colors_bgr, speed_max, speed_min = create_colorbar_overlay()
    
    if progress_callback:
        progress_callback(0.8)
    
    while True:
        ret, img = cap.read()
        if not ret or frame_idx >= num_frames:
            break
        frame_idx += 1
        frame_color = img.copy()
        
        # Draw trajectory (this part changes per frame)
        for I in ID:
            if I not in combined:
                continue
            available_frames = sorted(combined[I].keys())
            for i in range(1, len(available_frames)):
                if available_frames[i] < frame_idx:
                    prev_frame = available_frames[i-1]
                    curr_frame = available_frames[i]
                    pt1 = combined[I][prev_frame]
                    pt2 = combined[I][curr_frame]
                    if curr_frame in colors_bgr.get(I, {}):
                        color = colors_bgr[I][curr_frame]
                        cv2.line(frame_color, pt1, pt2, color, 2, cv2.LINE_AA)
        
        # CONVERT TO PIL FOR OVERLAYING
        pil_image = Image.fromarray(cv2.cvtColor(frame_color, cv2.COLOR_BGR2RGB))
        
        # Paste the static overlay (colorbar + text)
        pil_image.paste(static_overlay, (0, 0), static_overlay)
        
        # Draw particle-specific text that changes per frame
        draw = ImageDraw.Draw(pil_image)
        for I in ID:
            if I in combined and (frame_idx - 1) in combined[I] and (frame_idx - 1) in speeds.get(I, {}):
                cx, cy = combined[I][frame_idx - 1]
                cur_speed = speeds[I][frame_idx - 1] 
                
                # draw.text((max(10, cx - 85), max(25, cy - 50)), 
                #          f"Speed: {cur_speed:.2f} μm/s", fill=TEXT_COLOR, font=FONT_REGULAR)
    
        # CONVERT BACK TO OPENCV
        frame_color = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
        out.write(frame_color)
    
    cap.release()
    out.release()
    if progress_callback:
        progress_callback(1)