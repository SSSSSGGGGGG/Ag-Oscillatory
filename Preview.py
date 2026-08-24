# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 18:33:14 2025

@author: gaosh
"""

import cv2
import matplotlib.pyplot as plt
import os

def Preview(file_path, threshold1, threshold2):
    # Read .avi
    file_name = os.path.basename(file_path)
    folder_path = os.path.dirname(file_path)
    os.chdir(folder_path)
    cap = cv2.VideoCapture(file_name)
    
    """----------------------------------------- Object detection -----------------------------------------"""
    # Read the first frame.
    ret, img = cap.read()
    cap.release()
    
    edges = cv2.Canny(img, threshold1=threshold1, threshold2=threshold2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
    positions = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        positions.append((cx, cy))
    
    # All the detected particles assigned with ID in the first frame.
    combined = {}
    
    offset = 10    
    for idx, i in enumerate(positions):
        # Store positions as dictionary with particle numbers as keys
        combined[idx + 1] = {0: i}  # Frame 0 position
        # Calculate label position ensuring it stays within left and top boundaries
        x_pos = int(i[0]) - 2 * offset
        y_pos = int(i[1]) - offset
        
        # Ensure label doesn't go beyond left boundary
        if x_pos < 5:
            x_pos = 5
            
        # Ensure label doesn't go beyond top boundary  
        if y_pos < 20:  # Need some space for text height
            y_pos = int(i[1]) + offset  # Place below the particle instead
        
        cv2.putText(img, f"{idx+1}:{i}", (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    number_particle = len(positions)
    plt.imsave("Preview.png", img)
    
    return img, combined, number_particle