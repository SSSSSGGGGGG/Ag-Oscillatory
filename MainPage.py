# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 14:36:32 2025

@author: gaosh
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from PIL import Image, ImageTk

# Create main window
root_pre = tk.Tk()
root_pre.title("Particle Track Tool")
root_pre.geometry("800x640")

# Variables
file_path = tk.StringVar()
thred1_var = tk.StringVar(value="80")
thred2_var = tk.StringVar(value="150")
speedmin_var = tk.StringVar(value="0")
speedmax_var = tk.StringVar(value="20")
particle_id_var = tk.StringVar(value="1")
speedup_var = tk.StringVar(value="1")
objective_var = tk.StringVar(value="63")
scale_var = tk.StringVar(value="10")

saved_preview_img = None
saved_combined_0 = None
saved_file_path = None

# Main container using PanedWindow for resizable left and right sections
main_paned = ttk.PanedWindow(root_pre, orient=tk.HORIZONTAL)
main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Left panel for controls
left_frame = ttk.Frame(main_paned, padding="10")
main_paned.add(left_frame, weight=1)  # weight=0 means fixed size for left panel

# File selection section
file_frame = ttk.LabelFrame(left_frame, text="Video File Selection", padding="20")
file_frame.pack(fill=tk.X, pady=(0, 10))

def select_video_file():
    global saved_file_path  # Add this to make it valid for all script
    filename = filedialog.askopenfilename(
        title="Select AVI file",
        filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
    )
    
    if filename:
        file_path.set(filename)
        file_label.config(text=f"Selected: {filename.split('/')[-1]}")
        saved_file_path = filename

ttk.Label(file_frame, text="Select .avi file").grid(row=0, column=0, sticky="w", padx=(0, 10))
ttk.Button(file_frame, text="Browse...", command=select_video_file).grid(row=0, column=1, padx=(0, 10))
file_label = ttk.Label(file_frame, text="No file selected", foreground="gray")
file_label.grid(row=0, column=2, sticky="w")

# Preview settings section
settings_frame = ttk.LabelFrame(left_frame, text="Preview Settings (Intensity Interval)", padding="20")
settings_frame.pack(fill=tk.X, pady=(0, 10))

# Threshold 1 (lower limit)
ttk.Label(settings_frame, text="Lower Limit").grid(row=0, column=0, sticky="w", padx=(0, 10))
ttk.Entry(settings_frame, textvariable=thred1_var, width=10).grid(row=0, column=1, sticky="w", padx=(0, 20))
ttk.Label(settings_frame, text="(Default: 80)", foreground="gray").grid(row=0, column=2, sticky="w")

# Threshold 2 (upper limit)
ttk.Label(settings_frame, text="Upper Limit").grid(row=1, column=0, sticky="w", padx=(0, 10))
ttk.Entry(settings_frame, textvariable=thred2_var, width=10).grid(row=1, column=1, sticky="w", padx=(0, 20))
ttk.Label(settings_frame, text="(Default: 150)", foreground="gray").grid(row=1, column=2, sticky="w")

# Right panel for preview
preview_frame = ttk.LabelFrame(main_paned, text="Preview Area", padding="5")
preview_frame.pack_propagate(False)  # Prevent frame from resizing to fit contents
preview_frame.config(width=400, height=400)  # Set fixed size
main_paned.add(preview_frame, weight=1)  # weight=1 means expandable for right panel

# Add a canvas inside the preview frame to see preview
preview_canvas = tk.Canvas(preview_frame, background="lightgray")
preview_canvas.pack(fill=tk.BOTH, expand=True)

import Preview as pr
canvas_width = 400
canvas_height = 400

def validate_thresholds(threshold1, threshold2):
    """Validate that threshold1 is smaller than threshold2"""
    if threshold1 >= threshold2:
        messagebox.showwarning("Warning", "Threshold 1 must be smaller than Threshold 2!")
        return False
    return True

def validate_speed_range(speedmin, speedmax):
    """Validate that speedmin is smaller than speedmax and both are within 0-100"""
    if speedmin >= speedmax:
        messagebox.showwarning("Warning", "Speed min must be smaller than Speed max!")
        return False
    if speedmin < 0 or speedmax > 100:
        messagebox.showwarning("Warning", "Speed values must be between 0 and 100!")
        return False
    return True

def refresh_preview():
    """Only refresh the preview image if it already exists"""
    if saved_preview_img is not None:
        try:
            # Convert array to PIL Image
            pil_image = Image.fromarray(saved_preview_img)
            # Resize image to fit canvas while maintaining aspect ratio
            pil_image.thumbnail((canvas_width-20, canvas_height-20), Image.Resampling.LANCZOS)
            # Convert to PhotoImage
            photo_image = ImageTk.PhotoImage(pil_image)
            # Update canvas with the preview image
            preview_canvas.delete("all")
            preview_canvas.create_image(canvas_width//2, canvas_height//2, image=photo_image, anchor=tk.CENTER)
            # Keep reference to prevent garbage collection
            preview_canvas.image = photo_image
        except Exception as e:
            print(f"Error refreshing preview: {e}")

# Save preview parameters and call preview
def preview_video():
    global saved_combined_0, saved_preview_img, num  
    
    if not file_path.get():
        messagebox.showwarning("Warning", "Please select a video file first!")
        return
    
    try:
        # Get thresholds as floats
        threshold1 = float(thred1_var.get())
        threshold2 = float(thred2_var.get()) 
        
        # Validate thresholds only when preview button is clicked
        if not validate_thresholds(threshold1, threshold2):
            return
        
        saved_preview_img, saved_combined_0, num = pr.Preview(file_path.get(), threshold1, threshold2)
        # Update the preview display
        refresh_preview()
        
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid input format: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"Error generating preview: {e}")

# REMOVE the trace_add calls for automatic refresh
# thred1_var.trace_add("write", refresh_preview)
# thred2_var.trace_add("write", refresh_preview)

# Instead, create a simple callback that just shows the previous preview if it exists
def on_threshold_change(*args):
    """Only refresh display if preview already exists, without re-running Preview"""
    if saved_preview_img is not None:
        refresh_preview()

# Add traces for simple refresh (not re-running Preview)
thred1_var.trace_add("write", on_threshold_change)
thred2_var.trace_add("write", on_threshold_change)

preview_button = ttk.Button(settings_frame, text="Preview IDs", command=preview_video)
preview_button.grid(row=7, column=0, columnspan=3, sticky="", pady=(20, 0))

# Configure grid to center the button
settings_frame.grid_columnconfigure(0, weight=1)
settings_frame.grid_columnconfigure(1, weight=0)
settings_frame.grid_columnconfigure(2, weight=1)

# Tracking settings section
settings_frame_tr = ttk.LabelFrame(left_frame, text="Track Settings", padding="20")
settings_frame_tr.pack(fill=tk.X, pady=(0, 10))

# ID 
ttk.Label(settings_frame_tr, text="Input IDs:").grid(row=0, column=0, sticky="w", padx=(0, 10))
ttk.Entry(settings_frame_tr, textvariable=particle_id_var, width=10).grid(row=0, column=1, sticky="w", padx=(0, 20))
ttk.Label(settings_frame_tr, text="(Default: 1,2)", foreground="gray").grid(row=0, column=2, sticky="w")

# Speedup 
ttk.Label(settings_frame_tr, text="Speedup").grid(row=1, column=0, sticky="w", padx=(0, 10))
ttk.Entry(settings_frame_tr, textvariable=speedup_var, width=10).grid(row=1, column=1, sticky="w", padx=(0, 20))
ttk.Label(settings_frame_tr, text="(Default: 1)", foreground="gray").grid(row=1, column=2, sticky="w")

# Objective
ttk.Label(settings_frame_tr, text="Objective").grid(row=2, column=0, sticky="w", padx=(0, 10))
ttk.Entry(settings_frame_tr, textvariable=objective_var, width=10).grid(row=2, column=1, sticky="w", padx=(0, 20))
ttk.Label(settings_frame_tr, text="(Default: 63)", foreground="gray").grid(row=2, column=2, sticky="w")

# Speed min.
ttk.Label(settings_frame_tr, text="Speed min.").grid(row=3, column=0, sticky="w", padx=(0, 10))
ttk.Entry(settings_frame_tr, textvariable=speedmin_var, width=10).grid(row=3, column=1, sticky="w", padx=(0, 20))
ttk.Label(settings_frame_tr, text="(Default: 0 μm/s)", foreground="gray").grid(row=3, column=2, sticky="w")

# Speed max.
ttk.Label(settings_frame_tr, text="Speed max.").grid(row=4, column=0, sticky="w", padx=(0, 10))
ttk.Entry(settings_frame_tr, textvariable=speedmax_var, width=10).grid(row=4, column=1, sticky="w", padx=(0, 20))
ttk.Label(settings_frame_tr, text="(Default: 20 μm/s)", foreground="gray").grid(row=4, column=2, sticky="w")

# Scale
ttk.Label(settings_frame_tr, text="Scale").grid(row=5, column=0, sticky="w", padx=(0, 10))
ttk.Entry(settings_frame_tr, textvariable=scale_var, width=10).grid(row=5, column=1, sticky="w", padx=(0, 20))
ttk.Label(settings_frame_tr, text="(Default: 10 μm)", foreground="gray").grid(row=5, column=2, sticky="w")

import Tracking_m as tr
# Process bar section 
process_frame = ttk.LabelFrame(root_pre, text="Process", padding="10")
process_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 10))

progress_bar = ttk.Progressbar(process_frame, mode='determinate', maximum=100)
progress_bar.pack(fill=tk.X, padx=5, pady=5)

def Finish():
    root_pre.destroy()

finish_button = ttk.Button(process_frame, text="Finish", command=Finish)
finish_button.pack(anchor=tk.E)

def update_progress(progress_value):
    """Update progress bar from another thread"""
    percentage = int(progress_value * 100)
    root_pre.after(0, lambda: progress_bar.config(value=percentage))
    
    if percentage >= 100:
        root_pre.after(0, lambda: finish_button.config(state=tk.NORMAL))
    
def StartTrack():
    if not file_path.get():
        messagebox.showwarning("Warning", "Please select a video file first!")
        return
    if saved_combined_0 is None:
        messagebox.showwarning("Warning", "Please generate preview first!")
        return
    try:
        # Get input values
        speedup = float(speedup_var.get())
        objective = float(objective_var.get()) 
        if objective not in (5, 10, 20, 40, 63, 100):
            messagebox.showwarning("Warning", "Please input the correct number!")
        speedmin = float(speedmin_var.get())
        speedmax = float(speedmax_var.get())
        # Validate speed range
        if not validate_speed_range(speedmin, speedmax):
            return
        
        scale = float(scale_var.get())
        
        # Get particle IDs as array of integers
        id_text = particle_id_var.get().strip()
        
        if not id_text:
            particle_ids = []
        else:
            particle_ids = [int(x.strip()) for x in id_text.split(',')]
        for pid in particle_ids:
                if pid < 1 or pid > num:
                    messagebox.showwarning("Warning", "ID number is not valid!")
                    return
        # Reset progress bar
        progress_bar.config(value=0)

        finish_button.config(state=tk.DISABLED)
        # Start processing in a separate thread - call external Tracking directly
        processing_thread = threading.Thread(
            target=tr.Tracking,
            args=(file_path.get(), particle_ids, saved_combined_0, scale, speedup, objective, speedmin, speedmax, update_progress)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid input format: {e}")
        finish_button.config(state=tk.NORMAL)

# THEN create the Start button
start_button = ttk.Button(settings_frame_tr, text="Start", command=StartTrack)
start_button.grid(row=7, column=0, columnspan=3, sticky="", pady=(20, 0))

# Add proper cleanup protocol
root_pre.protocol("WM_DELETE_WINDOW", root_pre.destroy)

root_pre.mainloop()