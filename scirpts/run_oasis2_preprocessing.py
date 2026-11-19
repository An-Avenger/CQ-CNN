"""
Automated Standalone OASIS-2 Preprocessing Pipeline
===================================================
1. Extracts OAS2_RAW_PART1.tar.gz and OAS2_RAW_PART2.tar.gz sequentially
   into datasets/OASIS-2/extracted/OAS2_RAW/
2. Converts 3D NIfTI/Analyze volumes (.img/.hdr) into 2D JPG slices
   across Axial, Coronal, and Sagittal views.
3. Maps CDR labels:
     CDR == 0.0 -> nondemented
     CDR == 2.0 -> moderate_dementia
4. Organizes slices into datasets/OASIS-2/views/{view}/{class}/
5. Cleans up large intermediate extracted raw folders to conserve disk space.
"""

import os
import sys
import shutil
import tarfile
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import nibabel as nib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "datasets", "OASIS-2", "downloads")
EXTRACTED_DIR = os.path.join(BASE_DIR, "datasets", "OASIS-2", "extracted")
RAW_DIR = os.path.join(EXTRACTED_DIR, "OAS2_RAW")
RAW_JPG_DIR = os.path.join(EXTRACTED_DIR, "OAS2_RAW_JPG")
VIEWS_DIR = os.path.join(BASE_DIR, "datasets", "OASIS-2", "views")
METADATA_CSV = os.path.join(DOWNLOADS_DIR, "oasis_longitudinal.csv")

def extract_archives():
    print("=" * 70)
    print("STEP 1: Extracting OASIS-2 tar.gz archives...")
    print("=" * 70)
    os.makedirs(RAW_DIR, exist_ok=True)
    
    archives = [
        os.path.join(DOWNLOADS_DIR, "OAS2_RAW_PART1.tar.gz"),
        os.path.join(DOWNLOADS_DIR, "OAS2_RAW_PART2.tar.gz")
    ]
    
    for arc in archives:
        if not os.path.exists(arc):
            raise FileNotFoundError(f"Missing archive: {arc}")
        print(f"Extracting {os.path.basename(arc)} -> {EXTRACTED_DIR} ...")
        with tarfile.open(arc, "r:gz") as tar:
            tar.extractall(path=EXTRACTED_DIR)
            
    # Normalize folder name if extracted under OAS2_RAW_PART1 / OAS2_RAW_PART2
    p1 = os.path.join(EXTRACTED_DIR, "OAS2_RAW_PART1")
    p2 = os.path.join(EXTRACTED_DIR, "OAS2_RAW_PART2")
    
    if os.path.exists(p1) and not os.path.exists(RAW_DIR):
        os.rename(p1, RAW_DIR)
    elif os.path.exists(p1) and os.path.exists(RAW_DIR):
        for item in os.listdir(p1):
            s = os.path.join(p1, item)
            d = os.path.join(RAW_DIR, item)
            if not os.path.exists(d):
                shutil.move(s, d)
        shutil.rmtree(p1, ignore_errors=True)
        
    if os.path.exists(p2):
        for item in os.listdir(p2):
            s = os.path.join(p2, item)
            d = os.path.join(RAW_DIR, item)
            if not os.path.exists(d):
                shutil.move(s, d)
        shutil.rmtree(p2, ignore_errors=True)
        
    print("Archives successfully extracted and consolidated into:", RAW_DIR)

def extract_and_save_views():
    print("=" * 70)
    print("STEP 2: Slicing 3D MRI volumes into 2D JPG images...")
    print("=" * 70)
    os.makedirs(RAW_JPG_DIR, exist_ok=True)
    
    total_files = 0
    for root, dirs, files in os.walk(RAW_DIR):
        if "RAW" in root:
            total_files += len([f for f in files if f.endswith((".img", ".hdr"))])
            
    print(f"Total volume files found: {total_files}")
    
    slides = 255
    l_axial, r_axial = 100, 90
    l_coronal, r_coronal = 90, 110
    l_sagittal, r_sagittal = 40, 40
    
    with tqdm(total=total_files, desc="Slicing volumes", ncols=85) as pbar:
        for root, dirs, files in os.walk(RAW_DIR):
            parent_folder = os.path.basename(os.path.dirname(root))
            relative_path = os.path.relpath(root, RAW_DIR)
            output_subfolder = os.path.join(RAW_JPG_DIR, relative_path)
            
            for file_name in files:
                if file_name.endswith(".img"): # nibabel loads .img along with .hdr
                    nifti_file_path = os.path.join(root, file_name)
                    try:
                        img = nib.load(nifti_file_path)
                        img_data = img.get_fdata()
                    except Exception as e:
                        pbar.update(1)
                        continue
                        
                    if len(img_data.shape) == 3:
                        num_slices = img_data.shape[2]
                    elif len(img_data.shape) == 4:
                        img_data = np.mean(img_data, axis=3)
                        num_slices = img_data.shape[2]
                    else:
                        pbar.update(1)
                        continue
                        
                    for view_name in ["axial", "coronal", "sagittal"]:
                        if view_name == "axial":
                            slices = [img_data[:, i, :] for i in range(0, img_data.shape[0], max(1, img_data.shape[0] // slides))]
                            l_skip, r_skip = l_axial, r_axial
                        elif view_name == "coronal":
                            slices = [img_data[i, :, :] for i in range(0, img_data.shape[1], max(1, img_data.shape[1] // slides))]
                            l_skip, r_skip = l_coronal, r_coronal
                        elif view_name == "sagittal":
                            slices = [img_data[:, :, i] for i in range(0, num_slices, max(1, num_slices // slides))]
                            l_skip, r_skip = l_sagittal, r_sagittal
                            
                        target_slices = slices[l_skip:-r_skip]
                        view_out = os.path.join(output_subfolder, view_name)
                        os.makedirs(view_out, exist_ok=True)
                        
                        for i, img_slice in enumerate(target_slices):
                            if img_slice.size == 0:
                                continue
                            try:
                                max_val = np.max(img_slice)
                                if max_val > 0:
                                    img_norm = np.uint8(255 * (img_slice / max_val))
                                else:
                                    img_norm = np.zeros_like(img_slice, dtype=np.uint8)
                                    
                                pil_img = Image.fromarray(img_norm)
                                if view_name in ["axial", "coronal"]:
                                    pil_img = pil_img.rotate(180, expand=True)
                                elif view_name == "sagittal":
                                    pil_img = pil_img.rotate(90, expand=True)
                                    
                                out_name = f"{parent_folder}_{file_name.replace('.img','')}_{view_name}_slice_{i}.jpg"
                                pil_img.save(os.path.join(view_out, out_name))
                            except Exception:
                                pass
                    pbar.update(1)
                elif file_name.endswith(".hdr"):
                    pbar.update(1)

def organize_into_classes():
    print("=" * 70)
    print("STEP 3: Organizing slices by CDR Clinical Class...")
    print("=" * 70)
    
    df = pd.read_csv(METADATA_CSV)
    print(f"Loaded clinical metadata: {len(df)} visits")
    
    views = ["axial", "coronal", "sagittal"]
    classes = ["nondemented", "moderate_dementia"]
    
    for v in views:
        for c in classes:
            os.makedirs(os.path.join(VIEWS_DIR, v, c), exist_ok=True)
            
    def get_cdr(mri_id):
        m = df[df["MRI ID"] == mri_id]
        if not m.empty:
            return m["CDR"].values[0]
        return None
        
    subfolders = [d for d in os.listdir(RAW_JPG_DIR) if os.path.isdir(os.path.join(RAW_JPG_DIR, d))]
    print(f"Sorting {len(subfolders)} patient scan folders...")
    
    for folder in tqdm(subfolders, desc="Classifying", ncols=85):
        cdr = get_cdr(folder)
        target_class = None
        if cdr == 0.0:
            target_class = "nondemented"
        elif cdr == 2.0:
            target_class = "moderate_dementia"
            
        if target_class:
            for v in views:
                src_view = os.path.join(RAW_JPG_DIR, folder, "RAW", v)
                dst_view = os.path.join(VIEWS_DIR, v, target_class)
                if os.path.exists(src_view):
                    for f in os.listdir(src_view):
                        sf = os.path.join(src_view, f)
                        df_path = os.path.join(dst_view, f)
                        if os.path.isfile(sf):
                            shutil.copy2(sf, df_path)
                            
    # Clean up huge raw extracted folder to save space
    print("Cleaning up intermediate extraction directory to free disk space...")
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    shutil.rmtree(RAW_JPG_DIR, ignore_errors=True)
    print("Organized view directories ready at:", VIEWS_DIR)

if __name__ == "__main__":
    extract_archives()
    extract_and_save_views()
    organize_into_classes()
    print("OASIS-2 preprocessing completed successfully!")
