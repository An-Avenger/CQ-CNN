import os
import re
import csv
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def extract_metadata_from_filename(filename):
    """
    Parses OASIS-2 filename formats:
    Real: OAS2_0001_MR1_mpr-1_axial_slice_45.jpg
    Synthetic: axial_Epoch_750_generated_image_1.png
    """
    is_synthetic = 'generated' in filename or 'Epoch' in filename
    if is_synthetic:
        plane = 'axial' if 'axial' in filename else ('coronal' if 'coronal' in filename else 'sagittal')
        return {
            'subject_id': 'SYNTHETIC',
            'session_id': 'SYNTHETIC',
            'plane': plane,
            'slice_idx': -1,
            'is_synthetic': True
        }
    
    # Real scan regex
    match = re.search(r'(OAS2_\d+)_([A-Za-z0-9]+)_.*?(axial|coronal|sagittal)_slice_(\d+)', filename, re.IGNORECASE)
    if match:
        return {
            'subject_id': match.group(1),
            'session_id': match.group(2),
            'plane': match.group(3).lower(),
            'slice_idx': int(match.group(4)),
            'is_synthetic': False
        }
    return {
        'subject_id': 'UNKNOWN',
        'session_id': 'UNKNOWN',
        'plane': 'unknown',
        'slice_idx': -1,
        'is_synthetic': False
    }

def build_manifest(views_dir, output_manifest_path, seed=42):
    """
    Traverses views/ or preprocessed/ directories, parses subject IDs,
    and assigns both Track A (slice-level) and Track B (subject-level) splits.
    """
    records = []
    
    # Walk dataset directory
    for root, _, files in os.walk(views_dir):
        for f in files:
            if f.endswith(('.jpg', '.png', '.jpeg')):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, start=views_dir)
                meta = extract_metadata_from_filename(f)
                
                # Determine label from directory hierarchy
                label = 'moderate_dementia' if 'moderate_dementia' in full_path.lower() else 'nondemented'
                is_skull_stripped = 'ss' in full_path.lower() or 'skullstrip' in full_path.lower()
                
                records.append({
                    'slice_id': f,
                    'subject_id': meta['subject_id'],
                    'session_id': meta['session_id'],
                    'plane': meta['plane'],
                    'slice_idx': meta['slice_idx'],
                    'is_skull_stripped': is_skull_stripped,
                    'label': label,
                    'is_synthetic': meta['is_synthetic'],
                    'file_path': rel_path
                })
                
    df = pd.DataFrame(records)
    if df.empty:
        print("No image records found to populate manifest.")
        return df

    # Track A Partitioning (Slice-level 90/10 random split per class and plane)
    df['track_a_split'] = 'train'
    for (plane, label), group in df.groupby(['plane', 'label']):
        idx = group.index
        if len(idx) > 1:
            _, test_idx = train_test_split(idx, test_size=0.1, random_state=seed)
            df.loc[test_idx, 'track_a_split'] = 'test'

    # Track B Partitioning (Subject-level: zero subject overlap between splits)
    real_subjects = [s for s in df['subject_id'].unique() if s not in ['SYNTHETIC', 'UNKNOWN']]
    random.seed(seed)
    random.shuffle(real_subjects)
    
    n_total = len(real_subjects)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    
    train_subs = set(real_subjects[:n_train])
    val_subs = set(real_subjects[n_train:n_train+n_val])
    test_subs = set(real_subjects[n_train+n_val:])
    
    def assign_track_b(row):
        if row['is_synthetic']:
            return 'train'  # Clinical safeguard: synthetics ONLY in train
        sub = row['subject_id']
        if sub in train_subs:
            return 'train'
        elif sub in val_subs:
            return 'val'
        elif sub in test_subs:
            return 'test'
        return 'train'

    df['track_b_split'] = df.apply(assign_track_b, axis=1)

    df.to_csv(output_manifest_path, index=False)
    print(f"Manifest written to {output_manifest_path} with {len(df)} records.")
    return df

if __name__ == '__main__':
    build_manifest('datasets/sample_test', 'data_manifest.csv')

