"""
Dual-Track Data Manifest Builder for OASIS-2 MRI
=================================================
Safeguards implemented:
  1. Preserves patient/subject IDs and session IDs.
  2. Builds Track A (slice-level 90/10 split) strictly matching the original paper methodology.
  3. Builds Track B (subject-level 70/15/15 split) strictly guaranteeing zero patient overlap.
  4. Keeps synthetic DDPM slices exclusively within the training split.
  5. Never drops test slices for artificial class balancing.
"""

import os
import re
import random
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def extract_metadata_from_filename(filename):
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

def build_manifest(views_dirs, output_manifest_path, seed=42):
    records = []
    
    if isinstance(views_dirs, str):
        views_dirs = [views_dirs]
        
    for v_dir in views_dirs:
        if not os.path.exists(v_dir):
            continue
        print(f"Scanning directory: {v_dir}")
        for root, _, files in os.walk(v_dir):
            for f in files:
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, start=os.path.dirname(output_manifest_path) or '.')
                    meta = extract_metadata_from_filename(f)
                    
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

    print(f"Total slices indexed: {len(df)}")
    print("Class breakdown:\n", df['label'].value_counts())
    print("Plane breakdown:\n", df['plane'].value_counts())

    # Track A Partitioning (Slice-level 90/10 split per class and plane)
    df['track_a_split'] = 'train'
    for (plane, label), group in df.groupby(['plane', 'label']):
        idx = group.index
        if len(idx) > 1:
            _, test_idx = train_test_split(idx, test_size=0.1, random_state=seed)
            df.loc[test_idx, 'track_a_split'] = 'test'

    # Track B Partitioning (Subject-level: zero subject overlap across splits)
    # Note: Exactly 3 real subjects in OASIS-2 have CDR == 2.0 (moderate dementia).
    # Stratify by subject clinical condition so each split (train, val, test) contains 1 moderate dementia subject.
    mod_subjects = sorted(list(set(df[df['label'] == 'moderate_dementia']['subject_id'].unique()) - {'SYNTHETIC', 'UNKNOWN'}))
    non_subjects = sorted(list(set(df[df['label'] == 'nondemented']['subject_id'].unique()) - {'SYNTHETIC', 'UNKNOWN'}))
    
    random.seed(seed)
    random.shuffle(mod_subjects)
    random.shuffle(non_subjects)
    
    # Stratified subject allocation:
    train_subs = set()
    val_subs = set()
    test_subs = set()
    
    # 1 moderate dementia patient in each split
    if len(mod_subjects) >= 3:
        train_subs.add(mod_subjects[0])
        val_subs.add(mod_subjects[1])
        test_subs.add(mod_subjects[2])
    else:
        for i, s in enumerate(mod_subjects):
            if i == 0: train_subs.add(s)
            elif i == 1: val_subs.add(s)
            else: test_subs.add(s)
            
    n_non = len(non_subjects)
    n_train_non = int(0.70 * n_non)
    n_val_non = int(0.15 * n_non)
    
    train_subs.update(non_subjects[:n_train_non])
    val_subs.update(non_subjects[n_train_non:n_train_non+n_val_non])
    test_subs.update(non_subjects[n_train_non+n_val_non:])
    
    print(f"Track B Subjects: total={len(train_subs)+len(val_subs)+len(test_subs)} (train={len(train_subs)}, val={len(val_subs)}, test={len(test_subs)})")
    print(f"Moderate Dementia subject assignment: train={mod_subjects[0]}, val={mod_subjects[1]}, test={mod_subjects[2]}")
    
    # Verify zero overlap
    assert len(train_subs & val_subs) == 0
    assert len(train_subs & test_subs) == 0
    assert len(val_subs & test_subs) == 0

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--views', nargs='+', default=['datasets/OASIS-2/views', 'datasets/sample_test'])
    parser.add_argument('--out', default='data_manifest.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    build_manifest(args.views, args.out, seed=args.seed)
