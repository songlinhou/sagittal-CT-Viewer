"""
Post-processing for AI-predicted vertebrae segmentation masks.

This script implements vertebrae-specific refinement inspired by ShapeKit
(https://github.com/BodyMaps/ShapeKit, MICCAI 2025 Workshop on Shape in Medical Imaging)
and optimized with techniques from the VerSe anatomic consistency cycle
(Meng et al., "Vertebrae localization, segmentation and identification using
a graph optimization and an anatomic consistency cycle", 2022,
https://gitlab.inria.fr/spine/vertebrae_segmentation).

Two modes of operation:
  1. --use_shapekit : Clone and run ShapeKit directly as a plug-and-play tool.
  2. (default)      : Run the standalone built-in postprocessing, which implements
                      the key ShapeKit vertebrae functions locally.

Key postprocessing steps (ShapeKit + VerSe-inspired):
  - Remove small connected components (noise)
  - Suppress non-largest connected components per vertebra
  - Fill holes inside vertebrae volumes
  - Spine adjacent pairs correction (reassign fragments to neighboring vertebrae)
  - Anatomical size consistency validation (lumbar > thoracic > cervical)
  - Residual connected component reassignment (VerSe: recover unassigned spine voxels)
  - Gap detection and filling (VerSe: detect missing vertebrae from Z-axis spacing)
  - Fishing for boundary vertebrae (VerSe: extrapolate beyond detected boundaries)
  - Duplicate vertebrae removal (VerSe: IoU-based merge of overlapping detections)
  - Reallocate based on size (merge extra-small, split extra-large vertebrae)
  - Relabel by Z-axis ordering (enforce anatomical order: L5 at bottom, C1 at top)
  - Balance protrusion between adjacent vertebrae
  - Iterative refinement until convergence (VerSe: anatomic consistency cycle)

Usage:
  # Standalone mode (default)
  python postprocessing_vertebrae.py \
      --input_folder /path/to/AbdomenAtlasDemoPredict \
      --output_folder /path/to/AbdomenAtlasDemoPredict_postprocessed

  # Direct ShapeKit mode (clones and runs ShapeKit)
  python postprocessing_vertebrae.py \
      --input_folder /path/to/AbdomenAtlasDemoPredict \
      --output_folder /path/to/AbdomenAtlasDemoPredict_postprocessed \
      --use_shapekit
"""

import os
import sys
import argparse
import shutil
import subprocess
import numpy as np
import nibabel as nib
import cc3d
from copy import deepcopy
from scipy.ndimage import binary_fill_holes, center_of_mass
from tqdm import tqdm

# ─── Label mapping ───────────────────────────────────────────────────────────
# SuPreM prediction labels: 1=L5, 2=L4, ..., 24=C1
# ShapeKit labels: 26=L5, 27=L4, ..., 49=C1
# We use the SuPreM 1-based mapping internally.

VERTEBRAE_NAMES = [
    "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
    "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8",
    "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3",
    "vertebrae_T2", "vertebrae_T1",
    "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4", "vertebrae_C3",
    "vertebrae_C2", "vertebrae_C1",
]

# label 1..24 → name
LABEL_TO_NAME = {i + 1: name for i, name in enumerate(VERTEBRAE_NAMES)}
NAME_TO_LABEL = {name: i + 1 for i, name in enumerate(VERTEBRAE_NAMES)}


# ─── Utility functions (from ShapeKit/utils/utils.py) ────────────────────────

def fill_holes(mask):
    """Fill small 2D holes in a binary mask."""
    return binary_fill_holes(mask)


def remove_small_components(mask, threshold):
    """Remove connected components smaller than `threshold` voxels."""
    if not np.any(mask):
        return mask
    cc = cc3d.connected_components(mask.astype(np.uint8), connectivity=6)
    labels, counts = np.unique(cc, return_counts=True)
    cleaned = np.zeros_like(mask)
    for lbl, cnt in zip(labels, counts):
        if lbl == 0:
            continue
        if cnt >= threshold:
            cleaned[cc == lbl] = 1
    return cleaned


def suppress_non_largest_components_binary(mask, keep_top=2):
    """Keep only the top-N largest connected components."""
    if not np.any(mask):
        return mask
    cc = cc3d.connected_components(mask.astype(np.uint8), connectivity=6)
    labels_all, counts_all = np.unique(cc, return_counts=True)
    nonzero = labels_all != 0
    labels = labels_all[nonzero]
    counts = counts_all[nonzero]
    if len(labels) > keep_top:
        top_labels = labels[np.argsort(counts)[::-1][:keep_top]]
    else:
        top_labels = labels
    return np.isin(cc, top_labels).astype(mask.dtype)


# ─── Vertebrae-specific functions (from ShapeKit/utils/vertebrae_postprocessing.py) ─

def fill_segmentation(segmentation):
    """Fill holes inside each vertebrae label."""
    result = np.zeros_like(segmentation)
    for label_id in np.unique(segmentation):
        if label_id == 0:
            continue
        mask = (segmentation == label_id).astype(int)
        mask = fill_holes(mask)
        result[mask > 0] = label_id
    return result


def suppress_non_largest_vertebrae(segmentation, default_val=0):
    """Keep only the 2 largest connected components per vertebra label."""
    result = deepcopy(segmentation)
    new_background = np.zeros(segmentation.shape, dtype=bool)
    for label_id in LABEL_TO_NAME:
        cc = cc3d.connected_components(segmentation == label_id, connectivity=6)
        uv, uc = np.unique(cc, return_counts=True)
        if len(uv) < 2:
            continue
        dominant_vals = uv[np.argsort(uc)[::-1][:2]]
        if len(dominant_vals) >= 2:
            keep_mask = (cc == dominant_vals[0]) | (cc == dominant_vals[1])
            new_background |= ~keep_mask & (segmentation == label_id)
    result[new_background] = default_val
    return result


def _merge_cc_of_adjacent(cc_cur, cc_above, voxel_supression_threshold):
    """Merge connected components of adjacent vertebrae that are misplaced."""
    nof_voxels_cc = [(x, np.sum(cc_cur == x)) for x in np.unique(cc_cur)]
    relevant_cc = [(idx, cnt) for idx, cnt in nof_voxels_cc if cnt > voxel_supression_threshold]
    relevant_cc = sorted(relevant_cc, key=lambda x: x[1], reverse=True)[1:]  # remove background

    nof_voxels_above = [(x, np.sum(cc_above == x)) for x in np.unique(cc_above)]
    relevant_cc_above = [(idx, cnt) for idx, cnt in nof_voxels_above if cnt > voxel_supression_threshold]
    relevant_cc_above = sorted(relevant_cc_above, key=lambda x: x[1], reverse=True)[2:]  # remove bg + main

    if len(relevant_cc_above) > 0:
        pool = np.zeros(cc_cur.shape, dtype=bool)
        for idx, _ in relevant_cc_above:
            pool |= cc_above == idx
        for idx, _ in relevant_cc:
            pool |= cc_cur == idx
        cc_pool = cc3d.connected_components(pool)
        rel_pool = sorted([(x, np.sum(cc_pool == x)) for x in np.unique(cc_pool)], key=lambda x: x[1], reverse=True)[1:]
        if len(rel_pool) > 0:
            return cc_pool == rel_pool[0][0]
    return None


def spine_adjacent_pairs(segmentation, voxel_supression_threshold=1000, default_val=0):
    """
    Check alternating connected components to identify fragments assigned to the wrong vertebra.
    For each vertebra, examine its neighbors above and below, and reassign misplaced fragments.
    """
    labels = list(LABEL_TO_NAME.keys())  # 1..24
    mod_img = deepcopy(segmentation)

    for idx, current in enumerate(labels):
        above = labels[idx - 1] if idx > 0 else None
        below = labels[idx + 1] if idx < len(labels) - 1 else None

        msk_cur = mod_img == current
        cc_cur = cc3d.connected_components(msk_cur, connectivity=6)

        # Suppress small fragments
        nof_voxels = [(x, np.sum(cc_cur == x)) for x in np.unique(cc_cur)]
        for cc_id, cnt in nof_voxels:
            if cnt <= voxel_supression_threshold and cc_id != 0:
                mod_img[cc_cur == cc_id] = default_val

        if above is not None:
            msk_above = mod_img == above
            cc_above = cc3d.connected_components(msk_above, connectivity=6)
            consolidated = _merge_cc_of_adjacent(cc_cur, cc_above, voxel_supression_threshold)
            if consolidated is not None:
                mod_img[consolidated] = current

        if below is not None:
            msk_below = mod_img == below
            cc_below = cc3d.connected_components(msk_below, connectivity=6)
            consolidated = _merge_cc_of_adjacent(cc_cur, cc_below, voxel_supression_threshold)
            if consolidated is not None:
                mod_img[consolidated] = current

    return mod_img


def relabel_by_z_order(segmentation, label_z_centers, start_label=1):
    """
    Relabel vertebrae based on Z-axis center ordering (bottom to top).
    The lowest vertebra (largest Z in image coords, i.e. inferior) gets label 1 (L5),
    the highest gets the largest label (C1).
    """
    sorted_labels = sorted(label_z_centers.items(), key=lambda x: x[1], reverse=False)
    new_seg = segmentation.copy()
    new_z_centers = {}
    for new_id, (old_id, z_center) in enumerate(sorted_labels, start=start_label):
        new_seg[segmentation == old_id] = new_id
        new_z_centers[new_id] = z_center
    return new_seg, new_z_centers


def split_overmerged_triplets(merged_seg, size_dict, label_z_centers, counter, size_threshold_ratio=1.5):
    """
    Split over-merged vertebrae: if label i is much larger than min(label i-1, i-2), split by Z-axis.
    """
    sorted_labels = sorted(size_dict.keys(), reverse=True)
    next_new_label = np.max(merged_seg) + 1

    for i in range(2, len(sorted_labels)):
        i2, i1, i0 = sorted_labels[i - 2], sorted_labels[i - 1], sorted_labels[i]
        if i0 not in size_dict or i1 not in size_dict or i2 not in size_dict:
            continue
        threshold = size_threshold_ratio * min(size_dict[i1], size_dict[i2])
        if size_dict[i0] > threshold and counter > 0:
            mask = merged_seg == i0
            coords = np.argwhere(mask)
            if coords.shape[0] == 0:
                continue
            sorted_coords = coords[np.argsort(coords[:, 2])[::-1]]
            half = len(sorted_coords) // 2
            coords_upper = sorted_coords[:half]
            coords_lower = sorted_coords[half:]
            for voxel in coords_lower:
                merged_seg[tuple(voxel)] = next_new_label
            size_dict[i0] = len(coords_upper)
            size_dict[next_new_label] = len(coords_lower)
            label_z_centers[i0] = np.median(coords_upper[:, 2])
            label_z_centers[next_new_label] = np.median(coords_lower[:, 2])
            next_new_label += 1
            counter -= 1
    return merged_seg, label_z_centers


def balance_protrusion(segmentation, label_z_centers, min_cc_voxel=1000):
    """
    For each adjacent pair (A=lower label, B=higher label):
    - If a sub-region of A protrudes above B's center, reassign it to B.
    - If a sub-region of B drops below A's center, reassign it to A.
    """
    corrected = segmentation.copy()
    sorted_labels = sorted(label_z_centers.keys())
    for i in range(len(sorted_labels) - 1):
        A = sorted_labels[i]
        B = sorted_labels[i + 1]
        z_A = label_z_centers[A]
        z_B = label_z_centers[B]

        cc_A = cc3d.connected_components(corrected == A, connectivity=6)
        for cc_id in np.unique(cc_A):
            if cc_id == 0:
                continue
            coords = np.argwhere(cc_A == cc_id)
            if coords.shape[0] < min_cc_voxel:
                continue
            if np.median(coords[:, 2]) > z_B:
                corrected[cc_A == cc_id] = B

        cc_B = cc3d.connected_components(corrected == B, connectivity=6)
        for cc_id in np.unique(cc_B):
            if cc_id == 0:
                continue
            coords = np.argwhere(cc_B == cc_id)
            if coords.shape[0] < min_cc_voxel:
                continue
            if np.median(coords[:, 2]) < z_A:
                corrected[cc_B == cc_id] = A
    return corrected


def reallocate_based_on_size(segmentation):
    """
    Handle extra-small (merge into nearest neighbor) and extra-large (split) vertebrae.
    Then relabel by Z-order and balance protrusion.
    """
    size_dict = {}
    label_z_centers = {}
    for label_id in np.unique(segmentation):
        if label_id == 0:
            continue
        mask = segmentation == label_id
        mask = remove_small_components(mask, threshold=max(int(np.sum(mask) / 10), 100))
        coords = np.argwhere(mask)
        if coords.shape[0] == 0:
            continue
        label_z_centers[label_id] = np.median(coords[:, 2])
        size_dict[label_id] = np.sum(mask)

    # Merge extra-small vertebrae
    size_threshold_ratio = 2 / 3
    merged_seg = segmentation.copy()
    to_merge = []
    for label_id in label_z_centers:
        neighbors = [label_id - 1, label_id + 1]
        neighbor_sizes = [size_dict.get(n, 0) for n in neighbors if n in size_dict]
        if len(neighbor_sizes) < 2:
            continue
        if size_dict[label_id] < size_threshold_ratio * np.mean(neighbor_sizes):
            to_merge.append(label_id)

    split_counter = len(to_merge)
    for label_id in to_merge:
        min_dist = np.inf
        nearest = None
        z = label_z_centers[label_id]
        for other_id, other_z in label_z_centers.items():
            if other_id == label_id:
                continue
            dist = abs(z - other_z)
            if dist < min_dist:
                min_dist = dist
                nearest = other_id
        if nearest is not None:
            merged_seg[merged_seg == label_id] = nearest
            size_dict[nearest] = size_dict.get(nearest, 0) + size_dict[label_id]
            del size_dict[label_id]
            del label_z_centers[label_id]

    # Split extra-large vertebrae
    split_seg, label_z_centers = split_overmerged_triplets(
        merged_seg, size_dict, label_z_centers, counter=split_counter
    )

    # Relabel by Z-order
    new_seg, label_z_centers = relabel_by_z_order(split_seg, label_z_centers)

    # Balance protrusion
    new_seg = balance_protrusion(new_seg, label_z_centers)

    return new_seg


# ─── VerSe-inspired functions (Meng et al., 2022) ────────────────────────────
# Adapted from https://gitlab.inria.fr/spine/vertebrae_segmentation
# These functions implement key ideas from the anatomic consistency cycle:
# residual connected components, gap detection, fishing, and iterative refinement.

def find_residual_components(binary_spine, segmentation, min_size=500):
    """
    Find connected components in the residual between the binary spine mask
    and the current individual vertebrae labels.
    Adapted from VerSe's filtered_connected_components (Meng et al., 2022).
    """
    individual_union = (segmentation > 0).astype(np.uint8)
    residual = binary_spine.astype(np.uint8) - individual_union
    residual[residual != 1] = 0

    if not np.any(residual):
        return []

    from scipy.ndimage import binary_erosion
    if np.sum(residual) > min_size * 5:
        residual = binary_erosion(residual).astype(np.uint8)

    cc = cc3d.connected_components(residual, connectivity=6)
    components = []
    labels_cc, counts = np.unique(cc, return_counts=True)

    for lbl, cnt in zip(labels_cc, counts):
        if lbl == 0:
            continue
        if cnt >= min_size:
            components.append((cc == lbl, cnt))

    return components


def reassign_residual_to_nearest(segmentation, residual_components, label_z_centers):
    """
    Reassign residual connected components to the nearest vertebra
    based on 3D centroid distance.
    """
    if not residual_components or not label_z_centers:
        return segmentation

    result = segmentation.copy()

    # Precompute vertebrae centroids
    centroids = {}
    for label_id in label_z_centers:
        mask = segmentation == label_id
        if not np.any(mask):
            continue
        coords = np.argwhere(mask)
        centroids[label_id] = (np.median(coords[:, 0]),
                               np.median(coords[:, 1]),
                               np.median(coords[:, 2]))

    for component, cnt in residual_components:
        coords = np.argwhere(component)
        if coords.shape[0] == 0:
            continue
        cx, cy, cz = np.median(coords[:, 0]), np.median(coords[:, 1]), np.median(coords[:, 2])

        min_dist = np.inf
        nearest_label = 0
        for label_id, (mx, my, mz) in centroids.items():
            dist = np.sqrt((cx - mx)**2 + (cy - my)**2 + (cz - mz)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_label = label_id

        if nearest_label > 0:
            result[component] = nearest_label

    return result


def detect_and_fill_gaps(segmentation, label_z_centers, binary_spine, min_size=500, gap_threshold=1.8):
    """
    Detect anomalously large gaps between consecutive vertebrae Z-centers.
    Try to assign residual components in the gap region to a new vertebra.
    Adapted from VerSe's get_extra_locations_from_intervals (Meng et al., 2022).
    """
    if len(label_z_centers) < 3:
        return segmentation

    result = segmentation.copy()
    sorted_labels = sorted(label_z_centers.keys())
    z_centers = [label_z_centers[l] for l in sorted_labels]

    gaps = [abs(z_centers[i+1] - z_centers[i]) for i in range(len(z_centers)-1)]
    median_gap = np.median(gaps)
    if median_gap == 0:
        return result

    for i, gap in enumerate(gaps):
        if gap > gap_threshold * median_gap:
            z_low = min(z_centers[i], z_centers[i+1])
            z_high = max(z_centers[i], z_centers[i+1])

            # Look for residual components in the gap region
            gap_mask = binary_spine.copy()
            gap_mask[result > 0] = 0
            gap_mask[:, :, :int(z_low)] = 0
            gap_mask[:, :, int(z_high)+1:] = 0

            if np.any(gap_mask):
                cc = cc3d.connected_components(gap_mask.astype(np.uint8), connectivity=6)
                labels_cc, counts = np.unique(cc, return_counts=True)
                for lbl, cnt in zip(labels_cc, counts):
                    if lbl == 0 or cnt < min_size:
                        continue
                    # Assign to the label that should be in this gap
                    # The missing label is between sorted_labels[i] and sorted_labels[i+1]
                    expected_label = min(sorted_labels[i], sorted_labels[i+1]) + 1
                    if expected_label not in label_z_centers:
                        result[cc == lbl] = expected_label
                        print(f"    Gap filled: assigned {cnt} voxels to label {expected_label}")

    return result


def fishing_for_boundary_vertebrae(segmentation, label_z_centers, binary_spine, min_size=500):
    """
    Check for missing vertebrae at the inferior and superior boundaries.
    If the lowest detected vertebra is not L5 (label 1) or highest is not C1 (label 24),
    look for residual components beyond the boundaries.
    Adapted from VerSe's fish_up and fish_down (Meng et al., 2022).
    """
    if not label_z_centers:
        return segmentation

    result = segmentation.copy()
    sorted_labels = sorted(label_z_centers.keys())
    z_centers = [label_z_centers[l] for l in sorted_labels]

    # Inferior boundary: should reach label 1 (L5), which has the smallest z
    lowest_label = sorted_labels[0]
    lowest_z = z_centers[0]

    if lowest_label > 1:
        below_mask = binary_spine.copy()
        below_mask[result > 0] = 0
        below_mask[:, :, int(lowest_z + 5):] = 0

        if np.any(below_mask):
            cc = cc3d.connected_components(below_mask.astype(np.uint8), connectivity=6)
            labels_cc, counts = np.unique(cc, return_counts=True)
            for lbl, cnt in zip(labels_cc, counts):
                if lbl == 0 or cnt < min_size:
                    continue
                new_label = lowest_label - 1
                result[cc == lbl] = new_label
                print(f"    Fished inferior: assigned {cnt} voxels to label {new_label}")

    # Superior boundary: should reach label 24 (C1), which has the largest z
    highest_label = sorted_labels[-1]
    highest_z = z_centers[-1]

    if highest_label < 24:
        above_mask = binary_spine.copy()
        above_mask[result > 0] = 0
        above_mask[:, :, :int(highest_z - 5)] = 0

        if np.any(above_mask):
            cc = cc3d.connected_components(above_mask.astype(np.uint8), connectivity=6)
            labels_cc, counts = np.unique(cc, return_counts=True)
            for lbl, cnt in zip(labels_cc, counts):
                if lbl == 0 or cnt < min_size:
                    continue
                new_label = highest_label + 1
                result[cc == lbl] = new_label
                print(f"    Fished superior: assigned {cnt} voxels to label {new_label}")

    return result


def remove_duplicate_vertebrae(segmentation, iou_threshold=0.5):
    """
    Check for overlapping vertebrae masks (duplicated detections).
    If two adjacent labels have IoU > threshold, merge the smaller into the larger.
    Adapted from VerSe's duplicated_locations_index (Meng et al., 2022).
    """
    labels_present = [l for l in np.unique(segmentation) if l > 0]
    if len(labels_present) < 2:
        return segmentation

    result = segmentation.copy()

    for i in range(len(labels_present) - 1):
        l1 = labels_present[i]
        l2 = labels_present[i + 1]
        m1 = result == l1
        m2 = result == l2

        intersection = np.logical_and(m1, m2).sum()
        if intersection == 0:
            continue

        union = np.logical_or(m1, m2).sum()
        iou = intersection / union if union > 0 else 0

        if iou > iou_threshold:
            # Merge smaller into larger
            if m1.sum() < m2.sum():
                result[m1] = l2
            else:
                result[m2] = l1
            print(f"    Merged duplicate labels {l1} and {l2} (IoU={iou:.3f})")

    return result


def check_anatomical_size_consistency(segmentation, label_z_centers, size_dict):
    """
    Validate vertebrae sizes against anatomical expectations.
    Lumbar (L1-L5) should be larger than thoracic (T1-T12), which should be larger than cervical (C1-C7).
    Flag outliers that deviate significantly from their group median.
    Adapted from VerSe's statistical prior validation (Meng et al., 2022).
    """
    if not size_dict or len(size_dict) < 3:
        return segmentation, label_z_centers, size_dict

    result = segmentation.copy()
    new_z_centers = dict(label_z_centers)
    new_size = dict(size_dict)

    sorted_labels = sorted(new_z_centers.keys())

    # Group by region: 1-5 = lumbar, 6-17 = thoracic, 18-24 = cervical
    lumbar = [l for l in sorted_labels if l <= 5]
    thoracic = [l for l in sorted_labels if 6 <= l <= 17]
    cervical = [l for l in sorted_labels if l >= 18]

    lumbar_median = np.median([new_size[l] for l in lumbar]) if lumbar else 0
    thoracic_median = np.median([new_size[l] for l in thoracic]) if thoracic else 0
    cervical_median = np.median([new_size[l] for l in cervical]) if cervical else 0

    # Remove cervical vertebrae that are abnormally small (< 20% of cervical median)
    for label_id in cervical:
        if cervical_median > 0 and new_size.get(label_id, 0) < 0.2 * cervical_median:
            print(f"    Anatomical check: label {label_id} too small for cervical, removing")
            result[result == label_id] = 0
            if label_id in new_z_centers:
                del new_z_centers[label_id]
            if label_id in new_size:
                del new_size[label_id]

    # Remove lumbar vertebrae that are abnormally small (< 20% of lumbar median)
    for label_id in lumbar:
        if lumbar_median > 0 and new_size.get(label_id, 0) < 0.2 * lumbar_median:
            print(f"    Anatomical check: label {label_id} too small for lumbar, removing")
            result[result == label_id] = 0
            if label_id in new_z_centers:
                del new_z_centers[label_id]
            if label_id in new_size:
                del new_size[label_id]

    return result, new_z_centers, new_size


def iterative_refinement(segmentation, max_iter=3, voxel_supression_threshold=1000):
    """
    Run the postprocessing pipeline iteratively until convergence.
    Each iteration: clean → residual reassignment → gap fill → fishing → reallocate.
    Adapted from VerSe's consistency_refinement_close_loop (Meng et al., 2022).
    """
    prev_seg = None

    for iteration in range(max_iter):
        print(f"  ─ Iteration {iteration+1}/{max_iter} ─")

        # Save binary spine mask before processing (union of all labels)
        binary_spine = (segmentation > 0).astype(np.uint8)

        # Step 1: Remove small noise components per vertebra
        for label_id in LABEL_TO_NAME:
            mask = (segmentation == label_id).astype(np.uint8)
            if mask.sum() == 0:
                continue
            cleaned = remove_small_components(mask, threshold=500)
            segmentation[~cleaned.astype(bool) & (segmentation == label_id)] = 0

        # Step 2: Suppress non-largest connected components (keep top 2)
        segmentation = suppress_non_largest_vertebrae(segmentation)

        # Step 3: Fill holes
        segmentation = fill_segmentation(segmentation)

        # Step 4: Spine adjacent pairs correction
        segmentation = spine_adjacent_pairs(segmentation, voxel_supression_threshold=voxel_supression_threshold)

        # Step 5: Compute label centers and sizes
        label_z_centers = {}
        size_dict = {}
        for label_id in np.unique(segmentation):
            if label_id == 0:
                continue
            mask = segmentation == label_id
            coords = np.argwhere(mask)
            if coords.shape[0] == 0:
                continue
            label_z_centers[label_id] = np.median(coords[:, 2])
            size_dict[label_id] = np.sum(mask)

        # Step 6: Anatomical size consistency check
        segmentation, label_z_centers, size_dict = check_anatomical_size_consistency(
            segmentation, label_z_centers, size_dict
        )

        # Step 7: Reassign residual components to nearest vertebrae
        residual_components = find_residual_components(binary_spine, segmentation, min_size=500)
        if residual_components:
            print(f"    Found {len(residual_components)} residual components, reassigning...")
            segmentation = reassign_residual_to_nearest(segmentation, residual_components, label_z_centers)

        # Step 8: Detect and fill gaps
        segmentation = detect_and_fill_gaps(segmentation, label_z_centers, binary_spine, min_size=500)

        # Step 9: Fishing for boundary vertebrae
        segmentation = fishing_for_boundary_vertebrae(segmentation, label_z_centers, binary_spine, min_size=500)

        # Step 10: Remove duplicates
        segmentation = remove_duplicate_vertebrae(segmentation, iou_threshold=0.5)

        # Step 11: Reallocate based on size (merge small, split large, relabel, balance)
        segmentation = reallocate_based_on_size(segmentation)

        # Step 12: Fill holes again
        segmentation = fill_segmentation(segmentation)

        # Check convergence
        if prev_seg is not None:
            changed = np.sum(prev_seg != segmentation)
            total = np.sum((prev_seg > 0) | (segmentation > 0))
            if total > 0:
                change_pct = changed / total
                print(f"    Change rate: {change_pct:.4f}")
                if change_pct < 0.01:
                    print("  Convergence reached.")
                    break

        prev_seg = segmentation.copy()

    # Final cleanup
    segmentation[segmentation > len(VERTEBRAE_NAMES)] = 0
    segmentation[segmentation < 0] = 0

    return segmentation


# ─── Main postprocessing pipeline ────────────────────────────────────────────

def postprocess_case(case_dir, output_dir, voxel_supression_threshold=1000):
    """
    Postprocess a single case: read combined_labels.nii.gz, apply vertebrae
    refinement, save corrected combined_labels and individual segmentations.
    """
    combined_path = os.path.join(case_dir, "combined_labels.nii.gz")
    if not os.path.exists(combined_path):
        print(f"[skip] No combined_labels.nii.gz in {case_dir}")
        return

    nii = nib.load(combined_path)
    seg = nii.get_fdata().astype(np.int32)
    if seg.ndim == 4:
        seg = np.squeeze(seg, axis=0)
    affine = nii.affine

    print(f"  Input labels: {sorted(np.unique(seg).tolist())}")
    print(f"  Shape: {seg.shape}")

    # Run iterative refinement pipeline (VerSe-inspired consistency loop)
    seg = iterative_refinement(seg, max_iter=3, voxel_supression_threshold=voxel_supression_threshold)

    print(f"  Output labels: {sorted(np.unique(seg).tolist())}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    seg_uint8 = seg.astype(np.uint8)

    # Save combined labels
    combined_out = os.path.join(output_dir, "combined_labels.nii.gz")
    nib.save(nib.Nifti1Image(seg_uint8, affine), combined_out)
    print(f"  [saved] {combined_out}")

    # Save individual segmentations
    seg_dir = os.path.join(output_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    for label_id, name in LABEL_TO_NAME.items():
        organ_mask = (seg == label_id).astype(np.uint8)
        organ_path = os.path.join(seg_dir, f"{name}.nii.gz")
        nib.save(nib.Nifti1Image(organ_mask, affine), organ_path)
    print(f"  [saved] {len(os.listdir(seg_dir))} individual masks in {seg_dir}")

    # Copy ct.nii.gz if present
    ct_src = os.path.join(case_dir, "ct.nii.gz")
    if os.path.exists(ct_src):
        ct_dst = os.path.join(output_dir, "ct.nii.gz")
        if not os.path.exists(ct_dst):
            shutil.copy(ct_src, ct_dst)
            print(f"  [copied] ct.nii.gz")


def run_shapekit(input_folder, output_folder):
    """
    Direct mode: clone ShapeKit, configure for vertebrae, and run it.
    """
    shapekit_dir = os.path.join(os.path.dirname(__file__), "ShapeKit")
    if not os.path.isdir(shapekit_dir):
        print("[ShapeKit] Cloning repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/BodyMaps/ShapeKit.git", shapekit_dir],
            check=True,
        )

    # Configure config.yaml for vertebrae-only processing
    config_path = os.path.join(shapekit_dir, "config.yaml")
    config_content = """\
subfolder_name: segmentations

class_map:
  26: vertebrae_L5
  27: vertebrae_L4
  28: vertebrae_L3
  29: vertebrae_L2
  30: vertebrae_L1
  31: vertebrae_T12
  32: vertebrae_T11
  33: vertebrae_T10
  34: vertebrae_T9
  35: vertebrae_T8
  36: vertebrae_T7
  37: vertebrae_T6
  38: vertebrae_T5
  39: vertebrae_T4
  40: vertebrae_T3
  41: vertebrae_T2
  42: vertebrae_T1
  43: vertebrae_C7
  44: vertebrae_C6
  45: vertebrae_C5
  46: vertebrae_C4
  47: vertebrae_C3
  48: vertebrae_C2
  49: vertebrae_C1

target_organs:
  - vertebrae

organ_adjacency_map: {}

affine_reference_file_name: vertebrae_L1.nii.gz

if_save_combined_label: True
"""
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"[ShapeKit] Config written to {config_path}")

    # Prepare input: ShapeKit expects labels 26-49, but SuPreM uses 1-24.
    # We need to remap the combined_labels and segmentation filenames.
    temp_input = os.path.join(output_folder + "_shapekit_input")
    os.makedirs(temp_input, exist_ok=True)
    for case_name in sorted(os.listdir(input_folder)):
        case_dir = os.path.join(input_folder, case_name)
        if not os.path.isdir(case_dir):
            continue
        combined_path = os.path.join(case_dir, "combined_labels.nii.gz")
        if not os.path.exists(combined_path):
            continue
        nii = nib.load(combined_path)
        seg = nii.get_fdata().astype(np.int32)
        if seg.ndim == 4:
            seg = np.squeeze(seg, axis=0)
        # Remap 1-24 → 26-49
        remapped = np.zeros_like(seg)
        for old_label in range(1, 25):
            new_label = old_label + 25
            remapped[seg == old_label] = new_label
        temp_case_dir = os.path.join(temp_input, case_name)
        os.makedirs(temp_case_dir, exist_ok=True)
        temp_seg_dir = os.path.join(temp_case_dir, "segmentations")
        os.makedirs(temp_seg_dir, exist_ok=True)
        nib.save(nib.Nifti1Image(remapped.astype(np.uint8), nii.affine),
                 os.path.join(temp_case_dir, "combined_labels.nii.gz"))
        for label_id, name in LABEL_TO_NAME.items():
            mask = (remapped == label_id + 25).astype(np.uint8)
            nib.save(nib.Nifti1Image(mask, nii.affine),
                     os.path.join(temp_seg_dir, f"{name}.nii.gz"))

    # Run ShapeKit
    log_dir = os.path.join(output_folder + "_logs")
    os.makedirs(log_dir, exist_ok=True)
    cmd = [
        sys.executable, "-W", "ignore", "main.py",
        "--input_folder", temp_input,
        "--output_folder", output_folder,
        "--log_folder", log_dir,
        "--cpu_count", str(min(os.cpu_count() or 4, 4)),
        "--continue_prediction",
    ]
    print(f"[ShapeKit] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=shapekit_dir, check=True)

    # Remap output back from 26-49 → 1-24
    for case_name in sorted(os.listdir(output_folder)):
        case_dir = os.path.join(output_folder, case_name)
        combined_path = os.path.join(case_dir, "combined_labels.nii.gz")
        if not os.path.exists(combined_path):
            continue
        nii = nib.load(combined_path)
        seg = nii.get_fdata().astype(np.int32)
        if seg.ndim == 4:
            seg = np.squeeze(seg, axis=0)
        remapped = np.zeros_like(seg)
        for old_label in range(26, 50):
            new_label = old_label - 25
            remapped[seg == old_label] = new_label
        nib.save(nib.Nifti1Image(remapped.astype(np.uint8), nii.affine), combined_path)
        seg_dir = os.path.join(case_dir, "segmentations")
        if os.path.isdir(seg_dir):
            for label_id, name in LABEL_TO_NAME.items():
                mask_path = os.path.join(seg_dir, f"{name}.nii.gz")
                if os.path.exists(mask_path):
                    mask_nii = nib.load(mask_path)
                    mask = mask_nii.get_fdata().astype(np.uint8)
                    nib.save(nib.Nifti1Image(mask, mask_nii.affine), mask_path)

    # Cleanup temp
    shutil.rmtree(temp_input, ignore_errors=True)
    print("[ShapeKit] Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process AI-predicted vertebrae segmentation masks (ShapeKit-inspired)"
    )
    parser.add_argument("--input_folder", required=True,
                        help="Folder containing case subdirs with combined_labels.nii.gz")
    parser.add_argument("--output_folder", required=True,
                        help="Folder to save postprocessed results")
    parser.add_argument("--use_shapekit", action="store_true",
                        help="Clone and run ShapeKit directly (requires git + network)")
    parser.add_argument("--voxel_threshold", type=int, default=1000,
                        help="Minimum voxel count for a fragment to be considered (default: 1000)")
    args = parser.parse_args()

    if args.use_shapekit:
        run_shapekit(args.input_folder, args.output_folder)
        return

    # Standalone mode
    cases = sorted(
        d for d in os.listdir(args.input_folder)
        if os.path.isdir(os.path.join(args.input_folder, d))
        and os.path.exists(os.path.join(args.input_folder, d, "combined_labels.nii.gz"))
    )
    if not cases:
        print(f"No cases with combined_labels.nii.gz found under {args.input_folder}")
        return

    print(f"Found {len(cases)} cases to postprocess\n")
    for case_name in cases:
        case_dir = os.path.join(args.input_folder, case_name)
        output_dir = os.path.join(args.output_folder, case_name)
        print(f"─ Processing: {case_name} ─")
        postprocess_case(case_dir, output_dir, voxel_supression_threshold=args.voxel_threshold)
        print()

    print("All cases postprocessed.")


if __name__ == "__main__":
    main()
