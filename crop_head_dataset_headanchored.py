#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crop_head_dataset_headanchored.py
---------------------------------
Simple, robust pre-crop for *upper-body* scans when the target is the HEAD.
Anchors the Z-range near the 'top' (superior) or 'bottom' (inferior) end so the head is included.
XY is centered with optional shifts. Never uses labels to decide the crop.

Usage:
  python crop_head_dataset_headanchored.py <src_root> <dst_root> [xy_radius] [z_extent] [xy_target] [anchor] [z_offset] [y_shift] [x_shift]
"""
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
# Read NIfTI via SimpleITK
def read_image(path): return sitk.ReadImage(str(path))
def write_image(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))
# Compute ROI bounds as (zmin,zmax,ymin,ymax,xmin,xmax)
def compute_bbox(shape_zyx, xy_radius, z_extent, anchor='top', z_offset=8, y_shift=0, x_shift=0):
    Z, Y, X = shape_zyx
    z_extent = int(z_extent)
    if z_extent <= 0: raise ValueError("z_extent must be > 0")
    if anchor.lower() == 'top':
        zmin = max(0, int(z_offset))
        zmax = min(Z - 1, zmin + z_extent - 1)
    elif anchor.lower() == 'bottom':
        zmax = min(Z - 1, Z - 1 - int(z_offset))
        zmin = max(0, zmax - z_extent + 1)
    else:
        raise ValueError("anchor must be 'top' or 'bottom'")
    yc = Y // 2 + int(y_shift); xc = X // 2 + int(x_shift)
    ymin = max(0, yc - xy_radius); ymax = min(Y - 1, yc + xy_radius)
    xmin = max(0, xc - xy_radius); xmax = min(X - 1, xc + xy_radius)
    return (zmin, zmax, ymin, ymax, xmin, xmax)
# Crop ROI using (Z,Y,X) bounds; SimpleITK expects (X,Y,Z) start/size
def roi_crop(img, bbox):
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    size_xyz  = [int(xmax-xmin+1), int(ymax-ymin+1), int(zmax-zmin+1)]
    start_xyz = [int(xmin), int(ymin), int(zmin)]
    return sitk.RegionOfInterest(img, size_xyz, start_xyz)
# Center pad/crop in XY to (xy_target, xy_target); keep Z
def center_pad_or_crop_xy(img, xy_target):
    arr = sitk.GetArrayFromImage(img)  # (z,y,x)
    Z, Y, X = arr.shape
    out = np.zeros((Z, xy_target, xy_target), dtype=arr.dtype)
    yd = max(0, (xy_target - Y)//2); xd = max(0, (xy_target - X)//2)
    yd_end = yd + min(Y, xy_target);  xd_end = xd + min(X, xy_target)
    ys = max(0, (Y - xy_target)//2);  xs = max(0, (X - xy_target)//2)
    ys_end = ys + (yd_end - yd);      xs_end = xs + (xd_end - xd)
    out[:, yd:yd_end, xd:xd_end] = arr[:, ys:ys_end, xs:xs_end]
    out_img = sitk.GetImageFromArray(out)
    out_img.SetSpacing(img.GetSpacing()); out_img.SetDirection(img.GetDirection()); out_img.SetOrigin(img.GetOrigin())
    return out_img
# Process one case folder
def process_case(case_dir, out_dir, xy_radius, z_extent, xy_target, anchor, z_offset, y_shift, x_shift):
    t1c_path = case_dir / "COR_T1_C.nii"
    if not t1c_path.exists():
        return False, "skip: no COR_T1_C.nii"
    t1c = read_image(t1c_path)
    shape_zyx = sitk.GetArrayFromImage(t1c).shape
    bbox = compute_bbox(shape_zyx, xy_radius, z_extent, anchor, z_offset, y_shift, x_shift)
    t1c_roi = roi_crop(t1c, bbox)
    t1c_xy  = center_pad_or_crop_xy(t1c_roi, xy_target)
    mask_path = case_dir / "mask.nii"
    m_xy = None
    if mask_path.exists():
        m = read_image(mask_path)
        m_roi = roi_crop(m, bbox)
        m_xy  = center_pad_or_crop_xy(m_roi, xy_target)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_image(t1c_xy, out_dir / "COR_T1_C.nii")
    if m_xy is not None: write_image(m_xy, out_dir / "mask.nii")
    return True, "ok"
# Process all cases in a split (train/val)
def process_split(split_src, split_dst, xy_radius, z_extent, xy_target, anchor, z_offset, y_shift, x_shift):
    kept, skipped = 0, 0
    for case in sorted(p for p in split_src.iterdir() if p.is_dir()):
        ok, msg = process_case(case, split_dst / case.name, xy_radius, z_extent, xy_target, anchor, z_offset, y_shift, x_shift)
        if ok: kept += 1
        else: skipped += 1
        print(f"[{split_src.name}] {case.name}: {msg}")
    print(f"[{split_src.name}] kept={kept}, skipped={skipped}")
# CLI entry
def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    src_root = Path(sys.argv[1]); dst_root = Path(sys.argv[2])
    xy_radius = int(sys.argv[3]) if len(sys.argv) > 3 else 144
    z_extent  = int(sys.argv[4]) if len(sys.argv) > 4 else 96
    xy_target = int(sys.argv[5]) if len(sys.argv) > 5 else 288
    anchor    =       sys.argv[6] if len(sys.argv) > 6 else 'top'
    z_offset  = int(sys.argv[7]) if len(sys.argv) > 7 else 8
    y_shift   = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    x_shift   = int(sys.argv[9]) if len(sys.argv) > 9 else 0

    for split in ("train", "val"):
        ss = src_root / split
        if not ss.exists():
            print(f"skip split: {split} (not found)"); continue
        dd = dst_root / split
        process_split(ss, dd, xy_radius, z_extent, xy_target, anchor, z_offset, y_shift, x_shift)

if __name__ == "__main__":
    main()
