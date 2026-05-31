#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pred2feat(seg, info):
    seg = seg.cpu()
    feat = torch.zeros([80 + 54, 320, 512], dtype=torch.float32)

    for pred in info:
        mask = (seg == pred["id"]).float()
        if pred["isthing"]:
            feat[pred["category_id"], :, :] = mask * float(pred["score"])
        else:
            feat[pred["category_id"] + 80, :, :] = mask

    feat = F.interpolate(
        feat.unsqueeze(0),
        size=[20, 32],
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    return feat


def get_dcbs(img_path, predictor, blur_radius=1):
    high = Image.open(img_path).convert("RGB").resize((512, 320))
    low = high.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    high_panoptic_seg, high_segments_info = predictor(np.array(high))["panoptic_seg"]
    low_panoptic_seg, low_segments_info = predictor(np.array(low))["panoptic_seg"]

    high_feat = pred2feat(high_panoptic_seg, high_segments_info)
    low_feat = pred2feat(low_panoptic_seg, low_segments_info)

    return high_feat, low_feat


def build_predictor(config_path, weights_path=None, score_thresh=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)

    if weights_path is not None:
        cfg.MODEL.WEIGHTS = weights_path

    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    predictor = DefaultPredictor(cfg)
    return predictor


def main():
    # ======== 改成你自己的路径 ========
    image_root = Path("/home/zx2/COSP/COD10K_HAT/images")
    hr_root = Path("/home/zx2/COSP/COD10K_HAT/DCBs/HR")
    lr_root = Path("/home/zx2/COSP/COD10K_HAT/DCBs/LR")

    detectron2_config = "/home/zx2/COSP/COSP-main/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
    detectron2_weights = "detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl"

    blur_radius = 1
    overwrite = False
    # =================================

    predictor = build_predictor(
        config_path=detectron2_config,
        weights_path=detectron2_weights,
        score_thresh=0.5,
    )

    task_dirs = sorted([p for p in image_root.iterdir() if p.is_dir()])
    if not task_dirs:
        print("images 根目录下没有 task 子目录。")
        return

    processed = 0
    skipped = 0
    failed = []

    for task_dir in task_dirs:
        task_name = task_dir.name
        hr_task_dir = hr_root / task_name
        lr_task_dir = lr_root / task_name
        hr_task_dir.mkdir(parents=True, exist_ok=True)
        lr_task_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in task_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTS
        )

        for img_path in tqdm(image_paths, desc=f"Processing {task_name}"):
            feat_name = img_path.name[:-3] + "pth.tar"
            hr_save_path = hr_task_dir / feat_name
            lr_save_path = lr_task_dir / feat_name

            if hr_save_path.exists() and lr_save_path.exists() and not overwrite:
                skipped += 1
                continue

            try:
                high_feat, low_feat = get_dcbs(
                    img_path=img_path,
                    predictor=predictor,
                    blur_radius=blur_radius
                )

                torch.save(high_feat.cpu(), hr_save_path)
                torch.save(low_feat.cpu(), lr_save_path)

                processed += 1

            except Exception as e:
                failed.append((str(img_path), str(e)))

    print("\n========== DCB Precompute Summary ==========")
    print(f"成功处理: {processed}")
    print(f"跳过已有: {skipped}")
    print(f"失败数: {len(failed)}")

    if failed:
        print("\n失败示例（前20个）:")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()