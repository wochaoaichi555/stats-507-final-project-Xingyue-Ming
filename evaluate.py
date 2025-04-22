# stats507/evaluate.py

import os
import time
import random
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.models.detection import maskrcnn_resnet50_fpn

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation
)


def load_maskrcnn(ckpt_pth, num_classes, device):
    """Load a Mask R‑CNN model from checkpoint."""
    model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    sd = torch.load(ckpt_pth, map_location='cpu')
    model.load_state_dict(sd)
    return model.eval().to(device)


def load_mask2former(ckpt_pth, num_classes, device):
    """Load a Mask2Former model and its processor from checkpoint."""
    proc = Mask2FormerImageProcessor.from_pretrained(
        "facebook/mask2former-swin-base-coco-instance",
        ignore_mismatched_sizes=True
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-instance",
        id2label={i: str(i) for i in range(num_classes)},
        ignore_mismatched_sizes=True
    )
    sd = torch.load(ckpt_pth, map_location='cpu')
    model.load_state_dict(sd, strict=False)
    return proc, model.eval().to(device)


def to_coco_results_maskrcnn(pred, img_id):
    """Convert Mask R‑CNN output into COCO‐style detection list."""
    results = []
    boxes  = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    for box, sc, lb in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        results.append({
            "image_id":    int(img_id),
            "category_id": int(lb),
            "bbox":        [float(x1), float(y1), float(w), float(h)],
            "score":       float(sc)
        })
    return results


def to_coco_results_mask2former(vis, img_id):
    """Convert Mask2Former post‐processed output into COCO‐style list."""
    results = []
    seg_map = vis['segmentation'].cpu().numpy()  # H×W int mask of instance IDs
    for seginfo in vis['segments_info']:
        inst_id = seginfo['id']
        label   = seginfo['label_id']
        score   = seginfo['score']
        # binary mask for this instance
        mask = (seg_map == inst_id).astype(np.uint8)
        # RLE encode
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode('ascii')
        # bounding box
        x, y, w, h = mask_utils.toBbox(rle)
        results.append({
            "image_id":     int(img_id),
            "category_id":  int(label),
            "segmentation": rle,
            "bbox":         [float(x), float(y), float(w), float(h)],
            "score":        float(score)
        })
    return results


def eval_coco(coco_gt, results, iou_type='segm'):
    """Run COCOeval and return key metrics."""
    coco_dt = coco_gt.loadRes(results)
    E = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    E.evaluate()
    E.accumulate()
    E.summarize()
    s = E.stats
    return {"mAP@[.5:.95]": s[0], "mAP@.5": s[1], "mAP@.75": s[2]}


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coco_gt = COCO(args.ann)
    img_ids = sorted(coco_gt.getImgIds())
    to_tensor = ToTensor()

    # 1) Load models
    print(">>> Loading Mask R‑CNN …")
    mrcnn = load_maskrcnn(args.maskrcnn_pth, args.num_classes, device)
    print(">>> Loading Mask2Former …")
    m2f_proc, m2f = load_mask2former(args.mask2former_pth, args.num_classes, device)

    # 2) Count GT instances per image
    inst_counts = [
        sum(1 for a in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
            if a.get("iscrowd", 0) == 0)
        for img_id in img_ids
    ]

    # 3) Mask R‑CNN inference, timing, and peak memory
    print("\n>>> Running Mask R‑CNN inference …")
    res_mrcnn, times_mrcnn = [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for img_id in tqdm(img_ids):
        info   = coco_gt.loadImgs(img_id)[0]
        img_pil= Image.open(os.path.join(args.images, info["file_name"])).convert("RGB")
        img_t  = to_tensor(img_pil).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out1 = mrcnn([img_t])[0]
        t1 = time.perf_counter()

        times_mrcnn.append(t1 - t0)
        res_mrcnn += to_coco_results_maskrcnn(out1, img_id)

    peak_mem_mrcnn = (torch.cuda.max_memory_allocated(device) / 1024**3
                      if device.type == "cuda" else 0.0)

    # 4) Mask2Former inference, timing, and peak memory
    print("\n>>> Running Mask2Former inference …")
    res_m2f, times_m2f = [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for img_id in tqdm(img_ids):
        info    = coco_gt.loadImgs(img_id)[0]
        img_pil = Image.open(os.path.join(args.images, info["file_name"])).convert("RGB")
        enc     = m2f_proc(images=img_pil, return_tensors="pt").to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out2 = m2f(**enc)
            vis  = m2f_proc.post_process_instance_segmentation(
                out2, target_sizes=[(info["height"], info["width"])]
            )[0]
        t1 = time.perf_counter()

        times_m2f.append(t1 - t0)
        res_m2f += to_coco_results_mask2former(vis, img_id)

    peak_mem_m2f = (torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda" else 0.0)

    # 5) Compute mAP metrics
    print("\n>>> Mask R‑CNN Evaluation")
    stats1 = eval_coco(coco_gt, res_mrcnn, "segm")
    print("\n>>> Mask2Former Evaluation")
    stats2 = eval_coco(coco_gt, res_m2f, "segm")

    # 6) Aggregate inference time by instance density
    df_time = pd.DataFrame({
        "instances":  inst_counts,
        "mrcnn_time": times_mrcnn,
        "m2f_time":   times_m2f,
    })
    df_time["bin"] = pd.cut(
        df_time["instances"],
        bins=[0,5,10,15,20,30,50,100],
        right=False
    )
    agg_time = df_time.groupby("bin")[["mrcnn_time","m2f_time"]].mean()

    # 7) Summary table combining accuracy and efficiency
    df_metrics = pd.DataFrame([
        {
            "Model":                      "Mask R‑CNN",
            "mAP@[.5:.95]":               stats1["mAP@[.5:.95]"],
            "mAP@.5":                     stats1["mAP@.5"],
            "mAP@.75":                    stats1["mAP@.75"],
            "Avg Inference Time (s/img)": np.mean(times_mrcnn),
            "Peak GPU Mem (GB)":          peak_mem_mrcnn,
        },
        {
            "Model":                      "Mask2Former",
            "mAP@[.5:.95]":               stats2["mAP@[.5:.95]"],
            "mAP@.5":                     stats2["mAP@.5"],
            "mAP@.75":                    stats2["mAP@.75"],
            "Avg Inference Time (s/img)": np.mean(times_m2f),
            "Peak GPU Mem (GB)":          peak_mem_m2f,
        },
    ])
    print("\n=== Summary Table ===")
    print(df_metrics.to_markdown(index=False))

    # 8) Save results
    os.makedirs("eval_results", exist_ok=True)
    df_metrics.to_csv("eval_results/summary_metrics.csv", index=False)
    with open("eval_results/summary_table.md", "w") as f:
        f.write(df_metrics.to_markdown(index=False))
    agg_time.to_csv("eval_results/inference_time_bins.csv")

    # 9) (可选) Qualitative visualizations…

    print("\n✅ All results saved to ./eval_results/")


if __name__ == "__main__":
    import os
    p = argparse.ArgumentParser()
    p.add_argument(
        "--maskrcnn-pth",
        default="best_maskrcnn_armbench.pth",
        help="Path to Mask R‑CNN checkpoint (default: best_maskrcnn_armbench.pth)"
    )
    p.add_argument(
        "--mask2former-pth",
        default="best_mask2former_armbench.pth",
        help="Path to Mask2Former checkpoint (default: best_mask2former_armbench.pth)"
    )
    p.add_argument(
        "--images",
        default=os.path.join(
            "/root/autodl-tmp/data/armbench-segmentation-0.1",
            "mix-object-tote", "images"
        ),
        help="Directory containing validation images"
    )
    p.add_argument(
        "--ann",
        default=os.path.join(
            "/root/autodl-tmp/data/armbench-segmentation-0.1",
            "mix-object-tote", "val.json"
        ),
        help="COCO-format validation annotation JSON"
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of semantic classes (default: 3)"
    )
    args = p.parse_args()
    main(args)