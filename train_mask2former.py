# train_mask2former.py
import os, random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
    get_linear_schedule_with_warmup
)
from torch.cuda.amp import GradScaler
from torch import amp 
from pycocotools.coco import COCO

# ——— 超参数区 ——————————————————————————————
IMG_DIR         = "/root/autodl-tmp/data/armbench-segmentation-0.1/mix-object-tote/images"
TRAIN_JSON      = "/root/autodl-tmp/data/armbench-segmentation-0.1/mix-object-tote/train.json"
VAL_JSON        = "/root/autodl-tmp/data/armbench-segmentation-0.1/mix-object-tote/val.json"
OUTPUT_MODEL    = "best_mask2former_armbench.pth"

NUM_CLASSES     = 3
BATCH_SIZE      = 4
LR              = 1e-5
WEIGHT_DECAY    = 1e-4
WARMUP_RATIO    = 0.1
EPOCHS          = 50
PATIENCE        = 5
DROPOUT         = 0.1
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ————————————————————————————————————————————————

class ARMBenchInstanceSeg(Dataset):
    def __init__(self, img_dir, ann_file, processor):
        self.img_dir   = img_dir
        self.coco      = COCO(ann_file)
        self.ids       = list(sorted(self.coco.imgs.keys()))
        self.processor = processor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info   = self.coco.loadImgs(img_id)[0]
        img_pth= os.path.join(self.img_dir, info["file_name"])
        img    = Image.open(img_pth).convert("RGB")

        H, W = info["height"], info["width"]
        inst_map, inst2cls = np.zeros((H,W),np.int32), {}
        cur = 1
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
            if ann.get("iscrowd",0): continue
            m = self.coco.annToMask(ann)
            inst_map[m>0] = cur
            inst2cls[cur] = ann["category_id"]
            cur += 1
        if len(inst2cls)==0:
            inst_map[:] = 1
            inst2cls={1:0}

        enc = self.processor(
            images=img,
            segmentation_maps=inst_map,
            instance_id_to_semantic_id=inst2cls,
            reduce_labels=True,
            ignore_index=255,
            return_tensors="pt"
        )
        return {
            "pixel_values": enc.pixel_values[0], 
            "pixel_mask":   enc.pixel_mask[0],
            "mask_labels":  enc.mask_labels[0],
            "class_labels": enc.class_labels[0],
        }

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "pixel_mask":   torch.stack([x["pixel_mask"]   for x in batch]),
        "mask_labels":  [x["mask_labels"]  for x in batch],
        "class_labels": [x["class_labels"] for x in batch],
    }

def train():
    print(">>> Loading Mask2Former …")
    processor = Mask2FormerImageProcessor.from_pretrained(
        "facebook/mask2former-swin-base-coco-instance",
        ignore_mismatched_sizes=True
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-instance",
        id2label={i:str(i) for i in range(NUM_CLASSES)},
        ignore_mismatched_sizes=True
    )

    # Dropout
    if DROPOUT > 0:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and "class_predictor" in name:
                mod.dropout = nn.Dropout(DROPOUT)

    model.to(DEVICE)

    # dataloader
    train_ds = ARMBenchInstanceSeg(IMG_DIR, TRAIN_JSON, processor)
    val_ds   = ARMBenchInstanceSeg(IMG_DIR, VAL_JSON,   processor)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=1,           shuffle=False,
                              num_workers=2, pin_memory=True, collate_fn=collate_fn)

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p:p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # AMP
    scaler = GradScaler(enabled=(DEVICE.type=="cuda"))

    # training loop
    best_val, no_imp = float("inf"), 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        loss_tr=0.0
        for batch in tqdm(train_loader, desc=f"[Train] {epoch}/{EPOCHS}"):
            optimizer.zero_grad()
            pv = batch["pixel_values"].to(DEVICE)
            pm = batch["pixel_mask"].to(DEVICE)
            cl = [c.to(DEVICE) for c in batch["class_labels"]]
            ml = [m.to(DEVICE) for m in batch["mask_labels"]]

            with amp.autocast(device_type="cuda"):  # ✅ 替代原始 autocast
                out = model(pixel_values=pv,
                            pixel_mask=pm,
                            class_labels=cl,
                            mask_labels=ml)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_tr += loss.item()
        avg_tr = loss_tr / len(train_loader)

        model.train()
        loss_v=0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Val]   {epoch}/{EPOCHS}"):
                pv = batch["pixel_values"].to(DEVICE)
                pm = batch["pixel_mask"].to(DEVICE)
                cl = [c.to(DEVICE) for c in batch["class_labels"]]
                ml = [m.to(DEVICE) for m in batch["mask_labels"]]

                with amp.autocast(device_type="cuda"):  
                    out = model(pixel_values=pv,
                                pixel_mask=pm,
                                class_labels=cl,
                                mask_labels=ml)
                    loss_v += out.loss.item()
        avg_v = loss_v / len(val_loader)

        print(f"Epoch {epoch}/{EPOCHS}  TrainLoss={avg_tr:.4f}  ValLoss={avg_v:.4f}")

        if avg_v < best_val:
            best_val, no_imp = avg_v, 0
            torch.save(model.state_dict(), OUTPUT_MODEL)
            print(f" ↳ New best: {best_val:.4f}, saved ▶ {OUTPUT_MODEL}")
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f">>> Early stop at epoch {epoch}")
                break

    print(f">>> Done. Best val loss={best_val:.4f}")

if __name__=="__main__":
    train()