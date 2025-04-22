import os
import copy
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import optuna
from optuna.pruners import MedianPruner
from pycocotools.coco import COCO

# 1) Dataset
class ARMBenchDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir    = img_dir
        self.coco       = COCO(ann_file)
        self.ids        = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info   = self.coco.loadImgs(img_id)[0]
        path   = os.path.join(self.img_dir, info["file_name"])
        img    = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)

        boxes, masks, labels, areas, iscrowd = [], [], [], [], []
        for ann in anns:
            if ann.get("iscrowd", 0) == 1: continue
            x,y,w,h = ann["bbox"]
            boxes.append([x, y, x+w, y+h])
            masks.append(self.coco.annToMask(ann))
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(0)

        if not boxes:
            H,W = info["height"], info["width"]
            boxes   = [[0,0,1,1]]
            masks   = [np.zeros((H,W),np.uint8)]
            labels  = [0]
            areas   = [0.0]
            iscrowd = [0]

        boxes   = torch.as_tensor(boxes, dtype=torch.float32)
        masks   = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        labels  = torch.as_tensor(labels, dtype=torch.int64)
        areas   = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "masks":    masks,
            "image_id": torch.tensor([img_id]),
            "area":     areas,
            "iscrowd":  iscrowd,
        }

        if self.transforms:
            img = self.transforms(img)
        return img, target

def get_transform(train: bool):
    t = [T.ToTensor()]
    if train: t.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(t)

def collate_fn(batch):
    return tuple(zip(*batch))

# 2) Model builder
def get_model(num_classes:int, dropout:float, freeze_backbone:bool):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)

    if dropout>0:
        model.backbone.body.layer4[2].relu = nn.Sequential(
            model.backbone.body.layer4[2].relu,
            nn.Dropout(dropout)
        )
    if freeze_backbone:
        for _, p in model.backbone.named_parameters():
            p.requires_grad = False

    return model

# 3) 训练+验证
def train_one_fold(model, train_loader, val_loader, optimizer, device,
                   patience:int, max_epochs:int):
    best_val, no_imp = float("inf"), 0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, max_epochs+1):
        # — train —
        model.train()
        tl = 0.0
        for imgs, tgts in train_loader:
            imgs = [i.to(device) for i in imgs]
            tgts = [{k:v.to(device) for k,v in t.items()} for t in tgts]
            optimizer.zero_grad()
            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()
            tl += loss.item()
        avg_tr = tl / len(train_loader)

        # — val（保留 train 模式）—
        model.train()
        vl = 0.0
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs = [i.to(device) for i in imgs]
                tgts = [{k:v.to(device) for k,v in t.items()} for t in tgts]
                loss_dict = model(imgs, tgts)
                vl += sum(loss_dict.values()).item()
        avg_val = vl / len(val_loader)

        print(f"Epoch {epoch}/{max_epochs}  Train={avg_tr:.3f}  Val={avg_val:.3f}")

        if avg_val < best_val:
            best_val, best_wts, no_imp = avg_val, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_wts)
    return model, best_val

# 4) Optuna objective
def objective(trial):
    lr      = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1])
    bs      = trial.suggest_categorical("batch_size", [2,4])

    ds_sub = Subset(ds_full, list(range(min(2000, len(ds_full)))))
    n_val  = int(0.2 * len(ds_sub))
    tr_ds, vl_ds = random_split(ds_sub, [len(ds_sub)-n_val, n_val])

    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                           num_workers=4, pin_memory=True, collate_fn=collate_fn)
    vl_loader = DataLoader(vl_ds, batch_size=1, shuffle=False,
                           num_workers=2, pin_memory=True, collate_fn=collate_fn)

    model     = get_model(NUM_CLASSES, dropout, freeze_backbone=True).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p:p.requires_grad, model.parameters()), lr=lr
    )

    best_val = float("inf")
    for epoch in range(1, 9):  
        model.train()
        for imgs, tgts in tr_loader:
            imgs = [i.to(device) for i in imgs]
            tgts = [{k:v.to(device) for k,v in t.items()} for t in tgts]
            optimizer.zero_grad()
            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

        model.train()
        vl = 0.0
        with torch.no_grad():
            for imgs, tgts in vl_loader:
                imgs = [i.to(device) for i in imgs]
                tgts = [{k:v.to(device) for k,v in t.items()} for t in tgts]
                loss_dict = model(imgs, tgts)
                vl += sum(loss_dict.values()).item()
        avg_val = vl / len(vl_loader)

        trial.report(avg_val, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        best_val = min(best_val, avg_val)

    return best_val

# 5) 主流程：search → full‑train
if __name__ == "__main__":
    BASE        = "/root/autodl-tmp/data/armbench-segmentation-0.1"
    IMG_DIR     = os.path.join(BASE, "mix-object-tote/images")
    ANN_FILE    = os.path.join(BASE, "mix-object-tote/train.json")
    NUM_CLASSES = 3

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_full = ARMBenchDataset(IMG_DIR, ANN_FILE, transforms=get_transform(True))

    pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=3)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=10)

    best = study.best_trial.params
    print(">>>> Best hyperparams:", best)

    # 全量微调
    n_val = int(0.2 * len(ds_full))
    tr_ds, vl_ds = random_split(ds_full, [len(ds_full)-n_val, n_val])

    train_loader = DataLoader(tr_ds, batch_size=8, shuffle=True,
                              num_workers=8, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(vl_ds, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)

    model     = get_model(NUM_CLASSES, best["dropout"], freeze_backbone=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model, _ = train_one_fold(
        model, train_loader, val_loader, optimizer, device,
        patience=5, max_epochs=30
    )
    scheduler.step()

    torch.save(model.state_dict(), "best_maskrcnn_armbench.pth")
    print("Done, model saved.")