#!/usr/bin/env python3
r"""
Windows-CPU batch inference with Grad-CAM++ (no resizing) for EfficientNet-B2, ResNet-50 & MobileNet-V3
-------------------------------------------------------------------------------------------------------
• Uses your exact class names by default (11 classes), can be overridden with --classes
• Auto-detects num_classes from checkpoint (prefers fc.weight/fc.1.weight/classifier.3.weight)
• **No image resizing** — uses original image size
• Saves (Original | Grad-CAM++) per image + per-model CSV (filename, prediction, prob, runtime_ms)
"""

import os, glob, time, warnings, argparse, re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
import scipy.ndimage as ndi

import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b2, EfficientNet_B2_Weights,
    resnet50, ResNet50_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights
)

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------------------- DEFAULT CLASS NAMES (exact order) ----------------- #
DEFAULT_CLASSES = [
    "Conglomerate",
    "Cross bedding",
    "Lenticular bedding",
    "Low-angle bedding",
    "Wavy bedding",
    "Bioturbated muddy media",
    "Bioturbated sandy media",
    "Massive mudstone",
    "Massive sandstone",
    "mud drape",
    "Parallel lamination",
]  # 11 classes

# ---------------------- models ------------------------------------------- #
class EffB2(nn.Module):
    def __init__(self, n):
        super().__init__()
        base = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.drop     = nn.Dropout(0.5)
        self.fc       = nn.Linear(base.classifier[1].in_features, n)
    def forward(self,x):
        x=self.features(x); x=self.pool(x); x=torch.flatten(x,1); x=self.drop(x)
        return self.fc(x)

class Res50(nn.Module):
    def __init__(self, n):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_f = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.fc       = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, n))
    def forward(self,x):
        feats = self.backbone(x)
        return self.fc(feats)

class MobV3(nn.Module):
    def __init__(self, n):
        super().__init__()
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        base.classifier[-1] = nn.Identity()  # get 1280-d features
        self.backbone = base
        self.fc       = nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, n))
    def forward(self,x):
        feats = self.backbone(x)
        return self.fc(feats)

BUILD = {
    'efficientnet_b2': EffB2,
    'resnet50'      : Res50,
    'mobilenet_v3'  : MobV3
}

def cam_layer(name, model):
    if name == 'efficientnet_b2':
        return [model.features[-1]]
    if name == 'resnet50':
        return [model.backbone.layer4[-1]]
    return [model.backbone.features[-1]]  # MobileNet-V3

# ---------------------- helpers ------------------------------------------ #
def read_classes(arg: str | None):
    """Return list of class names from file path or CSV string; None if not provided."""
    if not arg:
        return None
    candidate = arg.strip().strip('"').strip("'")
    if os.path.exists(candidate):
        with open(candidate, 'r', encoding='utf-8') as f:
            names = [ln.strip() for ln in f if ln.strip()]
        return names if names else None
    parts = [p.strip() for p in candidate.split(',') if p.strip()]
    return parts if parts else None

def infer_num_classes(state: dict) -> int | None:
    """Prefer final classifier heads; avoid picking classifier.0.weight (1280x960 in MobileNet)."""
    # Flatten nested state_dict
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']

    if not isinstance(state, dict):
        return None
    # Priority-ordered candidate keys
    preferred = [
        r'^fc\.weight$',
        r'^fc\.1\.weight$',
        r'^classifier\.3\.weight$',   # final head in many torchvision nets
        r'^classifier\.1\.weight$',   # sometimes used
    ]
    for pat in preferred:
        for k, v in state.items():
            if re.search(pat, k) and isinstance(v, torch.Tensor) and v.ndim == 2:
                return int(v.shape[0])
    # As a fallback, consider 'fc' or 'classifier' weights with small out_features
    small = []
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and (k.endswith('fc.weight') or 'classifier' in k):
            small.append(int(v.shape[0]))
    if small:
        return int(min(small))
    return None

def build_and_load(model_name: str, ckpt_path: str, class_names: list[str] | None):
    device = torch.device('cpu')
    raw = torch.load(ckpt_path, map_location=device)
    n_infer = infer_num_classes(raw)
    if n_infer is None:
        n_infer = len(class_names) if class_names else len(DEFAULT_CLASSES)
        print(f'⚠ Could not infer num_classes from checkpoint; using {n_infer}.')

    # Reconcile class names
    if class_names is None:
        class_names = DEFAULT_CLASSES.copy()
        # pad/truncate to n_infer
        if len(class_names) != n_infer:
            if len(class_names) > n_infer:
                class_names = class_names[:n_infer]
            else:
                class_names += [f'Class_{i}' for i in range(len(class_names), n_infer)]
        print('✓ Using default class names:', class_names)
    elif len(class_names) != n_infer:
        old = len(class_names)
        if old > n_infer:
            class_names = class_names[:n_infer]
            print(f'⚠ Truncated class list from {old} to {n_infer} to match checkpoint.')
        else:
            class_names = class_names + [f'Class_{i}' for i in range(old, n_infer)]
            print(f'⚠ Padded class list from {old} to {n_infer} to match checkpoint.')
        print('✓ Using class names:', class_names)
    else:
        print('✓ Using class names:', class_names)

    model = BUILD[model_name](len(class_names)).to(device)

    state = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw and isinstance(raw['state_dict'], dict) else raw
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  ⚠ Missing keys: {len(missing)}')
    if unexpected:
        print(f'  ⚠ Unexpected keys: {len(unexpected)}')

    return model, class_names

def list_images(test_dir: str, exts: list[str]):
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(test_dir, f'*.{e}')))
        paths.extend(glob.glob(os.path.join(test_dir, f'*.{e.upper()}')))
    return sorted(set(paths))

def run_one_model(m_name: str, ckpt_path: str, class_names_arg, test_dir: str,
                  out_dir: str, img_size: int = 224, no_cam: bool = False):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f'❌ {m_name}: checkpoint not provided or not found – skipping.')
        return

    device = torch.device('cpu')
    torch.backends.cudnn.enabled = False
    warnings.filterwarnings('ignore', category=UserWarning)

    # Build model & class names resolved to checkpoint
    model, class_names = build_and_load(m_name, ckpt_path, class_names_arg)
    idx2cls = {i: c for i, c in enumerate(class_names)}
    model.eval()

    cam = None
    if not no_cam:
        cam = GradCAMPlusPlus(model=model, target_layers=cam_layer(m_name, model))

    # NO RESIZE: keep original size
    tfm = A.Compose([A.Normalize(), ToTensorV2()])

    img_paths = list_images(test_dir, exts=['jpg','jpeg','png'])
    if not img_paths:
        print(f'⚠ {m_name}: No images found in: {test_dir}')
        return

    print(f'▶ {m_name}: {len(img_paths)} images | ckpt: {ckpt_path}')
    rows, runtimes = [], []
    t0 = time.perf_counter()

    for pth in img_paths:
        name = Path(pth).name
        try:
            orig = Image.open(pth).convert('RGB')
            rgb  = np.asarray(orig).astype(np.float32) / 255.0
            tensor = tfm(image=np.asarray(orig))['image'].unsqueeze(0)

            t_start = time.perf_counter()
            with torch.inference_mode():
                logits = model(tensor)
                probs  = torch.softmax(logits, 1).squeeze()
                idx    = int(torch.argmax(probs))
                pred   = idx2cls[idx]
                conf   = float(probs[idx])

            cam_img = None
            if cam is not None:
                grayscale = cam(tensor, targets=[ClassifierOutputTarget(idx)])[0]
                grayscale = ndi.gaussian_filter(grayscale, sigma=1)
                cam_img   = show_cam_on_image(rgb, grayscale, use_rgb=True)

            runtime_ms = (time.perf_counter() - t_start) * 1e3
            runtimes.append(runtime_ms)

            # Plot & save
            if cam_img is not None:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(orig);    ax[0].set_title('Original'); ax[0].axis('off')
                ax[1].imshow(cam_img); ax[1].set_title(pred);       ax[1].axis('off')
            else:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.imshow(orig); ax.set_title(pred); ax.axis('off')

            plt.tight_layout()
            save_name = f'{Path(pth).stem}_{m_name}.jpg'
            plt.savefig(os.path.join(out_dir, save_name), dpi=300)
            plt.close(fig)

            rows.append((name, pred, conf, runtime_ms))
            print(f'  ✓ {name:25s} → {pred:25s}  ({conf:6.2%})  {runtime_ms:7.1f} ms')

        except Exception as e:
            print(f'  ⚠ {name} skipped: {e}')

    total = time.perf_counter() - t0
    avg   = float(np.mean(runtimes)) if runtimes else 0.0
    print(f'⏱  {m_name}: total {total:5.2f} s | avg {avg:5.1f} ms / image\n')

    df = pd.DataFrame(rows, columns=['filename','prediction','prob','runtime_ms'])
    csv_path = os.path.join(out_dir, f'preds_{m_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f'✔ CSV saved: {csv_path}\n')

def main():
    parser = argparse.ArgumentParser(description='Windows-CPU Multi-Model Inference with Grad-CAM++ (no resize, auto class-detect)')
    parser.add_argument('--test_dir', type=str,
                        default=r'C:\Users\Hp\Desktop\Multi-model XAI\Blind test images')
    parser.add_argument('--out_dir', type=str,
                        default=r'C:\Users\Hp\Desktop\Multi-model XAI\Outputs')

    parser.add_argument('--eff_ckpt', type=str,
                        default=r'C:\Users\Hp\Desktop\Multi-model XAI\Results\kfold_training_outputs\efficientnet_b2_fold1.pth')
    parser.add_argument('--res_ckpt', type=str,
                        default=r'C:\Users\Hp\Desktop\Multi-model XAI\Results\kfold_training_outputs\resnet50_fold1.pth')
    parser.add_argument('--mob_ckpt', type=str,
                        default=r'C:\Users\Hp\Desktop\Multi-model XAI\Results\kfold_training_outputs\mobilenet_v3_fold1.pth')

    parser.add_argument('--classes', type=str, default='',
                        help='Optional: comma-separated list or path to a text file (one class per line).')
    parser.add_argument('--no_cam', action='store_true', help='Skip Grad-CAM generation.')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # If user provided classes, use them; otherwise use the DEFAULT_CLASSES above
    user_classes = read_classes(args.classes)
    if user_classes is None:
        class_names_arg = DEFAULT_CLASSES.copy()
    else:
        class_names_arg = user_classes

    run_one_model('efficientnet_b2', args.eff_ckpt, class_names_arg, args.test_dir, args.out_dir, no_cam=args.no_cam)
    run_one_model('resnet50',       args.res_ckpt, class_names_arg, args.test_dir, args.out_dir, no_cam=args.no_cam)
    run_one_model('mobilenet_v3',   args.mob_ckpt, class_names_arg, args.test_dir, args.out_dir, no_cam=args.no_cam)

    print(f'🎉 All done. Outputs are in: {args.out_dir}')

if __name__ == '__main__':
    main()
