#!/usr/bin/env python3
r"""
Windows-CPU batch inference with Grad-CAM++ + XAI evaluation
-----------------------------------------------------------
✅ PyTorch 2.6: torch.load(..., weights_only=False) for trusted checkpoints
✅ Avoid torchvision downloads: weights=None
✅ Practical CPU speed: run model/CAM at --input_size (default 224), then upsample CAM to original for saving
✅ Saves (Original | Grad-CAM++) per image + per-model CSV (filename, prediction, prob, runtime_ms)

NEW (optional):
✅ XAI Faithfulness (Deletion test): mask top-k CAM pixels, measure prob drop
✅ XAI Stability: compare CAM maps under perturbations (SSIM if available + Pearson)

Outputs:
- preds_<model>.csv
- (if --xai_eval) xai_<model>_deletion.csv, xai_<model>_stability.csv
- (if --xai_eval) xai_<model>_deletion_summary.png, xai_<model>_stability_summary.png

Run:
python infer_xai.py --xai_eval
"""

import os, glob, time, warnings, argparse, re, traceback
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
import torch.nn.functional as F

from torchvision.models import (
    efficientnet_b2,
    resnet50,
    mobilenet_v3_large,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2  # NEW

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Optional SSIM
try:
    from skimage.metrics import structural_similarity as sk_ssim
except Exception:
    sk_ssim = None


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
        base = efficientnet_b2(weights=None)  # IMPORTANT: no download
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.drop     = nn.Dropout(0.5)
        self.fc       = nn.Linear(base.classifier[1].in_features, n)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.fc(x)


class Res50(nn.Module):
    def __init__(self, n):
        super().__init__()
        base = resnet50(weights=None)  # IMPORTANT: no download
        in_f = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.fc       = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, n))

    def forward(self, x):
        feats = self.backbone(x)
        return self.fc(feats)


class MobV3(nn.Module):
    def __init__(self, n):
        super().__init__()
        base = mobilenet_v3_large(weights=None)  # IMPORTANT: no download
        base.classifier[-1] = nn.Identity()  # get 1280-d features
        self.backbone = base
        self.fc       = nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, n))

    def forward(self, x):
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
    if not arg:
        return None
    candidate = arg.strip().strip('"').strip("'")
    if os.path.exists(candidate):
        with open(candidate, 'r', encoding='utf-8') as f:
            names = [ln.strip() for ln in f if ln.strip()]
        return names if names else None
    parts = [p.strip() for p in candidate.split(',') if p.strip()]
    return parts if parts else None


def safe_torch_load(ckpt_path: str, device: torch.device):
    # PyTorch 2.6+ changed default weights_only=True; for your trusted checkpoints use False
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def extract_state_dict(raw):
    if isinstance(raw, nn.Module):
        return raw.state_dict()

    if isinstance(raw, dict):
        if 'state_dict' in raw and isinstance(raw['state_dict'], dict):
            return raw['state_dict']
        for k in ['model', 'net', 'weights', 'params']:
            if k in raw and isinstance(raw[k], dict):
                return raw[k]
        tensor_like = sum(1 for v in raw.values() if isinstance(v, torch.Tensor))
        if tensor_like >= max(1, int(0.3 * len(raw))):
            return raw

    return None


def infer_num_classes(state: dict) -> int | None:
    if not isinstance(state, dict):
        return None

    preferred = [
        r'^fc\.weight$',
        r'^fc\.1\.weight$',
        r'^classifier\.3\.weight$',
        r'^classifier\.1\.weight$',
    ]
    for pat in preferred:
        for k, v in state.items():
            if re.search(pat, k) and isinstance(v, torch.Tensor) and v.ndim == 2:
                return int(v.shape[0])

    small = []
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and (k.endswith('fc.weight') or 'classifier' in k):
            small.append(int(v.shape[0]))
    if small:
        return int(min(small))

    return None


def build_and_load(model_name: str, ckpt_path: str, class_names: list[str] | None):
    device = torch.device('cpu')

    raw = safe_torch_load(ckpt_path, device)
    state = extract_state_dict(raw)
    if state is None:
        raise RuntimeError(f"Could not extract state_dict from: {ckpt_path} (type={type(raw)})")

    n_infer = infer_num_classes(state)
    if n_infer is None:
        n_infer = len(class_names) if class_names else len(DEFAULT_CLASSES)
        print(f'⚠ Could not infer num_classes; using {n_infer}.', flush=True)

    if class_names is None:
        class_names = DEFAULT_CLASSES.copy()

    if len(class_names) != n_infer:
        old = len(class_names)
        if old > n_infer:
            class_names = class_names[:n_infer]
            print(f'⚠ Truncated class list from {old} to {n_infer}.', flush=True)
        else:
            class_names = class_names + [f'Class_{i}' for i in range(old, n_infer)]
            print(f'⚠ Padded class list from {old} to {n_infer}.', flush=True)

    print('✓ Using class names:', class_names, flush=True)

    model = BUILD[model_name](len(class_names)).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f'  ⚠ Missing keys: {len(missing)}', flush=True)
    if unexpected:
        print(f'  ⚠ Unexpected keys: {len(unexpected)}', flush=True)

    return model, class_names


def list_images(test_dir: str, exts: list[str]):
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(test_dir, f'*.{e}')))
        paths.extend(glob.glob(os.path.join(test_dir, f'*.{e.upper()}')))
    return sorted(set(paths))


def to_rgb_float01(pil_img: Image.Image) -> np.ndarray:
    return np.asarray(pil_img).astype(np.float32) / 255.0


def resize_pil(pil_img: Image.Image, size: int) -> Image.Image:
    return pil_img.resize((size, size), Image.BILINEAR)


# ===========================
# XAI Evaluation (NEW)
# ===========================
def _normalize_0_1(x: np.ndarray, eps=1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + eps)

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    if sk_ssim is None:
        return float("nan")
    a = _normalize_0_1(a)
    b = _normalize_0_1(b)
    return float(sk_ssim(a, b, data_range=1.0))

def _perturbations(rgb_uint8: np.ndarray) -> dict:
    """Mild perturbations to test stability of explanations."""
    out = {}
    x = rgb_uint8.astype(np.float32)

    # brightness +/-10%
    out["bright_plus10"]  = np.clip(x * 1.10, 0, 255).astype(np.uint8)
    out["bright_minus10"] = np.clip(x * 0.90, 0, 255).astype(np.uint8)

    # gaussian noise
    noise = np.random.normal(0, 6.0, size=rgb_uint8.shape).astype(np.float32)
    out["noise_sigma6"] = np.clip(x + noise, 0, 255).astype(np.uint8)

    # blur
    out["blur_sigma2"] = cv2.GaussianBlur(rgb_uint8, (0, 0), sigmaX=2, sigmaY=2)

    # small shift
    M = np.float32([[1, 0, 5], [0, 1, 5]])
    out["shift_5px"] = cv2.warpAffine(
        rgb_uint8, M, (rgb_uint8.shape[1], rgb_uint8.shape[0]),
        borderMode=cv2.BORDER_REFLECT
    )

    return out

def xai_deletion_test(model, tfm, cam_obj, pil_model: Image.Image, class_idx: int,
                      fracs=(0.05, 0.10, 0.20, 0.30), mask_mode="blur"):
    """
    Faithfulness: mask top-k CAM pixels and measure drop in prob(target class).
    Runs on model input size (pil_model).
    """
    model.eval()
    rgb_uint8 = np.asarray(pil_model)  # HxWx3 uint8 (resized)
    rgb_float = rgb_uint8.astype(np.float32) / 255.0

    tensor = tfm(image=rgb_uint8)['image'].unsqueeze(0)

    # baseline prob
    with torch.no_grad():
        probs = torch.softmax(model(tensor), 1).squeeze()
        prob_base = float(probs[class_idx].item())

    # base heatmap for the target class
    heat = cam_obj(tensor, targets=[ClassifierOutputTarget(class_idx)])[0]
    heat = ndi.gaussian_filter(heat, sigma=1)
    heat = _normalize_0_1(heat)

    H, W = heat.shape
    flat = heat.reshape(-1)
    order = np.argsort(-flat)  # descending importance
    N = H * W

    # mask reference
    if mask_mode == "blur":
        blurred = cv2.GaussianBlur(rgb_uint8, (0, 0), sigmaX=7, sigmaY=7)
    else:
        blurred = np.zeros_like(rgb_uint8)

    rows = []
    for frac in fracs:
        k = int(max(1, min(N, round(frac * N))))
        mask = np.zeros((N,), dtype=np.uint8)
        mask[order[:k]] = 1
        mask = mask.reshape(H, W)

        mod = rgb_uint8.copy()
        mod[mask == 1] = blurred[mask == 1]

        t_mod = tfm(image=mod)['image'].unsqueeze(0)
        with torch.no_grad():
            probs_mod = torch.softmax(model(t_mod), 1).squeeze()
            prob_after = float(probs_mod[class_idx].item())

        drop_abs = prob_base - prob_after
        drop_rel = drop_abs / (prob_base + 1e-12)

        rows.append({
            "frac_masked": float(frac),
            "prob_before": float(prob_base),
            "prob_after": float(prob_after),
            "drop_abs": float(drop_abs),
            "drop_rel": float(drop_rel),
        })

    return rows

def xai_stability_test(model, tfm, cam_obj, pil_model: Image.Image, class_idx: int):
    """
    Stability: compute similarity between base CAM and perturbed CAM for SAME target class_idx.
    """
    model.eval()
    base_uint8 = np.asarray(pil_model)
    base_tensor = tfm(image=base_uint8)['image'].unsqueeze(0)

    base_heat = cam_obj(base_tensor, targets=[ClassifierOutputTarget(class_idx)])[0]
    base_heat = ndi.gaussian_filter(base_heat, sigma=1)

    # baseline preds
    with torch.no_grad():
        base_probs = torch.softmax(model(base_tensor), 1).squeeze()
        pred_base = int(torch.argmax(base_probs))
        prob_class_base = float(base_probs[class_idx].item())

    rows = []
    perts = _perturbations(base_uint8)

    for name, p_uint8 in perts.items():
        p_tensor = tfm(image=p_uint8)['image'].unsqueeze(0)

        # probability + pred under perturbation
        with torch.no_grad():
            p_probs = torch.softmax(model(p_tensor), 1).squeeze()
            pred_p = int(torch.argmax(p_probs))
            prob_class_p = float(p_probs[class_idx].item())

        p_heat = cam_obj(p_tensor, targets=[ClassifierOutputTarget(class_idx)])[0]
        p_heat = ndi.gaussian_filter(p_heat, sigma=1)

        rows.append({
            "perturbation": name,
            "ssim": _ssim(base_heat, p_heat),
            "pearson": _pearson(_normalize_0_1(base_heat), _normalize_0_1(p_heat)),
            "pred_base": pred_base,
            "pred_pert": pred_p,
            "pred_changed": int(pred_p != pred_base),
            "prob_class_base": prob_class_base,
            "prob_class_pert": prob_class_p,
        })

    return rows

def plot_deletion_summary(df: pd.DataFrame, save_path: str):
    g = df.groupby("frac_masked")["prob_after"].mean().reset_index()
    plt.figure()
    plt.plot(g["frac_masked"], g["prob_after"], marker="o")
    plt.xlabel("Fraction masked (top Grad-CAM++ pixels)")
    plt.ylabel("Mean P(target class)")
    plt.title("Deletion faithfulness (mean over images)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_stability_summary(df: pd.DataFrame, save_path: str):
    g = df.groupby("perturbation")[["ssim", "pearson"]].mean().reset_index()
    plt.figure(figsize=(11, 4))
    x = np.arange(len(g))
    plt.bar(x - 0.2, g["ssim"].fillna(0.0).values, width=0.4, label="SSIM (if available)")
    plt.bar(x + 0.2, g["pearson"].fillna(0.0).values, width=0.4, label="Pearson")
    plt.xticks(x, g["perturbation"], rotation=25, ha="right")
    plt.ylabel("Similarity (higher = more stable)")
    plt.title("Grad-CAM++ stability under perturbations (mean over images)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ---------------------- main runner -------------------------------------- #
def run_one_model(m_name: str, ckpt_path: str, class_names_arg, test_dir: str,
                  out_dir: str, input_size: int = 224, no_resize: bool = False, no_cam: bool = False,
                  xai_eval: bool = False, del_fracs=(0.05, 0.10, 0.20, 0.30), max_eval_images: int = 0):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f'❌ {m_name}: checkpoint not provided or not found – skipping.', flush=True)
        return

    device = torch.device('cpu')
    torch.backends.cudnn.enabled = False
    warnings.filterwarnings('ignore', category=UserWarning)

    model, class_names = build_and_load(m_name, ckpt_path, class_names_arg)
    idx2cls = {i: c for i, c in enumerate(class_names)}
    model.eval()

    cam = None
    if not no_cam:
        cam = GradCAMPlusPlus(model=model, target_layers=cam_layer(m_name, model))

    # Albumentations expects uint8 0..255 then normalizes
    tfm = A.Compose([A.Normalize(), ToTensorV2()])

    img_paths = list_images(test_dir, exts=['jpg', 'jpeg', 'png'])
    if not img_paths:
        print(f'⚠ {m_name}: No images found in: {test_dir}', flush=True)
        return

    print(f'\n▶ {m_name}: {len(img_paths)} images | ckpt: {ckpt_path}', flush=True)
    print(f'   CAM: {"OFF" if no_cam else "ON"} | no_resize={no_resize} | input_size={input_size}', flush=True)
    if xai_eval:
        print(f'   XAI_EVAL: ON | deletion_fracs={del_fracs} | max_eval_images={max_eval_images or "ALL"}', flush=True)
        if sk_ssim is None:
            print("   ⚠ SSIM not available (install scikit-image). Will use Pearson only.", flush=True)

    rows, runtimes = [], []
    del_rows_all, stab_rows_all = [], []

    t0 = time.perf_counter()

    # Optional: evaluate only a subset for speed
    eval_limit = max_eval_images if (xai_eval and max_eval_images and max_eval_images > 0) else None
    eval_count = 0

    for pth in img_paths:
        name = Path(pth).name
        try:
            orig = Image.open(pth).convert('RGB')

            # For speed: run model/CAM on resized image (default 224), but save overlay at original size
            if no_resize:
                pil_model = orig
            else:
                pil_model = resize_pil(orig, input_size)

            rgb_model = to_rgb_float01(pil_model)  # float01 resized
            img_uint8 = np.asarray(pil_model)      # uint8 resized
            tensor = tfm(image=img_uint8)['image'].unsqueeze(0)

            t_start = time.perf_counter()

            # Prediction (no gradients needed)
            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits, 1).squeeze()
                idx    = int(torch.argmax(probs))
                pred   = idx2cls[idx]
                conf   = float(probs[idx])

            cam_img_up = None
            grayscale = None

            if cam is not None:
                # Grad-CAM++ needs gradients internally
                grayscale = cam(tensor, targets=[ClassifierOutputTarget(idx)])[0]
                grayscale = ndi.gaussian_filter(grayscale, sigma=1)

                cam_model = show_cam_on_image(rgb_model, grayscale, use_rgb=True)

                # Upsample overlay to original size for saving
                cam_img_up = Image.fromarray(cam_model).resize(orig.size, Image.BILINEAR)

            runtime_ms = (time.perf_counter() - t_start) * 1e3
            runtimes.append(runtime_ms)

            # Plot & save (Original | CAM)
            if cam_img_up is not None:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(orig);       ax[0].set_title('Original'); ax[0].axis('off')
                ax[1].imshow(cam_img_up); ax[1].set_title(pred);      ax[1].axis('off')
            else:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.imshow(orig); ax.set_title(pred); ax.axis('off')

            plt.tight_layout()
            save_name = f'{Path(pth).stem}_{m_name}.jpg'
            plt.savefig(os.path.join(out_dir, save_name), dpi=300)
            plt.close(fig)

            rows.append((name, pred, conf, runtime_ms))
            print(f'  ✓ {name:25s} → {pred:25s}  ({conf:6.2%})  {runtime_ms:7.1f} ms', flush=True)

            # ------------- XAI Evaluation (NEW) -------------
            if xai_eval and (cam is not None):
                if eval_limit is None or eval_count < eval_limit:
                    # Faithfulness: deletion test on model-size image
                    drows = xai_deletion_test(
                        model=model, tfm=tfm, cam_obj=cam,
                        pil_model=pil_model,
                        class_idx=idx,
                        fracs=del_fracs,
                        mask_mode="blur"
                    )
                    for r in drows:
                        r.update({
                            "filename": name,
                            "model": m_name,
                            "class_idx": idx,
                            "class_name": pred,
                            "prob_pred": conf
                        })
                        del_rows_all.append(r)

                    # Stability under perturbations
                    srows = xai_stability_test(
                        model=model, tfm=tfm, cam_obj=cam,
                        pil_model=pil_model,
                        class_idx=idx
                    )
                    for r in srows:
                        r.update({
                            "filename": name,
                            "model": m_name,
                            "class_idx": idx,
                            "class_name": pred,
                            "prob_pred": conf
                        })
                        stab_rows_all.append(r)

                    eval_count += 1

        except Exception as e:
            print(f'  ⚠ {name} skipped: {e}', flush=True)

    total = time.perf_counter() - t0
    avg   = float(np.mean(runtimes)) if runtimes else 0.0
    print(f'⏱  {m_name}: total {total:5.2f} s | avg {avg:5.1f} ms / image', flush=True)

    # Predictions CSV
    df = pd.DataFrame(rows, columns=['filename', 'prediction', 'prob', 'runtime_ms'])
    csv_path = os.path.join(out_dir, f'preds_{m_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f'✔ CSV saved: {csv_path}', flush=True)

    # XAI CSVs + plots
    if xai_eval and (cam is not None) and (len(del_rows_all) or len(stab_rows_all)):
        if len(del_rows_all):
            df_del = pd.DataFrame(del_rows_all)
            del_csv = os.path.join(out_dir, f'xai_{m_name}_deletion.csv')
            df_del.to_csv(del_csv, index=False)
            print(f'✔ XAI deletion CSV saved: {del_csv}', flush=True)

            del_plot = os.path.join(out_dir, f'xai_{m_name}_deletion_summary.png')
            plot_deletion_summary(df_del, del_plot)
            print(f'✔ XAI deletion plot saved: {del_plot}', flush=True)

        if len(stab_rows_all):
            df_st = pd.DataFrame(stab_rows_all)
            st_csv = os.path.join(out_dir, f'xai_{m_name}_stability.csv')
            df_st.to_csv(st_csv, index=False)
            print(f'✔ XAI stability CSV saved: {st_csv}', flush=True)

            st_plot = os.path.join(out_dir, f'xai_{m_name}_stability_summary.png')
            plot_stability_summary(df_st, st_plot)
            print(f'✔ XAI stability plot saved: {st_plot}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Windows-CPU Multi-Model Inference with Grad-CAM++ + XAI evaluation')
    parser.add_argument('--test_dir', type=str, default=r'C:\Users\Hp\Desktop\Multi-model XAI\Blind test images')
    parser.add_argument('--out_dir',  type=str, default=r'C:\Users\Hp\Desktop\Multi-model XAI\Outputs1')

    parser.add_argument('--eff_ckpt', type=str, default=r'C:\Users\Hp\Desktop\Multi-model XAI\Results\kfold_training_outputs\efficientnet_b2_fold1.pth')
    parser.add_argument('--res_ckpt', type=str, default=r'C:\Users\Hp\Desktop\Multi-model XAI\Results\kfold_training_outputs\resnet50_fold1.pth')
    parser.add_argument('--mob_ckpt', type=str, default=r'C:\Users\Hp\Desktop\Multi-model XAI\Results\kfold_training_outputs\mobilenet_v3_fold1.pth')

    parser.add_argument('--classes', type=str, default='', help='Comma-separated list or path to a text file (one class per line).')
    parser.add_argument('--no_cam', action='store_true', help='Skip Grad-CAM generation (debug only).')

    # Speed/behavior controls
    parser.add_argument('--input_size', type=int, default=224, help='Model/CAM size (default 224).')
    parser.add_argument('--no_resize', action='store_true', help='Use original image size for model/CAM (VERY SLOW on CPU).')

    # NEW: XAI evaluation controls
    parser.add_argument('--xai_eval', action='store_true', help='Run XAI evaluation (deletion + stability).')
    parser.add_argument('--del_fracs', type=str, default="0.05,0.10,0.20,0.30",
                        help='Comma list of deletion fractions, e.g., "0.05,0.1,0.2,0.3"')
    parser.add_argument('--max_eval_images', type=int, default=0,
                        help='Limit number of images used for XAI eval (0=all). Useful on CPU.')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    user_classes = read_classes(args.classes)
    class_names_arg = user_classes if user_classes is not None else DEFAULT_CLASSES.copy()

    # Parse deletion fractions
    try:
        del_fracs = tuple(float(x.strip()) for x in args.del_fracs.split(",") if x.strip())
        if not del_fracs:
            del_fracs = (0.05, 0.10, 0.20, 0.30)
    except Exception:
        del_fracs = (0.05, 0.10, 0.20, 0.30)

    # Run each model and show any exceptions clearly
    for name, ckpt in [
        ('efficientnet_b2', args.eff_ckpt),
        ('resnet50', args.res_ckpt),
        ('mobilenet_v3', args.mob_ckpt),
    ]:
        try:
            run_one_model(
                name, ckpt, class_names_arg,
                args.test_dir, args.out_dir,
                input_size=args.input_size,
                no_resize=args.no_resize,
                no_cam=args.no_cam,
                xai_eval=args.xai_eval,
                del_fracs=del_fracs,
                max_eval_images=args.max_eval_images
            )
        except Exception:
            print(f'\n❌ {name} failed with an exception:', flush=True)
            traceback.print_exc()

    print(f'\n🎉 All done. Outputs are in: {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
