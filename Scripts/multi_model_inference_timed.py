#!/usr/bin/env python3
"""
Timed batch inference for EfficientNet-B2, ResNet-50 & MobileNet-V3
-------------------------------------------------------------------
✓ loads     <model>_fold<k>.pth
✓ saves     <img>_<model>_fold<k>.jpg   (Original | Grad-CAM++)
✓ writes    preds_<model>_fold<k>.csv   (filename, prediction, prob, runtime_ms)
✓ prints    total & average latency
"""

# ---------- std lib -------------------------------------------------------- #
import os, glob, time, warnings
from pathlib import Path

# ---------- scientific stack ---------------------------------------------- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as ndi

# ---------- torch / vision ------------------------------------------------- #
import torch, torch.nn as nn
from torchvision.models import (efficientnet_b2, EfficientNet_B2_Weights,
                                 resnet50,        ResNet50_Weights,
                                 mobilenet_v3_large, MobileNet_V3_Large_Weights)

# ---------- augmentations & CAM ------------------------------------------- #
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------------------- CONFIG -------------------------------------------- #
BASE_PATH  = '/home/hamad-alhajri/Desktop/EfficientNet B0/train'
CKPT_DIR   = os.path.join(BASE_PATH, 'kfold_training_outputs')

TEST_DIR   = '/home/hamad-alhajri/Desktop/EfficientNet B0/Blind test images'
OUT_DIR    = '/home/hamad-alhajri/Desktop/EfficientNet B0/BR_kfold5_batch_results_timedv2'
os.makedirs(OUT_DIR, exist_ok=True)

FOLD_CHOICE = {'efficientnet_b2': 5,
               'resnet50'      : 5,
               'mobilenet_v3'  : 5}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore', category=UserWarning)
torch.backends.cudnn.benchmark = True               # speed

# ---------------------- class list ---------------------------------------- #
LABELS_CSV = os.path.join(BASE_PATH, 'image_labels.csv')
EXCL       = {'Fissile shale','Flaser bedding','Rip-up',
              'Soft-sediment deformation','current ripple'}
df_lbl     = pd.read_csv(LABELS_CSV)
CLASS_NAMES= sorted(df_lbl[~df_lbl['class'].isin(EXCL)]['class'].unique())
IDX2CLS    = {i:c for i,c in enumerate(CLASS_NAMES)}
N_CLASS    = len(CLASS_NAMES)

# ---------------------- transforms ---------------------------------------- #
VAL_TFM = A.Compose([ A.Resize(224,224), A.Normalize(), ToTensorV2()])

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
    def forward(self,x): return self.fc(self.backbone(x))

class MobV3(nn.Module):
    def __init__(self, n):
        super().__init__()
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        base.classifier[-1] = nn.Identity()
        self.backbone = base
        self.fc       = nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, n))
    def forward(self,x): return self.fc(self.backbone(x))

BUILD = {'efficientnet_b2': EffB2,
         'resnet50'      : Res50,
         'mobilenet_v3'  : MobV3}

# ---------------------- CAM target layer helper --------------------------- #
def cam_layer(name, model):
    if name == 'efficientnet_b2':
        return [model.features[5]]
    if name == 'resnet50':
        return [model.backbone.layer4[-1]]
    return [model.backbone.features[13]]            # MobileNet-V3

# ---------------------- per-backbone routine ------------------------------ #
def run_model(m_name: str, fold: int):

    ckpt = Path(CKPT_DIR) / f'{m_name}_fold{fold}.pth'
    if not ckpt.exists():
        print(f'❌ {ckpt.name} missing – skipping {m_name}')
        return

    # build & load
    model = BUILD[m_name](N_CLASS).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
    model.eval()

    cam = GradCAMPlusPlus(model=model,
                          target_layers=cam_layer(m_name, model))

    img_paths = sorted(glob.glob(os.path.join(TEST_DIR, '*.[jp][pn]g')))
    rows, runtimes = [], []

    print(f'▶ {m_name}  (fold {fold})  –  {len(img_paths)} images')
    t0 = time.perf_counter()

    for pth in img_paths:
        name = Path(pth).name
        try:
            # ---------- load & preprocess ------------------------------------ #
            orig = Image.open(pth).convert('RGB').resize((224,224))
            rgb  = np.asarray(orig).astype(np.float32) / 255.0
            tensor = VAL_TFM(image=np.asarray(orig))['image'].unsqueeze(0).to(DEVICE)

            # ---------- forward + Grad-CAM + timing ------------------------- #
            t_start = time.perf_counter()

            logits = model(tensor)               # keep grads enabled
            probs  = torch.softmax(logits.detach(),1).squeeze()
            idx    = int(torch.argmax(probs))
            pred   = IDX2CLS[idx]; conf = float(probs[idx])

            grayscale = cam(tensor,
                            targets=[ClassifierOutputTarget(idx)])[0]
            grayscale = ndi.gaussian_filter(grayscale, sigma=1)
            cam_img   = show_cam_on_image(rgb, grayscale, use_rgb=True)

            runtime_ms = (time.perf_counter() - t_start) * 1e3
            runtimes.append(runtime_ms)

            # ---------- save visual ---------------------------------------- #
            fig, ax = plt.subplots(1,2,figsize=(8,4))
            ax[0].imshow(orig);    ax[0].set_title('Original'); ax[0].axis('off')
            ax[1].imshow(cam_img); ax[1].set_title(pred);       ax[1].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(
                OUT_DIR, f'{Path(pth).stem}_{m_name}_fold{fold}.jpg'),
                dpi=300)
            plt.close(fig)

            rows.append((name, pred, conf, runtime_ms))
            print(f'  ✓ {name:20s} → {pred:25s}  ({conf:5.2%})  {runtime_ms:7.1f} ms')

        except Exception as e:
            print(f'  ⚠ {name} skipped: {e}')

    total = time.perf_counter() - t0
    avg   = np.mean(runtimes) if runtimes else 0.0
    print(f'⏱  {m_name}:  total {total:5.2f} s   |   avg {avg:5.1f} ms / image\n')

    # ---------- CSV -------------------------------------------------------- #
    pd.DataFrame(rows, columns=['filename','prediction','prob','runtime_ms']) \
      .to_csv(os.path.join(OUT_DIR, f'preds_{m_name}_fold{fold}.csv'),
              index=False)
    print(f'✔ CSV saved: preds_{m_name}_fold{fold}.csv\n')

# ---------------------- MAIN ---------------------------------------------- #
if __name__ == '__main__':
    for key, fold in FOLD_CHOICE.items():
        run_model(key, fold)

    print(f'🎉  All done.  Outputs are in  {OUT_DIR}')
