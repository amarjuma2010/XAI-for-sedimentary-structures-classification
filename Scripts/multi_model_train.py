# MULTI-MODEL TRAINING & COMPARISON (k-Fold) – EfficientNet-B2, ResNet-50, MobileNet-V3
# -----------------------------------------------------------------------------#
# Stratified k-fold CV (k = 5). Produces per-fold & overall confusion matrices,
# CSV reports, comparison plots, and summary JSONs.
# -----------------------------------------------------------------------------#

import os, random, xmltodict, warnings, json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight

from torchvision.models import (efficientnet_b2, EfficientNet_B2_Weights,
                                 resnet50, ResNet50_Weights,
                                 mobilenet_v3_large, MobileNet_V3_Large_Weights)

import albumentations as A
from albumentations.pytorch import ToTensorV2

# --------------------- CONFIG -------------------------------------------------
BASE_PATH = '../train'
ANNOT_DIR = os.path.join(BASE_PATH, 'Annotations')
IMG_DIR   = os.path.join(BASE_PATH, 'JPEGImages')
CSV_PATH  = os.path.join(BASE_PATH, 'image_labels.csv')
OUT_DIR   = os.path.join(BASE_PATH, 'kfold_training_outputs')
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDED_CLASSES = {'Fissile shale', 'Flaser bedding', 'Rip-up',
                    'Soft-sediment deformation', 'current ripple'}

K_FOLDS  = 5
EPOCHS   = 60
BATCH_SZ = 32
PATIENCE = 5
SEED     = 42
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
warnings.filterwarnings('ignore', category=UserWarning)
torch.backends.cudnn.benchmark = True
# -----------------------------------------------------------------------------


# --------------------- DATA ---------------------------------------------------
def load_labels():
    """Parse XMLs once or read cached CSV."""
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        records = []
        for xml_file in tqdm(os.listdir(ANNOT_DIR), desc='XML'):
            try:
                with open(os.path.join(ANNOT_DIR, xml_file)) as fd:
                    doc = xmltodict.parse(fd.read())
                fn   = doc['annotation']['filename']
                obj  = doc['annotation'].get('object', None)
                if obj is None:
                    continue
                cls = obj[0]['name'] if isinstance(obj, list) else obj['name']
                records.append([fn, cls])
            except Exception:
                continue
        df = pd.DataFrame(records, columns=['filename', 'class'])
        df.to_csv(CSV_PATH, index=False)

    df = df[~df['class'].isin(EXCLUDED_CLASSES)].dropna().reset_index(drop=True)
    if df.empty:
        raise RuntimeError('No data after filtering')
    return df


DF      = load_labels()
CLASSES = sorted(DF['class'].unique())
N_CLASS = len(CLASSES)
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

T_TRAIN = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
T_VAL   = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])


class ImageDS(Dataset):
    def __init__(self, df, tfm):
        self.df  = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y   = CLS2IDX[row['class']]
        img = Image.open(os.path.join(IMG_DIR, row['filename'])).convert('RGB')
        x   = self.tfm(image=np.array(img))['image']
        return x, y


# --------------------- MODELS -------------------------------------------------
class EffB2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base          = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.drop     = nn.Dropout(0.5)
        self.fc       = nn.Linear(base.classifier[1].in_features, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.fc(x)


class Res50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base         = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_feats     = base.fc.in_features
        base.fc      = nn.Identity()
        self.backbone = base
        self.fc       = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_feats, n_classes))

    def forward(self, x):
        return self.fc(self.backbone(x))


class MobV3(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base                 = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        base.classifier[-1]  = nn.Identity()
        self.backbone        = base
        self.fc              = nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, n_classes))

    def forward(self, x):
        return self.fc(self.backbone(x))


MODELS = {'efficientnet_b2': EffB2,
          'resnet50':        Res50,
          'mobilenet_v3':    MobV3}


# --------------------- TRAIN / VAL HELPERS -----------------------------------
def train_epoch(model, loader, crit, opt):
    model.train(); tot = 0.
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out  = model(x)
        loss = crit(out, y)
        loss.backward(); opt.step()
        tot += loss.item()
    return tot / len(loader)


def val_epoch(model, loader, crit):
    model.eval(); tot = 0.; y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x)
            loss = crit(out, y)
            tot += loss.item()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.argmax(out, 1).cpu().numpy())
    loss = tot / len(loader)
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average='macro')
    return loss, acc, f1, np.array(y_true), np.array(y_pred)


# --------------------- k-FOLD TRAINING ---------------------------------------
class_wts = torch.tensor(compute_class_weight('balanced',
                                              classes=np.unique(DF['class']),
                                              y=DF['class']),
                         dtype=torch.float).to(DEVICE)

history   = defaultdict(lambda: defaultdict(list))
fold_summ = defaultdict(list)

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

for name, Net in MODELS.items():
    print(f'\n==== {name.upper()} : {K_FOLDS}-Fold CV ====')
    all_true, all_pred = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(DF, DF['class']), 1):
        print(f'Fold {fold}/{K_FOLDS}')
        tr_df, va_df = DF.iloc[tr_idx], DF.iloc[va_idx]

        tr_ld = DataLoader(ImageDS(tr_df, T_TRAIN), batch_size=BATCH_SZ,
                           shuffle=True, num_workers=4, pin_memory=True)
        va_ld = DataLoader(ImageDS(va_df, T_VAL), batch_size=BATCH_SZ,
                           shuffle=False, num_workers=4, pin_memory=True)

        model = Net(N_CLASS).to(DEVICE)
        crit  = nn.CrossEntropyLoss(weight=class_wts, label_smoothing=0.1)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch   = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

        best, wait, curves = np.inf, 0, {'tr': [], 'vl': [], 'acc': [], 'f1': []}

        for ep in range(1, EPOCHS + 1):
            tr = train_epoch(model, tr_ld, crit, opt)
            vl, acc, f1, yt, yp = val_epoch(model, va_ld, crit)

            curves['tr'].append(tr); curves['vl'].append(vl)
            curves['acc'].append(acc); curves['f1'].append(f1)
            sch.step(vl)

            print(f'  Ep{ep:02}  tr={tr:.3f}  vl={vl:.3f}  acc={acc:.3f}  f1={f1:.3f}')
            if vl < best:
                best, wait = vl, 0
                torch.save(model.state_dict(), os.path.join(OUT_DIR, f'{name}_fold{fold}.pth'))
            else:
                wait += 1
                if wait >= PATIENCE:
                    break

        model.load_state_dict(torch.load(os.path.join(OUT_DIR, f'{name}_fold{fold}.pth')))
        _, _, _, yt, yp = val_epoch(model, va_ld, crit)
        all_true.extend(yt.tolist()); all_pred.extend(yp.tolist())

        # per-fold outputs
        cm = confusion_matrix(yt, yp)
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap='Blues', xticks_rotation=45)
        plt.title(f'{name.upper()} – Fold {fold}'); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'cm_{name}_fold{fold}.jpg'), dpi=300); plt.close()

        pd.DataFrame(classification_report(
            yt, yp, target_names=CLASSES, output_dict=True)).T.to_csv(
            os.path.join(OUT_DIR, f'report_{name}_fold{fold}.csv'))

        for k in curves:
            history[name][k].append(curves[k])
        fold_summ[name].append({'acc': curves['acc'][-1], 'f1': curves['f1'][-1]})

    # overall CM & report
    cm_tot = confusion_matrix(all_true, all_pred)
    ConfusionMatrixDisplay(cm_tot, display_labels=CLASSES).plot(cmap='Greens', xticks_rotation=45)
    plt.title(f'{name.upper()} – Overall'); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'cm_{name}_overall.jpg'), dpi=300); plt.close()

    pd.DataFrame(classification_report(
        all_true, all_pred, target_names=CLASSES, output_dict=True)).T.to_csv(
        os.path.join(OUT_DIR, f'report_{name}_overall.csv'))

    # summary JSON
    m_acc = np.mean([d['acc'] for d in fold_summ[name]]); s_acc = np.std([d['acc'] for d in fold_summ[name]])
    m_f1  = np.mean([d['f1'] for d in fold_summ[name]]); s_f1  = np.std([d['f1'] for d in fold_summ[name]])
    json.dump({'mean_acc': m_acc, 'std_acc': s_acc,
               'mean_f1': m_f1, 'std_f1': s_f1},
              open(os.path.join(OUT_DIR, f'summary_{name}.json'), 'w'), indent=2)


# --------------------- COMPARISON PLOTS --------------------------------------
def avg_curve(curves):
    max_len = max(len(c) for c in curves)
    pad     = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        pad[i, :len(c)] = c
    return np.nanmean(pad, axis=0)


COL = {'efficientnet_b2': 'tab:blue',
       'resnet50':        'tab:green',
       'mobilenet_v3':    'tab:orange'}

for metric, ylabel in [('vl', 'Loss'), ('acc', 'Accuracy'), ('f1', 'Macro F1')]:
    plt.figure(figsize=(10, 6))
    for m, hist in history.items():
        if metric == 'vl':
            plt.plot(avg_curve(hist['tr']), ':', color=COL[m], label=f'{m} train')
        plt.plot(avg_curve(hist[metric]), color=COL[m], label=m)
    plt.xlabel('Epoch'); plt.ylabel(ylabel)
    plt.title(f'Validation {ylabel} – average of {K_FOLDS} folds')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{ylabel.lower()}_comparison.jpg'), dpi=300); plt.close()

print(f'\n✅ Finished. Outputs in {OUT_DIR}')
