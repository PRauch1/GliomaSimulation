"""Probe preprocessing variants against the earlier per-patient evaluation (lesion masks).
Usage: python3 repro_probe2.py variant1,variant2 id1,id2"""
import os, sys, json, numpy as np, nibabel as nib, torch, SimpleITK as sitk, pandas as pd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get('LGG_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'lgg_regrowth')))
from safetensors.torch import load_file
from core.models import ResizeConvUNet3D_FiLM
from eval.eval_io import zscore_normalize, pad_to_multiple, unpad_volume, build_time_channel
from eval.eval_metrics import dice_score
torch.set_num_threads(2)
cfg = json.load(open(os.path.join(BASE, 'infer/model_config.json')))
m = ResizeConvUNet3D_FiLM(in_channels=3, out_channels=1, channels=tuple(cfg['channels']),
    cat_vocab_sizes=[len(cfg['vocabs'][k]) for k in cfg['cat_order']], meta_emb_dim=16, meta_dim=64, meta_dropout=0.0)
m.load_state_dict(load_file(os.path.join(BASE, 'infer/model.safetensors'))); m.eval()
pub = pd.read_csv(os.path.join(BASE, 'data/nc/per_followup_metrics_0.5_idh_histo_cortex.csv')); pub['id'] = pub.patient_id.str[8:].astype(int); pub = pub.set_index('id')
fv = pd.read_excel(os.path.join(BASE, 'data/new/feat_val.xlsx')); fv['pid'] = fv.id.str[8:].astype(int); fv = fv.set_index('pid')
def idx(cat, val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return 0
    return int(cfg['vocabs'][cat].get(str(val).strip(), 0))
R = os.path.join(BASE, 'data/ucsf/raw')
def n4(path):
    im = sitk.ReadImage(path, sitk.sitkFloat32); mask = sitk.Cast(im > 0, sitk.sitkUInt8)
    sh = sitk.Shrink(im, [4] * 3); ms = sitk.Shrink(mask, [4] * 3)
    c = sitk.N4BiasFieldCorrectionImageFilter(); c.Execute(sh, ms)
    lb = c.GetLogBiasFieldAsImage(im); out = im / sitk.Exp(lb)
    return np.transpose(sitk.GetArrayFromImage(out), (2, 1, 0)) * (np.asanyarray(nib.load(path).dataobj) != 0)
variants = sys.argv[1].split(','); ids = [int(x) for x in sys.argv[2].split(',')]
for i in ids:
    path = f'{R}/{i}/{i}_time1_t2.nii.gz'
    pathF = f'{R}/{i}/{i}_time1_flair.nii.gz'
    raw0 = np.asanyarray(nib.load(path).dataobj).astype(np.float32)
    s10 = np.asanyarray(nib.load(f'{R}/{i}/{i}_time1_seg.nii.gz').dataobj) > 0
    s20 = np.asanyarray(nib.load(f'{R}/{i}/{i}_time2_seg.nii.gz').dataobj) > 0
    meta = [idx(c, fv.loc[i, c]) for c in cfg['cat_order']]; day = float(pub.loc[i, 'fu_day'])
    n4img = None
    for v in variants:
        base = np.asanyarray(nib.load(pathF).dataobj).astype(np.float32) if 'flair' in v else raw0
        if 'n4' in v:
            if n4img is None: n4img = n4(pathF if 'flair' in v else path).astype(np.float32)
            base = n4img
        ax = tuple(int(c) for c in v.split('_')[0].replace('flip', '')) if v.startswith('flip') else ()
        f = (lambda a: np.ascontiguousarray(np.flip(a, ax))) if ax else (lambda a: a)
        img = zscore_normalize(f(base)); s1 = f(s10); s2 = f(s20)
        t = build_time_channel(img, day, 4536.0)
        x, pads = pad_to_multiple(np.stack([img, s1.astype(np.float32), t], 0).astype(np.float32), 8)
        with torch.no_grad(): p = torch.sigmoid(m(torch.from_numpy(x[None]), torch.tensor([meta]))).numpy()
        p = unpad_volume(p, pads); P = np.maximum(p, s1) >= 0.5
        print(f'{v:12s} {i} raw {int((p>=0.5).sum()):7d} (orig {int(pub.loc[i,"vol_pred_raw_mm3"]):7d}) enf {int(P.sum()):7d} (orig {int(pub.loc[i,"vol_pred_enf_mm3"]):7d}) '
              f'dice {dice_score(P, s2):.3f} (orig {pub.loc[i,"dice_enf"]:.3f}) rawdice {dice_score(p>=0.5, s2):.3f} (orig {pub.loc[i,"dice_raw"]:.3f})', flush=True)
